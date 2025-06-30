import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from glob import glob
import math

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Plot hexbin grid for test results.")
parser.add_argument('--root_dir', type=str, required=True, help="Root directory (e.g. data1)")
parser.add_argument('--output_file', type=str, default="hexbin_grid.png", help="Path to save combined plot")
parser.add_argument('--log_target', action='store_true', help="Unlog predictions and targets if trained on log1p")
parser.add_argument('--owp_plot', action='store_true', help="Plot flood_depth vs owp_hand_fim instead of predictions")
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
sns.set(style="ticks")
test_files = glob(os.path.join(args.root_dir, "**", "*_test.pkl"), recursive=True)
print(f"📂 Found {len(test_files)} test files.")

records = []

for path in test_files:
    try:
        df = joblib.load(path)

        required_columns = {"flood_depth", "owp_hand_fim"} if args.owp_plot else {"flood_depth", "predicted"}
        if not required_columns.issubset(df.columns):
            print(f"⚠️ Skipping {path}: missing {required_columns}")
            continue

        huc8 = os.path.basename(os.path.dirname(path))
        return_period = os.path.basename(path).replace("_test.pkl", "")

        if args.owp_plot:
            df = df[["flood_depth", "owp_hand_fim"]].copy()
        else:
            df = df[["flood_depth", "predicted"]].copy()

        df["huc8"] = huc8
        df["return_period"] = return_period

        if args.log_target:
            df["flood_depth"] = np.expm1(df["flood_depth"])
            if not args.owp_plot:
                df["predicted"] = np.expm1(df["predicted"])

        # Filter out very small values (optional)
        threshold = 0.01 if not args.owp_plot else 0.0
        df = df[df["flood_depth"] > threshold]

        if not df.empty:
            records.append(df)

    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")

if not records:
    print("❌ No valid data found.")
    exit()

df_all = pd.concat(records, ignore_index=True).dropna()
print(f"✅ Total rows for plotting: {len(df_all)}")

# -----------------------------
# Group and plot
# -----------------------------
grouped = df_all.groupby(["huc8", "return_period"])
n = len(grouped)
ncols = 3
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
axes = axes.flatten()

last_hb = None

for i, ((huc8, period), group) in enumerate(grouped):
    ax = axes[i]

    if args.owp_plot:
        x = group["owp_hand_fim"].values
        y = group["flood_depth"].values
        xlabel = "OWP HAND FIM"
        ylabel = "Flood Depth"
    else:
        x = group["predicted"].values
        y = group["flood_depth"].values
        xlabel = "Predicted Flood Depth"
        ylabel = "True Flood Depth"

    # Compute R² and RMSE directly (no regression fit)
    r2 = r2_score(y, x)
    rmse = np.sqrt(mean_squared_error(y, x))
    mse = mean_squared_error(y, x)

    hb = ax.hexbin(
        x,
        y,
        gridsize=50,
        cmap="viridis",
        mincnt=1,
        zorder=1
    )
    last_hb = hb

    # Optional: add regression line just for visual context
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, color="red", linewidth=2, zorder=2)
    except Exception:
        pass  # Skip line if regression fails (e.g. constant x)

    ax.set_title(f"HUC8: {huc8}, Period: {period}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    textstr = f"R² = {r2:.2f}\nRMSE = {rmse:.2f}\nMSE = {mse:.2f}"
    ax.text(
        0.95, 0.05, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
    )

    ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Colorbar and save
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
fig.colorbar(last_hb, cax=cbar_ax, label="counts")

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
plt.savefig(args.output_file, dpi=300)
print(f"✅ Saved hexbin grid to: {args.output_file}")
