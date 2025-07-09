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
parser = argparse.ArgumentParser(description="Plot hexbin grid for benchmark results with model predictions.")
parser.add_argument('--root_dir', type=str, required=True, help="Directory with data_*.pkl files")
parser.add_argument('--model', type=str, help="Trained model (.pkl) for prediction")
parser.add_argument('--output_file', type=str, default="hexbin_benchmark.png", help="Path to save hexbin plot")
parser.add_argument('--log_target', action='store_true', help="Apply expm1 to reverse log1p transform (only if model trained with log1p)")
parser.add_argument('--owp_plot', action='store_true', help="Plot vs OWP-HAND-FIM instead of model prediction")
parser.add_argument('--min_depth', type=float, default=0.0, help="Minimum flood depth to include (default 0.0 for consistency with CLI)")
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
sns.set(style="ticks")
benchmark_files = sorted(glob(os.path.join(args.root_dir, "data_*.pkl")))
print(f"📂 Found {len(benchmark_files)} benchmark files.")

# Load model if needed
model = None
if not args.owp_plot:
    if not args.model:
        raise ValueError("You must provide --model unless using --owp_plot")
    print(f"📦 Loading model from: {args.model}")
    model = joblib.load(args.model)

records = []

for path in benchmark_files:
    try:
        df = joblib.load(path)
        period = os.path.basename(path).replace("data_", "").replace(".pkl", "")

        required_cols = {"flood_depth", "owp_hand_fim"}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ Skipping {path}: missing columns {required_cols}")
            continue

        # Make predictions if needed
        if not args.owp_plot:
            if "predicted" not in df.columns:
                X = df.drop(columns=["flood_depth", "huc8", "return_period"], errors='ignore')
                X = X.select_dtypes(include=[np.number])
                preds = model.predict(X)
                if args.log_target:
                    preds = np.expm1(preds)
                df["predicted"] = preds

        # Apply expm1 if log-transformed
        if args.log_target:
            df["flood_depth"] = np.expm1(df["flood_depth"])

        # Apply optional depth filter
        if args.min_depth > 0:
            df = df[df["flood_depth"] > args.min_depth]

        df["return_period"] = period
        records.append(df)

    except Exception as e:
        print(f"❌ Failed to process {path}: {e}")

if not records:
    print("❌ No valid benchmark data found.")
    exit()

df_all = pd.concat(records, ignore_index=True).dropna()
print(f"✅ Total rows for plotting: {len(df_all)}")

# -----------------------------
# Group and plot
# -----------------------------
grouped = df_all.groupby("return_period")
n = len(grouped)
ncols = 3
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
axes = axes.flatten()
last_hb = None

for i, (period, group) in enumerate(grouped):
    ax = axes[i]

    if args.owp_plot:
        x = group["owp_hand_fim"].values
        xlabel = "OWP HAND FIM"
    else:
        x = group["predicted"].values
        xlabel = "Predicted Flood Depth"

    y = group["flood_depth"].values
    ylabel = "True Flood Depth"

    # Match CLI: replace NaN/Inf with 0
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute metrics
    r2 = r2_score(y, x)
    rmse = np.sqrt(mean_squared_error(y, x))
    mse = mean_squared_error(y, x)

    print(f"\n📊 Period {period} → R²: {r2:.4f}, RMSE: {rmse:.4f}, MSE: {mse:.4f}")

    hb = ax.hexbin(x, y, gridsize=50, cmap="viridis", mincnt=1, zorder=1)
    last_hb = hb

    # Optional regression line
    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, color="red", linewidth=2, zorder=2)
    except Exception:
        pass

    ax.set_title(f"Period: {period}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Annotate metrics
    metrics_text = f"R² = {r2:.2f}\nRMSE = {rmse:.2f}\nMSE = {mse:.2f}"
    ax.text(
        0.95, 0.05, metrics_text,
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

# Add colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
fig.colorbar(last_hb, cax=cbar_ax, label="counts")

# Save figure
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
plt.savefig(args.output_file, dpi=300)
print(f"✅ Saved hexbin plot to: {args.output_file}")
