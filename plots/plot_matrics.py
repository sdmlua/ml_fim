import os
import argparse
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import r2_score, mean_squared_error

def calculate_metrics(df_all):
    """
    Compute R² and RMSE for predicted and owp_hand_fim by HUC8 and return period.
    """
    metrics = []

    grouped = df_all.groupby(["huc8", "return_period"])
    for (huc8, period), group in grouped:
        y_true = group["flood_depth"].values

        if "predicted" in group:
            y_pred = group["predicted"].values
            r2_pred = r2_score(y_true, y_pred)
            rmse_pred = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics.append((huc8, period, "Predicted", r2_pred, rmse_pred))

        if "owp_hand_fim" in group:
            x_owp = group["owp_hand_fim"].values
            r2_owp = r2_score(y_true, x_owp)
            rmse_owp = np.sqrt(mean_squared_error(y_true, x_owp))
            metrics.append((huc8, period, "OWP_HAND", r2_owp, rmse_owp))

    return pd.DataFrame(metrics, columns=["huc8", "return_period", "Type", "R2", "RMSE"])


def plot_metrics(df_metrics, output_dir):
    """
    Plot bar charts for R² and RMSE metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_metrics["Label"] = df_metrics["huc8"] + "_" + df_metrics["return_period"]
    df_metrics.sort_values("Label", inplace=True)

    # R² plot
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_metrics, x="Label", y="R2", hue="Type")
    plt.title("R² Score by HUC8 and Return Period")
    plt.ylabel("R²")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison_r2.png"))
    print("✅ Saved R² bar plot.")

    # RMSE plot
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_metrics, x="Label", y="RMSE", hue="Type")
    plt.title("RMSE by HUC8 and Return Period")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison_rmse.png"))
    print("✅ Saved RMSE bar plot.")


def load_data(root_dir, log_target):
    """
    Load and merge all *_test.pkl files into a single DataFrame.
    """
    test_files = glob(os.path.join(root_dir, "**", "*_test.pkl"), recursive=True)
    print(f"📂 Found {len(test_files)} test files.")

    records = []

    for path in test_files:
        try:
            df = joblib.load(path)
            if not {"flood_depth"}.issubset(df.columns):
                continue

            huc8 = os.path.basename(os.path.dirname(path))
            return_period = os.path.basename(path).replace("_test.pkl", "")

            df = df.copy()
            df["huc8"] = huc8
            df["return_period"] = return_period

            if log_target:
                df["flood_depth"] = np.expm1(df["flood_depth"])
                if "predicted" in df.columns:
                    df["predicted"] = np.expm1(df["predicted"])

            df = df[df["flood_depth"] > 0.01]  # Optional filter
            records.append(df)
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")

    if not records:
        print("❌ No valid test data found.")
        exit(1)

    return pd.concat(records, ignore_index=True).dropna()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare R² and RMSE for predictions and OWP HAND.")
    parser.add_argument("--root_dir", type=str, required=True, help="Directory containing *_test.pkl files.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save output plots.")
    parser.add_argument("--log_target", action="store_true", help="Unlog flood_depth/predicted with expm1")
    args = parser.parse_args()

    df_all = load_data(args.root_dir, args.log_target)
    print(f"✅ Loaded total rows: {len(df_all)}")

    df_metrics = calculate_metrics(df_all)
    plot_metrics(df_metrics, args.output_dir)
