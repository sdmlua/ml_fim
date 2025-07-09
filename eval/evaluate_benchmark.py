import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_pair(y_true, y_pred, label=""):
    """Return R², RMSE, MSE for predictions"""
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    return r2, rmse, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--benchmark", required=True, help="Directory containing benchmark data_*.pkl files")
    parser.add_argument("--target", required=True, help="Target column name (e.g., flood_depth)")
    parser.add_argument("--log_target", action="store_true", help="Use log1p target transformation (as used in training)")
    args = parser.parse_args()

    # Load trained model
    print(f"📦 Loading model from: {args.model}")
    model = joblib.load(args.model)

    # List benchmark files
    files = sorted([
        f for f in os.listdir(args.benchmark)
        if f.endswith(".pkl") and f.startswith("data_")
    ])

    if not files:
        print("⚠️ No benchmark data_*.pkl files found.")
        return

    print(f"\n🔍 Evaluating on {len(files)} benchmark files:\n")

    results = []

    for fname in files:
        path = os.path.join(args.benchmark, fname)
        try:
            df = joblib.load(path)

            y_true = df[args.target].copy()
            if args.log_target:
                y_true_log = np.log1p(y_true)
            else:
                y_true_log = y_true

            # Prepare model input
            X = df.drop(columns=[args.target, "huc8", "return_period"], errors='ignore')
            X = X.select_dtypes(include=[np.number])

            y_pred_model = model.predict(X)

            if args.log_target:
                y_true = np.expm1(y_true_log)
                y_pred_model = np.expm1(y_pred_model)

            # Baseline: raw OWP vs actual
            y_pred_owp = df["owp_hand_fim"].copy()

            # Metrics
            r2_m, rmse_m, mse_m = evaluate_pair(y_true, y_pred_model)
            r2_o, rmse_o, mse_o = evaluate_pair(y_true, y_pred_owp)

            results.append({
                "file": fname,
                "model_r2": r2_m,
                "model_rmse": rmse_m,
                "model_mse": mse_m,
                "owp_r2": r2_o,
                "owp_rmse": rmse_o,
                "owp_mse": mse_o
            })

            print(f"{fname:<18} | Model R²: {r2_m:.4f}, RMSE: {rmse_m:.4f} | OWP R²: {r2_o:.4f}, RMSE: {rmse_o:.4f}")

        except Exception as e:
            print(f"❌ Failed to evaluate {fname}: {e}")

    # Summary
    if results:
        df_res = pd.DataFrame(results)
        print("\n📊 Average Metrics Across All Benchmarks:")
        print(f"Model   → R²: {df_res['model_r2'].mean():.4f}, RMSE: {df_res['model_rmse'].mean():.4f}, MSE: {df_res['model_mse'].mean():.4f}")
        print(f"OWP-HAND→ R²: {df_res['owp_r2'].mean():.4f}, RMSE: {df_res['owp_rmse'].mean():.4f}, MSE: {df_res['owp_mse'].mean():.4f}")

        # Optional: Save metrics as CSV
        csv_path = os.path.join(args.benchmark, "benchmark_metrics.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"\n📝 Saved metrics to: {csv_path}")

if __name__ == "__main__":
    main()
