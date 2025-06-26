import argparse
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input .pkl file')
    parser.add_argument('--output', required=True, help='Path to save trained model (.pkl)')
    parser.add_argument('--config', required=True, help='YAML config file with model parameters')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--log_target', action='store_true', help='Apply log1p transform to target')
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input}")
    df = joblib.load(args.input)
    df[args.target] = df[args.target] + 1e-5  # Avoid log(0)

    print("Splitting dataset (80% train, 20% validation)")
    y = df[args.target].copy()
    if args.log_target:
        print("Applying log1p transform to target")
        y = np.log1p(y)

    drop_cols = [args.target, 'huc8', 'return_period']
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])  # keep numeric features only

    print(f"Training columns ({len(X.columns)}): {list(X.columns)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Loading config from: {args.config}")
    params = load_config(args.config)

    print("Training XGBoost model...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    preds = model.predict(X_val)

    if args.log_target:
        preds = np.expm1(preds)
        y_val = np.expm1(y_val)

    # Replace NaNs with 0
    preds = np.nan_to_num(preds, nan=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0)

    rmse = np.sqrt(np.mean((y_val - preds) ** 2))
    r2 = r2_score(y_val, preds)

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")

    print(f"Saving model to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
