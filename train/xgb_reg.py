import argparse
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    max_error
)


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

    # Ensure target is numeric and in a valid range
    df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
    df[args.target] = np.minimum(df[args.target], 35) + 1e-5

    # Prepare target vector
    y = df[args.target].copy()
    if args.log_target:
        print("Applying log1p transform to target")
        y = np.log1p(y)

    # Drop unused columns and select numeric features
    drop_cols = [args.target, 'huc8', 'return_period']
    X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])

    print(f"Training columns ({X.shape[1]}): {list(X.columns)}")
    print("Splitting dataset (80% train, 20% validation)")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sanity check for NaNs
    assert X_train.isna().sum().sum() == 0, "X_train contains NaNs"
    assert X_val.isna().sum().sum() == 0,   "X_val contains NaNs"
    assert pd.isna(y_train).sum() == 0,     "y_train contains NaNs"
    assert pd.isna(y_val).sum() == 0,       "y_val contains NaNs"

    # Convert to numpy arrays
    Xtr, ytr = X_train.values, y_train.values
    Xvl, yvl = X_val.values,   y_val.values

    print(f"Loading config from: {args.config}")
    params = load_config(args.config)

    # Extract early stopping and metric from config, if present
    early_rounds = params.pop("early_stopping_rounds", None)
    eval_metric  = params.pop("eval_metric", None)

    # Inject them into the constructor so we don't pass unsupported kwargs to fit()
    if eval_metric is not None:
        params["eval_metric"] = eval_metric
    if early_rounds is not None:
        params["early_stopping_rounds"] = early_rounds

    print("Training XGBoost model...")
    model = xgb.XGBRegressor(**params)

    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xtr, ytr), (Xvl, yvl)],
        verbose=True
    )

    print("Evaluating model...")
    train_preds = model.predict(Xtr)
    val_preds   = model.predict(Xvl)

    # Invert log1p if applied
    if args.log_target:
        train_preds = np.expm1(train_preds)
        ytr         = np.expm1(ytr)
        val_preds   = np.expm1(val_preds)
        yvl         = np.expm1(yvl)

    # Clip any Inf/NaN
    train_preds = np.nan_to_num(train_preds, nan=0.0, posinf=0.0, neginf=0.0)
    val_preds   = np.nan_to_num(val_preds,   nan=0.0, posinf=0.0, neginf=0.0)

    # Compute metrics
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n--- Training Results ---")
    print(f"RMSE:       {rmse(ytr, train_preds):.4f}")
    print(f"R²:         {r2_score(ytr, train_preds):.4f}")
    print(f"MAE:        {mean_absolute_error(ytr, train_preds):.4f}")
    print(f"Max Error:  {max_error(ytr, train_preds):.4f}")

    print("\n--- Validation Results ---")
    print(f"RMSE:       {rmse(yvl, val_preds):.4f}")
    print(f"R²:         {r2_score(yvl, val_preds):.4f}")
    print(f"MAE:        {mean_absolute_error(yvl, val_preds):.4f}")
    print(f"Max Error:  {max_error(yvl, val_preds):.4f}")

    # Save the trained model
    print(f"\nSaving model to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
