import argparse
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
from pyswarm import pso


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Bounds for PSO optimization
param_bounds = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'n_estimators': (100, 500)
}


def pso_objective(params_list, Xtr, ytr, Xvl, yvl, fixed_params):
    keys = list(param_bounds.keys())
    tuned_params = dict(zip(keys, params_list))
    tuned_params['max_depth'] = int(tuned_params['max_depth'])
    tuned_params['n_estimators'] = int(tuned_params['n_estimators'])

    model_params = {**fixed_params, **tuned_params}
    model = xgb.XGBRegressor(**model_params)

    model.fit(Xtr, ytr)
    preds = model.predict(Xvl)
    score = mean_squared_error(yvl, preds)

    print(f"Tested params: {tuned_params}, MSE: {score:.4f}")
    return score


def optimize_with_pso(Xtr, ytr, Xvl, yvl, fixed_params):
    lb = [param_bounds[k][0] for k in param_bounds]
    ub = [param_bounds[k][1] for k in param_bounds]

    print("⚙️ Running Particle Swarm Optimization on reduced sample...")
    best_params, _ = pso(
        func=lambda p: pso_objective(p, Xtr, ytr, Xvl, yvl, fixed_params),
        lb=lb,
        ub=ub,
        swarmsize=8,       # Reduce swarm size for speed
        maxiter=6          # Fewer iterations for quicker convergence
    )

    keys = list(param_bounds.keys())
    best_params_dict = dict(zip(keys, best_params))
    best_params_dict['max_depth'] = int(best_params_dict['max_depth'])
    best_params_dict['n_estimators'] = int(best_params_dict['n_estimators'])

    return best_params_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input .pkl file')
    parser.add_argument('--output', required=True, help='Path to save trained model (.pkl)')
    parser.add_argument('--config', required=True, help='YAML config file with model parameters')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--log_target', action='store_true', help='Apply log1p transform to target')
    parser.add_argument('--pso_sample_frac', type=float, default=0.1, help='Fraction of data to use during PSO (default=0.1)')
    args = parser.parse_args()

    print(f"📂 Loading dataset: {args.input}")
    df = joblib.load(args.input)

    df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
    df[args.target] = np.minimum(df[args.target], 25.01)

    y = df[args.target].copy()
    if args.log_target:
        print("🔢 Applying log1p transform to target")
        y = np.log1p(y)

    drop_cols = [args.target, 'huc8', 'return_period']
    X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])

    print(f"🧠 Training features: {list(X.columns)}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sample a small portion for faster PSO
    Xtr = X_train.sample(frac=args.pso_sample_frac, random_state=42)
    ytr = y_train.loc[Xtr.index]
    Xvl = X_val.sample(frac=args.pso_sample_frac, random_state=42)
    yvl = y_val.loc[Xvl.index]

    print(f"📘 Loading model config: {args.config}")
    config_params = load_config(args.config)
    fixed_params = {k: v for k, v in config_params.items() if k not in param_bounds}

    early_rounds = fixed_params.pop("early_stopping_rounds", None)
    eval_metric = fixed_params.pop("eval_metric", None)

    tuned_params = optimize_with_pso(Xtr.values, ytr.values, Xvl.values, yvl.values, fixed_params)
    print("✅ Best parameters found:", tuned_params)

    # Combine fixed and tuned params
    final_params = {**fixed_params, **tuned_params}
    if eval_metric:
        final_params["eval_metric"] = eval_metric
    if early_rounds:
        final_params["early_stopping_rounds"] = early_rounds

    print("🚀 Training final XGBoost model...")
    model = xgb.XGBRegressor(**final_params)
    model.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values), (X_val.values, y_val.values)], verbose=True)

    print("📊 Evaluating model...")
    def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

    for name, Xd, yd in [('Train', X_train, y_train), ('Validation', X_val, y_val)]:
        preds = model.predict(Xd)
        if args.log_target:
            preds = np.expm1(preds)
            yd = np.expm1(yd)

        preds = np.nan_to_num(preds)
        print(f"\n📈 {name} Metrics:")
        print(f"  RMSE:       {rmse(yd, preds):.4f}")
        print(f"  R²:         {r2_score(yd, preds):.4f}")
        print(f"  MAE:        {mean_absolute_error(yd, preds):.4f}")
        print(f"  Max Error:  {max_error(yd, preds):.4f}")

    print(f"\n💾 Saving model to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print("✅ Done.")


if __name__ == "__main__":
    main()
