# models/train.py

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import functions and configurations from other modules
from utils.helpers import print_step, print_done
from utils.plot_utils import (
    plot_and_save_violin_and_scatter,
    save_lulc_graphs,
    plot_single_model_feature_importance,
    plot_combined_feature_importance, # Used in run_training_pipeline for combined plots
    plot_combined_prediction_comparison # Used in run_training_pipeline for combined plots
)
from models.modeling import get_xgboost_regressor, get_random_forest_regressor
from models.model_utils import save_model_artifacts, save_model_evaluation_report
from config.settings import (
    FEATURES, LULC_LABEL_MAP, TEST_SIZE, RANDOM_STATE,
    TARGET, WEIGHTS_COLUMN # Ensure TARGET and WEIGHTS_COLUMN are imported
)


def train_and_evaluate_model(
    model_instance, model_name, X_train, y_train, w_train, X_test, y_test,
    features, save_dir, common_test_data_for_eval
):
    """
    Trains, evaluates, and saves results for a single machine learning model.

    Args:
        model_instance: The initialized model object (e.g., XGBRegressor, RandomForestRegressor).
        model_name (str): A descriptive name for the model (e.g., "XGBoost", "Random Forest").
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (log-transformed).
        w_train (pd.Series): Training sample weights.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target (log-transformed).
        features (list): List of feature names.
        save_dir (str): Directory to save model-specific results.
        common_test_data_for_eval (pd.DataFrame): DataFrame containing test features
                                                   and original scale actual flood depth for common plots.

    Returns:
        dict: A dictionary containing evaluation metrics, feature importances, and predictions.
    """
    start_time = time.time()
    print(f"\n--- Training {model_name} Model ---")
    os.makedirs(save_dir, exist_ok=True)

    print_step(f"Training {model_name} model")
    # Fit the model using training data and sample weights
    model_instance.fit(X_train, y_train, sample_weight=w_train)
    print_done()

    print_step(f"Making predictions for {model_name}")
    # Predict on test and train sets
    # Inverse transform log-transformed predictions back to original scale
    y_test_pred = np.expm1(model_instance.predict(X_test))
    y_test_actual = np.expm1(y_test) # Original scale actual values for evaluation and plotting
    y_train_pred = np.expm1(model_instance.predict(X_train))
    y_train_actual = np.expm1(y_train) # Original scale actual values for evaluation and plotting
    print_done()

    # --- Generate Model-Specific Plots ---
    # Scatter and Violin plot for test set predictions vs actual
    plot_and_save_violin_and_scatter(
        y_test_actual, y_test_pred, "Flood Depth (m)",
        os.path.join(save_dir, "violin_test.png"), model_name=f"{model_name} prediction"
    )
    # Scatter and Violin plot for OWP HAND vs actual flood depth (common visualization)
    # This plot is intended to show the baseline comparison of OWP HAND with actual flood depth.
    plot_and_save_violin_and_scatter(
        y_test_actual, common_test_data_for_eval["owp_hand_fim"], "Flood Depth (m)",
        os.path.join(save_dir, "owp_vs_flood.png"), model_name="OWP HAND vs Flood Depth"
    )

    # Prepare DataFrame for LULC-wise scatter plots
    df_eval_lulc = common_test_data_for_eval.copy()
    df_eval_lulc["actual_flood_depth"] = y_test_actual
    df_eval_lulc["predicted_flood_depth"] = y_test_pred
    save_lulc_graphs(df_eval_lulc, save_dir, model_name=model_name)

    # Feature importance plot for the current model
    # Safely plot feature importance only if the model has the attribute
    if hasattr(model_instance, 'feature_importances_'):
        plot_single_model_feature_importance(
            model_instance.feature_importances_, features,
            os.path.join(save_dir, "feature_importance.png"), model_name=model_name
        )
    else:
        print(f"Skipping feature importance plot for {model_name}: Model does not have 'feature_importances_' attribute.")

    print_step(f"Evaluating {model_name} model")
    # Calculate evaluation metrics for the test set
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = mean_squared_error(y_test_actual, y_test_pred, squared=False)
    r2_test = r2_score(y_test_actual, y_test_pred)

    # Calculate evaluation metrics for the training set
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = mean_squared_error(y_train_actual, y_train_pred, squared=False)
    r2_train = r2_score(y_train_actual, y_train_pred)
    print_done()

    # --- Save Model Info and Artifacts using model_utils ---
    # Call the new utility function to save the evaluation report
    save_model_evaluation_report(
        model_name=model_name,
        X_train_shape=X_train.shape,
        X_test_shape=X_test.shape,
        mae_train=mae_train, rmse_train=rmse_train, r2_train=r2_train,
        mae_test=mae_test, rmse_test=rmse_test, r2_test=r2_test,
        feature_importances=getattr(model_instance, 'feature_importances_', None), # Safely get importances
        feature_names=features,
        save_dir=save_dir
    )

    # Call the new utility function to save model artifacts (model and data splits)
    save_model_artifacts(model_instance, X_train, y_train, X_test, y_test, save_dir)

    print(f"\n✅ {model_name} training complete. All results saved to {save_dir}")
    print(f"⏱️ Total Time for {model_name}: {time.time() - start_time:.2f} seconds")

    # Return key results for combined visualizations/summary
    return {
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "feature_importances": getattr(model_instance, 'feature_importances_', None),
        "y_test_pred": y_test_pred,
        "y_test_actual": y_test_actual
    }

# --- Main Training Workflow Function ---

def run_training_pipeline(pickle_path, save_base):
    """
    Main pipeline to train and evaluate both XGBoost and Random Forest models.
    Generates individual and combined plots/statistics.

    Args:
        pickle_path (str): Path to the input pickle file containing the dataset.
        save_base (str): Base directory for saving all results.
    """
    total_pipeline_start_time = time.time()
    print(f"\n🚀 Starting combined training pipeline using data from {pickle_path}")
    os.makedirs(save_base, exist_ok=True)

    print_step("Loading and preprocessing data")
    df = pd.read_pickle(pickle_path)
    print(f"Loaded {len(df)} rows.")

    # Filter data and create new features
    df = df[df['flood_depth'] > 0.01].copy()
    df["flood_depth_log"] = np.log1p(df["flood_depth"]) # Log transform target
    df["aspect_sin"] = np.sin(np.radians(df["aspect"]))
    df["aspect_cos"] = np.cos(np.radians(df["aspect"]))
    print_done()

    # Define features, target, and weights using settings from config
    X = df[FEATURES].fillna(df[FEATURES].mean()) # Fill NaNs with column means
    y = df[TARGET]
    weights = df[WEIGHTS_COLUMN] + 0.1 # Use original flood depth for weights, adding a small constant

    print_step("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print_done()

    # Store common test data that might be used for plots that are not model-specific
    common_test_data_for_eval = X_test.copy()

    results = {} # Dictionary to store results from each model

    # --- Train and Evaluate XGBoost Model ---
    xgboost_model = get_xgboost_regressor() # Get initialized model from models.modeling
    results["xgboost"] = train_and_evaluate_model(
        xgboost_model, "XGBoost", X_train, y_train, w_train, X_test, y_test,
        FEATURES, os.path.join(save_base, "xgboost_results"), common_test_data_for_eval
    )

    # --- Train and Evaluate Random Forest Model ---
    random_forest_model = get_random_forest_regressor() # Get initialized model from models.modeling
    results["random_forest"] = train_and_evaluate_model(
        random_forest_model, "Random Forest", X_train, y_train, w_train, X_test, y_test,
        FEATURES, os.path.join(save_base, "random_forest_results"), common_test_data_for_eval
    )

    # --- Generate Combined Visualizations ---
    print_step("Generating combined visualizations")
    combined_plots_dir = os.path.join(save_base, "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)

    # Combined Feature Importance Plot
    plot_combined_feature_importance(
        results["xgboost"]["feature_importances"],
        results["random_forest"]["feature_importances"],
        FEATURES,
        os.path.join(combined_plots_dir, "combined_feature_importance.png")
    )

    # Combined Prediction Comparison Plot (Scatter & Violin for both models)
    # y_test_actual is effectively the same for both as it comes from the same y_test split
    plot_combined_prediction_comparison(
        results["xgboost"]["y_test_actual"],
        results["xgboost"]["y_test_pred"],
        results["random_forest"]["y_test_pred"],
        "Flood Depth (m)",
        os.path.join(combined_plots_dir, "combined_prediction_comparison.png")
    )
    print_done()

    # --- Generate Combined Model Summary ---
    print_step("Generating combined model summary")
    summary_path = os.path.join(save_base, "combined_model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Combined Model Training Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write("Full Dataset (Inputs + Target) Description:\n")
        full_combined_data_desc = X.copy()
        full_combined_data_desc[TARGET] = y
        f.write(str(full_combined_data_desc.describe()) + "\n\n")

        f.write("--- XGBoost Results ---\n")
        f.write(f"Train MAE: {results['xgboost']['mae_train']:.4f}\n")
        f.write(f"Train RMSE: {results['xgboost']['rmse_train']:.4f}\n")
        f.write(f"Train R2 Score: {results['xgboost']['r2_train']:.4f}\n\n")
        f.write(f"Test MAE: {results['xgboost']['mae_test']:.4f}\n")
        f.write(f"Test RMSE: {results['xgboost']['rmse_test']:.4f}\n")
        f.write(f"Test R2 Score: {results['xgboost']['r2_test']:.4f}\n\n")

        f.write("--- Random Forest Results ---\n")
        f.write(f"Train MAE: {results['random_forest']['mae_train']:.4f}\n")
        f.write(f"Train RMSE: {results['random_forest']['rmse_train']:.4f}\n")
        f.write(f"Train R2 Score: {results['random_forest']['r2_train']:.4f}\n\n")
        f.write(f"Test MAE: {results['random_forest']['mae_test']:.4f}\n")
        f.write(f"Test RMSE: {results['random_forest']['rmse_test']:.4f}\n")
        f.write(f"Test R2 Score: {results['random_forest']['r2_test']:.4f}\n\n")

        f.write("Comparison of Test R2 Scores:\n")
        f.write(f"XGBoost R2: {results['xgboost']['r2_test']:.4f}\n")
        f.write(f"Random Forest R2: {results['random_forest']['r2_test']:.4f}\n\n")

    print_done()

    print(f"\n✅ Combined pipeline complete. All results saved to {save_base}")
    print(f"⏱️ Total Pipeline Time: {time.time() - total_pipeline_start_time:.2f} seconds")

# --- Main execution block ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost and Random Forest models.")
    parser.add_argument("--data", type=str, required=True, help="Path to input pickle file containing the dataset.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for saving all models, individual results, and combined results.")
    args = parser.parse_args()
    run_training_pipeline(args.data, args.save_dir)