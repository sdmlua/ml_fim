# scripts/run_training.py

import os
import time
import pandas as pd
import numpy as np
import argparse

# Import helper functions for consistent output messages
from utils.helpers import print_step, print_done

# Import plotting utilities for combined visualizations
from utils.plot_utils import plot_combined_feature_importance, plot_combined_prediction_comparison

# Import model initialization function from the models directory
from models.modeling import get_xgboost_regressor, get_random_forest_regressor

# Import the core training and evaluation function from the models directory
from models.train_model import train_and_evaluate_model

# Import configuration settings
from config.settings import (
    FEATURES, TEST_SIZE, RANDOM_STATE, TARGET, WEIGHTS_COLUMN, LULC_LABEL_MAP # Ensure LULC_LABEL_MAP is imported if needed for data understanding/summary
)

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
    # Ensure the base save directory exists
    os.makedirs(save_base, exist_ok=True)

    print_step("Loading and preprocessing data")
    # Load the dataset from the specified pickle file
    df = pd.read_pickle(pickle_path)
    print(f"Loaded {len(df)} rows.")

    # Apply data filtering and create new features as defined in the original script
    df = df[df['flood_depth'] > 0.01].copy() # Filter out very small flood depths
    df["flood_depth_log"] = np.log1p(df["flood_depth"]) # Log transform the target variable
    df["aspect_sin"] = np.sin(np.radians(df["aspect"])) # Engineer sine of aspect feature
    df["aspect_cos"] = np.cos(np.radians(df["aspect"])) # Engineer cosine of aspect feature
    print_done()

    # Define features (X), target (y), and weights (weights) for modeling
    # Features are filled with their column means to handle any missing values
    X = df[FEATURES].fillna(df[FEATURES].mean())
    y = df[TARGET] # Use the target column defined in settings
    weights = df[WEIGHTS_COLUMN] + 0.1 # Use original flood depth for weights, adding a small constant to avoid zero weights
    
    print_step("Splitting data into training and test sets")
    # Split the data into training and testing sets, ensuring weights are also split consistently
    from sklearn.model_selection import train_test_split # Ensure this is imported here if not globally
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print_done()

    # Create a copy of the test features. This is passed to the train_and_evaluate_model
    # function to allow for plots that use original data (like LULC-wise plots)
    common_test_data_for_eval = X_test.copy()

    results = {} # Dictionary to store results from each model's training and evaluation

    # --- Train and Evaluate XGBoost Model ---
    # Initialize the XGBoost model using the function from models.modeling
    xgboost_model = get_xgboost_regressor()
    # Call the train_and_evaluate_model function for XGBoost
    results["xgboost"] = train_and_evaluate_model(
        model_instance=xgboost_model,
        model_name="XGBoost",
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_test=X_test,
        y_test=y_test,
        features=FEATURES,
        save_dir=os.path.join(save_base, "xgboost_results"), # Define output directory for XGBoost specific results
        common_test_data_for_eval=common_test_data_for_eval
    )

    # --- Train and Evaluate Random Forest Model ---
    # Initialize the Random Forest model using the function from models.modeling
    random_forest_model = get_random_forest_regressor()
    # Call the train_and_evaluate_model function for Random Forest
    results["random_forest"] = train_and_evaluate_model(
        model_instance=random_forest_model,
        model_name="Random Forest",
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_test=X_test,
        y_test=y_test,
        features=FEATURES,
        save_dir=os.path.join(save_base, "random_forest_results"), # Define output directory for Random Forest specific results
        common_test_data_for_eval=common_test_data_for_eval
    )

    # --- Generate Combined Visualizations ---
    print_step("Generating combined visualizations")
    combined_plots_dir = os.path.join(save_base, "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)

    # Plot combined feature importance for both models
    plot_combined_feature_importance(
        xgboost_importance=results["xgboost"]["feature_importances"],
        rf_importance=results["random_forest"]["feature_importances"],
        feature_names=FEATURES,
        save_path=os.path.join(combined_plots_dir, "combined_feature_importance.png")
    )

    # Plot combined prediction comparison (scatter and violin plots) for both models
    plot_combined_prediction_comparison(
        y_true_actual=results["xgboost"]["y_test_actual"], # Actual values are the same for both
        y_pred_xgboost=results["xgboost"]["y_test_pred"],
        y_pred_rf=results["random_forest"]["y_test_pred"],
        variable_name="Flood Depth (m)",
        save_path=os.path.join(combined_plots_dir, "combined_prediction_comparison.png")
    )
    print_done()

    # --- Generate Combined Model Summary ---
    print_step("Generating combined model summary")
    summary_path = os.path.join(save_base, "combined_model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Combined Model Training Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write("Full Dataset (Inputs + Target) Description:\n")
        # Combine features and target for a descriptive summary of the full dataset
        full_combined_data_desc = X.copy()
        full_combined_data_desc[TARGET] = y
        f.write(str(full_combined_data_desc.describe()) + "\n\n")

        # Write out XGBoost results
        f.write("--- XGBoost Results ---\n")
        f.write(f"Train MAE: {results['xgboost']['mae_train']:.4f}\n")
        f.write(f"Train RMSE: {results['xgboost']['rmse_train']:.4f}\n")
        f.write(f"Train R2 Score: {results['xgboost']['r2_train']:.4f}\n\n")
        f.write(f"Test MAE: {results['xgboost']['mae_test']:.4f}\n")
        f.write(f"Test RMSE: {results['xgboost']['rmse_test']:.4f}\n")
        f.write(f"Test R2 Score: {results['xgboost']['r2_test']:.4f}\n\n")

        # Write out Random Forest results
        f.write("--- Random Forest Results ---\n")
        f.write(f"Train MAE: {results['random_forest']['mae_train']:.4f}\n")
        f.write(f"Train RMSE: {results['random_forest']['rmse_train']:.4f}\n")
        f.write(f"Train R2 Score: {results['random_forest']['r2_train']:.4f}\n\n")
        f.write(f"Test MAE: {results['random_forest']['mae_test']:.4f}\n")
        f.write(f"Test RMSE: {results['random_forest']['rmse_test']:.4f}\n")
        f.write(f"Test R2 Score: {results['random_forest']['r2_test']:.4f}\n\n")

        # Comparative summary of R2 scores
        f.write("Comparison of Test R2 Scores:\n")
        f.write(f"XGBoost R2: {results['xgboost']['r2_test']:.4f}\n")
        f.write(f"Random Forest R2: {results['random_forest']['r2_test']:.4f}\n\n")

    print_done()

    print(f"\n✅ Combined pipeline complete. All results saved to {save_base}")
    print(f"⏱️ Total Pipeline Time: {time.time() - total_pipeline_start_time:.2f} seconds")

# --- Main execution block ---
# This block ensures that run_training_pipeline is called when the script is executed directly.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost and Random Forest models.")
    parser.add_argument("--data", type=str, required=True, help="Path to input pickle file containing the dataset.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for saving all models, individual results, and combined results.")
    args = parser.parse_args()
    run_training_pipeline(args.data, args.save_dir)

