import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Helper Functions (re-used and slightly modified) ---

def print_step(message):
    """Prints a formatted step message."""
    print(f"\n🟡 {message}...")

def print_done():
    """Prints a formatted done message."""
    print("✅ Done!")

def plot_scatter_with_metrics(ax, y_true, y_pred, variable_name, title):
    """
    Plots a scatter plot of predicted vs actual values with regression lines and metrics.
    Can be used as a standalone plot or as a subplot within a larger figure.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        variable_name (str): Name of the target variable (e.g., "Flood Depth (m)").
        title (str): Title of the plot.
    """
    y_true_np = np.array(y_true).reshape(-1, 1)
    y_pred_np = np.array(y_pred)

    # Fit a linear regression line through the origin
    reg_origin = LinearRegression(fit_intercept=False)
    reg_origin.fit(y_true_np, y_pred_np)
    x_line = np.linspace(0, max(np.max(y_true), np.max(y_pred)), 100).reshape(-1, 1)
    y_line = reg_origin.predict(x_line)

    # Compute evaluation metrics
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = mean_squared_error(y_true_np, y_pred_np, squared=False)
    r2 = r2_score(y_true_np, y_pred_np)
    metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"

    # Plot scatter points, regression line, and ideal line
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, edgecolors='k', linewidths=0.2)
    ax.plot(x_line, y_line, color='red', linewidth=2, label='Fit from origin')
    ax.plot(x_line, x_line, color='green', linestyle='--', linewidth=2.5, label='Ideal: y = x')
    ax.set_title(title)
    ax.set_xlabel(f"Actual {variable_name}")
    ax.set_ylabel(f"Predicted {variable_name}")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    # Add metrics text to the plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    text_x = xlim[0] + 0.95 * (xlim[1] - xlim[0])
    text_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.text(text_x, text_y, metrics_text,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

def plot_and_save_violin_and_scatter(y_true, y_pred, variable_name, save_path=None, model_name="Model"):
    """
    Plots violin and scatter plots of predictions vs actuals and saves them as a single figure.
    This function creates a standalone figure with two subplots.

    Args:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        variable_name (str): Name of the target variable.
        save_path (str, optional): Full path to save the plot. If None, the plot will be shown.
        model_name (str): Name of the model for plot titles.
    """
    print_step(f"Plotting violin and scatter for {model_name} on {variable_name}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot scatter
    plot_scatter_with_metrics(ax1, y_true, y_pred, variable_name, f"{model_name} Prediction vs Actual (Scatter)")

    # Plot violin
    sns.violinplot(data=[y_true, y_pred], ax=ax2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Actual", "Predicted"])
    ax2.set_title(f"{model_name} Distribution of Actual vs Predicted")
    ax2.set_ylabel(variable_name)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print_done()
    else:
        plt.show()

def plot_single_model_feature_importance(importance, feature_names, save_path, model_name="Model"):
    """
    Plots and saves feature importance for a single model.

    Args:
        importance (array-like): Array of feature importances.
        feature_names (list): List of feature names corresponding to importance values.
        save_path (str): Full path to save the plot.
        model_name (str): Name of the model for plot title.
    """
    print_step(f"Plotting {model_name} feature importance")
    importance_percent = 100.0 * (importance / np.sum(importance))
    sorted_idx = np.argsort(importance_percent)[::-1]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        [feature_names[i] for i in sorted_idx],
        [importance_percent[i] for i in sorted_idx],
        color='skyblue', edgecolor='black'
    )

    plt.ylabel("Mean Decrease in Impurity (%)")
    plt.xlabel("Features")
    plt.title(f"{model_name} Feature Importances")
    plt.xticks(rotation=45, ha='right')

    # Add percentage labels on top of bars
    for bar, percent in zip(bars, importance_percent[sorted_idx]):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{percent:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print_done()

def save_lulc_graphs(df, save_dir, model_name="Model"):
    """
    Saves LULC-wise scatter plots for a given model's predictions.

    Args:
        df (pd.DataFrame): DataFrame containing 'lulc', 'actual_flood_depth', and 'predicted_flood_depth'.
        save_dir (str): Directory to save the plots.
        model_name (str): Name of the model for plot titles and filename prefix.
    """
    print_step(f"Saving LULC-wise scatter plots for {model_name}")

    lulc_label_map = {
        1: "Water", 2: "Trees / Forest", 3: "Grassland", 4: "Flooded Vegetation",
        5: "Crops", 6: "Shrubland", 7: "Built-up / Urban", 8: "Bare / Sparse Vegetation",
        9: "Snow / Ice", 10: "Clouds", 11: "Rangeland"
    }

    categories = sorted(df['lulc'].unique())
    cols = 3
    rows = int(np.ceil(len(categories) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, lulc_val in enumerate(categories):
        subset = df[df['lulc'] == lulc_val]
        y_true = subset["actual_flood_depth"].values
        y_pred = subset["predicted_flood_depth"].values
        title = lulc_label_map.get(lulc_val, f"LULC {lulc_val}")
        plot_scatter_with_metrics(axes[i], y_true, y_pred, "Flood Depth", f"{model_name}: {title}")

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_lulc_scatter_plots.png"))
    plt.close()
    print_done()

# --- New Plotting Functions for Combined Visualizations ---

def plot_combined_feature_importance(xgboost_importance, rf_importance, feature_names, save_path):
    """
    Plots combined feature importance for XGBoost and Random Forest.
    Shows two bars per feature (one for each model).

    Args:
        xgboost_importance (array-like): Feature importances from the XGBoost model.
        rf_importance (array-like): Feature importances from the Random Forest model.
        feature_names (list): List of feature names.
        save_path (str): Full path to save the plot.
    """
    print_step("Plotting combined feature importance")
    importance_xgboost_percent = 100.0 * (xgboost_importance / np.sum(xgboost_importance))
    importance_rf_percent = 100.0 * (rf_importance / np.sum(rf_importance))

    # Sort features by the average importance for consistent ordering
    avg_importance = (importance_xgboost_percent + importance_rf_percent) / 2
    sorted_idx = np.argsort(avg_importance)[::-1]

    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_xgboost_importance = [importance_xgboost_percent[i] for i in sorted_idx]
    sorted_rf_importance = [importance_rf_percent[i] for i in sorted_idx]

    x = np.arange(len(sorted_features))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(x - width/2, sorted_xgboost_importance, width, label='XGBoost Importance', color='teal', alpha=0.8)
    bars2 = plt.bar(x + width/2, sorted_rf_importance, width, label='Random Forest Importance', color='orange', alpha=0.8)

    plt.ylabel("Mean Decrease in Impurity (%)")
    plt.xlabel("Features")
    plt.title("Combined Feature Importances: XGBoost vs Random Forest")
    plt.xticks(x, sorted_features, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage values on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print_done()

def plot_combined_prediction_comparison(y_true_actual, y_pred_xgboost, y_pred_rf, variable_name, save_path):
    """
    Plots a combined figure comparing predictions from both models.
    Includes scatter plots and violin plots for both models in a 2x2 grid.

    Args:
        y_true_actual (array-like): Actual target values (same for both models).
        y_pred_xgboost (array-like): Predicted values from the XGBoost model.
        y_pred_rf (array-like): Predicted values from the Random Forest model.
        variable_name (str): Name of the target variable.
        save_path (str): Full path to save the plot.
    """
    print_step("Plotting combined prediction comparison")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # 2 rows for plot types, 2 columns for models

    # Top-left: XGBoost Scatter Plot
    plot_scatter_with_metrics(axes[0, 0], y_true_actual, y_pred_xgboost, variable_name, "XGBoost Prediction vs Actual")

    # Top-right: Random Forest Scatter Plot
    plot_scatter_with_metrics(axes[0, 1], y_true_actual, y_pred_rf, variable_name, "Random Forest Prediction vs Actual")

    # Bottom-left: XGBoost Violin Plot
    sns.violinplot(data=[y_true_actual, y_pred_xgboost], ax=axes[1, 0])
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Actual", "XGBoost Predicted"])
    axes[1, 0].set_title("XGBoost Distribution of Actual vs Predicted")
    axes[1, 0].set_ylabel(variable_name)

    # Bottom-right: Random Forest Violin Plot
    sns.violinplot(data=[y_true_actual, y_pred_rf], ax=axes[1, 1])
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Actual", "Random Forest Predicted"])
    axes[1, 1].set_title("Random Forest Distribution of Actual vs Predicted")
    axes[1, 1].set_ylabel(variable_name)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print_done()

# --- Main Training Workflow Function ---

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
        y_train (pd.Series): Training target.
        w_train (pd.Series): Training sample weights.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
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
    model_instance.fit(X_train, y_train, sample_weight=w_train)
    print_done()

    print_step(f"Making predictions for {model_name}")
    # Inverse transform log-transformed predictions back to original scale
    y_test_pred = np.expm1(model_instance.predict(X_test))
    y_test_actual = np.expm1(y_test) # Original scale actual values
    y_train_pred = np.expm1(model_instance.predict(X_train))
    y_train_actual = np.expm1(y_train) # Original scale actual values
    print_done()

    # --- Generate Model-Specific Plots ---
    # Scatter and Violin plot for test set predictions vs actual
    plot_and_save_violin_and_scatter(
        y_test_actual, y_test_pred, "Flood Depth (m)",
        os.path.join(save_dir, "violin_test.png"), model_name=f"{model_name} prediction"
    )
    # Scatter and Violin plot for OWP HAND vs actual flood depth (common visualization)
    plot_and_save_violin_and_scatter(
        y_test_actual, common_test_data_for_eval["owp_hand_fim"], "Flood Depth (m)",
        os.path.join(save_dir, "owp_vs_flood.png"), model_name="OWP HAND vs Flood Depth"
    )

    # LULC-wise scatter plots
    df_eval_lulc = common_test_data_for_eval.copy()
    df_eval_lulc["actual_flood_depth"] = y_test_actual
    df_eval_lulc["predicted_flood_depth"] = y_test_pred
    save_lulc_graphs(df_eval_lulc, save_dir, model_name=model_name)

    # Feature importance plot for the current model
    plot_single_model_feature_importance(
        model_instance.feature_importances_, features,
        os.path.join(save_dir, "feature_importance.png"), model_name=model_name
    )

    print_step(f"Evaluating {model_name} model")
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = mean_squared_error(y_test_actual, y_test_pred, squared=False)
    r2_test = r2_score(y_test_actual, y_test_pred)

    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = mean_squared_error(y_train_actual, y_train_pred, squared=False)
    r2_train = r2_score(y_train_actual, y_train_pred)
    print_done()

    print_step(f"Saving {model_name} model info")
    info_path = os.path.join(save_dir, "model_info.txt")

    # Save model evaluation statistics and feature importances to a text file
    with open(info_path, "w") as f:
        f.write(f"{model_name} Model Info and Evaluation\n")
        f.write("=" * 40 + "\n")
        f.write("Train Shape: {}\n".format(X_train.shape))
        f.write("Test Shape: {}\n\n".format(X_test.shape))

        f.write("Training Metrics:\n")
        f.write(f"MAE: {mae_train:.4f}\nRMSE: {rmse_train:.4f}\nR2 Score: {r2_train:.4f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"MAE: {mae_test:.4f}\nRMSE: {rmse_test:.4f}\nR2 Score: {r2_test:.4f}\n\n")
        f.write("Feature Importances (0-100%):\n")
        importance_percent = model_instance.feature_importances_ / np.sum(model_instance.feature_importances_) * 100
        for name, score in zip(features, importance_percent):
            f.write(f"{name}: {score:.2f}%\n")
    print_done()

    print_step(f"Saving {model_name} model and data splits")
    with open(os.path.join(save_dir, "train.pkl"), "wb") as f:
        pickle.dump((X_train, y_train), f)
    with open(os.path.join(save_dir, "test.pkl"), "wb") as f:
        pickle.dump((X_test, y_test), f)
    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(model_instance, f)
    print_done()

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
        "feature_importances": model_instance.feature_importances_,
        "y_test_pred": y_test_pred,
        "y_test_actual": y_test_actual # This will be the same for both but good to keep for consistency
    }

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

    # Define features, target, and weights
    features = ["owp_hand_fim", "lulc", "slope", "curvature", "aspect_sin", "aspect_cos", "rem", "dem"]
    X = df[features].fillna(df[features].mean()) # Fill NaNs with column means
    y = df["flood_depth_log"]
    weights = df["flood_depth"] + 0.1 # Use original flood depth for weights, adding a small constant

    print_step("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.3, random_state=42
    )
    print_done()

    # Store common test data that might be used for plots that are not model-specific
    common_test_data_for_eval = X_test.copy()
    # Note: 'actual_flood_depth' is calculated and passed within train_and_evaluate_model
    # from y_test_actual, which is np.expm1(y_test)

    results = {} # Dictionary to store results from each model

    # --- Train and Evaluate XGBoost Model ---
    xgboost_model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        objective="reg:squarederror", tree_method="hist", verbosity=0
    )
    results["xgboost"] = train_and_evaluate_model(
        xgboost_model, "XGBoost", X_train, y_train, w_train, X_test, y_test,
        features, os.path.join(save_base, "xgboost_results"), common_test_data_for_eval
    )

    # --- Train and Evaluate Random Forest Model ---
    random_forest_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        n_jobs=-1, # Use all available cores
        random_state=42
    )
    results["random_forest"] = train_and_evaluate_model(
        random_forest_model, "Random Forest", X_train, y_train, w_train, X_test, y_test,
        features, os.path.join(save_base, "random_forest_results"), common_test_data_for_eval
    )

    # --- Generate Combined Visualizations ---
    print_step("Generating combined visualizations")
    combined_plots_dir = os.path.join(save_base, "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)

    # Combined Feature Importance Plot
    plot_combined_feature_importance(
        results["xgboost"]["feature_importances"],
        results["random_forest"]["feature_importances"],
        features,
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
        full_combined_data_desc["flood_depth_log"] = y
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
