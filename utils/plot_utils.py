# utils/plot_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Import helper functions for consistent output messages
from utils.helpers import print_step, print_done

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
    # This shows how well the model's predictions align with the actual values.
    reg_origin = LinearRegression(fit_intercept=False)
    reg_origin.fit(y_true_np, y_pred_np)
    # Create a line from 0 up to the maximum of actual or predicted values
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
    ax.plot(x_line, x_line, color='green', linestyle='--', linewidth=2.5, label='Ideal: y = x') # Ideal 1:1 relationship
    ax.set_title(title)
    ax.set_xlabel(f"Actual {variable_name}")
    ax.set_ylabel(f"Predicted {variable_name}")
    ax.set_xlim(left=0) # Ensure X-axis starts at 0
    ax.set_ylim(bottom=0) # Ensure Y-axis starts at 0
    ax.legend()

    # Add evaluation metrics text to the plot for easy viewing
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    text_x = xlim[0] + 0.95 * (xlim[1] - xlim[0]) # Position at 95% of x-range
    text_y = ylim[0] + 0.05 * (ylim[1] - ylim[0]) # Position at 5% of y-range
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

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot scatter on the first axes
    plot_scatter_with_metrics(ax1, y_true, y_pred, variable_name, f"{model_name} Prediction vs Actual (Scatter)")

    # Plot violin on the second axes
    sns.violinplot(data=[y_true, y_pred], ax=ax2)
    ax2.set_xticks([0, 1]) # Set x-ticks for 'Actual' and 'Predicted'
    ax2.set_xticklabels(["Actual", "Predicted"])
    ax2.set_title(f"{model_name} Distribution of Actual vs Predicted")
    ax2.set_ylabel(variable_name)

    if save_path:
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.savefig(save_path) # Save the figure to the specified path
        plt.close() # Close the figure to free up memory
        print_done()
    else:
        plt.show() # Display the figure if no save path is provided

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
    # Convert raw importance values to percentages for better readability
    importance_percent = 100.0 * (importance / np.sum(importance))
    # Sort features by importance in descending order
    sorted_idx = np.argsort(importance_percent)[::-1]

    plt.figure(figsize=(8, 6)) # Create a new figure
    bars = plt.bar(
        [feature_names[i] for i in sorted_idx], # Sorted feature names
        [importance_percent[i] for i in sorted_idx], # Sorted importance percentages
        color='skyblue', edgecolor='black'
    )

    plt.ylabel("Mean Decrease in Impurity (%)") # Standard metric for tree-based models
    plt.xlabel("Features")
    plt.title(f"{model_name} Feature Importances")
    plt.xticks(rotation=45, ha='right') # Rotate labels for better fit

    # Add percentage labels on top of bars
    for bar, percent in zip(bars, importance_percent[sorted_idx]):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{percent:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout() # Adjust layout
    plt.savefig(save_path) # Save the plot
    plt.close() # Close the figure
    print_done()

def save_lulc_graphs(df, save_dir, model_name="Model"):
    """
    Saves LULC-wise scatter plots for a given model's predictions.
    This function leverages plot_scatter_with_metrics to create multiple subplots
    based on Land Use/Land Cover categories.

    Args:
        df (pd.DataFrame): DataFrame containing 'lulc', 'actual_flood_depth', and 'predicted_flood_depth'.
        save_dir (str): Directory to save the plots.
        model_name (str): Name of the model for plot titles and filename prefix.
    """
    print_step(f"Saving LULC-wise scatter plots for {model_name}")

    # Define a mapping for LULC numerical codes to human-readable labels
    # This could also be imported from config.settings
    lulc_label_map = {
        1: "Water", 2: "Trees / Forest", 3: "Grassland", 4: "Flooded Vegetation",
        5: "Crops", 6: "Shrubland", 7: "Built-up / Urban", 8: "Bare / Sparse Vegetation",
        9: "Snow / Ice", 10: "Clouds", 11: "Rangeland"
    }

    categories = sorted(df['lulc'].unique()) # Get unique LULC categories
    cols = 3 # Number of columns for the subplot grid
    rows = int(np.ceil(len(categories) / cols)) # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows)) # Create grid of subplots
    axes = axes.flatten() # Flatten the axes array for easy iteration

    # Iterate through each LULC category and create a scatter plot
    for i, lulc_val in enumerate(categories):
        subset = df[df['lulc'] == lulc_val] # Subset data for the current LULC
        y_true = subset["actual_flood_depth"].values
        y_pred = subset["predicted_flood_depth"].values
        # Get the descriptive label for the LULC value
        title = lulc_label_map.get(lulc_val, f"LULC {lulc_val}")
        # Plot on the current subplot axis
        plot_scatter_with_metrics(axes[i], y_true, y_pred, "Flood Depth", f"{model_name}: {title}")

    # Remove any empty subplots if the number of categories doesn't perfectly fill the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout() # Adjust layout
    # Save the figure to the specified directory with a descriptive filename
    plt.savefig(os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_lulc_scatter_plots.png"))
    plt.close() # Close the figure
    print_done()

def plot_combined_feature_importance(xgboost_importance, rf_importance, feature_names, save_path):
    """
    Plots combined feature importance for XGBoost and Random Forest.
    Shows two bars per feature (one for each model) for direct comparison.

    Args:
        xgboost_importance (array-like): Feature importances from the XGBoost model.
        rf_importance (array-like): Feature importances from the Random Forest model.
        feature_names (list): List of feature names.
        save_path (str): Full path to save the plot.
    """
    print_step("Plotting combined feature importance")
    # Convert importances to percentages for comparison
    importance_xgboost_percent = 100.0 * (xgboost_importance / np.sum(xgboost_importance))
    importance_rf_percent = 100.0 * (rf_importance / np.sum(rf_importance))

    # Sort features by the average importance for consistent ordering across models
    avg_importance = (importance_xgboost_percent + importance_rf_percent) / 2
    sorted_idx = np.argsort(avg_importance)[::-1]

    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_xgboost_importance = [importance_xgboost_percent[i] for i in sorted_idx]
    sorted_rf_importance = [importance_rf_percent[i] for i in sorted_idx]

    x = np.arange(len(sorted_features))  # the label locations for grouped bars
    width = 0.35  # the width of the bars for each model

    plt.figure(figsize=(12, 7)) # Create a new figure
    bars1 = plt.bar(x - width/2, sorted_xgboost_importance, width, label='XGBoost Importance', color='teal', alpha=0.8)
    bars2 = plt.bar(x + width/2, sorted_rf_importance, width, label='Random Forest Importance', color='orange', alpha=0.8)

    plt.ylabel("Mean Decrease in Impurity (%)")
    plt.xlabel("Features")
    plt.title("Combined Feature Importances: XGBoost vs Random Forest")
    plt.xticks(x, sorted_features, rotation=45, ha='right') # Set feature names as x-axis labels
    plt.legend() # Show legend for model types
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a horizontal grid for readability

    # Add percentage values on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout() # Adjust layout
    plt.savefig(save_path) # Save the plot
    plt.close() # Close the figure
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

    # Create a 2x2 grid of subplots
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

    plt.tight_layout() # Adjust layout
    plt.savefig(save_path) # Save the plot
    plt.close() # Close the figure
    print_done()