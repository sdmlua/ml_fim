import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_step(message):
    print(f"\n🟡 {message}...")

def print_done():
    print("✅ Done!")

def plot_scatter_with_metrics(ax, y_true, y_pred, variable_name, title):
    y_true_np = np.array(y_true).reshape(-1, 1)
    y_pred_np = np.array(y_pred)

    reg_origin = LinearRegression(fit_intercept=False)
    reg_origin.fit(y_true_np, y_pred_np)
    x_line = np.linspace(0, max(np.max(y_true), np.max(y_pred)), 100).reshape(-1, 1)
    y_line = reg_origin.predict(x_line)

    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = mean_squared_error(y_true_np, y_pred_np, squared=False)
    r2 = r2_score(y_true_np, y_pred_np)
    metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"

    ax.scatter(y_true, y_pred, alpha=0.4, s=10, edgecolors='k', linewidths=0.2)
    ax.plot(x_line, y_line, color='red', linewidth=2, label='Fit from origin')
    ax.plot(x_line, x_line, color='green', linestyle='--', linewidth=2.5, label='Ideal: y = x')
    ax.set_title(title)
    ax.set_xlabel(f"Actual {variable_name}")
    ax.set_ylabel(f"Predicted {variable_name}")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    text_x = xlim[0] + 0.95 * (xlim[1] - xlim[0])
    text_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])

    ax.text(text_x, text_y, metrics_text,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))



def plot_and_save_violin(y_true, y_pred, variable_name, save_path=None, title=None, ax1=None, ax2=None):
    print_step(f"Plotting violin and scatter for {variable_name}")
    
    y_true_np = np.array(y_true).reshape(-1, 1)
    y_pred_np = np.array(y_pred)

    reg_origin = LinearRegression(fit_intercept=False)
    reg_origin.fit(y_true_np, y_pred_np)
    x_line = np.linspace(0, max(np.max(y_true), np.max(y_pred)), 100).reshape(-1, 1)
    y_line = reg_origin.predict(x_line)

    # Compute metrics
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = mean_squared_error(y_true_np, y_pred_np, squared=False)
    r2 = r2_score(y_true_np, y_pred_np)
    metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"

    if ax1 is None and ax2 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        save = True
    else:
        save = False

    # Scatter
    ax1.scatter(y_true, y_pred, alpha=0.4, s=10, edgecolors='k', linewidths=0.2)
    ax1.plot(x_line, y_line, color='red', linewidth=2, label='Fit from origin')
    ax1.plot(x_line, x_line, color='green', linestyle='--', linewidth=2.5, label='Ideal: y = x')
    ax1.set_title(title or "Predicted vs Actual (Scatter)")
    ax1.set_xlabel(f"Actual {variable_name}")
    ax1.set_ylabel(f"Predicted {variable_name}")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.legend()

    text_x = 0.95 * ax1.get_xlim()[1]
    text_y = 0.05 * ax1.get_ylim()[1]
    ax1.text(text_x, text_y, metrics_text,
             ha='right', va='bottom', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Violin
    sns.violinplot(data=[y_true, y_pred], ax=ax2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Actual", "Predicted"])
    ax2.set_title(title or "Distribution of Actual vs Predicted")

    if save and save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print_done()



def plot_feature_importance(importance, feature_names, save_path):
    print_step("Plotting feature importance")
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
    plt.title("Random Forest Feature Importances")
    plt.xticks(rotation=45, ha='right')

    for bar, percent in zip(bars, importance_percent[sorted_idx]):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{percent:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print_done()

def save_lulc_graphs(df, save_dir):
    print_step("Saving LULC-wise scatter plots")

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
        plot_scatter_with_metrics(axes[i], y_true, y_pred, "Flood Depth", title)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lulc_scatter_plots.png"))
    plt.close()
    print_done()



def train_and_save_model(pickle_path, save_base):
    start = time.time()
    print(f"\n🔷 Training model using data from {pickle_path}")
    os.makedirs(save_base, exist_ok=True)

    print_step("Loading data")
    df = pd.read_pickle(pickle_path)
    print(f"Loaded {len(df)} rows.")

    print_step("Filtering and transforming data")
    df = df[df['flood_depth'] > 0.01].copy()
    df["flood_depth_log"] = np.log1p(df["flood_depth"])
    df["aspect_sin"] = np.sin(np.radians(df["aspect"]))
    df["aspect_cos"] = np.cos(np.radians(df["aspect"]))
    print_done()

    features = ["owp_hand_fim", "lulc", "slope", "curvature", "aspect_sin", "aspect_cos", "rem", "dem"]
    X = df[features].fillna(df[features].mean())
    y = df["flood_depth_log"]
    weights = df["flood_depth"] + 0.1

    print_step("Splitting train/test")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.3, random_state=42
    )
    print_done()

    print_step("Training Random Forest model")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    print_step("Making predictions")
    y_test_pred = np.expm1(model.predict(X_test))
    y_test_actual = np.expm1(y_test)
    y_train_pred = np.expm1(model.predict(X_train))
    y_train_actual = np.expm1(y_train)
    print_done()

    plot_and_save_violin(y_test_actual, y_test_pred, "Flood Depth (m)", os.path.join(save_base, "violin_test.png"), title="Random Forest prediction vs Flood Depth (test set)")
    plot_and_save_violin(y_test_actual,X_test["owp_hand_fim"], "Flood Depth (m)", os.path.join(save_base, "owp_vs_flood.png"), title="OWP HAND vs Flood Depth")

    df_eval_lulc = X_test.copy()
    df_eval_lulc["actual_flood_depth"] = y_test_actual
    df_eval_lulc["predicted_flood_depth"] = y_test_pred
    save_lulc_graphs(df_eval_lulc, save_base)

    plot_feature_importance(model.feature_importances_, features, os.path.join(save_base, "feature_importance.png"))

    print_step("Evaluating model")
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = mean_squared_error(y_test_actual, y_test_pred, squared=False)
    r2_test = r2_score(y_test_actual, y_test_pred)

    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = mean_squared_error(y_train_actual, y_train_pred, squared=False)
    r2_train = r2_score(y_train_actual, y_train_pred)
    print_done()

    print_step("Saving model info")
    info_path = os.path.join(save_base, "model_info.txt")

    full_combined = X.copy()
    full_combined["flood_depth_log"] = y

    train_combined = X_train.copy()
    train_combined["flood_depth_log"] = y_train
    
    test_combined = X_test.copy()
    test_combined["flood_depth_log"] = y_test

    
    with open(info_path, "w") as f:
        f.write("Model Info and Evaluation\n")
        f.write("=" * 40 + "\n")
        f.write("Train Shape: {}\n".format(X_train.shape))
        f.write("Test Shape: {}\n\n".format(X_test.shape))
        
        f.write("Full Dataset (Inputs + Target) Description:\n")
        f.write(str(full_combined.describe()) + "\n\n")
        
        f.write("Training Dataset (Inputs + Target) Description:\n")
        f.write(str(train_combined.describe()) + "\n\n")
        
        f.write("Test Dataset (Inputs + Target) Description:\n")
        f.write(str(test_combined.describe()) + "\n\n")

        f.write("Training Metrics:\n")
        f.write(f"MAE: {mae_train:.4f}\nRMSE: {rmse_train:.4f}\nR2 Score: {r2_train:.4f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"MAE: {mae_test:.4f}\nRMSE: {rmse_test:.4f}\nR2 Score: {r2_test:.4f}\n\n")
        f.write("Feature Importances (0-100%):\n")
        importance = model.feature_importances_ / np.sum(model.feature_importances_) * 100
        for name, score in zip(features, importance):
            f.write(f"{name}: {score:.2f}%\n")
    print_done()

    print_step("Saving model and data")
    with open(os.path.join(save_base, "train.pkl"), "wb") as f:
        pickle.dump((X_train, y_train), f)
    with open(os.path.join(save_base, "test.pkl"), "wb") as f:
        pickle.dump((X_test, y_test), f)
    with open(os.path.join(save_base, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print_done()

    print(f"\n✅ Training complete. All results saved to {save_base}")
    print(f"⏱️ Total Time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input pickle file")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for saving model and results")
    args = parser.parse_args()
    train_and_save_model(args.data, args.save_dir)
