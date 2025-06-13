# models/model_utils.py

import os
import pickle
import numpy as np

from utils.helpers import print_step, print_done

def save_model_artifacts(model, X_train, y_train, X_test, y_test, save_dir):
    """
    Saves the trained model and data splits (X_train, y_train, X_test, y_test)
    to a specified directory using pickle.

    Args:
        model: The trained machine learning model object.
        X_train (pd.DataFrame): Training features DataFrame.
        y_train (pd.Series): Training target Series.
        X_test (pd.DataFrame): Test features DataFrame.
        y_test (pd.Series): Test target Series.
        save_dir (str): The directory path where the artifacts should be saved.
                        Directories will be created if they don't exist.
    """
    print_step("Saving model and data splits")
    # Ensure the target directory exists before saving files
    os.makedirs(save_dir, exist_ok=True)

    # Save training data
    with open(os.path.join(save_dir, "train.pkl"), "wb") as f:
        pickle.dump((X_train, y_train), f)

    # Save test data
    with open(os.path.join(save_dir, "test.pkl"), "wb") as f:
        pickle.dump((X_test, y_test), f)

    # Save the trained model
    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print_done()

def save_model_evaluation_report(
    model_name, X_train_shape, X_test_shape,
    mae_train, rmse_train, r2_train,
    mae_test, rmse_test, r2_test,
    feature_importances, feature_names, save_dir
):
    """
    Generates and saves a detailed text report of model evaluation metrics
    and feature importances to a specified directory.

    Args:
        model_name (str): A descriptive name for the model (e.g., "XGBoost", "Random Forest").
        X_train_shape (tuple): The shape (rows, columns) of the training feature set.
        X_test_shape (tuple): The shape (rows, columns) of the test feature set.
        mae_train (float): Mean Absolute Error calculated on the training set.
        rmse_train (float): Root Mean Squared Error calculated on the training set.
        r2_train (float): R-squared score calculated on the training set.
        mae_test (float): Mean Absolute Error calculated on the test set.
        rmse_test (float): Root Mean Squared Error calculated on the test set.
        r2_test (float): R-squared score calculated on the test set.
        feature_importances (array-like or None): An array of feature importance values.
                                                  Pass None if the model does not provide them.
        feature_names (list): A list of strings representing the names of the features.
        save_dir (str): The directory path where the report should be saved.
    """
    print_step(f"Saving {model_name} model info")
    # Define the full path for the info text file
    info_path = os.path.join(save_dir, "model_info.txt")

    with open(info_path, "w") as f:
        f.write(f"{model_name} Model Info and Evaluation\n")
        f.write("=" * 40 + "\n") # A simple separator line
        f.write(f"Train Shape: {X_train_shape}\n")
        f.write(f"Test Shape: {X_test_shape}\n\n")

        f.write("Training Metrics:\n")
        f.write(f"MAE: {mae_train:.4f}\n")
        f.write(f"RMSE: {rmse_train:.4f}\n")
        f.write(f"R2 Score: {r2_train:.4f}\n\n")

        f.write("Test Metrics:\n")
        f.write(f"MAE: {mae_test:.4f}\n")
        f.write(f"RMSE: {rmse_test:.4f}\n")
        f.write(f"R2 Score: {r2_test:.4f}\n\n")

        f.write("Feature Importances (0-100%):\n")
        # Check if feature importances are provided and non-empty
        if feature_importances is not None and len(feature_importances) > 0:
            # Convert raw importances to percentages
            importance_percent = feature_importances / np.sum(feature_importances) * 100
            # Write each feature and its importance score
            for name, score in zip(feature_names, importance_percent):
                f.write(f"{name}: {score:.2f}%\n")
        else:
            f.write("No feature importances available for this model type.\n")
    print_done()