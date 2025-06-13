# models/evaluate.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming utils/helpers.py exists and contains print_step, print_done
from utils.helpers import print_step, print_done
from utils.plot_utils import plot_and_save_violin_and_scatter

def load_model(model_path):
    """
    Loads a trained machine learning model from a specified pickle file.

    Args:
        model_path (str): The full path to the pickled model file (e.g., 'results/xgboost_results/model.pkl').

    Returns:
        object: The loaded machine learning model.
    """
    print_step(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print_done()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def predict_with_model(model, X_data):
    """
    Makes predictions using a loaded model on new input data.

    Assumes the model was trained on log-transformed targets and
    inverse transforms the predictions back to the original scale.

    Args:
        model: The trained machine learning model.
        X_data (pd.DataFrame): The input features for making predictions.

    Returns:
        np.array: Predicted values on the original scale.
    """
    if model is None:
        print("Error: No model provided for prediction.")
        return None

    print_step("Making predictions")
    # Predict using the model (assumes log-transformed target during training)
    predictions_log = model.predict(X_data)
    # Inverse transform predictions back to original scale (e.g., meters for flood depth)
    predictions_original_scale = np.expm1(predictions_log)
    print_done()
    return predictions_original_scale

def evaluate_predictions(y_true, y_pred, variable_name="Target", save_path=None, title_suffix="Evaluation"):
    """
    Evaluates predictions against actual values and optionally saves a scatter/violin plot.

    Args:
        y_true (array-like): Actual target values (on original scale).
        y_pred (array-like): Predicted target values (on original scale).
        variable_name (str): Name of the target variable for plot labels.
        save_path (str, optional): Full path to save the plot. If None, the plot will be shown.
        title_suffix (str): Suffix for the plot title (e.g., "Evaluation").
    """
    print_step(f"Evaluating predictions for {variable_name}")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    print_done()

    # Generate and save/show evaluation plot
    plot_and_save_violin_and_scatter(
        y_true, y_pred, variable_name,
        save_path=save_path, model_name=f"Model {title_suffix}"
    )
    return {"mae": mae, "rmse": rmse, "r2": r2}

# Example usage (for demonstration, typically called from a script)
if __name__ == "__main__":
    # This block demonstrates how evaluate.py functions might be used.
    # In a real scenario, you'd have a main script (like scripts/run_inference.py)
    # to orchestrate loading data, models, and making predictions.

    # Dummy data for demonstration
    # In a real scenario, you'd load actual data here (e.g., from data/processed/)
    print_step("Creating dummy data for demonstration")
    np.random.seed(42)
    sample_size = 100
    dummy_X = pd.DataFrame(np.random.rand(sample_size, 5), columns=[f'feature_{i}' for i in range(5)])
    dummy_y_true = np.expm1(np.random.rand(sample_size) * 3) # Simulate original scale flood depth
    print_done()

    # Placeholder for where a trained model would be
    # For this example, we'll create a simple dummy model
    class DummyModel:
        def predict(self, X):
            # Simulate log-transformed predictions
            return np.log1p(X.iloc[:, 0] * 2 + X.iloc[:, 1] * 0.5 + 0.1)

    dummy_model = DummyModel()
    dummy_model_path = "dummy_model.pkl" # This won't actually be loaded as we create a dummy
    with open(dummy_model_path, 'wb') as f:
        pickle.dump(dummy_model, f)

    # Demonstrate loading and prediction
    loaded_model = load_model(dummy_model_path)
    if loaded_model:
        predictions = predict_with_model(loaded_model, dummy_X)
        if predictions is not None:
            # Demonstrate evaluation
            results = evaluate_predictions(
                dummy_y_true, predictions,
                variable_name="Dummy Value (m)",
                save_path="dummy_evaluation_plot.png",
                title_suffix="Dummy Test"
            )
            print("\nDummy Evaluation Results:", results)

    if os.path.exists(dummy_model_path):
        os.remove(dummy_model_path) # Clean up dummy model file
