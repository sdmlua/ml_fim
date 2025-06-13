# models/modeling.py

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Import model parameters from the centralized configuration file
from config.settings import XGBOOST_PARAMS, RANDOM_FOREST_PARAMS

def get_xgboost_regressor():
    """
    Returns an initialized XGBoost Regressor model.

    The hyperparameters for the model are loaded from config.settings.
    """
    print("Initializing XGBoost Regressor...")
    return XGBRegressor(**XGBOOST_PARAMS)

def get_random_forest_regressor():
    """
    Returns an initialized Random Forest Regressor model.

    The hyperparameters for the model are loaded from config.settings.
    """
    print("Initializing Random Forest Regressor...")
    return RandomForestRegressor(**RANDOM_FOREST_PARAMS)