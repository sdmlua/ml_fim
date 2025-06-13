# config/settings.py

# Data processing and feature definitions
LULC_LABEL_MAP = {
    1: "Water", 2: "Trees / Forest", 3: "Grassland", 4: "Flooded Vegetation",
    5: "Crops", 6: "Shrubland", 7: "Built-up / Urban", 8: "Bare / Sparse Vegetation",
    9: "Snow / Ice", 10: "Clouds", 11: "Rangeland"
}

FEATURES = ["owp_hand_fim", "lulc", "slope", "curvature", "aspect_sin", "aspect_cos", "rem", "dem"]
TARGET = "flood_depth_log"
WEIGHTS_COLUMN = "flood_depth"

# Train/Test Split parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42

# XGBoost Model Parameters
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "verbosity": 0
}

# Random Forest Model Parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "n_jobs": -1, # Use all available cores
    "random_state": RANDOM_STATE
}
