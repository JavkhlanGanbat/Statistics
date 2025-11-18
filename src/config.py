"""
Configuration file for the Income Prediction project.
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data files
TRAIN_DATA = os.path.join(DATA_DIR, "train.csv")
TEST_DATA = os.path.join(DATA_DIR, "test.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "logreg_model.pkl")

# Model configuration
MODEL_PARAMS = {
    "max_iter": 1000,
    "random_state": 42
}

# Train/validation split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature configuration
NUMERIC_FEATURES = [
    "age", 
    "educational-num", 
    "capital-gain",
    "capital-loss", 
    "hours-per-week"
]

CATEGORICAL_FEATURES = [
    "workclass", 
    "education", 
    "marital-status",
    "occupation", 
    "relationship", 
    "race",
    "gender", 
    "native-country"
]

# Visualization settings
FIGURE_SIZE = (10, 6)
DPI = 100
STYLE = "seaborn-v0_8-darkgrid"
