import os
import joblib

MODEL_DIR = "../models"

def save_model(model, filename="logreg_model.pkl"):
    """Save model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

def load_model(filename="logreg_model.pkl"):
    """Load model from disk."""
    return joblib.load(os.path.join(MODEL_DIR, filename))
