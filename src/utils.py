import pandas as pd
import joblib
import os

DATA_DIR = "../data"
MODEL_DIR = "../models"

def load_data(split="train"):
    path = os.path.join(DATA_DIR, f"{split}.csv")
    return pd.read_csv(path)

def save_model(model, filename="logreg_model.pkl"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

def load_model(filename="logreg_model.pkl"):
    return joblib.load(os.path.join(MODEL_DIR, filename))
