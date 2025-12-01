import os
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor
from logistic_regression import LogisticRegressionGD

DEFAULT_KEEP_COLS = [
    "age", "educational-num", "capital-gain", "capital-loss", "hours-per-week",
    "education", "marital-status", "occupation", "gender"
]

def _root():
    return Path(__file__).resolve().parent.parent

def load_model(keep_cols=None, random_state=42):
    df = pd.read_csv(_root() / "data" / "data.csv")
    if keep_cols is None:
        keep_cols = [c for c in DEFAULT_KEEP_COLS if c in df.columns]
    X = df[keep_cols].copy()
    y = df["income_>50K"].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in keep_cols if c not in num_cols]
    pre = build_preprocessor(num_cols, cat_cols)
    logreg = LogisticRegressionGD(lr=0.1, lr_decay=1e-4, n_iter=800, l2=1e-4, random_state=random_state)
    pipe = Pipeline([
        ("preprocess", pre),
        ("logreg", logreg)
    ])
    pipe.fit(X_train, y_train)
    pipe.keep_cols = keep_cols
    return pipe

def save_model(model, name="logreg_pipeline.joblib"):
    out = _root() / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / name)
