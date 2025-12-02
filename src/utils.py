from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle

from preprocess import build_preprocessor
from logistic_regression import LogisticRegression

def _root():
    return Path(__file__).resolve().parent.parent

def load_model(random_state=42):
    # Train on trimmed training split
    train_df = pd.read_csv(_root() / "data" / "train_split.csv")
    X_train = train_df.drop(columns=["income_>50K"])
    y_train = train_df["income_>50K"].astype(int)

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    pre = build_preprocessor(num_cols, cat_cols)
    logreg = LogisticRegression(
        learning_rate=0.1,
        max_iter=800,
        reg_lambda=1e-4,
        lr_decay=1e-4,
        class_weight="balanced",
        threshold=0.5
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("logreg", logreg)
    ])
    pipe.fit(X_train, y_train)
    return pipe

def save_model(model, name="logreg_pipeline.pkl"):
    out_dir = _root() / "models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / name
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    return out_path
