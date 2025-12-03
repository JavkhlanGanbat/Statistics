"""
1. Хуваасан өгөгдлийг ачаалах
2. Өгөгдлийг боловсруулах
3. Logistic regression загварыг сургах
4. "validation" өгөгдөл дээр үнэлгээ хийх
5. Сургасан загварыг хадгалах
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import build_preprocessor
from logistic_regression import LogisticRegression

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    models_root = project_root / "models"
    models_root.mkdir(exist_ok=True)

    train_df = pd.read_csv(data_root / "train_split.csv")
    val_df = pd.read_csv(data_root / "val_split.csv")

    X_train = train_df.drop(columns=["income_>50K"])
    y_train = train_df["income_>50K"].astype(int)
    X_val = val_df.drop(columns=["income_>50K"])
    y_val = val_df["income_>50K"].astype(int)

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    logreg = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-4,
        reg_lambda=1e-4,
        lr_decay=1e-4,
        class_weight="balanced",
        threshold=0.5
    )
    logreg.fit(X_train_proc, y_train.values, X_val=X_val_proc, y_val=y_val.values)

    logreg.optimize_threshold(X_val_proc, y_val.values, metric="f1")

    y_pred = logreg.predict(X_val_proc)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
        "threshold": float(getattr(logreg, "threshold", 0.5)),
        "iterations": len(getattr(logreg, "losses", []))
    }

    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("logreg", logreg)
    ])
    out_path = models_root / "logreg_pipeline.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {out_path}")

if __name__ == "__main__":
    main()
