import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from utils import load_data, save_model
from preprocess import build_preprocessor

def train():

    df = load_data(split="train")

    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    X = df.drop(columns=["income_>50K"])
    y = df["income_>50K"]

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    preprocessor = build_preprocessor()

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\n=== Model Evaluation on Validation Set ===")
    y_pred = model.predict(X_val)
    
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_val, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['<=50K', '>50K']))
    
    save_model(model)

    print("\nModel trained and saved successfully.")

if __name__ == "__main__":
    train()
