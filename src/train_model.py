import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils import save_model
from preprocess import build_preprocessor
from logistic_regression import LogisticRegressionCustom

def train():
    # Load the pre-split data
    print("Loading training data...")
    train_df = pd.read_csv("../data/train_split.csv")
    val_df = pd.read_csv("../data/val_split.csv")
    
    X_train = train_df.drop(columns=["income_>50K"])
    y_train = train_df["income_>50K"]
    
    X_val = val_df.drop(columns=["income_>50K"])
    y_val = val_df["income_>50K"]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build preprocessor and fit on training data
    print("\nPreprocessing data...")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Train our custom logistic regression
    print("\nTraining custom Logistic Regression...")
    print("=" * 60)
    
    logreg = LogisticRegressionCustom(
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-4,
        verbose=True
    )
    
    logreg.fit(X_train_processed, y_train.values)
    
    # Create pipeline for easy deployment
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("logreg", logreg)
    ])
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("VALIDATION SET EVALUATION")
    print("=" * 60)
    
    y_pred = logreg.predict(X_val_processed)
    
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_val, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['<=50K', '>50K']))
    
    save_model(model)
    print("\n✓ Model trained and saved successfully!")

if __name__ == "__main__":
    train()
