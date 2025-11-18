"""
Model Training Script

This script orchestrates the complete training pipeline:
1. Load pre-split data
2. Preprocess features (scaling + encoding)
3. Train custom logistic regression
4. Evaluate on validation set
5. Save trained model

The key advantage of using pre-split data is reproducibility -
every training run uses exactly the same train/validation split.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils import save_model
from preprocess import build_preprocessor
from logistic_regression import LogisticRegressionCustom

def train():
    """
    Train and evaluate the logistic regression model.
    
    Pipeline:
    ---------
    1. Load data: Use pre-split files for consistency
    2. Preprocess: Scale numerics, encode categoricals
    3. Train: Custom gradient descent implementation
    4. Evaluate: Compute metrics on held-out validation set
    5. Save: Store model for later predictions
    """
    
    # --- STEP 1: LOAD PRE-SPLIT DATA ---
    # Using pre-split files ensures reproducibility
    print("Loading training data...")
    train_df = pd.read_csv("../data/train_split.csv")
    val_df = pd.read_csv("../data/val_split.csv")
    
    # Separate features (X) from target (y)
    X_train = train_df.drop(columns=["income_>50K"])
    y_train = train_df["income_>50K"]
    
    X_val = val_df.drop(columns=["income_>50K"])
    y_val = val_df["income_>50K"]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # --- STEP 2: PREPROCESS FEATURES ---
    # Build preprocessor: StandardScaler for numbers, OneHotEncoder for categories
    print("\nPreprocessing data...")
    preprocessor = build_preprocessor()
    
    # Fit preprocessor on training data only (prevent data leakage)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform validation data using training statistics
    X_val_processed = preprocessor.transform(X_val)
    
    # --- STEP 3: TRAIN CUSTOM MODEL ---
    print("\nTraining custom Logistic Regression...")
    print("=" * 60)
    
    # Initialize our custom implementation
    logreg = LogisticRegressionCustom(
        learning_rate=0.1,  # Step size for gradient descent
        max_iter=1000,      # Maximum training iterations
        tol=1e-4,           # Convergence threshold
        verbose=True        # Print training progress
    )
    
    # Train using gradient descent
    logreg.fit(X_train_processed, y_train.values)
    
    # --- STEP 4: CREATE SKLEARN-COMPATIBLE PIPELINE ---
    # Wrap preprocessor + model for easy deployment
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("logreg", logreg)
    ])
    
    # --- STEP 5: EVALUATE ON VALIDATION SET ---
    print("\n" + "=" * 60)
    print("VALIDATION SET EVALUATION")
    print("=" * 60)
    
    y_pred = logreg.predict(X_val_processed)
    
    # Compute standard classification metrics
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_val, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['<=50K', '>50K']))
    
    # --- STEP 6: SAVE MODEL ---
    save_model(model)
    print("\n✓ Model trained and saved successfully!")

if __name__ == "__main__":
    train()
