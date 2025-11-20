"""
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

from utils import save_model
from preprocess import build_preprocessor
from logistic_regression import LogisticRegression

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
    
    # --- STEP 2: PREPROCESS FEATURES ---
    # Build preprocessor: StandardScaler for numbers, OneHotEncoder for categories
    print("\nPreprocessing data...")
    preprocessor = build_preprocessor()
    
    # Fit preprocessor on training data only (prevent data leakage)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform validation data using training statistics
    X_val_processed = preprocessor.transform(X_val)
    
    # --- STEP 3: TRAIN CUSTOM MODEL ---
    print("\nTraining model with improved hyperparameters...")
    
    # Improved hyperparameters for better metrics
    logreg = LogisticRegression(
        learning_rate=0.1,        # Adjusted for faster convergence
        max_iter=2000,            # More iterations
        tol=1e-5,                 # Stricter convergence
        reg_lambda=0.005,         # Lighter regularization
        lr_decay=0.0005,          # Slower decay
        class_weight='balanced',  # Handle imbalanced data
        threshold=0.5             # Will optimize this
    )
    
    # Train with validation set for early stopping
    logreg.fit(X_train_processed, y_train.values, 
               X_val=X_val_processed, y_val=y_val.values)
    
    # --- STEP 4: OPTIMIZE DECISION THRESHOLD ---
    optimal_threshold = logreg.optimize_threshold(X_val_processed, y_val.values, metric='f1')
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    
    # --- STEP 5: CREATE SKLEARN-COMPATIBLE PIPELINE ---
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("logreg", logreg)
    ])
    
    # --- STEP 6: EVALUATE ON VALIDATION SET ---
    y_pred = logreg.predict(X_val_processed)
    
    # --- STEP 7: SAVE MODEL ---
    save_model(model)

if __name__ == "__main__":
    train()
