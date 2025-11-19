"""
Random Forest Model Training Script

This script trains a Random Forest classifier using the same preprocessing
and data split as the logistic regression model for fair comparison.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import save_model
from preprocess import build_preprocessor

def train():
    """
    Train and evaluate the Random Forest model.
    
    Uses optimized hyperparameters for better performance while
    avoiding overfitting.
    """
    
    print("=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Load pre-split data (same as logistic regression)
    print("\n[1/6] Loading training data...")
    try:
        train_df = pd.read_csv("../data/train_split.csv")
        val_df = pd.read_csv("../data/val_split.csv")
        print(f"      ✓ Loaded {len(train_df)} training samples")
        print(f"      ✓ Loaded {len(val_df)} validation samples")
    except FileNotFoundError as e:
        print(f"      ✗ Error: {e}")
        print("\n      Make sure you're running from the 'src' directory:")
        print("      cd /home/javkhlan/Statistics/src")
        raise
    
    X_train = train_df.drop(columns=["income_>50K"])
    y_train = train_df["income_>50K"]
    
    X_val = val_df.drop(columns=["income_>50K"])
    y_val = val_df["income_>50K"]
    
    # Build preprocessor (same as logistic regression)
    print("\n[2/6] Preprocessing data...")
    preprocessor = build_preprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    print(f"      ✓ Created {X_train_processed.shape[1]} features")
    
    # Train Random Forest with optimized hyperparameters
    print("\n[3/6] Training Random Forest...")
    print("      This may take 1-2 minutes...")
    rf = RandomForestClassifier(
        n_estimators=200,        # More trees for better performance
        max_depth=25,            # Deeper trees to capture patterns
        min_samples_split=15,    # Prevent overfitting
        min_samples_leaf=5,      # Ensure leaf quality
        max_features='sqrt',     # Standard for classification
        class_weight='balanced', # Handle class imbalance
        random_state=42,
        n_jobs=-1,              # Use all CPU cores
        verbose=1               # Show training progress
    )
    
    rf.fit(X_train_processed, y_train)
    print("      ✓ Training complete!")
    
    # Create pipeline
    print("\n[5/6] Creating model pipeline...")
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf)
    ])
    print("      ✓ Pipeline created")
    
    # Evaluate on validation set
    print("\n[4/6] Evaluating on validation set...")
    y_pred = rf.predict(X_val_processed)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"      Accuracy:  {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1 Score:  {f1:.4f}")
    
    # Save model
    print("\n[6/6] Saving model...")
    save_model(model, filename="random_forest_model.pkl")
    print("      ✓ Model saved to: models/random_forest_model.pkl")
    
    print("\n" + "=" * 70)
    print("✓ RANDOM FOREST TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now run the analysis notebook:")
    print("  Open: notebooks/random_forest_analysis.ipynb")
    print("  Execute all cells to see the results")
    print("=" * 70)

if __name__ == "__main__":
    train()
