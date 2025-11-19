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

    # Load pre-split data (same as logistic regression)
    print("\nLoading training data...")
    try:
        train_df = pd.read_csv("../data/train_split.csv")
        val_df = pd.read_csv("../data/val_split.csv")
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
    print("\nPreprocessing data...")
    preprocessor = build_preprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Train Random Forest with optimized hyperparameters
    print("\nTraining random forest...")
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
    
    # Create pipeline
    print("\nCreating model pipeline...")
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf)
    ])
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
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
    print("\nSaving model...")
    save_model(model, filename="random_forest_model.pkl")

if __name__ == "__main__":
    train()
