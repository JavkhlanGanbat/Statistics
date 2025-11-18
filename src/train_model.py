import pandas as pd
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor
from logistic_regression import LogisticRegressionCustom

def train():
    # Load the pre-split data
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
    print("\nTraining model...")
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
    
    y_pred = logreg.predict(X_val_processed)
    
if __name__ == "__main__":
    train()
