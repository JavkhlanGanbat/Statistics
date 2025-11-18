import pandas as pd

from utils import load_data, load_model

def evaluate():
    """
    Generate predictions on the test set (without labels).
    """
    # Load test data
    df_test = load_data(split="test")
    
    # Drop fnlwgt if present
    if "fnlwgt" in df_test.columns:
        df_test = df_test.drop(columns=["fnlwgt"])
    
    # Test data has no labels - just generate predictions
    X_test = df_test
    
    # Load trained model
    model = load_model()
    
    print(f"Generating predictions for {len(X_test)} test samples...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Save predictions to file
    predictions_df = pd.DataFrame({
        "prediction": y_pred,
        "probability_class_0": y_pred_proba[:, 0],
        "probability_class_1": y_pred_proba[:, 1]
    })
    predictions_df.to_csv("../data/predictions.csv", index=False)
    
    print(f"Predictions saved to ../data/predictions.csv")
    print(f"\nPrediction distribution:")
    print(f"  Class 0 (<=50K): {(y_pred == 0).sum()} ({(y_pred == 0).sum() / len(y_pred) * 100:.1f}%)")
    print(f"  Class 1 (>50K):  {(y_pred == 1).sum()} ({(y_pred == 1).sum() / len(y_pred) * 100:.1f}%)")

if __name__ == "__main__":
    evaluate()
