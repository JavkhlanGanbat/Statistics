"""
Data Splitting Script

This script splits the training data into separate train and validation sets
to enable proper model evaluation. We do this once upfront to ensure:

1. Reproducibility: Same split every time we run experiments
2. No data leakage: Validation set never seen during training
3. Fair evaluation: Stratified split maintains class balance
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_and_save():
    """
    Split train.csv into train and validation sets.
    
    Strategy:
    ---------
    - 80% training: Used to learn model parameters
    - 20% validation: Used to evaluate model performance
    - Stratified: Maintains the same proportion of income classes in both sets
    - Random state: Fixed seed (42) for reproducibility
    
    Why stratification?
    - If original data has 75% low income, 25% high income
    - Both splits will maintain this 75/25 ratio
    - Prevents accidentally creating imbalanced splits
    - Ensures validation set is representative of full dataset
    
    Output Files:
    -------------
    - train_split.csv: For training the model
    - val_split.csv: For evaluating the model
    
    Note: The fnlwgt column is removed as it's a survey weight that
    isn't useful for our prediction task.
    """
    # Load original training data
    df = pd.read_csv("../data/train.csv")
    
    print(f"Original data size: {len(df)}")
    
    # Remove fnlwgt: This is a survey weighting variable, not a predictive feature
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])
    
    # Split with stratification to maintain class balance
    # random_state=42: Ensures we get the same split every time
    # stratify=df["income_>50K"]: Maintains proportion of income classes
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2,           # 20% for validation
        random_state=42,         # For reproducibility
        stratify=df["income_>50K"]  # Keep class proportions
    )
    
    # Save to disk
    train_df.to_csv("../data/train_split.csv", index=False)
    val_df.to_csv("../data/val_split.csv", index=False)
    
    # Report results
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    print(f"\nClass distribution in training:")
    print(train_df["income_>50K"].value_counts())
    
    print(f"\nClass distribution in validation:")
    print(val_df["income_>50K"].value_counts())
    
    print("\n✓ Data split complete!")
    print("  - train_split.csv: Use for training")
    print("  - val_split.csv: Use for evaluation")

if __name__ == "__main__":
    split_and_save()
