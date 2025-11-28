import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save():
    df = pd.read_csv("../data/data.csv")
    
    print(f"Original data size: {len(df)}")
    
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])
    
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2,
        random_state=42,
        stratify=df["income_>50K"]  # Классуудын пропорцыг хадгална
    )
    
    train_df.to_csv("../data/train_split.csv", index=False)
    val_df.to_csv("../data/val_split.csv", index=False)
    
#   print(f"Training set: {len(train_df)} samples")
#   print(f"Validation set: {len(val_df)} samples")
    
#   print(f"\nClass distribution in training:")
#   print(train_df["income_>50K"].value_counts())
    
#   print(f"\nClass distribution in validation:")
#   print(val_df["income_>50K"].value_counts())
#   print("  - train_split.csv: Сургалтад ашиглана")
#   print("  - val_split.csv: Үнэлгээнд ашиглана")

if __name__ == "__main__":
    split_and_save()
