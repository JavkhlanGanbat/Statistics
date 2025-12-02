import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

def main():
    df = pd.read_csv(DATA_ROOT / "data.csv")

    # Columns to keep (drop the rest) — done here so all downstream use trimmed files only.
    keep_cols = [
        "age", "educational-num", "capital-gain", "capital-loss", "hours-per-week",
        "education", "marital-status", "occupation", "gender", "income_>50K"
    ]
    df = df[keep_cols].copy()

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["income_>50K"]
    )

    train_df.to_csv(DATA_ROOT / "train_split.csv", index=False)
    val_df.to_csv(DATA_ROOT / "val_split.csv", index=False)
    print("Saved trimmed splits: train_split.csv, val_split.csv")

if __name__ == "__main__":
    main()
