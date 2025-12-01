"""
1. Хуваасан өгөгдлийг ачаалах
2. Өгөгдлийг боловсруулах
3. Logistic regression загварыг сургах
4. "validation" өгөгдөл дээр үнэлгээ хийх
5. Сургасан загварыг хадгалах
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from utils import load_model, save_model, DEFAULT_KEEP_COLS
from preprocess import build_preprocessor
from logistic_regression import LogisticRegression

def main():
    print("Сургалтын өгөгдлийг ачаалж байна...")
    train_df = pd.read_csv("../data/train_split.csv")
    val_df = pd.read_csv("../data/val_split.csv")
    
    # Шинж чанар (X) болон зорилтот хувьсагч (y)-г салгах
    X_train = train_df.drop(columns=["income_>50K"])
    y_train = train_df["income_>50K"]
    
    X_val = val_df.drop(columns=["income_>50K"])
    y_val = val_df["income_>50K"]
    
    # Өгөгдөл боловсруулах
    preprocessor = build_preprocessor()
    
    # Зөвхөн сургалтын өгөгдөл дээр тохируулах
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Баталгаажуулах өгөгдлийг хувиргах
    X_val_processed = preprocessor.transform(X_val)
    
    logreg = LogisticRegression(
        learning_rate=0.1,        # Илүү хурдтай ойролцоо шийдэлд хүрэх
        max_iter=2000,            # Илүү олон давталт
        tol=1e-5,                 # Ойролцоо шийдлийн илүү нарийвчлал
        reg_lambda=0.005,         # Хөнгөвтөр L2 регуляризаци
        lr_decay=0.0005,          # Сурах хурдны удаан бууралт
        class_weight='balanced',  # Ангиудын тэнцвэргүй байдлыг засах
        threshold=0.5             # Үүнийг дараа оновчлоно
    )
    
    logreg.fit(X_train_processed, y_train.values, 
               X_val=X_val_processed, y_val=y_val.values)
    
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("logreg", logreg)
    ])
    
    y_pred = logreg.predict(X_val_processed)
    
    save_model(model)

def main():
    model = load_model(keep_cols=DEFAULT_KEEP_COLS)
    saved_path = save_model(model)
    print(f"Model trained and saved to: {saved_path}")

if __name__ == "__main__":
    main()
