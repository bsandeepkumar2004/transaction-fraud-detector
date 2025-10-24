# ==========================================================
# preprocessing.py
# ==========================================================
# Purpose:
#   - Load and clean the dataset
#   - Handle missing values
#   - Extract datetime features
#   - Encode categorical variables
#   - Engineer new features
#   - Scale numerical variables
#   - Handle class imbalance (SMOTE)
#   - Split into train/test sets
#   - Save preprocessing objects (scalers, encoders)
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

# ----------------------------------------------------------
# 1️⃣ Load Dataset
# ----------------------------------------------------------
def load_dataset(path="transactions.csv"):
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ----------------------------------------------------------
# 2️⃣ Handle Missing Values
# ----------------------------------------------------------
def handle_missing_values(df):
    print("\n🧩 Missing Values Before:")
    print(df.isnull().mean() * 100)

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("\n✅ Missing Values After:")
    print(df.isnull().mean() * 100)
    return df


# ----------------------------------------------------------
# 3️⃣ Convert & Extract Date Features
# ----------------------------------------------------------
def extract_date_features(df):
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["day"] = df["transaction_date"].dt.day
    df["month"] = df["transaction_date"].dt.month
    df["hour"] = df["transaction_date"].dt.hour
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df.drop(columns=["transaction_date"], inplace=True)
    return df


# ----------------------------------------------------------
# 4️⃣ Encode Categorical Columns
# ----------------------------------------------------------
def encode_categorical(df):
    cat_cols = ["transaction_type", "location", "device_used"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Save encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.joblib")
    print("✅ Encoders saved to models/encoders.joblib")
    return df, encoders


# ----------------------------------------------------------
# 5️⃣ Feature Engineering
# ----------------------------------------------------------
def feature_engineering(df, freq_threshold=3):
    df["transaction_diff"] = df["present_transaction_amount"] - df["last_transaction_amount"]
    df["amount_ratio"] = df["present_transaction_amount"] / (df["avg_amount_per_trans"] + 1)
    df["balance_ratio"] = df["present_transaction_amount"] / (df["balance"] + 1)
    df["is_high_freq_user"] = np.where(df["frequent_transactions"] > freq_threshold, 1, 0)
    return df


# ----------------------------------------------------------
# 6️⃣ Outlier Treatment (Optional – clip extreme values)
# ----------------------------------------------------------
def treat_outliers(df, cols=None):
    if cols is None:
        cols = ["present_transaction_amount", "balance"]

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df


# ----------------------------------------------------------
# 7️⃣ Scale Numerical Columns
# ----------------------------------------------------------
def scale_numerical(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [col for col in num_cols if col not in ["fraud"]]  # exclude target

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    joblib.dump(scaler, "models/scaler.joblib")
    print("✅ Scaler saved to models/scaler.joblib")
    return df, scaler


# ----------------------------------------------------------
# 8️⃣ Handle Class Imbalance (SMOTE)
# ----------------------------------------------------------
def balance_classes(X, y):
    print(f"Before SMOTE: Fraud = {sum(y)}, Non-Fraud = {len(y) - sum(y)}")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"After SMOTE: Fraud = {sum(y_res)}, Non-Fraud = {len(y_res) - sum(y_res)}")
    return X_res, y_res


# ----------------------------------------------------------
# 9️⃣ Full Preprocessing Pipeline
# ----------------------------------------------------------
def preprocess_pipeline(csv_path="transactions.csv", test_size=0.2):
    df = load_dataset(csv_path)
    df = handle_missing_values(df)
    df = extract_date_features(df)
    df, encoders = encode_categorical(df)
    df = feature_engineering(df)
    df = treat_outliers(df)
    df, scaler = scale_numerical(df)

    # Separate features and target
    X = df.drop(columns=["fraud", "account_no"])
    y = df["fraud"]

    # Handle imbalance
    X_res, y_res = balance_classes(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, stratify=y_res, random_state=42
    )

    # Save processed data (optional)
    os.makedirs("data", exist_ok=True)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("\n✅ Data Preprocessing Complete:")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features: {X_train.columns.tolist()[:10]} ...")
    print(f"Class balance: {np.bincount(y_res)}")

    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------
# 🔟 Helper for Single Transaction (used in prediction.py)
# ----------------------------------------------------------
def preprocess_single(tx_dict):
    """Prepare a single transaction (dict) for prediction"""
    df = pd.DataFrame([tx_dict])

    # Load encoders/scaler
    encoders = joblib.load("models/encoders.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # Same transformations
    df = extract_date_features(df)
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    df = feature_engineering(df)
    df = treat_outliers(df)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    # Drop unused columns
    drop_cols = [c for c in ["fraud", "account_no"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    return df


# ----------------------------------------------------------
# Main (for standalone testing)
# ----------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_pipeline("transactions.csv")
