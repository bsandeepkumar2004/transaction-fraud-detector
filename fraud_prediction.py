# ==========================================================
# fraud_prediction.py
# ==========================================================
# Purpose:
#   - Load trained model and preprocessing artifacts
#   - Prepare new transactions for inference
#   - Predict fraud probability and label
#   - Designed for backend integration with app.py
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from preprocessing import extract_date_features, feature_engineering, treat_outliers

# ----------------------------------------------------------
# 1️⃣ Load model and preprocessing tools
# ----------------------------------------------------------
MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "models/encoders.joblib"
SCALER_PATH = "models/scaler.joblib"

print("🔄 Loading model and preprocessing objects...")
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model, encoders, and scaler loaded successfully.")


# ----------------------------------------------------------
# 2️⃣ Prepare a single transaction for prediction
# ----------------------------------------------------------
def prepare_transaction(tx_dict):
    """
    Convert a single transaction (dict) into model-ready format.
    Applies same preprocessing steps as training.
    """
    df = pd.DataFrame([tx_dict]).copy()

    # Convert transaction_date to datetime & extract features
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["day"] = df["transaction_date"].dt.day
        df["month"] = df["transaction_date"].dt.month
        df["hour"] = df["transaction_date"].dt.hour
        df["day_of_week"] = df["transaction_date"].dt.dayofweek
        df.drop(columns=["transaction_date"], inplace=True, errors="ignore")

    # Feature engineering (must match training)
    df = feature_engineering(df)

    # Encode categorical columns
    cat_cols = ["transaction_type", "location", "device_used"]
    for col in cat_cols:
        if col in df.columns:
            if df[col].iloc[0] not in encoders[col].classes_:
                # Handle unseen categories by assigning "unknown" code
                print(f"⚠️ Warning: unseen category '{df[col].iloc[0]}' in column '{col}'")
                # Append unknown to classes if needed
                encoders[col].classes_ = np.append(encoders[col].classes_, df[col].iloc[0])
            df[col] = encoders[col].transform(df[col])

    # Treat outliers & scale numerical columns
    df = treat_outliers(df)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Drop unused identifiers
    drop_cols = [c for c in ["account_no", "fraud"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Ensure columns order matches scaler
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# ----------------------------------------------------------
# 3️⃣ Predict Fraud for a Single Transaction
# ----------------------------------------------------------
def predict_fraud(tx_dict, threshold=0.4):
    """
    Predict fraud probability and label (0=Not Fraud, 1=Fraud)
    tx_dict: dict containing transaction details
    threshold: decision threshold (default 0.4)
    """
    df = prepare_transaction(tx_dict)
    prob = model.predict_proba(df)[:, 1][0]
    label = int(prob >= threshold)
    return {"fraud_probability": float(prob), "fraud_label": label}


# ----------------------------------------------------------
# 4️⃣ Batch Predictions (for multiple transactions)
# ----------------------------------------------------------
def predict_batch(csv_path, threshold=0.4, output_path="fraud_predictions.csv"):
    """
    Predict fraud for an entire CSV file of transactions.
    Adds probability and predicted label columns.
    """
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        pred = predict_fraud(row.to_dict(), threshold)
        results.append(pred)

    df["fraud_probability"] = [r["fraud_probability"] for r in results]
    df["fraud_predicted"] = [r["fraud_label"] for r in results]

    # Highlight high-risk transactions (prob > 0.7)
    high_risk = df[df["fraud_probability"] > 0.7]
    print(f"⚠️ High-risk transactions: {len(high_risk)}")

    # Save results
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    return df


# ----------------------------------------------------------
# 5️⃣ Example Usage (for quick testing)
# ----------------------------------------------------------
if __name__ == "__main__":
    sample_tx = {
        "account_no": "AC1050001",
        "balance": 3200.0,
        "account_age_days": 900,
        "last_transaction_amount": 800.0,
        "present_transaction_amount": 4000.0,
        "transaction_date": "2025-07-14 09:15:00",
        "transaction_type": "NEFT",
        "frequent_transactions": 2,
        "location": "Mumbai",
        "device_used": "Mobile",
        "avg_amount_per_trans": 2100.0,
        "ratio_transaction_outof10": 0.5
    }

    print("\n🧠 Running fraud prediction for sample transaction...\n")
    result = predict_fraud(sample_tx, threshold=0.4)
    print(f"Fraud Probability: {result['fraud_probability']:.3f}")
    print(f"Predicted Label: {'⚠️ FRAUD' if result['fraud_label'] == 1 else '✅ NOT FRAUD'}")
