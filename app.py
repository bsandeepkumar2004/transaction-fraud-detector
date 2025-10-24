# ==========================================================
# app.py
# ==========================================================
# Purpose:
#   - Central Streamlit interface for BFSI Fraud Detection
#   - Routes between preprocessing, training, prediction, and accuracy modules
# ==========================================================

import streamlit as st
import pandas as pd
from preprocessing import preprocess_pipeline
from train_test import train_and_evaluate, load_data, select_and_save_best
from fraud_prediction import predict_fraud, predict_batch
from accuracy import evaluate_model
import joblib
import os

st.set_page_config(page_title="BFSI Fraud Detection System", layout="wide")

# ----------------------------------------------------------
# Sidebar Navigation
# ----------------------------------------------------------
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["🏋️ Train & Test Model", "🔍 Predict Fraud", "📈 Model Accuracy"]
)

# ----------------------------------------------------------
# 🏋️ 1️⃣ Train & Test Model Section
# ----------------------------------------------------------
if menu == "🏋️ Train & Test Model":
    st.title("🏋️ Train & Test Model")
    st.markdown("Use this section to preprocess data, train models, and select the best performer.")

    if st.button("🚀 Run Preprocessing"):
        with st.spinner("Processing dataset..."):
            X_train, X_test, y_train, y_test = preprocess_pipeline("transactions.csv")
        st.success("✅ Preprocessing Complete!")

    if st.button("🎯 Train Models"):
        with st.spinner("Training models... please wait"):
            X_train, X_test, y_train, y_test = load_data()
            results = train_and_evaluate(X_train, X_test, y_train, y_test)
            best_name, best_model = select_and_save_best(results)
        st.success(f"✅ Training Complete! Best Model: {best_name}")

    if os.path.exists("models/best_model.pkl"):
        st.info("Model already trained and saved at: `models/best_model.pkl`")

# ----------------------------------------------------------
# 🔍 2️⃣ Predict Fraud Section
# ----------------------------------------------------------
elif menu == "🔍 Predict Fraud":
    st.title("🔍 Predict Fraudulent Transaction")

    st.markdown("### Enter transaction details below:")

    # Collect inputs
    account_no = st.text_input("Account Number", "AC100001")
    balance = st.number_input("Balance", min_value=0.0)
    account_age_days = st.number_input("Account Age (days)", min_value=0)
    last_transaction_amount = st.number_input("Last Transaction Amount", min_value=0.0)
    present_transaction_amount = st.number_input("Present Transaction Amount", min_value=0.0)
    transaction_date = st.text_input("Transaction Date (YYYY-MM-DD HH:MM:SS)", "2025-07-14 09:15:00")
    transaction_type = st.selectbox("Transaction Type", ["NEFT", "IMPS", "Wallet", "Card"])
    frequent_transactions = st.number_input("Frequent Transactions (count)", min_value=0)
    location = st.selectbox("Location", ["Delhi", "Mumbai", "Chennai", "Bengaluru", "Jaipur", "Lucknow"])
    device_used = st.selectbox("Device Used", ["Mobile", "Laptop", "POS", "Web"])
    avg_amount_per_trans = st.number_input("Average Amount per Transaction", min_value=0.0)
    ratio_transaction_outof10 = st.slider("Ratio Transaction out of 10", 0.0, 1.0, 0.5)

    tx = {
        "account_no": account_no,
        "balance": balance,
        "account_age_days": account_age_days,
        "last_transaction_amount": last_transaction_amount,
        "present_transaction_amount": present_transaction_amount,
        "transaction_date": transaction_date,
        "transaction_type": transaction_type,
        "frequent_transactions": frequent_transactions,
        "location": location,
        "device_used": device_used,
        "avg_amount_per_trans": avg_amount_per_trans,
        "ratio_transaction_outof10": ratio_transaction_outof10
    }

    if st.button("🔎 Predict Fraud"):
        with st.spinner("Predicting fraud..."):
            result = predict_fraud(tx, threshold=0.4)
        prob = result["fraud_probability"]
        label = "⚠️ Fraudulent Transaction" if result["fraud_label"] == 1 else "✅ Legit Transaction"

        st.metric(label="Fraud Probability", value=f"{prob:.3f}")
        st.success(label)

    st.markdown("---")
    st.subheader("📦 Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Processing batch file..."):
            df = pd.read_csv(uploaded_file)
            result_df = predict_batch(uploaded_file)
        st.success("✅ Predictions complete!")
        st.dataframe(result_df.head())

# ----------------------------------------------------------
# 📈 3️⃣ Accuracy Section
# ----------------------------------------------------------
elif menu == "📈 Model Accuracy":
    st.title("📈 Model Evaluation & Metrics")

    if os.path.exists("models/best_model.pkl"):
        with st.spinner("Evaluating model..."):
            metrics = evaluate_model()
        st.success("✅ Evaluation complete!")
        st.subheader("📊 Key Metrics")
        st.json(metrics)

        st.markdown("### 📉 Confusion Matrix & ROC Curve")
        st.image("reports/confusion_matrix.png")
        st.image("reports/roc_curve.png")
    else:
        st.warning("⚠️ No trained model found. Please train the model first.")
