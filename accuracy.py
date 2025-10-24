# ==========================================================
# accuracy.py
# ==========================================================
# Purpose:
#   - Evaluate saved best model on test data
#   - Show accuracy, precision, recall, f1, ROC-AUC
#   - Save confusion matrix and ROC plots
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

def evaluate_model():
    # Load model & test data
    model = joblib.load("models/best_model.pkl")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    }

    # Plots
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)

    import os
    os.makedirs("reports", exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    plt.close()

    # ROC Curve
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png")
    plt.close()

    return metrics


if __name__ == "__main__":
    results = evaluate_model()
    print("✅ Model Accuracy Metrics:")
    for k, v in results.items():
        print(f"{k}: {v}")
