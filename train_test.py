# ==========================================================
# train_test.py
# ==========================================================
# Purpose:
#   - Train and compare multiple models for fraud detection
#   - Handle class imbalance (optional with weights)
#   - Tune hyperparameters using GridSearchCV
#   - Evaluate via ROC-AUC, Recall, F1-score
#   - Visualize results & save best model
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb


# ----------------------------------------------------------
# 1️⃣ Load preprocessed data
# ----------------------------------------------------------
def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    print(f"✅ Data Loaded: Train = {X_train.shape}, Test = {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------
# 2️⃣ Define models and parameter grids
# ----------------------------------------------------------
def get_models():
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"C": [0.1, 1, 10]}
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                scale_pos_weight=1,
                random_state=42,
            ),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7]}
        ),
        "LightGBM": (
            LGBMClassifier(objective="binary", random_state=42),
            {"n_estimators": [100, 200], "num_leaves": [31, 50]}
        ),
    }
    return models


# ----------------------------------------------------------
# 3️⃣ Train & Evaluate Models
# ----------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}

    for name, (model, params) in models.items():
        print(f"\n🚀 Training {name} ...")

        grid = GridSearchCV(
            model,
            param_grid=params,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        recall = recall_score(y_test, preds)

        results[name] = {
            "model": best_model,
            "roc_auc": auc,
            "f1": f1,
            "recall": recall,
            "report": classification_report(y_test, preds, digits=4),
            "cm": confusion_matrix(y_test, preds),
        }

        print(f"✅ {name} done: AUC={auc:.4f}, F1={f1:.4f}, Recall={recall:.4f}")
        print("Best Params:", grid.best_params_)

    return results


# ----------------------------------------------------------
# 4️⃣ Compare and save best model
# ----------------------------------------------------------
def select_and_save_best(results):
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = results[best_name]["model"]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    print(f"\n🏆 Best Model: {best_name}")
    print(f"ROC-AUC: {results[best_name]['roc_auc']:.4f}")
    print(f"Saved → models/best_model.pkl")

    return best_name, best_model


# ----------------------------------------------------------
# 5️⃣ Plot confusion matrix and ROC curve
# ----------------------------------------------------------
def plot_evaluation(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_title(f"{model_name} - Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax[0].text(j, i, f"{val}", ha="center", va="center")

    # ROC Curve
    ax[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[1].set_title(f"{model_name} - ROC Curve")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 6️⃣ Show Feature Importance (for tree-based models)
# ----------------------------------------------------------
def show_feature_importance(model, X_train, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), X_train.columns[indices], rotation=90)
        plt.title(f"{model_name} - Feature Importance")
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------
# 7️⃣ Main
# ----------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_name, best_model = select_and_save_best(results)

    # Plot & analyze best model
    plot_evaluation(best_model, X_test, y_test, best_name)
    show_feature_importance(best_model, X_train, best_name)

    # Optional: Print classification report
    print("\nDetailed Report for Best Model:")
    print(results[best_name]["report"])
