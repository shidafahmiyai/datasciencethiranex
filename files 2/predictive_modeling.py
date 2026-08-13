"""
Predictive Modeling Using Machine Learning
============================================
Demonstrates a full supervised-learning workflow:
  1. Load & split data
  2. Train multiple algorithms (Logistic Regression, Decision Tree, Random Forest)
  3. Evaluate accuracy / precision / recall / F1
  4. Visualize performance (confusion matrices + ROC curves)

Dataset: Breast Cancer Wisconsin (built into scikit-learn) — a real binary
classification problem (malignant vs. benign tumor), used here as a stand-in
for whatever CSV data you plug in later.

To use your own data: replace the "LOAD DATA" section with
    df = pd.read_csv("your_file.csv")
    X = df.drop(columns=["target_column"])
    y = df["target_column"]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Class balance: {y.value_counts().to_dict()}\n")

# ----------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features (helps Logistic Regression converge / perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 3. TRAIN MODELS
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

results = {}

for name, model in models.items():
    # Logistic Regression benefits from scaled features; trees don't need it
    # but it doesn't hurt, so we keep the pipeline simple and consistent.
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }

# ----------------------------------------------------------------------
# 4. PRINT COMPARISON TABLE
# ----------------------------------------------------------------------
summary = pd.DataFrame({
    name: {k: v for k, v in r.items() if k in ("accuracy", "precision", "recall", "f1", "auc")}
    for name, r in results.items()
}).T.round(4)

print("Model comparison:")
print(summary.to_string())
print()

# ----------------------------------------------------------------------
# 5. VISUALIZE: CONFUSION MATRICES
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.3f}")

plt.tight_layout()
plt.savefig("/home/claude/confusion_matrices.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. VISUALIZE: ROC CURVES
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("/home/claude/roc_curves.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (Random Forest) — bonus insight
# ----------------------------------------------------------------------
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/feature_importance.png", dpi=150)
plt.close()

print("Saved: confusion_matrices.png, roc_curves.png, feature_importance.png")
print("\nBest model by accuracy:", summary["accuracy"].idxmax())
print("Best model by AUC:", summary["auc"].idxmax())
