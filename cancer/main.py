# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    classification_report, roc_curve, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# 👉 Fix for plot not showing (VERY IMPORTANT)
plt.style.use('default')
plt.switch_backend('TkAgg')   # Works in VS Code / Python

# Step 2: Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Class Labels:")
print("0 -> Malignant (Cancer)")
print("1 -> Benign (Non-Cancer)")

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Step 6: Predict Labels
y_pred = model.predict(X_test)

# Step 7: Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Step 8: Recall
print("Recall:", recall_score(y_test, y_pred))

# Step 9: Classification Report
print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=["Cancer (Malignant)", "Non-Cancer (Benign)"]
))

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix Values:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=["Cancer", "Non-Cancer"],
            yticklabels=["Cancer", "Non-Cancer"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Cancer vs Non-Cancer)")
plt.show()

# Step 11: ROC Curve

# Get probability scores
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute AUC score
auc_score = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", auc_score)

# Plot ROC Curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0,1], [0,1], linestyle='--')  # diagonal line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer Detection")
plt.legend()

# 👉 Important so window doesn't close
plt.show(block=True)