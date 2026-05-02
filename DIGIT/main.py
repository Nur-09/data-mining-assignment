# EXPERIMENT 1: Digit Recognition (Simple Version)

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Train model
model = SVC()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Confusion Matrixp
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Show some predictions
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i].reshape(8,8), cmap='gray')
    plt.title("Pred: " + str(y_pred[i]))
    plt.axis('off')

plt.tight_layout()
plt.show()
