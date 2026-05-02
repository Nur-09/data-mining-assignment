# Step 1: Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Create KNN model
k = 3
model = KNeighborsClassifier(n_neighbors=k)

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Test model
y_pred = model.predict(X_test)

# Step 7: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Take user input for prediction
print("\nEnter flower details:")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

# Step 9: Predict
user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(user_data)

# Step 10: Show result
print("\nPredicted Flower:", iris.target_names[prediction][0])
