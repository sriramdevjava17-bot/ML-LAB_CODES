import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("===== BREAST CANCER CLASSIFICATION USING KNN =====\n")

# Load dataset
data = load_breast_cancer()
print("Dataset loaded successfully")

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
X = df
y = data.target

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nData split completed")
print("Training samples:", X_train.shape)
print("Testing samples :", X_test.shape)

# Feature Scaling (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature scaling completed")

# =====================================
# Finding Optimal K Value
# =====================================
print("\n===== FINDING OPTIMAL K VALUE =====")

k_values = range(1, 21)
accuracies = []

for k in k_values:
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    accuracies.append(acc)
    print(f"K = {k} -> Accuracy = {acc:.4f}")

# Best K
optimal_k = k_values[accuracies.index(max(accuracies))]
print("\nOptimal K value:", optimal_k)

# Plot K vs Accuracy
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")
plt.show()

# =====================================
# Train Final Model with Optimal K
# =====================================

print("\n===== TRAINING FINAL MODEL =====")

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report
report = classification_report(y_test, y_pred)

print("\n===== FINAL RESULTS =====")

print("Optimal K:", optimal_k)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", report)