import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

print("===== BREAST CANCER CLASSIFICATION USING KNN =====\n")

# Load dataset
data = load_breast_cancer()
print("Dataset loaded successfully ✅")

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
X = df
y = data.target

print("\nFeature matrix (X) created")
print("Feature matrix shape:", X.shape)

print("\nTarget vector (y) created")
print("Target vector shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nData split into training and testing sets successfully ✅")
print("Training feature shape:", X_train.shape)
print("Testing feature shape :", X_test.shape)
print("Training target shape :", y_train.shape)
print("Testing target shape  :", y_test.shape)

# Feature scaling (VERY IMPORTANT for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature scaling completed using StandardScaler ✅")

# Train KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

print(f"\nKNN model trained successfully with k = {k} ✅")

# ===== TESTING PHASE =====
print("\n===== TESTING PHASE STARTED =====")

# Predict class labels
y_pred = knn.predict(X_test)
print("Class labels predicted for test data")

# Predict probabilities
y_prob = knn.predict_proba(X_test)[:, 1]
print("Prediction probabilities generated for test data")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy calculated successfully")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred)

# ROC & AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("ROC curve values computed")
print("AUC score computed")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN (Breast Cancer Dataset)")
plt.legend()
plt.show()

# Final Outputs
print("\n===== TESTING COMPLETED =====")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
print("Final AUC Score:", auc_score)