import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

print("===== BREAST CANCER CLASSIFICATION USING LOGISTIC REGRESSION =====\n")

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

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature scaling completed using StandardScaler ✅")

# Train Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully ✅")

# Testing phase
print("\n===== TESTING PHASE STARTED =====")

# Predict probabilities on test data
y_prob = model.predict_proba(X_test)[:, 1]
print("Prediction probabilities generated for test data")

# ROC Curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
print("ROC curve values computed")

# AUC score
auc_score = roc_auc_score(y_test, y_prob)
print("\nAUC Score computed successfully")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer Dataset")
plt.legend()
plt.show()

print("\n===== TESTING COMPLETED =====")
print("Final AUC Score:", auc_score)