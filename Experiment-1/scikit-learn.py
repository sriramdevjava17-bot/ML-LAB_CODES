from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Manual dataset
X = [[1],[2],[3],[4],[5],[6],[7],[8]]
y = [0,0,0,1,1,1,1,1]

print("========== SPLITTING ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

print("Training size:", len(X_train))
print("Testing size:", len(X_test))

print("\n========== SCALING ==========")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n========== MODEL TRAINING ==========")
model = LogisticRegression()
model.fit(X_train, y_train)

print("\n========== PREDICTION ==========")
y_pred = model.predict(X_test)
print("Predicted:", y_pred)

print("\n========== EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
