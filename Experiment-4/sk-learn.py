import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dt_data.csv")

# Encode categorical data
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("PlayTennis", axis=1)
y = df["PlayTennis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Decision Tree (Entropy → ID3 style)
dt_entropy = DecisionTreeClassifier(criterion="entropy")
dt_entropy.fit(X_train, y_train)

# Decision Tree (Gini → CART style)
dt_gini = DecisionTreeClassifier(criterion="gini")
dt_gini.fit(X_train, y_train)

# Predictions
pred_entropy = dt_entropy.predict(X_test)
pred_gini = dt_gini.predict(X_test)

print("Accuracy (Entropy):", accuracy_score(y_test, pred_entropy))
print("Accuracy (Gini):", accuracy_score(y_test, pred_gini))
