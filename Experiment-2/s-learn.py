import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


df = pd.read_csv("student_data.csv")
print("Original Dataset:\n", df)


X = df.drop("Result", axis=1)
y = df["Result"]

num_imputer = SimpleImputer(strategy="mean")
X[["Age", "Marks"]] = num_imputer.fit_transform(X[["Age", "Marks"]])

print("\nAfter Handling Missing Values:\n", X)

le = LabelEncoder()
X["Gender"] = le.fit_transform(X["Gender"])


y = pd.get_dummies(y)

print("\nAfter Encoding:\n", X)
print("\nEncoded Target:\n", y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Features:\n", X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
