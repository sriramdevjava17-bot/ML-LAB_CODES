import pandas as pd
import math

# Load dataset
df = pd.read_csv("knn_data.csv")

X = df[["Height", "Weight"]].values
y = df["Class"].values

# Euclidean distance
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Manual KNN
def knn(X, y, test_point, k):
    distances = []

    for i in range(len(X)):
        dist = euclidean(X[i], test_point)
        distances.append((dist, y[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)

# Test point
test_point = [172, 68]
k = 3

result = knn(X, y, test_point, k)
print("Predicted Class (Manual KNN):", result)
