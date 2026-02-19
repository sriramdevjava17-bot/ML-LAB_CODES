import pandas as pd
import math

df = pd.read_csv("dt_data.csv")

# Entropy function
def entropy(col):
    values = col.value_counts()
    total = len(col)
    ent = 0
    for count in values:
        p = count / total
        ent -= p * math.log2(p)
    return ent

# Information Gain
def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    values = df[attr].unique()
    weighted_entropy = 0

    for v in values:
        subset = df[df[attr] == v]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

target = "PlayTennis"

print("Total Entropy:", entropy(df[target]))
print("\nInformation Gain:")
for col in df.columns[:-1]:
    print(col, ":", info_gain(df, col, target))
