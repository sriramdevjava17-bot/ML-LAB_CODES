import pandas as pd

df = pd.read_csv("dt_data.csv")

# Gini Index
def gini(col):
    values = col.value_counts()
    total = len(col)
    g = 1
    for count in values:
        p = count / total
        g -= p ** 2
    return g

# Gini for attribute
def gini_attr(df, attr, target):
    values = df[attr].unique()
    gini_sum = 0

    for v in values:
        subset = df[df[attr] == v]
        gini_sum += (len(subset) / len(df)) * gini(subset[target])

    return gini_sum

target = "PlayTennis"

print("Gini Index for attributes:")
for col in df.columns[:-1]:
    print(col, ":", gini_attr(df, col, target))
