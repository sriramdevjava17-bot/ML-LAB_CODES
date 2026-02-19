import pandas as pd
import numpy as np

# Series
s1 = pd.Series([10, 20, 30, 40])
s2 = pd.Series([100, 200, 300], index=['A', 'B', 'C'])

print("Series1:\n", s1)
print("Series2:\n", s2)

# DataFrame
df = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "Marks": [80, 90, np.nan, 85],
    "Attendance": [90, 85, 95, 88]
})

print("\nDataFrame:\n", df)

# Access
print("Marks column:\n", df["Marks"])
print("First row:\n", df.iloc[0])
print("Row by loc:\n", df.loc[2])

# Boolean filtering
print("Marks > 85:\n", df[df["Marks"] > 85])

# Missing values
df["Marks"].fillna(df["Marks"].mean(), inplace=True)

# Column operations
df["Grade"] = ["B", "A", "A", "B"]
df["Result"] = df["Marks"] >= 85

print("\nUpdated DF:\n", df)

# Sorting
print("\nSorted by Marks:\n", df.sort_values(by="Marks"))

# Statistics
print("Mean Marks:", df["Marks"].mean())
print("Max Attendance:", df["Attendance"].max())
