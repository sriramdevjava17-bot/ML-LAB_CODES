import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y1 = [10,20,25,30,40]
y2 = [5,15,20,25,35]

print("Generating plots...")

# Line plot
plt.figure()
plt.plot(x, y1, label="Line1")
plt.plot(x, y2, label="Line2")
plt.title("Line Plot")
plt.legend()
plt.grid()
plt.show()

# Bar chart
plt.figure()
plt.bar(x, y1)
plt.title("Bar Chart")
plt.show()

# Scatter plot
plt.figure()
plt.scatter(x, y1)
plt.title("Scatter Plot")
plt.show()

# Histogram
plt.figure()
plt.hist(y1)
plt.title("Histogram")
plt.show()

# Pie chart
plt.figure()
labels = ["A","B","C","D","E"]
plt.pie(y1, labels=labels, autopct="%1.1f%%")
plt.title("Pie Chart")
plt.show()
