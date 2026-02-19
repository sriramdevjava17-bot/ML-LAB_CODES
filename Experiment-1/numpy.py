import numpy as np

# Array creation
a1 = np.array([10, 20, 30, 40])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
a3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("1D:", a1)
print("2D:\n", a2)
print("3D:\n", a3)

# Data types
print("Datatype:", a1.dtype)

# Indexing & slicing
print("a1[2]:", a1[2])
print("a1[1:3]:", a1[1:3])
print("a2[1][2]:", a2[1][2])

# Boolean indexing
print("Values > 20:", a1[a1 > 20])

# Arithmetic operations
print("Add:", a1 + 2)
print("Multiply:", a1 * 3)
print("Power:", a1 ** 2)

# Universal functions
print("Square root:", np.sqrt(a1))
print("Log:", np.log(a1))
print("Exponential:", np.exp(a1))

# Aggregations
print("Sum:", np.sum(a1))
print("Mean:", np.mean(a1))
print("Max:", np.max(a1))
print("Min:", np.min(a1))
print("Std:", np.std(a1))

# Reshape & flatten
r = np.array([1,2,3,4,5,6])
print("Reshaped:\n", r.reshape(2,3))
print("Flatten:", a2.flatten())

# Transpose
print("Transpose:\n", a2.T)

# Stacking
print("Vertical stack:\n", np.vstack((a2, a2)))
print("Horizontal stack:\n", np.hstack((a2, a2)))

# Splitting
print("Split:", np.split(a1, 2))

# Random numbers
print("Random int:", np.random.randint(1, 50, 5))
print("Random float:", np.random.rand(5))
