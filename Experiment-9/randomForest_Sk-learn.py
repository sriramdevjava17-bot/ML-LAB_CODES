import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names
rf = RandomForestClassifier(
    n_estimators = 3,
    max_depth = 3,
    random_state = 42,
    criterion = "entropy"
)
rf.fit(X, y)
fig, axes = plt.subplots(1, 3, figsize=(16,6))
for i in range(3):
    plot_tree(
        rf.estimators_[i],
        feature_names = feature_names,
        class_names = class_names,
        filled = True,
        ax = axes[i]
    )
    axes[i].set_title("tree "+str(i+1))
plt.tight_layout()
plt.show()