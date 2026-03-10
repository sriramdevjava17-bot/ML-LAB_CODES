import pandas as pd, math
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("ensemble_data.csv")

# Convert dataframe rows to dictionary indexed by No
data = {r["No"]: r.to_dict() for _, r in df.iterrows()}

# Bootstrap samples (row numbers)
boots = [[1,2,2,4,6],
         [1,3,4,5,5],
         [2,3,4,6,6]]

# Feature subsets for each tree
features_list = [["Age","Student"],
                 ["Income","Student"],
                 ["Age","Income"]]

# Entropy function
def entropy(rows):
    c = Counter(r["Class"] for r in rows)
    n = len(rows)
    return round(-sum((v/n)*math.log2(v/n) for v in c.values()),4)

# Split rows based on feature
def split(rows, f):
    g = defaultdict(list)
    for r in rows:
        g[r[f]].append(r)
    return g

# Information Gain
def ig(rows, f):
    parent = entropy(rows)
    groups = split(rows, f)
    n = len(rows)
    child = sum((len(g)/n)*entropy(g) for g in groups.values())
    return round(parent-child,4), groups

# Build Decision Tree (ID3)
def build(rows, feats):
    ids = [r["No"] for r in rows]
    h = entropy(rows)
    print("\nRows:", ids, "Entropy:", h)

    node = {"rows":ids,"entropy":h}

    if h == 0:
        node["leaf"] = rows[0]["Class"]
        print("Leaf:", node["leaf"])
        return node

    if not feats:
        node["leaf"] = "mixed"
        print("Leaf: mixed")
        return node

    best, best_ig, best_grp = None, 0, None
    for f in feats:
        val, grp = ig(rows,f)
        print("IG(",f,") =", val)
        if val > best_ig:
            best, best_ig, best_grp = f, val, grp

    if best_ig == 0:
        node["leaf"] = "mixed"
        print("Leaf: mixed")
        return node

    node.update({"feature":best,"ig":best_ig,"children":{}})
    print("Split on:", best)

    for val, sub in best_grp.items():
        print(" Branch", val, [r["No"] for r in sub])
        node["children"][val] = build(sub,[f for f in feats if f != best])

    return node

# Plot tree
def plot(tree, title):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.axis("off")

    def draw(node,x,y,w):
        if "leaf" in node:
            ax.text(x,y,"Leaf\n"+str(node["leaf"]),
                    ha="center",bbox=dict(boxstyle="round",fc="lightblue"))
        else:
            ax.text(x,y,node["feature"]+"\nIG="+str(node["ig"]),
                    ha="center",bbox=dict(boxstyle="round",fc="lightgreen"))
            kids=list(node["children"].items())
            step=w/len(kids)
            start=x-w/2+step/2
            for i,(v,c) in enumerate(kids):
                cx,cy=start+i*step,y-0.15
                ax.plot([x,cx],[y-0.02,cy+0.03])
                ax.text((x+cx)/2,(y+cy)/2,v)
                draw(c,cx,cy,step)

    draw(tree,0.5,0.9,0.8)
    ax.set_title(title)
    plt.show()

# Build and plot trees
for i in range(len(boots)):
    rows = [data[j] for j in boots[i]]
    print("\n\nTREE", i+1)
    t = build(rows, features_list[i])
    plot(t, "Tree "+str(i+1))