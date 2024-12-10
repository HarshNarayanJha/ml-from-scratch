# %% Cell 1
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from knn import KNN

# %% Cell 2
# X, y = datasets.make_blobs(n_features=2, centers=3, random_state=10, n_samples=500)
# X, y = datasets.make_classification(
#     n_samples=500,
#     n_features=2,
#     n_informative=2,
#     n_repeated=0,
#     n_redundant=0,
#     n_classes=3,
#     n_clusters_per_class=1,
#     random_state=101
# )

data = datasets.load_iris()

X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.figure()
plt.title("Iris Features (Sepal Length vs Sepal Width)")
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=ListedColormap(["green", "blue", "red"]))
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.scatter(X_test[:, 0], X_test[:, 1], c="gray", s=20)

plt.figure()
plt.title("Iris Features (Petal Length vs Petal Width)")
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.scatter(X[:, 2], X[:, 3], c=y, s=20, cmap=ListedColormap(["green", "blue", "red"]))
plt.scatter(X_test[:, 2], X_test[:, 3], c="gray", s=20)
plt.show()

# %% Cell 3

for i in range(1, 10):
    knn = KNN(k=i)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(f"{i}-Nearest Classifier: {acc * 100}%")
