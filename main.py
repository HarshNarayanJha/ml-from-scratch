# %% Cell 1
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree
from kmeans import KMeans
from perceptron import Perceptron
from knn import KNN
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes
from pca import PCA
from random_forest import RandomForest
from svm import SVM


# %% Cell KNN
def do_knn_classify():
    X, y = datasets.make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_repeated=0,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=109,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    plt.figure()
    plt.scatter(
        X[:, 0], X[:, 1], c=y, s=20, cmap=ListedColormap(["green", "blue", "red"])
    )

    plt.figure()
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        s=20,
        cmap=ListedColormap(["green", "blue", "red"]),
    )
    plt.scatter(X_test[:, 0], X_test[:, 1], c="gray", s=20)
    plt.show()

    for i in range(1, 10):
        knn = KNN(k=i)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        acc = np.sum(predictions == y_test) / len(y_test)
        print(f"{i}-Nearest Classifier: {acc * 100}%")


# do_knn_classify()


# %% Cell Linear Regression
def do_linear_regression():
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=10, random_state=200
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # plt.figure()
    # plt.scatter(X[:, 0], y, color="b", s=30)

    regressor = LinearRegression(lr=0.01, n_iterations=50000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    def mse(y_test, predictions):
        return np.mean((y_test - predictions) ** 2)

    print(mse(y_test, predictions))

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    plt.figure()
    plt.scatter(X_train, y_train, color=cmap(0.9), s=30, label="Train")
    plt.scatter(X_test, y_test, color=cmap(0.5), s=30, label="Test")
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.legend()
    plt.show()


# do_linear_regression()


# %% Cell Logistic Regression
def do_logistic_regression():
    X, y = datasets.make_classification(
        n_samples=500,
        n_features=3,
        n_informative=2,
        n_repeated=0,
        n_redundant=1,
        n_classes=2,
        random_state=100,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression(n_iterations=10000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_pred == y_test) / len(y_test)

    print(accuracy(y_test, predictions))


# do_logistic_regression()


# %% Cell Desicion Tree
def do_decision_tree():
    X, y = datasets.make_classification(
        n_samples=500,
        n_features=3,
        n_informative=2,
        n_repeated=0,
        n_redundant=1,
        n_classes=2,
        random_state=100,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTree(max_depth=200)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    print(accuracy(y_test, predictions))


do_decision_tree()


# %% Cell Random Forest
def do_random_forest():
    X, y = datasets.make_classification(
        n_samples=500,
        n_features=3,
        n_informative=2,
        n_repeated=0,
        n_redundant=1,
        n_classes=2,
        random_state=100,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForest(n_trees=100)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    print(accuracy(y_test, predictions))


# do_random_forest()


# %% Cell Naive Bayes
def do_naive_bayes():
    X, y = datasets.make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=50
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    print(accuracy(y_test, predictions))


# do_naive_bayes()


# %% Cell PCA
def do_pca():
    data = datasets.load_iris()
    X, y = data.data, data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print(f"Shape of X: {X.shape}")
    print(f"Shape of transformed X: {X_projected.shape}")

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolors="none", alpha=0.8, cmap=plt.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()


# do_pca()


# %% Cell Perceptron
def do_perceptron():
    X, y = datasets.make_blobs(
        n_samples=150,
        n_features=2,
        centers=2,
        cluster_std=1.05,
        random_state=100,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_pred == y_test) / len(y_test)

    print(accuracy(y_test, predictions) * 100)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()


# do_perceptron()


# %% Cell SVM
def do_svm():
    X, y = datasets.make_blobs(
        n_samples=50,
        n_features=2,
        centers=2,
        cluster_std=1.05,
        random_state=17,
    )
    y = np.where(y <= 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    print(accuracy(y_test, y_pred))


# do_svm()

# %% Cell KMeans
def do_kmeans():
    X, y = datasets.make_blobs(
        centers=5, n_samples=500, n_features=2, random_state=100
    )

    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=1500, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()

do_kmeans()
