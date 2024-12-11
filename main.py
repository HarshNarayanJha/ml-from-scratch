# %% Cell 1
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree
from naive_bayes import NaiveBayes
from random_forest import RandomForest
from knn import KNN
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression

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


do_naive_bayes()
