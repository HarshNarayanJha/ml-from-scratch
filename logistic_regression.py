import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iterations=1000, randomize=True) -> None:
        self.lr = lr
        self.n_iterations = n_iterations
        self.randomize = randomize
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.randomize:
            self.weights = np.random.random(n_features)
            self.bias = np.random.rand()
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0

        for _ in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(2 * X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(2 * (y_pred - y))

            self.weights = self.weights - self.lr * dw
            self.bia = self.bias - self.lr * db

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        assert self.weights is not None
        assert self.bias is not None

        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]

        return class_pred
