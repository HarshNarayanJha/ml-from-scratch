import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iterations=1000) -> None:
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize random weights and bias

        n_samples, n_features = X.shape
        self.weights = np.random.random(n_features)
        self.bias = np.random.rand()

        for _ in range(self.n_iterations):
            # calculate y predicted using y = wx + b
            y_pred = np.dot(X, self.weights) + self.bias

            # calculate error, gradient descent
            dw = (1 / n_samples) * np.dot(2 * X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(2 * (y_pred - y))

            # update weight and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        assert self.weights is not None
        assert self.bias is not None

        return np.dot(X, self.weights) + self.bias
