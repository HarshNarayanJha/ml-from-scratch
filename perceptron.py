import numpy as np


def unit_step_func(x):
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_function = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.random(n_features)
        self.bias = np.random.rand()

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_function(linear_output)

                dw = self.lr * (y_[idx] - y_pred) * x_i
                db = self.lr * (y_[idx] - y_pred)
                self.weights += dw
                self.bias += db

    def predict(self, X):
        assert self.weights is not None

        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_function(linear_output)

        return y_pred
