import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return predictions

    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_one(self, x):
        # compute the distances
        distances = [self.distance(x, x_train) for x_train in self.X_train]

        # get closest K
        k_closest_indices = np.argsort(distances)[: self.k]
        K_nearest_labels = [self.y_train[k] for k in k_closest_indices]

        # return majority vote for classification
        prediction = Counter(K_nearest_labels).most_common()[0][0]

        # return average for regression
        regression = np.average(K_nearest_labels)

        return prediction
