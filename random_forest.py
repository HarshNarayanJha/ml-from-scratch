from collections import Counter
import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = [self._create_tree(X, y) for _ in range(self.n_trees)]

    def _create_tree(self, X, y):
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features,
        )
        X_samples, y_samples = self._bootstrap_samples(X, y)
        tree.fit(X_samples, y_samples)
        return tree

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        tree_pred = np.array([self._most_common_label(pred) for pred in tree_preds])
        return tree_pred
