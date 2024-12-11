import numpy as np


class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        evecs, evals = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose for easier calculations
        evecs = evecs.T

        # sort eigenvectors
        idxs = np.argsort(evals)[::-1]
        evals = evals[idxs]
        evecs = evecs[idxs]

        self.components = evecs[: self.n_components]

    def transform(self, X):
        # projects data
        assert self.components is not None

        X = X - self.mean
        return np.dot(X, self.components.T)
