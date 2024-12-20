import matplotlib.pyplot as plt
import numpy as np

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K =  K
        self.max_iterms = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each clusters
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iterms):
            # assign samples to closest (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from the cluster
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify the samples as the index of thier clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sameple will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = idx

        return labels

    def _create_clusters(self, centroids):
        # assign the samples to the closes centroid
        clusters = [[] for _ in range(self.K)]

        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distance = [np.sqrt(np.sum((sample-point)**2)) for point in centroids]
        closest_idx = np.argmin(distance)
        return closest_idx

    def _get_centroids(self, clusters):
        # assign the mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean

        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [np.sqrt(np.sum((centroids_old[i] - centroids[i])**2)) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
