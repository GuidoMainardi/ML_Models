import numpy as np

class KMeans:

   
    def __init__(self, n_clusters=3, tries=10):
        self.n_clusters = n_clusters
        self.clusters = []
        self.tries = tries

    def fit(self, X):
        self.best_clusters = []
        min_dists = float('inf')
        for _ in range(self.tries):
            self.__init__(n_clusters=self.n_clusters, tries=self.tries)
            self.__fit(X)
            if self._clusters_variance() < min_dists:
                min_dists = self._clusters_variance()
                self.best_clusters = self.central_clusters.copy()

        self.clusters = self.best_clusters

    def __fit(self, X):
        self.data = X
        self.__initialize_clusters(X)
        
        self.__classify(X)
        while self._next_iter(X):
            self.clusters = self.central_clusters.copy()
            self.__classify(X)

    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            preds.append(self.__find_nearest_cluster(row, self.clusters))

        return preds

    def __initialize_clusters(self, X):
        for _ in range(self.n_clusters):
            self.clusters.append(X.sample().values.tolist()[0])
        
        self.classify = {cluster:[] for cluster in range(len(self.clusters))}
        self.central_clusters = [[0]] * self.n_clusters

    def __find_nearest_cluster(self, data, clusters):
        min_dist = float('inf')
        for index, cluster in enumerate(clusters):
            curr_dist = self.distance(data, cluster)
            if curr_dist < min_dist:
                min_dist = curr_dist
                nearest_cluster = index

        return nearest_cluster
        
    def __classify(self, X):
        for _, row in X.iterrows():
            row = row.values.tolist()
            self.classify[self.__find_nearest_cluster(row, self.clusters)].append(row)

        self.central_clusters = [self.__central_point(points) for points in self.classify.values()]

    def _clusters_variance(self):
        total = 0
        for k, v in self.classify.items():
            if len(v):
                distances = np.array(list(map(lambda x: self.distance(self.clusters[k], x), v)))
                total += distances.var()
        
        return total

    def __central_point(self, points):
        return [sum(x)/len(x) for x in zip(*points)]

    def _next_iter(self, X):
        for cluster in self.central_clusters:
            if not cluster:
                return False
        for _, row in X.iterrows():
            row = row.values.tolist()
            if self.__find_nearest_cluster(row, self.clusters) != self.__find_nearest_cluster(row, self.central_clusters):
                return True
        return False

    def distance(self, a, b):
        point_a = np.array(a)
        point_b = np.array(b)
        return np.linalg.norm(point_a - point_b)