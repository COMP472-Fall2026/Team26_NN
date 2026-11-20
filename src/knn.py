# Implements k-nearest neighbours from scratch in NumPy, with three distances and three voting rules.
import numpy as np

from src.base import BaseModel

METRICS = ("euclidean", "cosine", "manhattan")
WEIGHTS = ("uniform", "distance", "exp")


class KNN(BaseModel):
    name = "knn"

    # Checks the k-NN settings and stores them; temperature is only kept for exponential votes.
    def __init__(self, n_classes: int, k: int = 5, metric: str = "euclidean",
                 weights: str = "uniform", temperature: float = 0.03, batch_size: int = 512, **params):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if weights not in WEIGHTS:
            raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if weights == "exp":
            if temperature <= 0:
                raise ValueError(f"temperature must be positive, got {temperature}")
            params["temperature"] = temperature   # only logged when it is used, so older run configs stay valid
        super().__init__(n_classes, k=k, metric=metric, weights=weights, **params)
        self.batch_size = batch_size         # memory only, not a hyperparameter, so not logged

    # Stores the training rows and precomputes what the chosen distance needs.
    def fit(self, X, y, X_val=None, y_val=None) -> "KNN":
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=np.int64)
        if self.params["k"] > len(self.X_):
            raise ValueError(f"k={self.params['k']} is larger than the {len(self.X_)} training rows")
        if self.params["metric"] == "cosine":
            self.Xn_ = self.X_ / np.maximum(np.linalg.norm(self.X_, axis=1, keepdims=True), 1e-12)
        elif self.params["metric"] == "euclidean":
            self.sq_ = (self.X_ ** 2).sum(axis=1)
        else:
            self.Xt_ = np.ascontiguousarray(self.X_.T)   # one feature per row, so each column read below is contiguous
        return self

    # Computes distances from each query row to every training row with the chosen metric.
    def _distances(self, Q):
        metric = self.params["metric"]
        if metric == "euclidean":
            d2 = (Q ** 2).sum(axis=1)[:, None] + self.sq_[None, :] - 2 * Q @ self.X_.T
            return np.sqrt(np.maximum(d2, 0))
        if metric == "cosine":
            Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
            return 1 - Qn @ self.Xn_.T
        D = np.zeros((len(Q), len(self.X_)), dtype=np.float32)
        diff = np.empty_like(D)
        for j in range(Q.shape[1]):
            np.subtract(Q[:, j:j + 1], self.Xt_[j], out=diff)
            np.abs(diff, out=diff)
            D += diff
        return D

    # Finds the k nearest training rows per query, breaking ties by lower row index.
    @staticmethod
    def _nearest(D, k):
        if k >= D.shape[1]:
            return np.argsort(D, axis=1, kind="stable")
        cand = np.argpartition(D, k - 1, axis=1)[:, :k]         
        order = np.lexsort((cand, np.take_along_axis(D, cand, axis=1)), axis=1)   # by distance and then index
        idx = np.take_along_axis(cand, order, axis=1)
        kth = np.take_along_axis(D, idx[:, -1:], axis=1)
        tied = (D <= kth).sum(axis=1) > k
        if tied.any():
            idx[tied] = np.argsort(D[tied], axis=1, kind="stable")[:, :k]
        return idx

    # Returns distances and indices of the k nearest training rows, in batches.
    def kneighbors(self, X, k=None):
        k = self.params["k"] if k is None else k
        if not 1 <= k <= len(self.X_):
            raise ValueError(f"k must be between 1 and {len(self.X_)}, got {k}")
        X = np.asarray(X, dtype=np.float32)
        dist = np.empty((len(X), k), dtype=np.float32)
        idx = np.empty((len(X), k), dtype=np.int64)
        for start in range(0, len(X), self.batch_size):
            D = self._distances(X[start:start + self.batch_size])
            nn = self._nearest(D, k)
            dist[start:start + len(D)] = np.take_along_axis(D, nn, axis=1)
            idx[start:start + len(D)] = nn
        return dist, idx

    # Turns neighbour lists into class probabilities using uniform, distance or exponential weights.
    def vote(self, dist, idx, weights=None):
        weights = self.params["weights"] if weights is None else weights
        if weights == "distance":
            w = 1 / np.maximum(dist, 1e-8)
            exact = dist <= 1e-8
            hit = exact.any(axis=1)              # an exact match gets all the vote, as 1/0 would
            w[hit] = exact[hit]
        elif weights == "exp":
            w = np.exp(-(dist - dist[:, :1]) / self.params["temperature"])
        else:
            w = np.ones_like(dist)
        votes = np.zeros((len(dist), self.n_classes), dtype=np.float32)
        rows = np.repeat(np.arange(len(dist)), dist.shape[1])
        np.add.at(votes, (rows, self.y_[idx].ravel()), w.ravel())
        return votes / votes.sum(axis=1, keepdims=True)

    # Finds the k nearest neighbours of each row and turns their votes into probabilities.
    def predict_proba(self, X):
        return self.vote(*self.kneighbors(X))
