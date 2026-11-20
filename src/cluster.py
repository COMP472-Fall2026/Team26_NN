# Implements k-means from scratch in NumPy and scores clusters against the true labels with ARI and NMI.
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.config import N_CLASSES, SEED
from src.data import load
from src.logger import log_cluster
from src.seed import set_seed


class KMeans:
    name = "kmeans"

    # Stores the k-means settings
    def __init__(self, n_clusters, n_init=10, max_iter=300, tol=1e-4, seed=SEED):
        self.params = {"n_clusters": n_clusters, "n_init": n_init, "max_iter": max_iter, "tol": tol, "seed": seed}
        self.centers = None
        self.inertia = None

    # Runs k-means from several random starts and keeps the one with lowest inertia.
    def fit(self, X):
        rng = np.random.default_rng(self.params["seed"])
        for _ in range(self.params["n_init"]):
            centers, inertia = self._lloyd(X, self._init_centers(X, rng))
            if self.inertia is None or inertia < self.inertia:
                self.centers, self.inertia = centers, inertia
        return self

    # Assigns each row to the nearest fitted centre by squared distance.
    def predict(self, X):
        return sq_distances(X, self.centers).argmin(axis=1)

    # Picks starting centres with k-means++ so they begin far apart.
    def _init_centers(self, X, rng):
        # k-means++: each new center is drawn with probability proportional to its squared distance to the nearest center
        centers = [X[rng.integers(len(X))]]
        d2 = sq_distances(X, centers[0][None]).ravel().astype(np.float64)
        for _ in range(self.params["n_clusters"] - 1):
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
            d2 = np.minimum(d2, sq_distances(X, centers[-1][None]).ravel())
        return np.array(centers)

    # Alternates assigning rows to centres and moving centres to means until they settle.
    def _lloyd(self, X, centers):
        for _ in range(self.params["max_iter"]):
            d2 = sq_distances(X, centers)
            labels = d2.argmin(axis=1)
            new = centers.copy()
            for k in range(len(centers)):
                members = X[labels == k]
                if len(members):
                    new[k] = members.mean(axis=0)
                else:
                    # An empty cluster takes the point that is currently worst served
                    new[k] = X[d2.min(axis=1).argmax()]
            shift = np.abs(new - centers).max()
            centers = new
            if shift < self.params["tol"]:
                break
        d2 = sq_distances(X, centers)
        return centers, float(d2.min(axis=1).sum())


# Computes squared Euclidean distances from every row to every centre with matrix products.
def sq_distances(X, C):
    d2 = (X ** 2).sum(axis=1)[:, None] - 2 * X @ C.T + (C ** 2).sum(axis=1)[None, :]
    return np.maximum(d2, 0)


# Measures agreement between true labels and clusters with ARI and NMI.
def score(y, labels):
    return {"ari": adjusted_rand_score(y, labels), "nmi": normalized_mutual_info_score(y, labels)}


# Fits k-means on one dataset's training rows, logs the run and prints scores.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(N_CLASSES), required=True)
    parser.add_argument("--transform", default="raw")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    k = args.k or N_CLASSES[args.dataset]

    set_seed()
    Xtr, ytr, Xva, yva = load(args.dataset, args.transform)
    start = time.perf_counter()
    model = KMeans(k).fit(Xtr)
    fit_seconds = time.perf_counter() - start

    # Fitted on train rows only; val rows are just assigned to the nearest center
    train, val = score(ytr, model.predict(Xtr)), score(yva, model.predict(Xva))
    log_cluster(dataset=args.dataset, model=model, run_id=f"k{k}_{args.transform}", transform=args.transform,
                val_labels=model.predict(Xva), scores={"train": train, "val": val},
                fit_seconds=fit_seconds, owner="p3")
    if k <= 10:
        print(pd.crosstab(yva, model.predict(Xva), rownames=["label"], colnames=["cluster"]))


if __name__ == "__main__":
    main()
