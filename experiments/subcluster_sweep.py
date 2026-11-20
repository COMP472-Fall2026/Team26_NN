# Checks whether image classes split into sub-clusters, using nearest sub-centroid accuracy and silhouette scores.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src import config
from src.cluster import KMeans, sq_distances
from src.config import N_CLASSES, SEED
from src.data import load
from src.seed import set_seed

TRANSFORM = "logtfidf"
CENTROID_MS = [1, 2, 3, 5, 8, 12, 20]       # sub-clusters per class for test 1
SILHOUETTE_MS = [2, 3, 4, 5]                # test 2 (silhouette needs at least 2 clusters)
ONE_NN_VAL_ACC = 0.57675                    # best k-NN (k1_logtfidf) from results.csv, for reference
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]


# Fits m k-means centres per class and scores nearest-centre classification on validation rows.
def sub_centroid_accuracy(Xtr, ytr, Xva, yva, m):
    centers, center_class = [], []
    for c in range(N_CLASSES["image"]):
        model = KMeans(m, seed=SEED + c).fit(Xtr[ytr == c])
        centers.append(model.centers)
        center_class += [c] * m
    nearest = sq_distances(Xva, np.vstack(centers)).argmin(axis=1)
    return float((np.array(center_class)[nearest] == yva).mean())


# Shuffles each column independently to break structure, then L2-normalises each row again.
def shuffle_columns(X, rng):
    Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    return Xn / np.where(norms == 0, 1, norms)


# Computes silhouette scores of within-class clusters and of shuffled null data for each class.
def silhouette_table(Xtr, ytr):
    rng = np.random.default_rng(SEED)
    rows = []
    for c in range(N_CLASSES["image"]):
        Xc = Xtr[ytr == c]
        Xnull = shuffle_columns(Xc, rng)
        for m in SILHOUETTE_MS:
            labels = KMeans(m, seed=SEED + c).fit(Xc).predict(Xc)
            null_labels = KMeans(m, seed=SEED + c).fit(Xnull).predict(Xnull)
            rows.append({"class": c, "m": m, "n_rows": len(Xc),
                         "silhouette": silhouette_score(Xc, labels),
                         "silhouette_null": silhouette_score(Xnull, null_labels),
                         "largest_cluster_share": np.bincount(labels).max() / len(Xc)})
    return pd.DataFrame(rows)


# Plots sub-centroid accuracy against m and median silhouette against m, real versus null.
def plot(acc, sil):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    left.plot(acc["m"], acc["val_acc"], "o-", color=COLOURS[0], label="nearest sub-centroid")
    left.axhline(ONE_NN_VAL_ACC, color="gray", linestyle="--", label=f"1-NN ({ONE_NN_VAL_ACC:.3f})")
    left.set_xlabel("sub-clusters per class m (m=1: one mean per class)")
    left.set_ylabel("val accuracy")
    left.set_title("Test 1: more centers per class")
    left.legend()

    summary = sil.groupby("m")[["silhouette", "silhouette_null"]].median()
    right.plot(summary.index, summary["silhouette"], "o-", color=COLOURS[0], label="real class rows")
    right.plot(summary.index, summary["silhouette_null"], "o-", color=COLOURS[1], label="null (columns shuffled)")
    right.set_xlabel("sub-clusters per class m")
    right.set_ylabel("median silhouette over 50 classes")
    right.set_ylim(bottom=0)             # silhouette runs from -1 to 1; 0.03 is weak, the axis shouldn't hide that
    right.set_title("Test 2: are the sub-clusters tight?")
    right.legend()

    for ax in (left, right):
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Within-class k-means on image ({TRANSFORM}), train rows only")
    fig.tight_layout()
    fig.savefig(config.ROOT / "figures" / "subcluster_image.png", dpi=150)
    plt.close(fig)


# Runs both sub-cluster tests on image training rows, then saves tables and figure.
def main():
    set_seed()
    Xtr, ytr, Xva, yva = load("image", TRANSFORM)

    acc = []
    for m in CENTROID_MS:
        acc.append({"m": m, "val_acc": sub_centroid_accuracy(Xtr, ytr, Xva, yva, m)})
    acc = pd.DataFrame(acc)

    sil = silhouette_table(Xtr, ytr)
    out = config.RESULTS_CSV.parent
    acc.to_csv(out / "subcluster_image_accuracy.csv", index=False)
    sil.to_csv(out / "subcluster_image_silhouette.csv", index=False)
    plot(acc, sil)

    sil["above_null"] = sil["silhouette"] > sil["silhouette_null"]
    print(sil.groupby("m")[["silhouette", "silhouette_null", "above_null", "largest_cluster_share"]]
          .agg({"silhouette": "median", "silhouette_null": "median", "above_null": "sum",
                "largest_cluster_share": "median"}).round(3))


if __name__ == "__main__":
    main()
