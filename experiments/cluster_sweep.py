# Runs k-means over a range of k per dataset, logs each run and draws the k and PCA figures.
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.cluster import KMeans, score
from src.config import N_CLASSES
from src.data import load
from src.logger import log_cluster
from src.seed import set_seed

SETTINGS = {"image": ("logtfidf", [10, 20, 30, 40, 50, 75, 100, 150, 200]),
            "sentiment": ("standardise", [2, 3, 4, 5, 6, 8, 10, 15, 20])}
SCATTER_CLASSES = {"image": 8, "sentiment": 3}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# Fits and logs k-means with k clusters unless already logged, then returns validation labels.
def run(dataset, transform, k, data):
    run_dir = config.RUNS_DIR / dataset / "kmeans" / f"k{k}_{transform}"
    if run_dir.exists():
        return np.load(run_dir / "val_labels.npy")

    Xtr, ytr, Xva, yva = data
    start = time.perf_counter()
    model = KMeans(k).fit(Xtr)
    fit_seconds = time.perf_counter() - start
    val_labels = model.predict(Xva)
    scores = {"train": score(ytr, model.predict(Xtr)), "val": score(yva, val_labels)}
    log_cluster(dataset=dataset, model=model, run_id=f"k{k}_{transform}", transform=transform,
                val_labels=val_labels, scores=scores, fit_seconds=fit_seconds, owner="p3", notes="clustering sweep")
    return val_labels


# Saves the table of true labels against clusters for the validation rows.
def save_contingency(dataset, transform, yva, val_labels):
    table = pd.crosstab(yva, val_labels, rownames=["label"], colnames=["cluster"])
    table.to_csv(config.RESULTS_CSV.parent / f"cluster_contingency_{dataset}_{transform}.csv")
    return table


# Plots validation ARI, NMI and training inertia against k for one dataset.
def plot_k(dataset, transform):
    results = pd.read_csv(config.RESULTS_CSV.parent / "cluster_results.csv")
    results = results[(results["dataset"] == dataset) & (results["model"] == "kmeans")
                      & (results["transform"] == transform)]
    results = results.assign(k=results["params_json"].map(lambda p: json.loads(p)["n_clusters"])).sort_values("k")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(results["k"], results["val_ari"], "o-", color=COLOURS[0], label="ARI")
    left.plot(results["k"], results["val_nmi"], "o-", color=COLOURS[1], label="NMI")
    left.set_ylabel("agreement with true labels (val)")
    left.legend()
    right.plot(results["k"], results["inertia"], "o-", color=COLOURS[0])
    right.set_ylabel("inertia (train)")
    for ax in (left, right):
        ax.axvline(N_CLASSES[dataset], color="gray", linestyle="--")
        ax.set_xlabel("number of clusters k (dashed: number of classes)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"k-means on {dataset} ({transform})")
    fig.tight_layout()
    fig.savefig(config.ROOT / "figures" / f"cluster_k_{dataset}.png", dpi=150)
    plt.close(fig)


# Plots training rows on two principal components, coloured by true label and by cluster.
def plot_pca(dataset, transform, data):
    Xtr, ytr, _, _ = data
    Ptr, _, _, _ = load(dataset, "pca2")
    keep = ytr < SCATTER_CLASSES[dataset]
    n = SCATTER_CLASSES[dataset]
    clusters = KMeans(n).fit(Xtr[keep]).predict(Xtr[keep])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, groups, title in [(axes[0], ytr[keep], "true labels"), (axes[1], clusters, f"k-means, k={n}")]:
        for g in range(n):
            P = Ptr[keep][groups == g]
            ax.scatter(P[:, 0], P[:, 1], s=4, alpha=0.4, color=COLOURS[g], label=str(g))
        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.legend(markerscale=3, fontsize=8, ncol=2 if n > 4 else 1)
    axes[0].set_ylabel("PC 2")
    shown = "all classes" if n == N_CLASSES[dataset] else f"classes 0-{n - 1}"
    fig.suptitle(f"{dataset}: train rows of {shown} on the first two principal components "
                 f"(cluster colours don't match label colours)")
    fig.tight_layout()
    fig.savefig(config.ROOT / "figures" / f"cluster_pca_{dataset}.png", dpi=150)
    plt.close(fig)


# Runs the k sweep for one dataset, then saves the contingency table and figures.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(N_CLASSES), required=True)
    dataset = parser.parse_args().dataset
    transform, ks = SETTINGS[dataset]

    set_seed()
    data = load(dataset, transform)
    for k in ks:
        val_labels = run(dataset, transform, k, data)
        if k == N_CLASSES[dataset]:
            table = save_contingency(dataset, transform, data[3], val_labels)
    if dataset == "sentiment":
        print(table)
    plot_k(dataset, transform)
    plot_pca(dataset, transform, data)


if __name__ == "__main__":
    main()
