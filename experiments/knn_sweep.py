# Sweeps k-NN over k, distance, voting rule and transform, saves the curve and logs the key runs.
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.config import N_CLASSES
from src.data import load
from src.knn import KNN, METRICS
from src.logger import log_run
from src.seed import set_seed

TRANSFORMS = {"image": ["raw", "standardise", "logtfidf"],
              "sentiment": ["raw", "standardise", "pca50", "pca100", "pca200"]}
VOTES = ("uniform", "distance")   # k-NN sweep voting rules; exponential votes have their own sweep (knn_exp_sweep.py)
KS = list(range(1, 41)) + [50, 60, 70, 80, 90, 100, 120, 150, 200]
PAGE_KS = [1, 3, 5, 10, 20]                       # the k values the assignment page suggests
DEFAULTS = {"k": 5, "metric": "euclidean", "weights": "uniform"}
SHORT = {"euclidean": "euc", "cosine": "cos", "manhattan": "man", "uniform": "uni", "distance": "dist"}

# Categorical slots 1-3 of the reference palette: one colour per distance, line style for voting.
COLOURS = {"euclidean": "#2a78d6", "cosine": "#eb6834", "manhattan": "#1baf7a"}
STYLES = {"distance": "-", "uniform": "--"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# Builds the run name from the non-default settings, with the transform last.
def run_id(k, metric, weights, transform):
    parts = [f"k{k}"] if k != DEFAULTS["k"] else []
    parts += [SHORT[metric]] if metric != DEFAULTS["metric"] else []
    parts += [SHORT[weights]] if weights != DEFAULTS["weights"] else []
    return "_".join(parts + [transform])


# Scores the first k neighbours with the given voting rule against the true labels.
def accuracy(model, dist, idx, y, k, weights):
    return float((model.vote(dist[:, :k], idx[:, :k], weights).argmax(axis=1) == y).mean())


# Scores every transform, distance, voting rule and k on training and validation rows.
def sweep_curve(dataset):
    rows = []
    for transform in TRANSFORMS[dataset]:
        Xtr, ytr, Xva, yva = load(dataset, transform)
        for metric in METRICS:
            model = KNN(N_CLASSES[dataset], k=max(KS), metric=metric).fit(Xtr, ytr)
            d_va, i_va = model.kneighbors(Xva)
            d_tr, i_tr = model.kneighbors(Xtr)     # training rows find themselves first, as in log_run's train_acc
            for weights in VOTES:
                for k in KS:
                    rows.append({"dataset": dataset, "transform": transform, "metric": metric, "weights": weights,
                                 "k": k, "train_acc": accuracy(model, d_tr, i_tr, ytr, k, weights),
                                 "val_acc": accuracy(model, d_va, i_va, yva, k, weights)})
    return pd.DataFrame(rows)


# Returns the row with the highest validation accuracy, preferring the smaller k on ties.
def pick_best(df):
    return df.sort_values(["val_acc", "k"], ascending=[False, True], kind="stable").iloc[0]


# Lists the settings worth logging: each combination's best k and the suggested k values.
def key_runs(curve):
    picks = []
    for (transform, metric, weights), g in curve.groupby(["transform", "metric", "weights"], sort=False):
        picks.append((transform, metric, weights, int(pick_best(g)["k"])))
    top = pick_best(curve)
    picks += [(top["transform"], top["metric"], top["weights"], k) for k in PAGE_KS]
    return list(dict.fromkeys(picks))                  # drop repeats, keep order


# Logs each key setting as a full run and checks it matches the curve.
def log_key_runs(dataset, curve):
    n_classes = N_CLASSES[dataset]
    picks = key_runs(curve)
    cache = {}
    for transform, metric, weights, k in picks:
        rid = run_id(k, metric, weights, transform)
        if (config.RUNS_DIR / dataset / "knn" / rid).exists():
            continue
        if transform not in cache:
            cache = {transform: load(dataset, transform)}   # keep one transform in memory at a time
        Xtr, ytr, Xva, yva = cache[transform]
        t0 = time.perf_counter()
        model = KNN(n_classes, k=k, metric=metric, weights=weights).fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - t0
        rec = log_run(dataset=dataset, model=model, run_id=rid, transform=transform, y_train=ytr, X_train=Xtr,
                      y_val=yva, X_val=Xva, fit_seconds=fit_seconds, owner="p1", notes="k-NN sweep")
        want = curve[(curve["transform"] == transform) & (curve["metric"] == metric)
                     & (curve["weights"] == weights) & (curve["k"] == k)].iloc[0]
        assert rec["val_acc"] == want["val_acc"] and rec["train_acc"] == want["train_acc"], \
            f"{rid}: logged accuracy differs from the curve"


# Applies the shared log-scale axis and grid styling to one k-NN plot.
def _style(ax):
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10, 20, 40, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


# Draws the validation-against-k figure and the train-against-validation figure for one dataset.
def plot(dataset, curve):
    config.ROOT.joinpath("figures").mkdir(exist_ok=True)
    transforms = TRANSFORMS[dataset]

    # 1. Validation accuracy against k: one panel per transform, colour = distance, line style = voting.
    ncols = min(len(transforms), 3)
    nrows = -(-len(transforms) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows + 0.8), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(transforms):]:
        ax.set_visible(False)
    lo, hi = curve["val_acc"].min(), curve["val_acc"].max()
    for ax, transform in zip(axes, transforms):
        for metric in METRICS:
            for weights in VOTES:
                g = curve[(curve["transform"] == transform) & (curve["metric"] == metric)
                          & (curve["weights"] == weights)]
                ax.plot(g["k"], g["val_acc"], STYLES[weights], color=COLOURS[metric], linewidth=2,
                        label=f"{metric}, {weights}")
        best = pick_best(curve[curve["transform"] == transform])
        ax.plot(best["k"], best["val_acc"], "o", markersize=8, color=COLOURS[best["metric"]],
                markeredgecolor="white", markeredgewidth=2, zorder=5)
        at_edge = best["k"] == min(KS)                  # keep a k=1 label clear of the y-axis
        ax.annotate(f"{best['val_acc']:.3f} (k={best['k']})", (best["k"], best["val_acc"]),
                    xytext=(-4 if at_edge else 0, 8), textcoords="offset points",
                    ha="left" if at_edge else "center", fontsize=9, color=INK)
        ax.set_title(transform, fontsize=11, color=INK)
        ax.set_xlabel("k (log scale)", fontsize=9, color=MUTED)
        ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.15 * (hi - lo))   # headroom for the best-point label
        _style(ax)
    for ax in axes[::ncols]:
        ax.set_ylabel("validation accuracy", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False, labelcolor=INK)
    fig.suptitle(f"k-NN on {dataset}: validation accuracy against k", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.75 / fig.get_figheight(), 1, 1))
    fig.savefig(config.ROOT / "figures" / f"knn_k_curve_{dataset}.png", dpi=150)
    plt.close(fig)

    # 2. Train against validation accuracy for the best transform and distance, both voting rules.
    top = pick_best(curve)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for weights in VOTES:
        g = curve[(curve["transform"] == top["transform"]) & (curve["metric"] == top["metric"])
                  & (curve["weights"] == weights)]
        ax.plot(g["k"], g["train_acc"], STYLES[weights], color=COLOURS["euclidean"], linewidth=2, label=f"train, {weights}")
        ax.plot(g["k"], g["val_acc"], STYLES[weights], color=COLOURS["cosine"], linewidth=2, label=f"validation, {weights}")
    ax.set_xlabel("k (log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel("accuracy", fontsize=9, color=MUTED)
    ax.set_title(f"k-NN on {dataset} ({top['transform']}, {top['metric']}): train against validation",
                 fontsize=11, color=INK)
    _style(ax)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(config.ROOT / "figures" / f"knn_train_val_{dataset}.png", dpi=150)
    plt.close(fig)


# Runs the sweep or reads the saved curve, logs key runs, then draws the figures.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(N_CLASSES))
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from the saved curve file")
    args = ap.parse_args()
    set_seed()

    curve_path = config.RESULTS_CSV.parent / f"knn_curve_{args.dataset}.csv"
    if args.plot_only:
        curve = pd.read_csv(curve_path)
    else:
        curve = sweep_curve(args.dataset)
        curve_path.parent.mkdir(exist_ok=True)
        curve.to_csv(curve_path, index=False, float_format="%.6f")
        log_key_runs(args.dataset, curve)
    plot(args.dataset, curve)


if __name__ == "__main__":
    main()
