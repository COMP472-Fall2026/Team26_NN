# Refits every finalist setting on the five committed folds and logs its fold accuracies and out-of-fold predictions.
import os

# Each fold runs in its own process, so give each process one thread instead of letting them all fight over the cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.config import N_CLASSES
from src.data import load_fold
from src.forest import RandomForest
from src.knn import KNN
from src.logger import log_cv, restore_cv_arrays
from src.nn import MLP
from src.tree import DecisionTree

MODELS = {"knn": KNN, "tree": DecisionTree, "forest": RandomForest, "nn": MLP}
N_FOLDS = 5

# The best logged run of each model on each dataset (results.csv on 2026-11-13). Where the top runs are tied or within
# one standard error of each other on the fixed split, the runners-up are checked too, so the folds can separate them.
FINALISTS = [
    ("image", "knn", "k1_cos_logtfidf"),         # four k=1 runs tie at 0.5768; on L2-normalised rows they pick the same neighbour
    ("image", "tree", "dnone_ms2_ml1_gini_raw"),
    ("image", "forest", "n100_dnone_mfsqrt_logtfidf"),
    ("image", "nn", "h256_lr1e-3_do0.2_wd0_adam_gelu_bs256_standardise"),
    ("sentiment", "knn", "k90_dist_pca200"),
    ("sentiment", "knn", "k40_dist_raw"),        # the setting behind the first k-NN submission
    ("sentiment", "knn", "k90_man_dist_pca100"),
    ("sentiment", "tree", "d5_ms2_ml1_gini_n9600_raw"),
    ("sentiment", "forest", "n100_dnone_mf64_raw"),
    ("sentiment", "nn", "h256_lr1e-3_do0_wd0_adam_raw"),
    ("sentiment", "nn", "h256x64_lr1e-4_do0_wd0_adam_raw"),   # tied with the one above at 0.6679
    # Added 2026-11-15: new best tree and forest runs from the tree and forest update on main (commit 85ac804).
    ("image", "forest", "n200_dnone_mfsqrt_logtfidf"),
    ("image", "tree", "dnone_ms2_ml10_gini_mf32_b256_pca50"),
    ("sentiment", "tree", "d4_ms2_ml1_gini_raw"),             # ties with ml5 and ml10 at 0.5033; the simplest leaf rule
    # Added 2026-11-16: exponential vote weights (experiments/knn_exp_sweep.py). T=0.025 is the fixed-split best;
    # T=0.03 is the neighbouring setting, checked too because the two are within noise on the fixed split.
    ("image", "knn", "k100_cos_exp0.025_logtfidf"),
    ("image", "knn", "k100_cos_exp0.03_logtfidf"),
    ("sentiment", "knn", "k100_exp2_pca200"),
]


# Reads the saved settings of one logged run from its config file.
def run_config(dataset, model_name, run_id):
    return json.loads((config.RUNS_DIR / dataset / model_name / run_id / "config.json").read_text())


# Fits one finalist on four folds and predicts the held-out fold and test rows.
def run_fold(dataset, model_name, run_id, fold):
    import torch
    torch.set_num_threads(1)

    cfg = run_config(dataset, model_name, run_id)
    params = dict(cfg["params"])
    n_classes = params.pop("n_classes")
    Xtr, ytr, Xva, yva, Xte, va_rows = load_fold(dataset, cfg["transform"], fold)

    model = MODELS[model_name](n_classes, **params)
    t0 = time.perf_counter()
    if model_name == "nn":
        # Same protocol as the network sweep: early stopping watches the held-out rows and keeps the best epoch.
        model.fit(Xtr, ytr, Xva, yva)
    else:
        model.fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    va_proba = np.asarray(model.predict_proba(Xva), dtype=np.float32)
    te_proba = np.asarray(model.predict_proba(Xte), dtype=np.float32)
    acc = float((va_proba.argmax(axis=1) == yva).mean())
    return fold, va_rows, va_proba, te_proba, fit_seconds, acc


# Runs every finalist and fold in parallel processes and logs each finished finalist.
def check(finalists, workers, restore=False):
    jobs = [(d, m, r, f) for d, m, r in finalists for f in range(N_FOLDS)]
    jobs.sort(key=lambda j: j[1] != "forest")          # forests take far longest, so start them first
    pieces = {(d, m, r): [] for d, m, r in finalists}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_fold, *job): job for job in jobs}
        for fut in as_completed(futures):
            d, m, r, f = futures[fut]
            piece = fut.result()
            pieces[(d, m, r)].append(piece)
            if len(pieces[(d, m, r)]) == N_FOLDS:
                done = sorted(pieces.pop((d, m, r)), key=lambda p: p[0])
                try:
                    (restore_arrays if restore else save)(d, m, r, done)
                except ValueError as err:          # one mismatching finalist should not stop the others
                    print(err, flush=True)


# Puts each held-out fold's predictions back into one array in train-CSV row order.
def out_of_fold(dataset, pieces):
    n_train = len(np.load(config.SPLITS_DIR / f"{dataset}_split.npz")["fold"])
    oof = np.full((n_train, N_CLASSES[dataset]), np.nan, dtype=np.float32)
    for _, va_rows, va_proba, *_ in pieces:
        oof[va_rows] = va_proba
    assert not np.isnan(oof).any(), "some training rows were never held out"
    return oof


# Saves the refitted arrays of an already-logged check if its fold accuracies match.
def restore_arrays(dataset, model_name, run_id, pieces):
    # NumPy models are seeded and must match exactly; the network may differ slightly across machines.
    tolerance = 0.01 if model_name == "nn" else 1e-6
    restore_cv_arrays(dataset, model_name, run_id, fold_accs=[p[5] for p in pieces],
                            oof_proba=out_of_fold(dataset, pieces), test_proba=np.stack([p[3] for p in pieces]),
                            tolerance=tolerance)


# Joins the five fold predictions of one finalist and logs the 5-fold check.
def save(dataset, model_name, run_id, pieces):
    cfg = run_config(dataset, model_name, run_id)
    oof = out_of_fold(dataset, pieces)
    log_cv(dataset, model_name, run_id, cfg["transform"], cfg["params"],
                 fold_accs=[p[5] for p in pieces], oof_proba=oof, test_proba=np.stack([p[3] for p in pieces]),
                 fit_seconds=[p[4] for p in pieces], fixed_split_val_acc=cfg["val_acc"], owner="p1",
                 notes="5-fold check")


# Categorical slots 1-4 of the reference palette, one per model, in the board's model order.
COLOURS = {"knn": "#2a78d6", "tree": "#eb6834", "forest": "#1baf7a", "nn": "#8a5cd6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# Plots each finalist's five fold accuracies, their mean and its fixed-split score.
def plot(cv):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    for ax, dataset in zip(axes, ["image", "sentiment"]):
        g = cv[cv["dataset"] == dataset].sort_values("mean_acc").reset_index(drop=True)
        for i, row in g.iterrows():
            accs = [float(a) for a in row["fold_accs"].split()]
            colour = COLOURS[row["model"]]
            ax.plot([min(accs), max(accs)], [i, i], color=colour, linewidth=2, alpha=0.35, solid_capstyle="round")
            ax.plot(accs, [i] * len(accs), "o", markersize=5, color=colour, markeredgecolor="white",
                    markeredgewidth=1)
            ax.plot(row["mean_acc"], i, "|", markersize=16, markeredgewidth=2.5, color=INK)
            ax.plot(row["fixed_split_val_acc"], i, "x", markersize=7, markeredgewidth=1.5, color=MUTED)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f"{m}  {r}" if len(r) <= 28 else f"{m}  {r[:26]}…" for m, r in zip(g["model"], g["run_id"])],
                           fontsize=8, color=INK)
        ax.set_title(dataset, fontsize=11, color=INK)
        ax.set_xlabel("accuracy on the held-out fold", fontsize=9, color=MUTED)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8, left=False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=m) for m, c in COLOURS.items()]
    handles += [plt.Line2D([], [], marker="|", linestyle="", markersize=12, markeredgewidth=2.5, color=INK,
                           label="mean of 5 folds"),
                plt.Line2D([], [], marker="x", linestyle="", color=MUTED, label="fixed 80/20 split")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5, frameon=False, labelcolor=INK)
    fig.suptitle("5-fold check of the finalists: each dot is one held-out fold", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    config.ROOT.joinpath("figures").mkdir(exist_ok=True)
    fig.savefig(config.ROOT / "figures" / "cv_finalists.png", dpi=150)
    plt.close(fig)


# Parses options, runs the 5-fold checks that are not done yet, then plots.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(N_CLASSES))
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--restore-arrays", action="store_true",
                    help="refit logged checks that lack cv_oof_proba.npy; leaves cv.json and cv_results.csv alone")
    args = ap.parse_args()

    cv_path = config.RESULTS_CSV.parent / "cv_results.csv"
    if args.restore_arrays:
        todo = [(d, m, r) for d, m, r in FINALISTS
                if (args.dataset in (None, d)) and (args.model in (None, m))
                and (config.RUNS_DIR / d / m / r / "cv.json").exists()
                and not (config.RUNS_DIR / d / m / r / "cv_oof_proba.npy").exists()]
        check(todo, args.workers, restore=True)
        return
    if not args.plot_only:
        todo = [(d, m, r) for d, m, r in FINALISTS
                if (args.dataset in (None, d)) and (args.model in (None, m))
                and not (config.RUNS_DIR / d / m / r / "cv.json").exists()]
        check(todo, args.workers)
    if cv_path.exists():
        plot(pd.read_csv(cv_path))


if __name__ == "__main__":
    main()
