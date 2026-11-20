# Sweeps decision-tree depth, minimum split size, minimum leaf size and criterion, then plots the depth curve.
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.config import N_CLASSES, SEED
from src.logger import log_run
from src.seed import set_seed
from src.tree import DecisionTree


DEPTHS = [1, 3, 5, 10, 20, None]
MIN_SPLITS = [2, 10, 50]
MIN_LEAVES = [1, 5, 20]
CRITERIA = ["gini", "entropy"]
DEFAULTS = {"max_depth": 10, "min_samples_split": 2,
            "min_samples_leaf": 1, "criterion": "gini"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Reads the training CSV and the fixed split indices, optionally keeping fewer training rows.
def load_data(dataset, max_rows=None):
    frame = pd.read_csv(PROJECT_ROOT / "data" / f"{dataset}_train.csv")
    X = frame.drop(columns=["id", "label"]).to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    with np.load(PROJECT_ROOT / "splits" / f"{dataset}_split.npz") as split:
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

    if max_rows is not None:
        train_idx = train_idx[:max_rows]
    return X, y, train_idx, val_idx


# Builds the run name from depth, split size, leaf size, criterion and row count.
def run_id(max_depth, min_samples_split, min_samples_leaf, criterion,
           dataset=None, train_rows=None):
    depth = "none" if max_depth is None else max_depth
    short_criterion = "gini" if criterion == "gini" else "ent"
    sample_size = f"_n{train_rows}" if dataset == "sentiment" else ""
    return (f"d{depth}_ms{min_samples_split}_ml{min_samples_leaf}_"
            f"{short_criterion}{sample_size}_raw")


# Yields each setting to try, changing one parameter at a time from the defaults.
def settings():
    unique = set()
    for depth in DEPTHS:
        setting = (depth, DEFAULTS["min_samples_split"], DEFAULTS["min_samples_leaf"], DEFAULTS["criterion"])
        if setting not in unique:
            unique.add(setting)
            yield setting
    for min_split in MIN_SPLITS:
        setting = (DEFAULTS["max_depth"], min_split, DEFAULTS["min_samples_leaf"], DEFAULTS["criterion"])
        if setting not in unique:
            unique.add(setting)
            yield setting
    for min_leaf in MIN_LEAVES:
        setting = (DEFAULTS["max_depth"], DEFAULTS["min_samples_split"], min_leaf, DEFAULTS["criterion"])
        if setting not in unique:
            unique.add(setting)
            yield setting
    for criterion in CRITERIA[1:]:
        setting = (DEFAULTS["max_depth"], DEFAULTS["min_samples_split"], DEFAULTS["min_samples_leaf"], criterion)
        if setting not in unique:
            unique.add(setting)
            yield setting


# Fits and logs one tree setting, or reads its scores if already logged.
def score_setting(dataset, X, y, train_idx, val_idx, max_depth,
                  min_samples_split, min_samples_leaf, criterion, n_bins=64,
                  log=True):
    model = DecisionTree(
        n_classes=N_CLASSES[dataset],
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_bins=n_bins,
        seed=SEED,
    )
    start = time.perf_counter()
    rid = run_id(
        max_depth, min_samples_split, min_samples_leaf, criterion,
        dataset=dataset, train_rows=len(train_idx),
    )
    run_path = config.RUNS_DIR / dataset / "tree" / rid
    if log and run_path.exists():
        saved = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
        fit_seconds = float(saved["fit_seconds"])
        record = {
            "train_acc": float(saved["train_acc"]),
            "val_acc": float(saved["val_acc"]),
        }
    else:
        model.fit(X[train_idx], y[train_idx])
        fit_seconds = time.perf_counter() - start
        record = None
    if log and not run_path.exists():
        record = log_run(
            dataset=dataset, model=model, run_id=rid, transform="raw",
            y_train=y[train_idx], X_train=X[train_idx],
            y_val=y[val_idx], X_val=X[val_idx],
            fit_seconds=fit_seconds, owner="p2", notes="tree sweep",
            seed=SEED,
        )
    else:
        if record is None:
            train_pred = model.predict_proba(X[train_idx]).argmax(axis=1)
            val_pred = model.predict_proba(X[val_idx]).argmax(axis=1)
            record = {
                "train_acc": float(np.mean(train_pred == y[train_idx])),
                "val_acc": float(np.mean(val_pred == y[val_idx])),
            }

    return {
        "dataset": dataset,
        "train_rows": len(train_idx),
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
        "n_bins": n_bins,
        "train_acc": record["train_acc"],
        "val_acc": record["val_acc"],
        "fit_seconds": fit_seconds,
    }


# Plots training and validation accuracy against maximum depth, with the other settings at defaults.
def plot_depth_curve(dataset, curve):
    config.ROOT.joinpath("figures").mkdir(exist_ok=True)
    depth_curve = curve[
        (curve["min_samples_split"] == DEFAULTS["min_samples_split"])
        & (curve["min_samples_leaf"] == DEFAULTS["min_samples_leaf"])
        & (curve["criterion"] == DEFAULTS["criterion"])
    ].copy()
    depth_curve["plot_depth"] = depth_curve["max_depth"].fillna(
        depth_curve["max_depth"].max() + 5
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(depth_curve["plot_depth"], depth_curve["train_acc"],
            marker="o", label="Train accuracy")
    ax.plot(depth_curve["plot_depth"], depth_curve["val_acc"],
            marker="o", label="Validation accuracy")
    finite_depths = depth_curve["max_depth"].dropna()
    ticks = sorted(set(finite_depths.tolist()))
    unlimited = depth_curve["max_depth"].isna().any()
    if unlimited:
        ticks.append(depth_curve["plot_depth"].max())
    labels = [
        "unlimited" if unlimited and tick == depth_curve["plot_depth"].max()
        else str(int(tick))
        for tick in ticks
    ]
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("Maximum depth")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset.title()} decision tree: depth curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.ROOT / "figures" / f"tree_depth_curve_{dataset}.png", dpi=150)
    plt.close(fig)


# Runs every tree setting not done yet, saves the curve and plots the depth curve.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("image", "sentiment"), required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    set_seed(SEED)

    curve_path = config.ROOT / "results" / f"tree_curve_{args.dataset}.csv"
    if args.plot_only:
        curve = pd.read_csv(curve_path)
        plot_depth_curve(args.dataset, curve)
        return

    X, y, train_idx, val_idx = load_data(args.dataset, args.max_rows)
    train_rows = len(train_idx)
    if curve_path.exists():
        previous = pd.read_csv(curve_path)
        if "train_rows" in previous.columns:
            rows = previous[previous["train_rows"] == train_rows].to_dict("records")
        elif args.dataset == "image" and args.max_rows is None:
            rows = previous.to_dict("records")
        else:
            rows = []
    else:
        rows = []
    completed = set()
    for row in rows:
        depth = None if pd.isna(row["max_depth"]) else int(row["max_depth"])
        completed.add((
            depth,
            int(row["min_samples_split"]),
            int(row["min_samples_leaf"]),
            row["criterion"],
        ))
    for depth, min_split, min_leaf, criterion in settings():
        key = (depth, min_split, min_leaf, criterion)
        if key in completed:
            continue
        row = score_setting(
            args.dataset, X, y, train_idx, val_idx, depth, min_split,
            min_leaf, criterion, args.n_bins,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(curve_path, index=False)

    curve = pd.DataFrame(rows)
    curve.to_csv(curve_path, index=False)
    plot_depth_curve(args.dataset, curve)


if __name__ == "__main__":
    main()
