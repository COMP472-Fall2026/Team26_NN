# Sweeps random-forest settings on the validation split and compares the best forests with the best single tree.
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.config import N_CLASSES, SEED
from src.data import load
from src.forest import RandomForest
from src.logger import log_run
from src.seed import set_seed


TREE_COUNTS = [10, 50, 100]
DEPTHS = [5, 10, 20, None]
FEATURE_COUNTS = ["sqrt", 16, 64]
N_BINS = 64


# Returns the depth as text, with none standing for unlimited depth.
def short_depth(depth):
    return "none" if depth is None else str(depth)


# Returns the max_features setting as text for labels and printed messages.
def short_features(max_features):
    return "sqrt" if max_features == "sqrt" else str(max_features)


# Builds the run name from tree count, depth, feature count and transform.
def run_id(n_trees, depth, max_features, transform="raw"):
    features = "sqrt" if max_features == "sqrt" else max_features
    return (f"n{n_trees}_d{short_depth(depth)}_mf{features}_"
            f"{transform}")


# Yields every combination of tree count, depth and feature count to sweep.
def settings(n_trees=None, depth=None, max_features=None):
    counts = [n_trees] if n_trees is not None else TREE_COUNTS
    depths = [depth] if depth is not None else DEPTHS
    features = ([max_features] if max_features is not None
                else FEATURE_COUNTS)
    for trees in counts:
        for tree_depth in depths:
            for feature_count in features:
                yield trees, tree_depth, feature_count


# Turns the three --single values from text into tree count, depth and features.
def single_setting(trees, depth, features):
    return (int(trees), None if depth == "none" else int(depth),
            "sqrt" if features == "sqrt" else int(features))


# Reads the tree sweep curve and returns the best validation accuracy and its depth.
def best_tree(dataset):
    curve_path = config.ROOT / "results" / f"tree_curve_{dataset}.csv"
    if not curve_path.exists():
        raise FileNotFoundError(
            f"Run `python -m experiments.tree_sweep --dataset {dataset}` first"
        )
    curve = pd.read_csv(curve_path)
    row = curve.sort_values(
        ["val_acc", "train_acc"], ascending=[False, True], kind="stable"
    ).iloc[0]
    return float(row["val_acc"]), (
        None if pd.isna(row["max_depth"]) else int(row["max_depth"])
    )


# Saves a table comparing the best single tree with every forest setting.
def comparison_table(dataset, curve):
    tree_curve = pd.read_csv(
        config.ROOT / "results" / f"tree_curve_{dataset}.csv"
    )
    best = tree_curve.sort_values(
        ["val_acc", "train_acc"], ascending=[False, True], kind="stable"
    ).iloc[0]
    tree_features = best.get("max_features", "all")
    if pd.isna(tree_features):
        tree_features = "all"
    tree_transform = best.get("transform", "raw")
    if pd.isna(tree_transform):
        tree_transform = "raw"
    rows = [{
        "model": "tree",
        "n_trees": 1,
        "max_depth": best["max_depth"],
        "max_features": tree_features,
        "transform": tree_transform,
        "criterion": best["criterion"],
        "min_samples_leaf": best["min_samples_leaf"],
        "n_bins": best["n_bins"],
        "train_acc": best["train_acc"],
        "val_acc": best["val_acc"],
    }]
    forest_rows = curve.rename(columns={"max_features": "max_features"}).copy()
    forest_rows["model"] = "forest"
    forest_rows["transform"] = "logtfidf" if dataset == "image" else "raw"
    forest_rows["criterion"] = "gini"
    forest_rows["min_samples_leaf"] = 1
    rows.extend(forest_rows[
        ["model", "n_trees", "max_depth", "max_features", "transform",
         "criterion", "min_samples_leaf", "n_bins", "train_acc", "val_acc"]
    ].to_dict("records"))
    table = pd.DataFrame(rows)
    path = config.ROOT / "results" / f"forest_tree_comparison_{dataset}.csv"
    table.to_csv(path, index=False)
    return table


# Plots validation accuracy against tree count, one panel per feature count setting.
def plot(dataset, curve, tree_val, tree_depth):
    config.ROOT.joinpath("figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, feature_count in zip(axes, FEATURE_COUNTS):
        if feature_count == "sqrt":
            mask = curve["max_features"].astype(str) == "sqrt"
        else:
            mask = pd.to_numeric(
                curve["max_features"], errors="coerce"
            ) == feature_count
        g = curve[mask]
        for depth in DEPTHS:
            gd = g[g["max_depth"].fillna(-1) == (
                -1 if depth is None else depth
            )]
            if gd.empty:
                continue
            label = "unlimited" if depth is None else f"depth {depth}"
            ax.plot(gd["n_trees"], gd["val_acc"], marker="o", label=label)
        ax.axhline(tree_val, color="#777777", linestyle="--",
                   label=f"best tree ({tree_val:.3f})")
        ax.set_title(f"Features: {short_features(feature_count)}")
        ax.set_xlabel("Number of trees")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{dataset.title()} forest sweep (best single tree depth "
        f"{short_depth(tree_depth)})"
    )
    fig.tight_layout()
    path = config.ROOT / "figures" / f"forest_vs_tree_{dataset}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# Runs every forest setting not done yet, logs it, then saves tables and figures.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("image", "sentiment"), required=True)
    parser.add_argument("--max-trees", type=int, choices=(10, 50, 100))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--single", nargs=3,
                        metavar=("TREES", "DEPTH", "FEATURES"),
                        help="run one setting only, e.g. 200 none 64")
    args = parser.parse_args()
    set_seed(SEED)

    curve_path = config.ROOT / "results" / f"forest_curve_{args.dataset}.csv"
    if args.plot_only:
        curve = pd.read_csv(curve_path)
        tree_val, tree_depth = best_tree(args.dataset)
        comparison_table(args.dataset, curve)
        plot(args.dataset, curve, tree_val, tree_depth)
        return

    transform = "logtfidf" if args.dataset == "image" else "raw"
    X_train, y_train, X_val, y_val = load(args.dataset, transform)
    tree_val, tree_depth = best_tree(args.dataset)

    counts = TREE_COUNTS
    if args.max_trees is not None:
        counts = [count for count in counts if count <= args.max_trees]
    grid = list(settings())
    notes = "forest sweep"
    if args.single is not None:
        grid = [single_setting(*args.single)]
        counts = [grid[0][0]]
        notes = "targeted forest tree-count refinement"

    rows = []
    if curve_path.exists():
        rows.extend(pd.read_csv(curve_path).to_dict("records"))

    # Turns a max_features value into text so saved and new settings compare.
    def feature_key(value):
        return "sqrt" if value == "sqrt" else str(int(float(value)))

    completed = {
        (
            int(row["n_trees"]),
            None if pd.isna(row["max_depth"]) else int(row["max_depth"]),
            feature_key(row["max_features"]),
        )
        for row in rows
    }

    for n_trees, depth, max_features in grid:
        if n_trees not in counts:
            continue
        if (n_trees, depth, feature_key(max_features)) in completed:
            continue
        model = RandomForest(
            n_classes=N_CLASSES[args.dataset],
            n_trees=n_trees,
            max_depth=depth,
            max_features=max_features,
            min_samples_leaf=1,
            bootstrap=True,
            criterion="gini",
            n_bins=N_BINS,
            seed=SEED,
        )
        rid = run_id(n_trees, depth, max_features, transform)
        record = None
        run_path = config.RUNS_DIR / args.dataset / "forest" / rid
        if run_path.exists():
            saved = json.loads((run_path / "config.json").read_text())
            train_acc = saved["train_acc"]
            val_acc = saved["val_acc"]
            fit_seconds = saved["fit_seconds"]
        else:
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            record = log_run(
                dataset=args.dataset, model=model, run_id=rid,
                transform=transform, y_train=y_train, X_train=X_train,
                y_val=y_val, X_val=X_val, fit_seconds=fit_seconds,
                owner="p2", notes=notes, seed=SEED,
            )
        if record is not None:
            train_acc = record["train_acc"]
            val_acc = record["val_acc"]

        rows.append({
            "dataset": args.dataset,
            "n_trees": n_trees,
            "max_depth": depth,
            "max_features": max_features,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "fit_seconds": fit_seconds,
            "n_bins": N_BINS,
        })
        pd.DataFrame(rows).to_csv(curve_path, index=False)

    curve = pd.DataFrame(rows)
    curve.to_csv(curve_path, index=False)
    comparison_table(args.dataset, curve)
    plot(args.dataset, curve, tree_val, tree_depth)


if __name__ == "__main__":
    main()
