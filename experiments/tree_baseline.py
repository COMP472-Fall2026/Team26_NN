# Trains one untuned decision tree on raw features and logs it as the tree baseline.
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import N_CLASSES, SEED
from src.logger import log_run
from src.seed import set_seed
from src.tree import DecisionTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Fits one decision tree on raw features of the fixed split and logs it.
def run_baseline(dataset, criterion="gini", n_bins=64, max_depth=10):
    set_seed(SEED)
    train_path = PROJECT_ROOT / "data" / f"{dataset}_train.csv"
    split_path = PROJECT_ROOT / "splits" / f"{dataset}_split.npz"

    frame = pd.read_csv(train_path)
    X = frame.drop(columns=["id", "label"]).to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)

    with np.load(split_path) as split:
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

    start = time.perf_counter()
    model = DecisionTree(
        n_classes=N_CLASSES[dataset],
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        n_bins=n_bins,
        seed=SEED,
    )
    model.fit(X[train_idx], y[train_idx])
    elapsed = time.perf_counter() - start

    depth = "none" if max_depth is None else max_depth
    criterion_name = "gini" if criterion == "gini" else "ent"
    run_id = f"d{depth}_ms2_ml1_{criterion_name}_raw"
    log_run(
        dataset=dataset,
        model=model,
        run_id=run_id,
        transform="raw",
        y_train=y[train_idx],
        X_train=X[train_idx],
        y_val=y[val_idx],
        X_val=X[val_idx],
        fit_seconds=elapsed,
        owner="p2",
        seed=SEED,
    )


# Parses the dataset and tree options, then trains and logs the baseline tree.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("image", "sentiment"), required=True)
    parser.add_argument("--criterion", choices=("gini", "entropy"), default="gini")
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    args = parser.parse_args()
    run_baseline(
        args.dataset,
        criterion=args.criterion,
        n_bins=args.n_bins,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
