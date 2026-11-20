# Sweeps neural-network settings one at a time from the baseline, logs every run and plots the results.
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.nn_baseline import save_curve
from src import config
from src.config import N_CLASSES
from src.data import load
from src.logger import log_run
from src.nn import MLP
from src.seed import set_seed

BASE = {"hidden": (256, 64), "activation": "relu", "dropout": 0.0, "lr": 1e-3,
        "weight_decay": 0.0, "optimizer": "adam", "batch_size": 64}
TRANSFORMS = {"image": ["raw", "standardise", "logtfidf"], "sentiment": ["raw", "standardise"]}
VARIATIONS = [
    ("lr", {"lr": 1e-2}), ("lr", {"lr": 1e-4}),
    ("hidden", {"hidden": (256,)}), ("hidden", {"hidden": (256, 128, 64)}),
    ("hidden", {"hidden": (512, 128)}), ("hidden", {"hidden": (1024, 256)}),
    ("activation", {"activation": "tanh"}), ("activation", {"activation": "gelu"}),
    ("dropout", {"dropout": 0.2}), ("dropout", {"dropout": 0.5}),
    ("weight_decay", {"weight_decay": 1e-4}), ("weight_decay", {"weight_decay": 1e-3}),
    ("optimizer", {"optimizer": "sgd", "lr": 1e-2}), ("optimizer", {"optimizer": "sgd", "lr": 1e-1}),
    ("batch_size", {"batch_size": 32}), ("batch_size", {"batch_size": 256}),
]


# Formats a number in short scientific notation for run names, such as 1e-3.
def sci(x):
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# Builds the run name from layer sizes, learning rate, dropout, weight decay and optimiser.
def run_id(s, transform):
    parts = ["h" + "x".join(map(str, s["hidden"])), "lr" + sci(s["lr"]), f"do{s['dropout']:g}",
             "wd" + sci(s["weight_decay"]), s["optimizer"]]
    if s["activation"] != BASE["activation"]:
        parts.append(s["activation"])
    if s["batch_size"] != BASE["batch_size"]:
        parts.append(f"bs{s['batch_size']}")
    return "_".join(parts + [transform])


# Trains and logs one network setting, or reads its accuracy if already logged.
def train_and_log(dataset, transform, data, settings):
    rid = run_id(settings, transform)
    run_dir = config.RUNS_DIR / dataset / "nn" / rid
    if run_dir.exists():
        return rid, json.loads((run_dir / "config.json").read_text())["val_acc"]

    Xtr, ytr, Xva, yva = data
    model = MLP(N_CLASSES[dataset], max_epochs=100, patience=15, **settings)
    start = time.perf_counter()
    model.fit(Xtr, ytr, Xva, yva)
    fit_seconds = time.perf_counter() - start

    best_epoch = int(np.argmin(model.history["val_loss"])) + 1
    record = log_run(dataset=dataset, model=model, run_id=rid, transform=transform,
                     y_train=ytr, X_train=Xtr, y_val=yva, X_val=Xva,
                     fit_seconds=fit_seconds, owner="p3", notes=f"best_epoch={best_epoch}")
    save_curve(dataset, rid, model.history)
    return rid, record["val_acc"]


# Picks the best transform, varies one setting at a time, then combines the winners.
def sweep(dataset):
    rows = []

    scores = {}
    for transform in TRANSFORMS[dataset]:
        rid, scores[transform] = train_and_log(dataset, transform, load(dataset, transform), BASE)
        rows.append((f"transform {transform}", rid))
    transform = max(scores, key=scores.get)
    base_acc = scores[transform]

    data = load(dataset, transform)
    best = {}
    for name, change in VARIATIONS:
        rid, acc = train_and_log(dataset, transform, data, {**BASE, **change})
        rows.append((", ".join(f"{k} {v}" for k, v in change.items()), rid))
        if acc > best.get(name, (0, None))[0]:
            best[name] = (acc, change)

    combined = {}
    for acc, change in best.values():
        if acc > base_acc:
            combined.update(change)
    if len(combined) >= 2:
        rid, _ = train_and_log(dataset, transform, data, {**BASE, **combined})
        rows.append(("all winners", rid))
    return rows


# Plots validation accuracy of every sweep run against the baseline and the best k-NN.
def plot(dataset, rows):
    results = pd.read_csv(config.RESULTS_CSV)
    results = results[results["dataset"] == dataset]
    val_acc = dict(zip(results.loc[results["model"] == "nn", "run_id"], results.loc[results["model"] == "nn", "val_acc"]))
    accs = [val_acc[rid] for _, rid in rows]
    knn = results.loc[results["model"] == "knn", "val_acc"].max()

    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2))
    ax.plot(accs, y, "o", color="#2a78d6")
    ax.axvline(accs[0], color="gray", linestyle="--", label="baseline")
    ax.axvline(knn, color="gray", linestyle=":", label="best k-NN")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_xlabel("validation accuracy")
    ax.set_title(f"Network sweep on {dataset}")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.ROOT / "figures" / f"nn_sweep_{dataset}.png", dpi=150)
    plt.close(fig)


# Parses the dataset option, runs the network sweep and plots its results.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(N_CLASSES), required=True)
    dataset = parser.parse_args().dataset

    set_seed()
    plot(dataset, sweep(dataset))


if __name__ == "__main__":
    main()
