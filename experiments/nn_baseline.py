# Trains the untuned neural network on raw features, logs the run and plots its loss curves.
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src import config
from src.config import N_CLASSES
from src.data import load
from src.logger import log_run
from src.nn import MLP
from src.seed import set_seed

RUN_ID = "h256x64_lr1e-3_do0_wd0_adam_raw"
TRAIN_COLOUR = "#2a78d6"
VAL_COLOUR = "#eb6834"


# Returns the path of the per-epoch curve file for one dataset.
def curve_path(dataset):
    return config.ROOT / "results" / f"nn_curve_{dataset}.csv"


# Appends one row per epoch of training history to the dataset's curve file.
def save_curve(dataset, run_id, history):
    rows = pd.DataFrame(history)
    rows.insert(0, "epoch", range(1, len(rows) + 1))
    rows.insert(0, "run_id", run_id)

    path = curve_path(dataset)
    path.parent.mkdir(exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


# Plots training and validation loss and accuracy per epoch for one run.
def plot_curve(dataset, run_id, best_epoch):
    curve = pd.read_csv(curve_path(dataset))
    curve = curve[curve["run_id"] == run_id]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(axes, ["loss", "acc"], ["Cross-entropy loss", "Accuracy"]):
        ax.plot(curve["epoch"], curve[f"train_{metric}"], color=TRAIN_COLOUR, linewidth=2, label="Train")
        ax.plot(curve["epoch"], curve[f"val_{metric}"], color=VAL_COLOUR, linewidth=2, label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    # One shared legend below the panels, so it can't cover the lines
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{dataset.title()} network baseline: {run_id}")
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    figures = config.ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / f"nn_loss_curve_{dataset}.png", dpi=150)
    plt.close(fig)


# Trains the default network on raw features, logs it, then saves and plots its curves.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["image", "sentiment"], required=True)
    dataset = parser.parse_args().dataset

    set_seed()
    Xtr, ytr, Xva, yva = load(dataset, "raw")

    model = MLP(N_CLASSES[dataset])
    start = time.perf_counter()
    model.fit(Xtr, ytr, Xva, yva)
    fit_seconds = time.perf_counter() - start

    # argmin counts from 0, epochs count from 1
    best_epoch = int(np.argmin(model.history["val_loss"])) + 1

    # log_run goes first: it refuses a repeated run_id, so a rerun can't add duplicate curve rows
    log_run(dataset=dataset, model=model, run_id=RUN_ID, transform="raw",
                     y_train=ytr, X_train=Xtr, y_val=yva, X_val=Xva,
                     fit_seconds=fit_seconds, owner="p3", notes=f"best_epoch={best_epoch}")

    save_curve(dataset, RUN_ID, model.history)
    plot_curve(dataset, RUN_ID, best_epoch)


if __name__ == "__main__":
    main()
