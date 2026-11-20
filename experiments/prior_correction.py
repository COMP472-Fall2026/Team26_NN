# Tests whether correcting the sentiment class mix helps, using the out-of-fold and test predictions of each finalist.
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import load
from src.prior import adjust, balance_weights, confusion_estimate, em_estimate, prediction_share
from src.seed import set_seed

K = 3
CLASSES = ["negative", "neutral", "positive"]
SOURCE = np.full(K, 1 / K)                       # the training mix: exactly 4,000 rows per class
TARGETS = {"balanced": (1 / 3, 1 / 3, 1 / 3),
           "neg-heavy (old test estimate)": (0.50, 0.18, 0.32),
           "pos-heavy": (0.25, 0.25, 0.50)}
SAMPLE = 1500                                    # rows per resample, at most 750 of one class (each fold has 800)
REPEATS = 60                                     # 12 per fold


# Yields model name, run name and folder for every sentiment run with a 5-fold check.
def finalists():
    for cv in sorted(config.RUNS_DIR.glob("sentiment/*/*/cv.json")):
        rec = json.loads(cv.read_text())
        yield rec["model"], rec["run_id"], cv.parent


# Returns the share of rows whose most likely class matches the true label.
def acc(proba, y):
    return float((proba.argmax(axis=1) == y).mean())


# Estimates the test class mix three ways for each fold model of one finalist.
def test_mix(name, oof, test, y, folds):
    rows = []
    for f in range(len(test)):
        va = folds == f
        for method, q in [("prediction share", prediction_share(test[f], K)),
                          ("confusion inversion", confusion_estimate(oof[va], y[va], test[f])),
                          ("EM", em_estimate(test[f], SOURCE))]:
            rows.append({"finalist": name, "method": method, "fold": f, **dict(zip(CLASSES, np.round(q, 4)))})
    return rows


# Resamples held-out folds to a target class mix and scores each correction method.
def shift_sim(name, oof, y, folds, rng):
    rows = []
    for target_name, target in TARGETS.items():
        counts = np.round(np.asarray(target) * SAMPLE).astype(int)
        for r in range(REPEATS):
            f = r % 5
            fold_rows = np.flatnonzero(folds == f)
            pick = np.concatenate([rng.choice(fold_rows[y[fold_rows] == c], size=counts[c], replace=False)
                                   for c in range(K)])
            rest = np.setdiff1d(fold_rows, pick)          # the confusion matrix comes from rows outside the sample
            p, yy = oof[pick], y[pick]
            corrected = {
                "none": p,
                "true mix (upper bound)": adjust(p, target, SOURCE),
                "confusion inversion": adjust(p, confusion_estimate(oof[rest], y[rest], p), SOURCE),
                "EM": adjust(p, em_estimate(p, SOURCE), SOURCE),
                "rebalance": adjust(p, balance_weights(p), np.ones(K)),
            }
            for method, q in corrected.items():
                rows.append({"finalist": name, "target": target_name, "repeat": r, "method": method,
                             "acc": acc(q, yy)})
    return rows


# Compares accuracy and class shares on each held-out fold before and after rebalancing.
def balance_val(name, oof, y, folds):
    rows = []
    for f in range(5):
        va = folds == f
        p = oof[va]
        q = adjust(p, balance_weights(p), np.ones(K))
        rows.append({"finalist": name, "fold": f, "acc_none": acc(p, y[va]), "acc_rebalanced": acc(q, y[va]),
                     **{f"share_{c}_none": s for c, s in zip(CLASSES, prediction_share(p, K).round(4))},
                     **{f"share_{c}_rebalanced": s for c, s in zip(CLASSES, prediction_share(q, K).round(4))}})
    return rows


# Polarity order: warm for negative, grey for neutral, cool for positive.
CLASS_COLOURS = {"negative": "#eb6834", "neutral": "#b7b5ae", "positive": "#2a78d6"}
INK, MUTED = "#0b0b0b", "#52514e"


# Draws one stacked bar per finalist and method showing its estimated test class mix.
def plot(mix):
    g = mix.groupby(["finalist", "method"], sort=False)[CLASSES].mean().reset_index()
    labels = [f"{f}  ·  {m}" for f, m in zip(g["finalist"], g["method"])]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(g) + 1.6))
    y = np.arange(len(g))[::-1]
    left = np.zeros(len(g))
    for c in CLASSES:
        ax.barh(y, g[c], left=left, height=0.72, color=CLASS_COLOURS[c], edgecolor="white", linewidth=2, label=c)
        left += g[c].to_numpy()
    ax.axvline(1 / 3, color=INK, linewidth=1, linestyle=":")
    ax.axvline(2 / 3, color=INK, linewidth=1, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("estimated share of the test rows (dotted lines: the training mix, 1/3 each)", fontsize=9, color=MUTED)
    ax.set_title("Sentiment test mix as each model estimates it", fontsize=11, color=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, left=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.45 / fig.get_figheight(), 1, 1))
    config.ROOT.joinpath("figures").mkdir(exist_ok=True)
    fig.savefig(config.ROOT / "figures" / "prior_test_mix.png", dpi=150)
    plt.close(fig)


# Runs all three analyses for every sentiment finalist, then saves the tables and figure.
def main():
    set_seed()
    rng = np.random.default_rng(42)
    _, y, _ = load("sentiment", "raw", final=True)
    folds = np.load(config.SPLITS_DIR / "sentiment_split.npz")["fold"]

    mix, sim, bal = [], [], []
    for model, run_id, run_dir in finalists():
        name = f"{model} {run_id}"
        oof = np.load(run_dir / "cv_oof_proba.npy")
        test = np.load(run_dir / "cv_test_proba.npy")
        mix += test_mix(name, oof, test, y, folds)
        sim += shift_sim(name, oof, y, folds, rng)
        bal += balance_val(name, oof, y, folds)

    mix, sim, bal = pd.DataFrame(mix), pd.DataFrame(sim), pd.DataFrame(bal)
    out = config.RESULTS_CSV.parent
    mix.to_csv(out / "prior_test_mix.csv", index=False)
    sim.to_csv(out / "prior_shift_sim.csv", index=False, float_format="%.6f")
    bal.to_csv(out / "prior_balance_val.csv", index=False, float_format="%.6f")
    plot(mix)

    pd.set_option("display.width", 200)
    print("\ntest mix")
    print(mix.groupby(["finalist", "method"], sort=False)[CLASSES].mean().round(3).to_string())
    print("\nresampled acc")
    print(sim.pivot_table(index=["target", "finalist"], columns="method", values="acc", sort=False).round(4).to_string())
    print("\nrebalanced")
    print(bal.groupby("finalist", sort=False).mean(numeric_only=True).drop(columns="fold").round(4).to_string())


if __name__ == "__main__":
    main()
