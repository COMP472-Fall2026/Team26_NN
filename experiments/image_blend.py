# Tests whether mixing network probabilities into image k-NN, or balancing class counts, beats k-NN alone across folds.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import config
from src.data import load, load_fold
from src.prior import balance_counts

N_FOLDS = 5
NN_RUN = "h256_lr1e-3_do0.2_wd0_adam_gelu_bs256_standardise"   # best image network on validation and 5-fold checks
EPOCHS = [5, 10, 20, 30]
MIXES = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
MIN_WINS, MIN_GAIN = 4, 0.005


# Trains the image network on four folds for fixed epochs and predicts the fifth.
def nn_fold(args):
    fold, epochs = args
    import torch
    torch.set_num_threads(1)
    from src.nn import MLP
    cfg = json.loads((config.RUNS_DIR / "image/nn" / NN_RUN / "config.json").read_text())
    params = dict(cfg["params"], max_epochs=epochs)
    n_classes = params.pop("n_classes")
    Xtr, ytr, Xva, _, _, va = load_fold("image", cfg["transform"], fold)
    model = MLP(n_classes, **params).fit(Xtr, ytr)           # no validation rows: trains exactly `epochs` epochs
    return fold, epochs, va, np.asarray(model.predict_proba(Xva))


# Scores every mix and balance setting per fold and checks the best against k-NN alone.
def main():
    cv = pd.read_csv(config.RESULTS_CSV.parent / "cv_results.csv")
    knn_run = cv[(cv.dataset == "image") & (cv.model == "knn")].sort_values("mean_acc").iloc[-1].run_id
    knn = np.load(config.RUNS_DIR / "image/knn" / knn_run / "cv_oof_proba.npy")
    _, y, _ = load("image", "raw", final=True)
    folds = np.load(config.SPLITS_DIR / "image_split.npz")["fold"]

    nn = {e: np.zeros_like(knn) for e in EPOCHS}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        for fold, e, va, p in pool.map(nn_fold, itertools.product(range(N_FOLDS), EPOCHS)):
            nn[e][va] = p

    grid = [(e, a, bal) for e in EPOCHS for a in MIXES for bal in (False, True) if a > 0 or e == EPOCHS[0]]
    acc = np.zeros((N_FOLDS, len(grid)))
    for f in range(N_FOLDS):
        r = folds == f
        for g, (e, a, bal) in enumerate(grid):
            P = (1 - a) * knn[r] + a * nn[e][r]
            acc[f, g] = ((balance_counts(P) if bal else P.argmax(1)) == y[r]).mean()
    base = acc[:, grid.index((EPOCHS[0], 0, False))]           # k-NN alone

    name = lambda g: f"epochs={grid[g][0]} nn_share={grid[g][1]} balanced={grid[g][2]}" if grid[g][1] else \
        f"k-NN alone balanced={grid[g][2]}"
    rows = [{"setting": name(g), "epochs": e if a else None, "nn_share": a, "balanced": bal,
             "mean_acc": acc[:, g].mean(), "fold_accs": " ".join(f"{x:.4f}" for x in acc[:, g]),
             "gain_vs_knn": (acc[:, g] - base).mean(), "folds_better": int((acc[:, g] > base).sum())}
            for g, (e, a, bal) in enumerate(grid)]
    table = pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

    nested = []
    for f in range(N_FOLDS):
        g = int(np.argmax(np.delete(acc, f, axis=0).mean(axis=0)))   # chosen without looking at fold f
        nested.append(acc[f, g])
    nested = np.array(nested)
    gain = nested - base
    passes = (gain > 0).sum() >= MIN_WINS and gain.mean() >= MIN_GAIN
    table.to_csv(config.RESULTS_CSV.parent / "image_blend.csv", index=False, float_format="%.6f")

    print(table.drop(columns=["fold_accs"]).head(10).round(4).to_string(index=False))
    print(f"blend {nested.mean():.4f} vs knn {base.mean():.4f}: {'ADOPT' if passes else 'keep knn'}")


if __name__ == "__main__":
    main()
