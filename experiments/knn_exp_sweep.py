# Sweeps k and temperature for k-NN with exponential vote weights on the validation split.
import time

import numpy as np
import pandas as pd

from src import config
from src.data import load
from src.knn import KNN
from src.logger import log_run
from src.seed import set_seed

SETTINGS = {"image": ("logtfidf", "cosine"), "sentiment": ("pca200", "euclidean")}   # best k-NN sweep settings per dataset
KS = [5, 10, 20, 50, 100]
TEMPERATURES = {"image": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1],
                "sentiment": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]}   # Euclidean distances are ~10x larger
ALSO_LOG = {"image": [(100, 0.03)]}


# Builds the run name from k, distance, temperature and transform.
def run_id(k, temperature, transform, metric):
    parts = [f"k{k}"] + (["cos"] if metric == "cosine" else []) + [f"exp{temperature:g}", transform]
    return "_".join(parts)


# Scores every k and temperature from one neighbour list per dataset, then logs the best.
def main():
    set_seed()
    rows = []
    for dataset, (transform, metric) in SETTINGS.items():
        Xtr, ytr, Xva, yva = load(dataset, transform)
        model = KNN(len(np.unique(ytr)), k=max(KS), metric=metric, weights="exp").fit(Xtr, ytr)
        dist, idx = model.kneighbors(Xva)
        for k in KS:
            for T in TEMPERATURES[dataset]:
                model.params["temperature"] = T
                acc = float((model.vote(dist[:, :k], idx[:, :k]).argmax(axis=1) == yva).mean())
                rows.append({"dataset": dataset, "transform": transform, "metric": metric, "k": k,
                             "temperature": T, "val_acc": acc})
        curve = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
        best = curve.loc[curve["val_acc"].idxmax()]
        print(curve.pivot(index="temperature", columns="k", values="val_acc").round(4).to_string())

        for k, T in {(int(best.k), float(best.temperature)), *ALSO_LOG.get(dataset, [])}:
            rid = run_id(k, T, transform, metric)
            if (config.RUNS_DIR / dataset / "knn" / rid).exists():
                continue
            m = KNN(model.n_classes, k=k, metric=metric, weights="exp", temperature=T)
            t0 = time.perf_counter()
            m.fit(Xtr, ytr)
            rec = log_run(dataset, m, rid, transform, ytr, Xtr, yva, Xva, fit_seconds=time.perf_counter() - t0,
                          owner="p1", notes="k-NN sweep addendum: exponential vote weights")
            expected = curve[(curve.k == k) & (curve.temperature == T)].val_acc.item()
            assert abs(rec["val_acc"] - expected) < 1e-9, f"{rid}: logged {rec['val_acc']} vs curve {expected}"
    pd.DataFrame(rows).to_csv(config.RESULTS_CSV.parent / "knn_exp_curve.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
