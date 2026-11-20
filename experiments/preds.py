# Refits the k-NN, tree and forest finalists to regenerate their validation predictions for error analysis.
import argparse
import json

import numpy as np

from src import config
from src.data import load
from src.forest import RandomForest
from src.knn import KNN
from src.seed import set_seed
from src.tree import DecisionTree

MODELS = {"knn": KNN, "tree": DecisionTree, "forest": RandomForest}
CACHE_DIR = config.ROOT / "cache" / "preds"

# (dataset, model, run_id), fastest first
RUNS = [
    ("image", "knn", "k100_cos_exp0.025_logtfidf"),
    ("sentiment", "knn", "k100_exp2_pca200"),
    ("image", "tree", "dnone_ms2_ml10_gini_mf32_b256_pca50"),
    ("sentiment", "tree", "d4_ms2_ml1_gini_raw"),
    ("sentiment", "forest", "n100_dnone_mf64_raw"),
    ("image", "forest", "n200_dnone_mfsqrt_logtfidf"),
]


# Refits one run from its saved settings and caches its validation predictions.
def regenerate(dataset, model_name, run_id):
    out = CACHE_DIR / dataset / model_name / run_id
    if (out / "val_proba.npy").exists() and (out / "val_pred.npy").exists():
        return

    source = json.loads((config.RUNS_DIR / dataset / model_name / run_id / "config.json").read_text())
    params = dict(source["params"])
    n_classes = params.pop("n_classes")
    Xtr, ytr, Xva, yva = load(dataset, source["transform"])

    set_seed()
    model = MODELS[model_name](n_classes, **params).fit(Xtr, ytr)
    proba = model.predict_proba(Xva).astype(np.float32)
    pred = proba.argmax(axis=1).astype(np.int16)
    acc = float((pred == yva).mean())

    logged = source["val_acc"]
    if abs(acc - logged) > 0.001:
        print(f"WARNING {run_id} diff {acc - logged:+.4f}")

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_proba.npy", proba)
    np.save(out / "val_pred.npy", pred)


# Parses the filter option and regenerates predictions for every matching run.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help='prefix of "dataset/model/run_id" to run, e.g. image/forest')
    args = ap.parse_args()

    runs = [r for r in RUNS if args.only is None or "/".join(r).startswith(args.only)]
    if not runs:
        ap.error(f"no run matches --only {args.only!r}")
    for dataset, model_name, run_id in runs:
        regenerate(dataset, model_name, run_id)


if __name__ == "__main__":
    main()
