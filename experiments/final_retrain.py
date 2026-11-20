# Retrains a chosen validation run on all training rows and saves its test probabilities for submission.
import argparse
import json
import time

from src import config
from src.data import load
from src.forest import RandomForest
from src.knn import KNN
from src.logger import log_final
from src.nn import MLP
from src.seed import set_seed
from src.tree import DecisionTree

MODELS = {"knn": KNN, "tree": DecisionTree, "forest": RandomForest, "nn": MLP}


# Reads a run's settings, retrains it on all training rows and logs the final model.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["image", "sentiment"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--run", required=True, help="validation run_id whose settings to reuse")
    ap.add_argument("--owner", default="p1")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing final on purpose")
    ap.add_argument("--epochs", type=int, help="nn only: fixed number of epochs (no validation rows to stop on)")
    args = ap.parse_args()
    if (args.model == "nn") != (args.epochs is not None):
        ap.error("--epochs is required for nn and only used for nn")
    set_seed()

    source = json.loads((config.RUNS_DIR / args.dataset / args.model / args.run / "config.json").read_text())
    params = dict(source["params"])
    n_classes = params.pop("n_classes")
    if args.epochs is not None:
        params["max_epochs"] = args.epochs
    X_all, y_all, X_test = load(args.dataset, source["transform"], final=True)

    t0 = time.perf_counter()
    model = MODELS[args.model](n_classes, **params).fit(X_all, y_all)
    fit_seconds = time.perf_counter() - t0
    log_final(args.dataset, model, source["transform"], X_test, n_train=len(X_all), fit_seconds=fit_seconds,
                    owner=args.owner, source_run=args.run, overwrite=args.overwrite,
                    notes=f"retrained on all {len(X_all)} training rows; validation acc of {args.run}: "
                          f"{source['val_acc']:.4f}" + (f"; fixed {args.epochs} epochs" if args.epochs else ""))


if __name__ == "__main__":
    main()
