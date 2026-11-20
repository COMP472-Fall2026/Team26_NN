# Chooses the final model per dataset by comparing single models and averages on the same out-of-fold predictions.
import json
from itertools import combinations

import numpy as np
import pandas as pd

from src import config
from src.data import load

N_FOLDS = 5
MIN_WINS, MIN_GAIN = 4, 0.005


# Returns the run with the highest 5-fold mean for each dataset and model.
def best_finalists(cv):
    best = cv.sort_values("mean_acc", ascending=False).drop_duplicates(["dataset", "model"])
    return {(r.dataset, r.model): r.run_id for r in best.itertuples()}


# Computes the accuracy of the given probabilities on each of the five folds.
def fold_accs(proba, y, folds):
    pred = proba.argmax(axis=1)
    return np.array([(pred[folds == f] == y[folds == f]).mean() for f in range(N_FOLDS)])


# Yields every single model and every equal-weight combination of two or more models.
def candidates(dataset, finalists):
    models = sorted(m for d, m in finalists if d == dataset)
    for n in range(1, len(models) + 1):
        for combo in combinations(models, n):
            yield combo


# Scores every candidate fold by fold, applies the decision rule and saves the table.
def main():
    cv = pd.read_csv(config.RESULTS_CSV.parent / "cv_results.csv")
    finalists = best_finalists(cv)
    rows, decisions = [], {}
    for dataset in ["image", "sentiment"]:
        _, y, _ = load(dataset, "raw", final=True)
        folds = np.load(config.SPLITS_DIR / f"{dataset}_split.npz")["fold"]
        oof = {m: np.load(config.RUNS_DIR / dataset / m / finalists[(dataset, m)] / "cv_oof_proba.npy")
               for d, m in finalists if d == dataset}

        scored = {combo: fold_accs(np.mean([oof[m] for m in combo], axis=0), y, folds)
                  for combo in candidates(dataset, finalists)}
        best_single = max((c for c in scored if len(c) == 1), key=lambda c: scored[c].mean())
        base = scored[best_single]
        for combo, accs in sorted(scored.items(), key=lambda kv: -kv[1].mean()):
            diff = accs - base
            rows.append({"dataset": dataset, "candidate": " + ".join(combo), "n_models": len(combo),
                         "runs": " + ".join(finalists[(dataset, m)] for m in combo),
                         "mean_acc": accs.mean(), "std_acc": accs.std(ddof=1),
                         "fold_accs": " ".join(f"{a:.4f}" for a in accs),
                         "gain_vs_best_single": diff.mean(), "folds_better": int((diff > 0).sum())})

        winners = [c for c in scored if len(c) > 1
                   and (scored[c] - base).mean() >= MIN_GAIN and ((scored[c] - base) > 0).sum() >= MIN_WINS]
        choice = max(winners, key=lambda c: scored[c].mean()) if winners else best_single
        decisions[dataset] = {"models": list(choice), "runs": [finalists[(dataset, m)] for m in choice],
                              "mean_acc": float(scored[choice].mean()), "best_single": best_single[0],
                              "best_single_acc": float(base.mean())}

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_CSV.parent / "final_choice.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 200)
    print(table.drop(columns=["runs", "fold_accs"]).round(4).to_string(index=False))
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
