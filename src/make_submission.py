# Builds a Kaggle submission file from the final test probabilities of one image and one sentiment model.
import argparse
from datetime import date

import numpy as np
import pandas as pd

from src import config
from src.config import N_CLASSES
from src.prior import adjust, balance_counts, balance_weights


# Loads a final model's test probabilities and checks their shape.
def final_proba(dataset: str, model: str) -> np.ndarray:
    ids = np.load(config.SPLITS_DIR / "test_ids.npz")[dataset]
    path = config.RUNS_DIR / dataset / model / "final" / "test_proba.npy"
    proba = np.load(path)
    assert proba.shape == (len(ids), N_CLASSES[dataset]), f"{path}: shape {proba.shape}"
    return proba


# Turns a model's test probabilities into labels, with optional mixing, balancing or rebalancing.
def predictions(dataset: str, model: str, correction: str = "none", mix=None, balance=False) -> np.ndarray:
    ids = np.load(config.SPLITS_DIR / "test_ids.npz")[dataset]
    if model == "constant":
        y = pd.read_csv(config.DATA_DIR / f"{dataset}_train.csv", usecols=["label"])["label"]
        # classes are balanced, so sort first: ties go to the lowest label and the file is reproducible
        return np.full(len(ids), y.value_counts().sort_index().idxmax(), dtype=np.int64)
    proba = final_proba(dataset, model)
    if mix is not None:
        other, share = mix
        proba = (1 - share) * proba + share * final_proba(dataset, other)
    if balance:
        labels = balance_counts(proba)
        return labels
    if correction == "rebalance":
        w = balance_weights(proba)
        proba = adjust(proba, w, np.ones(len(w)))
    return proba.argmax(axis=1)


# Combines image and sentiment predictions in sample submission order and checks every row.
def build(image_model: str, sentiment_model: str, sentiment_correction: str = "none",
          image_mix=None, image_balance: bool = False, sentiment_mix=None) -> pd.DataFrame:
    ids = np.load(config.SPLITS_DIR / "test_ids.npz")
    sample = pd.read_csv(config.DATA_DIR / "sample_submission.csv")
    preds = pd.concat([
        pd.DataFrame({"id": ids["image"], "label": predictions("image", image_model, mix=image_mix,
                                                               balance=image_balance)}),
        pd.DataFrame({"id": ids["sentiment"], "label": predictions("sentiment", sentiment_model,
                                                                   sentiment_correction,
                                                                   mix=sentiment_mix)}),
    ])
    sub = sample[["id"]].merge(preds, on="id", how="left", validate="one_to_one")

    assert len(sub) == 10_000, f"{len(sub)} rows, expected 10,000"
    assert sub["label"].notna().all(), f"{sub['label'].isna().sum()} ids have no prediction"
    sub["label"] = sub["label"].astype(int)
    is_img = sub["id"].str.startswith("img_")
    is_text = sub["id"].str.startswith("text_")
    assert is_img.sum() == 5_000 and is_text.sum() == 5_000, "expected 5,000 img_ and 5,000 text_ ids"
    assert sub.loc[is_img, "label"].between(0, N_CLASSES["image"] - 1).all(), "image labels outside 0-49"
    assert sub.loc[is_text, "label"].between(0, N_CLASSES["sentiment"] - 1).all(), "sentiment labels outside 0-2"
    return sub


# Parses the model options, builds the submission and writes it under a dated name.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="model folder under runs/image/, or constant")
    ap.add_argument("--sentiment", required=True, help="model folder under runs/sentiment/, or constant")
    ap.add_argument("--sentiment-correction", choices=["none", "rebalance"], default="none")
    ap.add_argument("--image-mix", metavar="MODEL:SHARE", help="e.g. nn:0.2 averages in 20%% of runs/image/nn/final")
    ap.add_argument("--sentiment-mix", metavar="MODEL:SHARE",
                    help="e.g. forest:0.5 averages equal probabilities from tree and forest")
    ap.add_argument("--image-balance", action="store_true", help="equal predicted count per image class")
    args = ap.parse_args()

    mix = None
    if args.image_mix:
        other, share = args.image_mix.split(":")
        mix = (other, float(share))
        if not 0 < mix[1] < 1:
            ap.error("--image-mix share must be between 0 and 1")
    sentiment_mix = None
    if args.sentiment_mix:
        other, share = args.sentiment_mix.split(":")
        sentiment_mix = (other, float(share))
        if not 0 < sentiment_mix[1] < 1:
            ap.error("--sentiment-mix share must be between 0 and 1")
    sub = build(args.image, args.sentiment, args.sentiment_correction, mix, args.image_balance,
                sentiment_mix)
    image_name = args.image + (f"+{mix[0]}{mix[1]:g}" if mix else "") + ("-balanced" if args.image_balance else "")
    sentiment_name = args.sentiment + (
        f"+{sentiment_mix[0]}{sentiment_mix[1]:g}" if sentiment_mix else ""
    )
    suffix = "" if args.sentiment_correction == "none" else f"-{args.sentiment_correction}"
    out = config.ROOT / "submissions" / f"{date.today().isoformat()}_{image_name}_{sentiment_name}{suffix}.csv"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    sub.to_csv(out, index=False)


if __name__ == "__main__":
    main()
