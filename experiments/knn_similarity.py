# Groups image rows by similarity to their nearest training row and compares model accuracy in each group.
import numpy as np
import pandas as pd

from src import config
from src.data import load, load_fold
from src.knn import KNN

NN_RUN = "h256_lr1e-3_do0.2_wd0_adam_gelu_bs256_standardise"
FOREST_RUN = "n200_dnone_mfsqrt_logtfidf"


# Finds each row's nearest-neighbour similarity and tabulates model accuracy by similarity decile.
def main():
    folds = np.load(config.SPLITS_DIR / "image_split.npz")["fold"]
    _, y, _ = load("image", "raw", final=True)
    sim = np.zeros(len(y))
    knn_pred = np.zeros(len(y), dtype=int)
    for f in range(5):
        Xtr, ytr, Xva, _, _, va = load_fold("image", "logtfidf", f)
        dist, idx = KNN(50, k=1, metric="cosine").fit(Xtr, ytr).kneighbors(Xva, 1)
        sim[va] = 1 - dist[:, 0]
        knn_pred[va] = ytr[idx[:, 0]]

    runs = config.RUNS_DIR / "image"
    nn = np.load(runs / "nn" / NN_RUN / "cv_oof_proba.npy")
    forest = np.load(runs / "forest" / FOREST_RUN / "cv_oof_proba.npy")
    right = pd.DataFrame({"1-NN": knn_pred == y, "network": nn.argmax(1) == y, "forest": forest.argmax(1) == y,
                          "forest + network": (nn + forest).argmax(1) == y})
    decile = pd.qcut(sim, 10, labels=False)
    table = right.groupby(decile).mean()
    table.insert(0, "similarity_from", pd.Series(sim).groupby(decile).min())
    table.insert(1, "similarity_to", pd.Series(sim).groupby(decile).max())
    table.index.name = "decile"
    table.to_csv(config.RESULTS_CSV.parent / "knn_similarity_deciles.csv", float_format="%.4f")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
