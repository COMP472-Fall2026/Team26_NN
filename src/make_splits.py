# Creates the fixed validation split, the five folds and the test id order, once, with seed 42.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from src import config
from src.config import SEED, N_CLASSES


# Creates the 80/20 split, five stratified folds and test id order, refusing to overwrite.
def main() -> None:
    config.SPLITS_DIR.mkdir(exist_ok=True)
    targets = [config.SPLITS_DIR / f"{n}_split.npz" for n in N_CLASSES]
    targets.append(config.SPLITS_DIR / "test_ids.npz")
    existing = [p.name for p in targets if p.exists()]
    if existing:
        raise SystemExit(f"{existing} already exist. The split is made once and never regenerated.")

    for name in N_CLASSES:
        y = pd.read_csv(config.DATA_DIR / f"{name}_train.csv", usecols=["label"])["label"].to_numpy()
        idx = np.arange(len(y))
        tr, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
        fold = np.empty(len(y), dtype=np.int8)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (_, f) in enumerate(skf.split(idx, y)):
            fold[f] = k
        np.savez(config.SPLITS_DIR / f"{name}_split.npz",
                 train_idx=np.sort(tr), val_idx=np.sort(va), fold=fold)

    np.savez(config.SPLITS_DIR / "test_ids.npz", **{
        name: pd.read_csv(config.DATA_DIR / f"{name}_test.csv", usecols=["id"])["id"].to_numpy().astype(str)
        for name in N_CLASSES
    })


if __name__ == "__main__":
    main()
