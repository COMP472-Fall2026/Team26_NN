# Loads the competition CSVs, applies the committed splits and fits every transform on training rows only.
import re

import numpy as np
import pandas as pd
from src import config
from src.config import N_CLASSES

N_FEATURES = {"image": 500, "sentiment": 768}
N_ROWS = {"image": (20000, 5000), "sentiment": (12000, 5000)}


# Parses one competition CSV into ids, features and labels, caching the result.
def _read(name: str, part: str):
    csv = config.DATA_DIR / f"{name}_{part}.csv"
    cache = config.DATA_DIR / "cache" / f"{name}_{part}.npz"
    if cache.exists() and cache.stat().st_mtime >= csv.stat().st_mtime:
        z = np.load(cache)
        return z["ids"], z["X"], (z["y"] if "y" in z.files else None)

    df = pd.read_csv(csv)
    cols = [f"feature_{i}" for i in range(N_FEATURES[name])]
    ids = df["id"].to_numpy().astype(str)
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None

    n = N_ROWS[name][0 if part == "train" else 1]
    assert X.shape == (n, N_FEATURES[name]), f"{csv.name}: got {X.shape}"
    assert len(set(ids)) == len(ids), f"{csv.name}: duplicate ids"
    assert np.isfinite(X).all(), f"{csv.name}: non-finite features"
    if y is not None:
        assert y.min() == 0 and y.max() == N_CLASSES[name] - 1, f"{csv.name}: labels outside 0..{N_CLASSES[name] - 1}"

    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, ids=ids, X=X, **({"y": y} if y is not None else {}))
    return ids, X, y


# Scales each column to mean 0 and standard deviation 1, using the first array's statistics.
def _standardise(X_fit, *others):
    mu = X_fit.mean(axis=0)
    sd = X_fit.std(axis=0)
    sd[sd == 0] = 1.0                   # constant columns stay constant instead of dividing by zero
    return [((X - mu) / sd).astype(np.float32) for X in (X_fit, *others)]


# Recovers image word counts, then applies log counts times IDF and L2-normalises rows.
def _logtfidf(X_fit, *others):
    # Divides each row by its smallest non-zero value to recover integer counts.
    def counts(X):
        m = np.where(X > 0, X, np.inf).min(axis=1, keepdims=True)
        return np.rint(X / m)
    C_fit = counts(X_fit)
    df = (C_fit > 0).sum(axis=0)
    idf = np.log((1 + len(C_fit)) / (1 + df)) + 1      # smoothed, so unseen words get a finite weight
    out = []
    for C in (C_fit, *(counts(X) for X in others)):
        W = np.log1p(C) * idf
        out.append((W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32))
    return out


# Standardises, then projects all arrays onto the top n principal components of the fit rows.
def _pca(n, X_fit, *others):
    Z_fit, *Z_others = _standardise(X_fit, *others)
    if not 1 <= n <= Z_fit.shape[1]:
        raise ValueError(f"pca{n}: n must be between 1 and {Z_fit.shape[1]}")
    vals, vecs = np.linalg.eigh(Z_fit.T.astype(np.float64) @ Z_fit / (len(Z_fit) - 1))
    V = vecs[:, np.argsort(vals)[::-1][:n]]
    V *= np.sign(V[np.abs(V).argmax(axis=0), np.arange(n)])   # fix each component's sign
    return [(Z @ V).astype(np.float32) for Z in (Z_fit, *Z_others)]


# Fits the named transform on the first array only and applies it to all.
def apply_transform(transform: str, dataset: str, X_fit, *others):
    if transform == "raw":
        return [X_fit, *others]
    if transform == "standardise":
        return _standardise(X_fit, *others)
    if transform == "logtfidf":
        if dataset != "image":
            raise ValueError("logtfidf only applies to the image counts")
        return _logtfidf(X_fit, *others)
    m = re.fullmatch(r"pca(\d+)", transform)
    if m:
        return _pca(int(m.group(1)), X_fit, *others)
    raise ValueError(f"unknown transform {transform!r}: use raw, standardise, logtfidf or pca{{n}}")


# Loads one dataset as the fixed validation split, or as all training rows plus test.
def load(dataset: str, transform: str = "raw", final: bool = False):
    if dataset not in N_CLASSES:
        raise ValueError(f"dataset must be one of {list(N_CLASSES)}, got {dataset!r}")

    _, X, y = _read(dataset, "train")
    if final:
        ids_te, X_te, _ = _read(dataset, "test")
        expected = np.load(config.SPLITS_DIR / "test_ids.npz")[dataset]
        assert np.array_equal(ids_te, expected), "test CSV order differs from splits/test_ids.npz"
        X, X_te = apply_transform(transform, dataset, X, X_te)
        return X, y, X_te

    split = np.load(config.SPLITS_DIR / f"{dataset}_split.npz")
    tr, va = split["train_idx"], split["val_idx"]
    Xtr, Xva = apply_transform(transform, dataset, X[tr], X[va])
    return Xtr, y[tr], Xva, y[va]


# Loads one of the five folds as held-out rows, training on the other four.
def load_fold(dataset: str, transform: str, fold: int):
    if dataset not in N_CLASSES:
        raise ValueError(f"dataset must be one of {list(N_CLASSES)}, got {dataset!r}")
    folds = np.load(config.SPLITS_DIR / f"{dataset}_split.npz")["fold"]
    if fold not in set(folds.tolist()):
        raise ValueError(f"fold must be one of {sorted(set(folds.tolist()))}, got {fold}")

    _, X, y = _read(dataset, "train")
    ids_te, X_te, _ = _read(dataset, "test")
    assert np.array_equal(ids_te, np.load(config.SPLITS_DIR / "test_ids.npz")[dataset]), \
        "test CSV order differs from splits/test_ids.npz"
    tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    Xtr, Xva, Xte = apply_transform(transform, dataset, X[tr], X[va], X_te)
    return Xtr, y[tr], Xva, y[va], Xte, va
