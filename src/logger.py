# Saves every run's settings, scores and predictions to its run folder and appends one row to the results files.
import csv
import json
import pickle
import re
import subprocess
import time
from datetime import datetime

import numpy as np

from src import config
from src.config import N_CLASSES, SEED

COLUMNS = ["timestamp", "owner", "dataset", "model", "run_id", "transform", "params_json", "seed",
           "train_acc", "val_acc", "fit_seconds", "predict_seconds", "model_bytes", "git_commit", "notes"]
RUN_ID = re.compile(r"[a-z0-9][a-z0-9._\-]*")


# Returns the short commit hash, marked dirty if there are uncommitted changes.
def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=config.ROOT).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Gets class probabilities from a model and checks their shape and row sums.
def _proba(model, X, n_classes):
    p = np.asarray(model.predict_proba(X), dtype=np.float32)
    assert p.shape == (len(X), n_classes), f"bad proba shape {p.shape}"
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-3), "rows must sum to 1"
    return p


# Scores a fitted model and saves its run folder and results row.
def log_run(dataset, model, run_id, transform, y_train, X_train, y_val, X_val,
            fit_seconds, owner, notes="", seed=None) -> dict:
    if dataset not in N_CLASSES:
        raise ValueError(f"bad dataset {dataset!r}")
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"bad run_id {run_id!r}")
    k = N_CLASSES[dataset]
    run_dir = config.RUNS_DIR / dataset / model.name / run_id
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists")

    train_proba = _proba(model, X_train, k)
    t0 = time.perf_counter()
    val_proba = _proba(model, X_val, k)
    predict_seconds = time.perf_counter() - t0

    val_pred = val_proba.argmax(axis=1).astype(np.int16)
    train_acc = float((train_proba.argmax(axis=1) == y_train).mean())
    val_acc = float((val_pred == y_val).mean())

    record = {
        "run_id": run_id, "dataset": dataset, "model": model.name, "transform": transform,
        "params": model.get_params(), "seed": model.params.get("seed", SEED) if seed is None else seed,
        "owner": owner, "git_commit": _git_commit(), "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_acc": train_acc, "val_acc": val_acc, "fit_seconds": float(fit_seconds),
        "predict_seconds": predict_seconds, "model_bytes": len(pickle.dumps(model)), "notes": notes,
    }

    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(record, indent=2, default=str))
    np.save(run_dir / "val_pred.npy", val_pred)
    np.save(run_dir / "val_proba.npy", val_proba)

    csv_path = config.RESULTS_CSV
    csv_path.parent.mkdir(exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({**{c: record.get(c) for c in COLUMNS},
                    "params_json": json.dumps(record["params"], sort_keys=True, default=str)})
    return record


# Saves the test probabilities and settings of a model retrained on all rows.
def log_final(dataset, model, transform, X_test, n_train, fit_seconds, owner, source_run, notes="",
              overwrite=False, seed=None) -> dict:
    if dataset not in N_CLASSES:
        raise ValueError(f"bad dataset {dataset!r}")
    k = N_CLASSES[dataset]
    n_test = len(np.load(config.SPLITS_DIR / "test_ids.npz")[dataset])
    if len(X_test) != n_test:
        raise ValueError(f"X_test has {len(X_test)} rows, expected {n_test}")
    run_dir = config.RUNS_DIR / dataset / model.name / "final"
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"{run_dir} exists")

    t0 = time.perf_counter()
    test_proba = _proba(model, X_test, k)
    predict_seconds = time.perf_counter() - t0
    pred = test_proba.argmax(axis=1)

    record = {
        "run_id": "final", "source_run": source_run, "dataset": dataset, "model": model.name,
        "transform": transform, "params": model.get_params(),
        "seed": model.params.get("seed", SEED) if seed is None else seed, "owner": owner,
        "git_commit": _git_commit(), "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(n_train), "fit_seconds": float(fit_seconds), "predict_seconds": predict_seconds,
        "model_bytes": len(pickle.dumps(model)),
        "test_pred_share": (np.bincount(pred, minlength=k) / len(pred)).round(4).tolist(),
        "notes": notes,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(record, indent=2, default=str))
    np.save(run_dir / "test_proba.npy", test_proba)
    return record


CLUSTER_COLUMNS = ["timestamp", "owner", "dataset", "model", "run_id", "transform", "params_json",
                   "train_ari", "train_nmi", "val_ari", "val_nmi", "inertia", "fit_seconds", "git_commit", "notes"]


# Saves a clustering run's settings, validation labels and scores, and appends a results row.
def log_cluster(dataset, model, run_id, transform, val_labels, scores, fit_seconds, owner, notes="") -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"bad run_id {run_id!r}")
    run_dir = config.RUNS_DIR / dataset / model.name / run_id
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists")

    record = {
        "run_id": run_id, "dataset": dataset, "model": model.name, "transform": transform,
        "params": model.params, "owner": owner, "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_ari": scores["train"]["ari"], "train_nmi": scores["train"]["nmi"],
        "val_ari": scores["val"]["ari"], "val_nmi": scores["val"]["nmi"],
        "inertia": model.inertia, "fit_seconds": float(fit_seconds), "notes": notes,
    }
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(record, indent=2, default=str))
    np.save(run_dir / "val_labels.npy", np.asarray(val_labels, dtype=np.int16))

    csv_path = config.RESULTS_CSV.parent / "cluster_results.csv"
    csv_path.parent.mkdir(exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLUSTER_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({**{c: record.get(c) for c in CLUSTER_COLUMNS},
                    "params_json": json.dumps(record["params"], sort_keys=True)})
    return record


CV_COLUMNS = ["timestamp", "owner", "dataset", "model", "run_id", "transform", "params_json", "n_folds",
              "fold_accs", "mean_acc", "std_acc", "fixed_split_val_acc", "fit_seconds_mean", "git_commit", "notes"]


# Saves a 5-fold check's fold accuracies and predictions inside an existing run's folder.
def log_cv(dataset, model_name, run_id, transform, params, fold_accs, oof_proba, test_proba, fit_seconds,
           fixed_split_val_acc, owner, notes="") -> dict:
    run_dir = config.RUNS_DIR / dataset / model_name / run_id
    if not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"{run_dir} not logged")
    if (run_dir / "cv.json").exists():
        raise FileExistsError(f"{run_dir}/cv.json exists")
    k = N_CLASSES[dataset]
    oof_proba = np.asarray(oof_proba, dtype=np.float32)
    test_proba = np.asarray(test_proba, dtype=np.float32)
    assert oof_proba.ndim == 2 and oof_proba.shape[1] == k, f"oof_proba shape {oof_proba.shape}"
    assert test_proba.ndim == 3 and test_proba.shape[0] == len(fold_accs) and test_proba.shape[2] == k

    accs = np.asarray(fold_accs, dtype=float)
    record = {
        "run_id": run_id, "dataset": dataset, "model": model_name, "transform": transform, "params": params,
        "owner": owner, "git_commit": _git_commit(), "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_folds": len(accs), "fold_accs": accs.round(6).tolist(), "mean_acc": float(accs.mean()),
        "std_acc": float(accs.std(ddof=1)), "fixed_split_val_acc": float(fixed_split_val_acc),
        "fit_seconds_mean": float(np.mean(fit_seconds)), "notes": notes,
    }
    (run_dir / "cv.json").write_text(json.dumps(record, indent=2, default=str))
    np.save(run_dir / "cv_oof_proba.npy", oof_proba)
    np.save(run_dir / "cv_test_proba.npy", test_proba)

    csv_path = config.RESULTS_CSV.parent / "cv_results.csv"
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({**{c: record.get(c) for c in CV_COLUMNS},
                    "params_json": json.dumps(params, sort_keys=True, default=str),
                    "fold_accs": " ".join(f"{a:.4f}" for a in accs)})
    return record


# Recreates a logged 5-fold check's lost arrays, refusing them if fold accuracies differ.
def restore_cv_arrays(dataset, model_name, run_id, fold_accs, oof_proba, test_proba, tolerance) -> float:
    run_dir = config.RUNS_DIR / dataset / model_name / run_id
    logged = json.loads((run_dir / "cv.json").read_text())["fold_accs"]
    for name in ("cv_oof_proba.npy", "cv_test_proba.npy"):
        if (run_dir / name).exists():
            raise FileExistsError(f"{run_dir / name} exists")
    k = N_CLASSES[dataset]
    oof_proba = np.asarray(oof_proba, dtype=np.float32)
    test_proba = np.asarray(test_proba, dtype=np.float32)
    assert oof_proba.ndim == 2 and oof_proba.shape[1] == k, f"oof_proba shape {oof_proba.shape}"
    assert test_proba.ndim == 3 and test_proba.shape[0] == len(logged) and test_proba.shape[2] == k

    gap = float(np.abs(np.asarray(fold_accs) - np.asarray(logged)).max())
    if gap > tolerance:
        raise ValueError(f"{run_dir}: fold gap {gap:.6f} > {tolerance}")
    np.save(run_dir / "cv_oof_proba.npy", oof_proba)
    np.save(run_dir / "cv_test_proba.npy", test_proba)
    return gap
