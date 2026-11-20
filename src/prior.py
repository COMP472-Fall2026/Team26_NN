# Estimates the sentiment class mix of the test rows and reweights probabilities to correct for it.
import numpy as np


# Reweights each row's probabilities per class by target over source, then renormalises.
def adjust(proba, target, source):
    w = np.asarray(target, dtype=np.float64) / np.asarray(source, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64) * w
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


# Returns the share of rows predicted as each class, from the argmax.
def prediction_share(proba, n_classes):
    return np.bincount(np.asarray(proba).argmax(axis=1), minlength=n_classes) / len(proba)


# Estimates the test class mix by inverting the held-out confusion matrix.
def confusion_estimate(val_proba, y_val, test_proba):
    k = val_proba.shape[1]
    pred = val_proba.argmax(axis=1)
    C = np.zeros((k, k))
    for j in range(k):
        C[:, j] = np.bincount(pred[y_val == j], minlength=k) / max((y_val == j).sum(), 1)
    q = np.linalg.solve(C, prediction_share(test_proba, k))
    q = np.clip(q, 0, None)
    return q / q.sum()


# Estimates the test class mix with EM: correct, average, repeat until stable.
def em_estimate(test_proba, source, iters=1000, tol=1e-8):
    q = np.asarray(source, dtype=np.float64).copy()
    for _ in range(iters):
        new = adjust(test_proba, q, source).mean(axis=0).astype(np.float64)
        if np.abs(new - q).max() < tol:
            return new
        q = new
    return q


# Nudges per-class offsets until each class gets about the same number of predicted rows.
def balance_counts(proba, iters=3000, step=0.002):
    score = np.log(np.asarray(proba, dtype=np.float64) + 1e-12)
    k = score.shape[1]
    target = len(score) / k
    offset = np.zeros(k)
    for _ in range(iters):
        counts = np.bincount((score + offset).argmax(axis=1), minlength=k)
        offset -= step * (counts - target) / target
    return (score + offset).argmax(axis=1)


# Finds per-class weights that make the reweighted test probabilities average to uniform.
def balance_weights(test_proba, iters=1000, tol=1e-8):
    k = test_proba.shape[1]
    w = np.ones(k)
    for _ in range(iters):
        mean = adjust(test_proba, w, np.ones(k)).mean(axis=0)
        step = (1 / k) / mean
        w *= step
        w /= w.mean()
        if np.abs(step - 1).max() < tol:
            break
    return w
