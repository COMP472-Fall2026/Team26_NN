# Implements the decision tree from scratch in NumPy, with Gini or entropy splits and optional binned thresholds.
import numpy as np

try:
    from src.base import BaseModel
except ModuleNotFoundError:
    from base import BaseModel


class _Node:
    # Stores a node's class counts, or its split feature, threshold and children.
    def __init__(
        self,
        *,
        counts=None,
        feature=None,
        threshold=None,
        left=None,
        right=None,
    ):
        self.counts = counts
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    # A node is a leaf when it has no split feature.
    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTree(BaseModel):
    name = "tree"

    # Checks and stores the tree settings; the nodes are built when fitting.
    def __init__(
        self,
        n_classes: int,
        criterion: str = "gini",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_bins: int = None,
        max_features=None,
        seed: int = 42,
        **params,
    ):
        super().__init__(
            n_classes=n_classes,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_bins=n_bins,
            max_features=max_features,
            seed=seed,
            **params,
        )
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if n_bins is not None and n_bins < 2:
            raise ValueError("n_bins must be at least 2")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.max_features = max_features
        self.seed = seed
        self.root = None
        self.bin_thresholds_ = None
        self._rng = np.random.default_rng(seed)

    # Validates the data, bins it if asked, and grows the tree from the root.
    def fit(self, X, y, X_val=None, y_val=None):
        X = self._validate_X(X)
        y = self._validate_y(y, X.shape[0])

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")

        self._rng = np.random.default_rng(self.seed)
        X = self._preprocess_fit(X)
        self.root = self._build_node(X, y, depth=0)
        return self

    # Sends each row down the tree and returns its leaf's class shares.
    def predict_proba(self, X):
        if self.root is None:
            raise RuntimeError("DecisionTree must be fitted before prediction")

        X = self._validate_X(X)
        X = self._preprocess_predict(X)
        probabilities = np.empty((X.shape[0], self.n_classes), dtype=np.float32)

        for row_index, row in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            probabilities[row_index] = node.counts.astype(np.float64) / node.counts.sum()

        return probabilities

    # Returns every tree setting needed to rebuild the model, for the run's config.
    def get_params(self):
        return {
            "n_classes": self.n_classes,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "n_bins": self.n_bins,
            "max_features": self.max_features,
            "seed": self.seed,
        }

    # Learns quantile bin thresholds per feature from training rows and bins the data.
    def _preprocess_fit(self, X):
        if self.n_bins is None:
            return X

        self.bin_thresholds_ = []
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        binned = np.empty(X.shape, dtype=np.int32)

        for feature in range(X.shape[1]):
            thresholds = np.unique(np.quantile(X[:, feature], quantiles))
            self.bin_thresholds_.append(thresholds)
            binned[:, feature] = np.digitize(
                X[:, feature], thresholds, right=False
            )

        return binned

    # Bins new rows with the thresholds learned from the training rows.
    def _preprocess_predict(self, X):
        if self.n_bins is None:
            return X
        if len(self.bin_thresholds_) != X.shape[1]:
            raise ValueError("X has a different number of features than training data")

        binned = np.empty(X.shape, dtype=np.int32)
        for feature, thresholds in enumerate(self.bin_thresholds_):
            binned[:, feature] = np.digitize(
                X[:, feature], thresholds, right=False
            )
        return binned

    # Checks the features form a finite two-dimensional array.
    def _validate_X(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[0] == 0:
            return X
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

    # Checks the labels are integers in range, one per training row.
    def _validate_y(self, y, n_rows):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        if y.shape[0] != n_rows:
            raise ValueError("X and y must contain the same number of rows")
        if y.size == 0:
            raise ValueError("y must not be empty")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer class indices")
        if np.any(y < 0) or np.any(y >= self.n_classes):
            raise ValueError("y contains a class outside 0..n_classes - 1")
        return y.astype(np.int64, copy=False)

    # Grows one node: a leaf if a stop rule applies, otherwise split and recurse.
    def _build_node(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes).astype(np.int64)

        reached_depth_limit = (
            self.max_depth is not None and depth >= self.max_depth
        )
        if reached_depth_limit or y.size < self.min_samples_split or np.count_nonzero(counts) == 1:
            return _Node(counts=counts)

        split = self._best_split(X, y)
        if split is None:
            return _Node(counts=counts)

        feature, threshold = split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build_node(X[left_mask], y[left_mask], depth + 1)
        right = self._build_node(X[right_mask], y[right_mask], depth + 1)
        return _Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
        )

    # Tries every threshold on the candidate features and keeps the largest impurity decrease.
    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        parent_impurity = self._impurity(y)

        features = self._candidate_features(X.shape[1])
        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            labels = y[order]

            group_starts = np.flatnonzero(values[1:] != values[:-1]) + 1
            for split_index in group_starts:
                left_size = split_index
                right_size = y.size - left_size
                if (
                    left_size < self.min_samples_leaf
                    or right_size < self.min_samples_leaf
                ):
                    continue

                left_labels = labels[:split_index]
                right_labels = labels[split_index:]
                gain = parent_impurity - (
                    left_size * self._impurity(left_labels)
                    + right_size * self._impurity(right_labels)
                ) / y.size

                if gain > best_gain:
                    threshold = (
                        values[split_index - 1] + values[split_index]
                    ) / 2.0
                    if np.isfinite(threshold):
                        best_feature = feature
                        best_threshold = threshold
                        best_gain = gain

        if best_feature is None:
            return None
        return best_feature, best_threshold

    # Returns the features to try at this split: all, or a random subset.
    def _candidate_features(self, n_features):
        if self.max_features is None:
            return range(n_features)
        if isinstance(self.max_features, str):
            if self.max_features != "sqrt":
                raise ValueError("max_features string must be 'sqrt'")
            n_candidates = max(1, int(np.sqrt(n_features)))
        else:
            n_candidates = int(self.max_features)
        if n_candidates < 1 or n_candidates > n_features:
            raise ValueError(
                "max_features must be between 1 and the number of input features"
            )
        return self._rng.choice(
            n_features, size=n_candidates, replace=False
        )

    # Computes the Gini impurity or entropy of a set of labels.
    def _impurity(self, y):
        counts = np.bincount(y, minlength=self.n_classes)
        proportions = counts[counts != 0] / counts.sum()

        if self.criterion == "gini":
            return 1.0 - np.square(proportions).sum()

        return -np.sum(proportions * np.log(proportions))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=16000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "image_train.csv"
    split_path = project_root / "splits" / "image_split.npz"

    if not data_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            "Run 'python -m src.make_splits' after adding "
            "data/image_train.csv and data/image_test.csv."
        )

    data = pd.read_csv(data_path)
    X = data.drop(columns=["id", "label"]).to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.int64)

    split = np.load(split_path)
    train_idx = split["train_idx"][:args.max_rows]
    val_idx = split["val_idx"]
    split.close()

    for criterion in ("gini", "entropy"):
        tree = DecisionTree(
            n_classes=50,
            criterion=criterion,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            n_bins=args.n_bins,
        )
        tree.fit(X[train_idx], y[train_idx])

        validation_proba = tree.predict_proba(X[val_idx])
        validation_accuracy = np.mean(
            validation_proba.argmax(axis=1) == y[val_idx]
        )

        print(f"{criterion} val {validation_accuracy:.4f}")
