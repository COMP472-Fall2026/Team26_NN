# Implements the random forest in NumPy: bootstrapped decision trees whose class probabilities are averaged.
import numpy as np

from src.base import BaseModel
from src.tree import DecisionTree


class RandomForest(BaseModel):
    name = "forest"

    # Checks and stores the forest settings; the trees are built when fitting.
    def __init__(
        self,
        n_classes: int,
        n_trees: int = 100,
        max_depth: int = None,
        max_features="sqrt",
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        criterion: str = "gini",
        n_bins: int = 64,
        seed: int = 42,
    ):
        super().__init__(
            n_classes=n_classes,
            n_trees=n_trees,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            criterion=criterion,
            n_bins=n_bins,
            seed=seed,
        )
        if n_trees < 1:
            raise ValueError("n_trees must be at least 1")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if not isinstance(bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be boolean")

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bool(bootstrap)
        self.criterion = criterion
        self.n_bins = n_bins
        self.seed = seed
        self.trees = []

    # Trains each tree on a bootstrap sample of rows with its own seed.
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X and y must contain the same number of rows")
        if len(X) == 0:
            raise ValueError("X and y must not be empty")

        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            if self.bootstrap:
                row_indices = rng.integers(0, len(X), size=len(X))
            else:
                row_indices = np.arange(len(X))

            tree = DecisionTree(
                n_classes=self.n_classes,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=2,
                min_samples_leaf=self.min_samples_leaf,
                n_bins=self.n_bins,
                max_features=self.max_features,
                seed=int(rng.integers(0, 2**32)),
            )
            tree.fit(X[row_indices], y[row_indices])
            self.trees.append(tree)
        return self

    # Averages the class probabilities of all trees for each row.
    def predict_proba(self, X):
        if not self.trees:
            raise RuntimeError("RandomForest must be fitted before prediction")

        X = np.asarray(X)
        probabilities = np.zeros((len(X), self.n_classes), dtype=np.float32)
        for tree in self.trees:
            probabilities += tree.predict_proba(X)
        probabilities /= len(self.trees)
        return probabilities.astype(np.float32)

    # Returns every forest setting needed to rebuild the model, for the run's config.
    def get_params(self):
        return {
            "n_classes": self.n_classes,
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "bootstrap": self.bootstrap,
            "criterion": self.criterion,
            "n_bins": self.n_bins,
            "seed": self.seed,
        }
