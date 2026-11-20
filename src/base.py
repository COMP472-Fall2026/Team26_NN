# Defines BaseModel, the shared fit, predict and predict_proba interface every classifier in the project follows.
import numpy as np

class BaseModel:
    name = "base"                       # used in run folders: knn, tree, forest, nn

    # Stores the number of classes and the hyperparameters every model is built with.
    def __init__(self, n_classes: int, **params):
        self.n_classes = n_classes
        self.params = params

    # Learns from the training rows; each model must replace this method.
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "BaseModel":
        raise NotImplementedError

    # Returns class probabilities per row; each model must replace this method.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # Returns the most likely class per row, taken from predict_proba.
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # Returns every setting needed to rebuild the model, for the run's config file.
    def get_params(self) -> dict:
        return {"n_classes": self.n_classes, **self.params}
