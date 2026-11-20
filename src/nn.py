# Defines the feed-forward neural network in PyTorch and trains it with mini-batches and early stopping.
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.base import BaseModel
from src.config import SEED

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class NeuralNetwork(nn.Module):
    # Stacks linear, activation and dropout layers for each hidden size, then an output layer.
    def __init__(self, n_features=500, n_classes=50, hidden=(512, 512),
                 activation="relu", dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        in_size = n_features
        for h in hidden:
            layers.append(nn.Linear(in_size, h))
            layers.append(ACTIVATIONS[activation]())
            layers.append(nn.Dropout(dropout))
            in_size = h
        # Outside the loop: CrossEntropyLoss needs raw scores I think
        layers.append(nn.Linear(in_size, n_classes))
        self.linear_relu_stack = nn.Sequential(*layers)

    # Passes a batch through the layers and returns raw class scores.
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MLP(BaseModel):
    name = "nn"

    # Stores the network's hyperparameters; the network itself is built when fitting.
    def __init__(self, n_classes, hidden=(256, 64), activation="relu", dropout=0.0,
                 lr=1e-3, weight_decay=0.0, optimizer="adam", batch_size=64,
                 max_epochs=50, patience=15, seed=SEED):
        super().__init__(n_classes, hidden=hidden, activation=activation, dropout=dropout,
                         lr=lr, weight_decay=weight_decay, optimizer=optimizer,
                         batch_size=batch_size, max_epochs=max_epochs,
                         patience=patience, seed=seed)
        self.net = None
        self.history = None

    # Trains the network with mini-batches, keeping the epoch with lowest validation loss.
    def fit(self, X, y, X_val=None, y_val=None):
        p = self.params
        torch.manual_seed(p["seed"])

        self.net = NeuralNetwork(X.shape[1], self.n_classes, p["hidden"], p["activation"], p["dropout"])
        optimizer = OPTIMIZERS[p["optimizer"]](self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=p["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(p["seed"]))

        val_loader = None
        if X_val is not None:
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val, dtype=torch.long))
            val_loader = DataLoader(val_dataset, batch_size=p["batch_size"])

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(p["max_epochs"]):
            train_loss, train_acc = train_loop(loader, self.net, loss_fn, optimizer)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if val_loader is not None:
                val_loss, val_acc = test_loop(val_loader, self.net, loss_fn)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self.net.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= p["patience"]:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # Runs the trained network without gradients and turns scores into probabilities with softmax.
    def predict_proba(self, X):
        self.net.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.net(X_tensor)
            proba = torch.softmax(logits, dim=1)
        return proba.numpy()



# Runs one training epoch over all batches and returns mean loss and accuracy.
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss, correct = 0, 0
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * len(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return total_loss / size, correct / size


# Scores the network on a dataset without training, returning mean loss and accuracy.
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * len(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return total_loss / size, correct / size
