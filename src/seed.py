# Sets the random seed for Python, NumPy and PyTorch so every run is reproducible.
import random
import numpy as np
from src.config import SEED

# Seeds Python, NumPy and PyTorch random generators with the project seed 42.
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
