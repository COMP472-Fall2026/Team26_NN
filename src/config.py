# Holds the shared constants: random seed, number of classes per dataset and the main project folders.
from pathlib import Path

SEED = 42
N_CLASSES = {"image": 50, "sentiment": 3}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
RUNS_DIR = ROOT / "runs"
RESULTS_CSV = ROOT / "results" / "results.csv"
