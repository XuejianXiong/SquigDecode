"""
SquigDecode: Configuration for Nanopore Signal Simulation
Values represent standard R10.4.1 pore chemistry at 180mV bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Base-to-Picoampere (pA) Mapping
# Authentic levels where G is high and T is the deepest blockade.
BASE_PICOAMPERE_MAP = {
    "A": 95.0,
    "C": 110.0,
    "G": 125.0,
    "T": 80.0,
    "blank": 0.0,  # Used for CTC Loss padding
}

# Simulation Parameters
NUM_SEQUENCES = 1000 # Number of DNA sequences to simulate
MIN_LENGTH = 50 # Minimum length of DNA sequences
MAX_LENGTH = 100 # Maximum length of DNA sequences

DWELL_TIME_MEAN = 15 # Average number of samples a base is detected (depends on sequencing speed)
DWELL_TIME_STD = 4  # The "spread" of how fast the DNA moves
MIN_DWELL_TIME = 5  # The minimum samples to detect a base

# Weights for the 3-sample sliding window filter (current, prev, next).
WEIGHTS = (0.7, 0.2, 0.1)

NOISE_STD = 3.5  # Gaussian noise in pA
DRIFT_FACTOR = 0.01  # Simulates slight electrical fluctuations over time

WINDOW_SIZE = 3  # Size of the sliding window for smoothing

# Dataset generation & splitting parameters
DATASET_TRAIN_RATIO = 0.8
DATASET_RANDOM_SEED = 42

TRAIN_PATH = Path("data/train")
TEST_PATH = Path("data/test")

# When generating the training split, noise standard deviation is sampled
# uniformly per sequence from [min, max]. Defaults keep legacy behavior.
DATASET_TRAIN_NOISE_STD_MIN = NOISE_STD
DATASET_TRAIN_NOISE_STD_MAX = NOISE_STD

# Training configuration constants
TRAIN_NUM_EPOCHS = 50  # Default number of epochs for training
TRAIN_BATCH_SIZE = 32  # Default batch size for DataLoader
TRAIN_LEARNING_RATE = 1e-3  # Learning rate for optimizer
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoint files
MODEL_DIR = "models"  # Directory to save trained models
MODEL_FILE = "squig_model.pt"  # Default model filename
CHECKPOINT_FILE = "checkpoint.pt"  # Default checkpoint filename
LOSS_PLOT_DPI = 150  # DPI when saving loss curve


# DNA-to-integer mapping for CTC loss
BASE_TO_INT = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
}

# ---------------------------------------------------------------------------
# User configuration: load optional overrides from src/input.json
# ---------------------------------------------------------------------------


def load_user_config(path: Path = Path(__file__).parent / "input.json") -> dict[str, Any]:
    """Read a JSON file containing user-specified parameters.

    If the file does not exist or is invalid, an empty dict is returned.
    """
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# global dictionary that modules can import and query for overrides
USER_CONFIG: dict = load_user_config()

# Base index to character mapping
INT_TO_BASE = {
    0: "Blank",
    1: "A",
    2: "C",
    3: "G",
    4: "T",
}
