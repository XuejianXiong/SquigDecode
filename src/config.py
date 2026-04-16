"""
SquigDecode Configuration Module.

This module serves as the central authority for all hyperparameters and physical 
constants used across the SquigDecode pipeline. It supports dynamic overrides 
via a local 'input.json' file to facilitate rapid experimentation without 
modifying core source code.

Physical constants are modeled after the Oxford Nanopore R10.4.1 pore chemistry 
operating at a 180mV bias, roughly translating to 400 bps translocation at 4-5kHz.

Attributes:
    BASE_PICOAMPERE_MAP (dict): Mapping of DNA bases to their mean ionic current (pA).
    DWELL_TIME_MEAN (int): Expected number of signal samples per base.
    BASE_TO_INT (dict): Label encoding for Connectionist Temporal Classification (CTC).
    USER_CONFIG (dict): Dictionary containing overrides loaded from 'input.json'.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Final

# Setup minimal logging for configuration tracking
logger = logging.getLogger(__name__)

# ===========================================================================
# 1. BIOPHYSICAL & SIGNAL CONSTANTS
# ===========================================================================

#: Mean current levels (pA) for R10.4.1 chemistry.
BASE_PICOAMPERE_MAP: Final[Dict[str, float]] = {
    "A": 95.0,
    "C": 110.0,
    "G": 125.0,
    "T": 80.0,
    "blank": 0.0,  # Reserved for CTC padding/blank token
}

#: Dwell time statistics (samples per base).
DWELL_TIME_MEAN: Final[int] = 12  
DWELL_TIME_STD: Final[int] = 3  
MIN_DWELL_TIME: Final[int] = 5  

#: Signal smoothing parameters.
WEIGHTS: Final[tuple] = (0.7, 0.2, 0.1)
WINDOW_SIZE: Final[int] = 3

#: Electronic noise and drift parameters.
NOISE_STD_MIN: Final[float] = 1.5 
NOISE_STD_MAX: Final[float] = 3.5 
DRIFT_FACTOR: Final[float] = 0.01  

# ===========================================================================
# 2. DIRECTORY & PATH MANAGEMENT
# ===========================================================================

# Project root calculation (assumes config.py is in project_root/src/)
ROOT_DIR: Final[Path] = Path(__file__).parent.parent.resolve()

TRAIN_PATH: Final[Path] = ROOT_DIR / "data/train"
TEST_PATH: Final[Path] = ROOT_DIR / "data/test"
CHECKPOINT_DIR: Final[str] = "checkpoints"
MODEL_DIR: Final[str] = "models"

# ===========================================================================
# 3. MACHINE LEARNING HYPERPARAMETERS
# ===========================================================================

NUM_SEQUENCES: int = 1000  
MIN_LENGTH: int = 50  
MAX_LENGTH: int = 100  

DATASET_TRAIN_RATIO: float = 0.8
DATASET_RANDOM_SEED: int = 42

TRAIN_NUM_EPOCHS: int = 50  
TRAIN_BATCH_SIZE: int = 32  
TRAIN_LEARNING_RATE: float = 1e-3  
MODEL_FILE: str = "squig_model.pt"  
CHECKPOINT_FILE: str = "checkpoint.pt"  

# ===========================================================================
# 4. CTC ENCODING & INFERENCE
# ===========================================================================

INFERENCE_NOISE: float = 0.0
USE_NOISE_CURRICULUM: bool = False

#: Label mapping for CTCLoss. Index 0 is strictly reserved for 'blank'.
BASE_TO_INT: Final[Dict[str, int]] = {
    "blank": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
}

#: Reverse mapping for decoding.
INT_TO_BASE: Final[Dict[int, str]] = {
    0: "",  # CTC Blank decodes to an empty string
    1: "A",
    2: "C",
    3: "G",
    4: "T",
}

# ===========================================================================
# 5. DYNAMIC CONFIGURATION LOADER
# ===========================================================================

def load_user_config(filename: str = "input.json") -> Dict[str, Any]:
    """
    Loads user-defined overrides from a JSON file.

    The function searches for the configuration file in the same directory 
    as this script. If the file is missing or contains malformed JSON, 
    it defaults to an empty dictionary, preserving the hardcoded constants.

    Args:
        filename (str): Name of the configuration file.

    Returns:
        Dict[str, Any]: Key-value pairs of configuration overrides.
    """
    config_path = Path(__file__).parent / filename
    
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
            return overrides
    except (json.JSONDecodeError, OSError) as e:
        # Using a print here as logging might not be fully configured yet
        print(f"[WARNING] Failed to load {filename}: {e}. Using defaults.")
        return {}

# Instantiate global overrides
USER_CONFIG: Final[Dict[str, Any]] = load_user_config()