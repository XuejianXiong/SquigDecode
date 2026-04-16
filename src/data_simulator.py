"""
SquigDecode: Nanopore Signal Simulation Engine.

This module implements a multi-stage biophysical simulation pipeline to generate 
synthetic nanopore current signals (squiggles). The pipeline models:
1.  Stochastic base translocation (Dwell Time modeling).
2.  Electronic RC-lag (Signal smoothing).
3.  Linear baseline drift and Gaussian ionic noise.
4.  Robust Signal Normalization (MAD standardization and center-clipping).

The simulator includes a 'stress-test' DNA generator capable of creating 
homopolymer regions, which are critical for validating CTC-based basecallers.

Example:
    >>> from squigdecode.data_simulation import generate_squiggle
    >>> signal, dwells = generate_squiggle("ACGT")
    >>> print(signal.shape)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config import (
    BASE_PICOAMPERE_MAP,
    DATASET_RANDOM_SEED,
    DATASET_TRAIN_RATIO,
    DRIFT_FACTOR,
    DWELL_TIME_MEAN,
    DWELL_TIME_STD,
    MAX_LENGTH,
    MIN_DWELL_TIME,
    MIN_LENGTH,
    NOISE_STD_MAX,
    NOISE_STD_MIN,
    NUM_SEQUENCES,
    TEST_PATH,
    TRAIN_PATH,
    USER_CONFIG,
)

# Configure Professional Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SimulatorConfig:
    """Encapsulates simulation hyperparameters for reproducibility."""
    noise_std_min: float = float(USER_CONFIG.get("noise_std_min", NOISE_STD_MIN))
    noise_std_max: float = float(USER_CONFIG.get("noise_std_max", NOISE_STD_MAX))
    drift_factor: float = float(USER_CONFIG.get("drift_factor", DRIFT_FACTOR))
    random_seed: int = int(USER_CONFIG.get("random_seed", DATASET_RANDOM_SEED))

def get_dwell_time() -> int:
    """
    Samples dwell time from a normal distribution with a hard floor.

    Returns:
        int: The number of signal samples the base occupies in the pore.
    """
    dwell = int(np.random.normal(DWELL_TIME_MEAN, DWELL_TIME_STD))
    return max(dwell, MIN_DWELL_TIME)

def _step_map_and_expand(dna_sequence: str) -> Tuple[np.ndarray, List[int]]:
    """
    Maps DNA bases to their respective pA levels and expands them.

    Args:
        dna_sequence: The input DNA string.

    Returns:
        Tuple[np.ndarray, List[int]]: The raw expanded signal and the list 
            of dwell times per base.
    """
    valid_bases = set(BASE_PICOAMPERE_MAP.keys()) - {"blank"}
    if not all(base in valid_bases for base in dna_sequence):
        raise ValueError("Sequence contains invalid DNA bases for R10.4.1 map.")

    signal_list: List[float] = []
    dwell_times: List[int] = []
    for base in dna_sequence:
        base_level = BASE_PICOAMPERE_MAP[base]
        dwell_time = get_dwell_time()
        signal_list.extend([base_level] * dwell_time)
        dwell_times.append(dwell_time)

    return np.array(signal_list, dtype=np.float32), dwell_times

def _step_electronic_lag(signal: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Applies an exponential moving average to simulate electronic RC lag.

    Args:
        signal: The raw 'step' signal.
        alpha: Smoothing factor (0 < alpha < 1). Lower is more lag.

    Returns:
        np.ndarray: The refined signal with realistic transitions.
    """
    refined = np.zeros_like(signal)
    refined[0] = signal[0]
    for i in range(1, len(signal)):
        refined[i] = alpha * signal[i] + (1 - alpha) * refined[i-1]
    return refined

def _step_add_noise(
    signal: np.ndarray,
    config: SimulatorConfig,
    current_noise_std: float
) -> np.ndarray:
    """
    Injects linear electrical drift and Gaussian white noise.

    Args:
        signal: The input signal array.
        config: Simulation configuration.
        current_noise_std: The standard deviation for Gaussian noise.

    Returns:
        np.ndarray: The noisy signal.
    """
    mean_signal = np.mean(signal)
    drift = np.linspace(0, config.drift_factor * mean_signal, len(signal))
    noise = np.random.normal(0, current_noise_std, len(signal))
    return signal + drift + noise

def robust_standardize(signal: np.ndarray, clip_limit: float = 5.0) -> np.ndarray:
    """
    Performs Median Absolute Deviation (MAD) standardization.

    Standardization process:
    1. Subtract median.
    2. Scale by MAD (normalized by 1.4826 for Gaussian consistency).
    3. Center-clipping to mitigate the impact of outlier spikes.

    Args:
        signal: 1D signal array.
        clip_limit: The maximum allowed deviation from the median.

    Returns:
        np.ndarray: The standardized signal clipped to [-clip_limit, clip_limit].
    """
    median = np.median(signal)
    mad = np.median(np.abs(signal - median)) * 1.4826
    
    if mad == 0:
        return signal - median

    standardized = (signal - median) / mad
    return np.clip(standardized, -clip_limit, clip_limit)

def generate_squiggle(
    dna_sequence: str,
    config: Optional[SimulatorConfig] = None,
    noise_std_override: Optional[float] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Orchestrates the full generation pipeline for a single DNA read.

    Args:
        dna_sequence: Target DNA sequence.
        config: Simulator configuration settings.
        noise_std_override: Specific noise level to apply.

    Returns:
        Tuple[np.ndarray, List[int]]: Standardized signal and its dwell times.
    """
    cfg = config or SimulatorConfig()
    noise_val = noise_std_override or cfg.noise_std_max

    # 1. Expand bases to pA levels
    signal, dwell_times = _step_map_and_expand(dna_sequence)

    # 2. Apply RC-lag
    signal = _step_electronic_lag(signal)

    # 3. Add noise and drift
    signal = _step_add_noise(signal, cfg, noise_val)

    # 4. Standardize for Neural Network input
    signal = robust_standardize(signal)

    return signal, dwell_times

def generate_random_dna_sequence(length: int, homopolymer_prob: float = 0.1) -> str:
    """
    Generates random DNA with simulated homopolymer sequences.

    Args:
        length: Desired sequence length.
        homopolymer_prob: Probability of generating a homopolymer run.

    Returns:
        str: Synthetic DNA sequence.
    """
    bases = ["A", "C", "G", "T"]
    seq: List[str] = []
    while len(seq) < length:
        base = np.random.choice(bases)
        if np.random.random() < homopolymer_prob:
            repeat = np.random.randint(3, 7)
            seq.extend([base] * repeat)
        else:
            seq.append(base)
    return "".join(seq[:length])

def save_dataset(
    signals: List[np.ndarray],
    sequences: List[str],
    dwells: List[List[int]],
    output_dir: Path,
) -> None:
    """
    Serialized the generated dataset to disk using Torch and Pickle.

    Args:
        signals: List of signal arrays.
        sequences: List of DNA sequences.
        dwells: List of dwell time lists.
        output_dir: Target directory for storage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(signals, output_dir / "signals.pt")
    
    with open(output_dir / "sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)

    with open(output_dir / "dwell_times.pkl", "wb") as f:
        pickle.dump(dwells, f)

    metadata = {
        "num_sequences": len(sequences),
        "mean_signal_len": np.mean([len(s) for s in signals]),
        "normalization": "MAD_with_clipping_5.0",
        "format_version": "1.1"
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    logger.info("Dataset persisted to: %s", output_dir)

def main() -> None:
    """Main entry point for dataset generation."""
    config = SimulatorConfig()
    np.random.seed(config.random_seed)

    num_seqs = int(USER_CONFIG.get("num_sequences", NUM_SEQUENCES))
    min_len = int(USER_CONFIG.get("min_length", MIN_LENGTH))
    max_len = int(USER_CONFIG.get("max_length", MAX_LENGTH))
    train_ratio = float(USER_CONFIG.get("train_ratio", DATASET_TRAIN_RATIO))

    logger.info("Starting simulation of %d sequences...", num_seqs)

    signals, sequences, dwells = [], [], []

    for _ in range(num_seqs):
        length = np.random.randint(min_len, max_len + 1)
        dna = generate_random_dna_sequence(length)
        
        # Stochastic noise level per read
        noise_val = np.random.uniform(config.noise_std_min, config.noise_std_max)
        
        signal, dwell = generate_squiggle(dna, config, noise_val)
        
        signals.append(signal)
        sequences.append(dna)
        dwells.append(dwell)

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    train_p = base_dir / Path(USER_CONFIG.get("data_dir", TRAIN_PATH))
    test_p = base_dir / Path(USER_CONFIG.get("test_data_dir", TEST_PATH))

    # Split logic
    indices = np.random.permutation(len(signals))
    split_idx = int(len(signals) * train_ratio)
    
    save_dataset(
        [signals[i] for i in indices[:split_idx]],
        [sequences[i] for i in indices[:split_idx]],
        [dwells[i] for i in indices[:split_idx]],
        train_p
    )
    save_dataset(
        [signals[i] for i in indices[split_idx:]],
        [sequences[i] for i in indices[split_idx:]],
        [dwells[i] for i in indices[split_idx:]],
        test_p
    )

if __name__ == "__main__":
    main()