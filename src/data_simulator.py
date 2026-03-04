"""
SquigDecode: Data Simulator for Nanopore Signal Generation

This module provides functionality to simulate realistic nanopore squiggle signals
from DNA sequences, including noise, drift, and base-specific signal characteristics.
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from config import (
    BASE_PICOAMPERE_MAP,
    NUM_SEQUENCES,
    MIN_LENGTH,
    MAX_LENGTH,
    DWELL_TIME_MEAN,
    DWELL_TIME_STD,
    MIN_DWELL_TIME,
    WEIGHTS,
    NOISE_STD,
    DRIFT_FACTOR,
    WINDOW_SIZE,
    USER_CONFIG,
)


# apply optional overrides from user config
NOISE_STD = USER_CONFIG.get("noise_std", NOISE_STD)
DRIFT_FACTOR = USER_CONFIG.get("drift_factor", DRIFT_FACTOR)


def get_dwell_time() -> int:
    """
    Generate a random dwell time (number of samples) for a base.

    Dwell time is sampled from a normal distribution with mean DWELL_TIME_MEAN,
    std DWELL_TIME_STD, and enforces a minimum of MIN_DWELL_TIME.

    Returns:
        int: Random dwell time >= MIN_DWELL_TIME
    """
    dwell = int(np.random.normal(DWELL_TIME_MEAN, DWELL_TIME_STD))
    return max(dwell, MIN_DWELL_TIME)


def _step_map_and_expand(dna_sequence: str) -> Tuple[np.ndarray, List[int]]:
    """Step 1 & 2: Validate sequence, map bases to levels and expand by dwell times.

    Returns:
        Tuple[np.ndarray, List[int]]: Expanded raw signal and list of dwell times.
    """
    valid_bases = set(BASE_PICOAMPERE_MAP.keys()) - {"blank"}
    if not all(base in valid_bases for base in dna_sequence):
        raise ValueError(f"Invalid bases in sequence. Valid bases: {valid_bases}")

    signal_list: List[float] = []
    dwell_times: List[int] = []
    for base in dna_sequence:
        base_level = BASE_PICOAMPERE_MAP[base]
        dwell_time = get_dwell_time()
        signal_list.extend([base_level] * dwell_time)
        dwell_times.append(dwell_time)

    return np.array(signal_list, dtype=np.float32), dwell_times


def _step_sliding_window_filter(
    signal: np.ndarray, weights: Tuple[float, float, float] = WEIGHTS
) -> np.ndarray:
    """Step 3: Apply a 3-sample sliding window filter (current, prev, next).

    Weights default to (current=0.7, prev=0.2, next=0.1).
    """
    filtered = np.zeros_like(signal)
    w_cur, w_prev, w_next = weights
    for i in range(len(signal)):
        prev_idx = max(0, i - 1)
        next_idx = min(len(signal) - 1, i + 1)
        filtered[i] = (
            w_cur * signal[i] + w_prev * signal[prev_idx] + w_next * signal[next_idx]
        )
    return filtered


def _step_add_noise(
    signal: np.ndarray, 
    drift_factor: float = DRIFT_FACTOR,
    noise_std: float = NOISE_STD
) -> np.ndarray:
    """Step 4a: Add linear drift proportional to the mean signal."""
    mean_signal = np.mean(signal)
    drift = np.linspace(0, drift_factor * mean_signal, len(signal))

    """Step 4b: Add Gaussian noise with provided standard deviation."""
    noise = np.random.normal(0, noise_std, len(signal))

    return signal + drift + noise


def _step_smooth(
    signal: np.ndarray, window_size: int = WINDOW_SIZE
) -> np.ndarray:
    """Step 5: Smooth signal with a moving average of given window size."""
    smoothed = np.zeros_like(signal)
    for i in range(len(signal)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(signal), i + window_size // 2 + 1)
        window = signal[start_idx:end_idx]
        smoothed[i] = np.mean(window)
    return smoothed


def generate_squiggle(dna_sequence: str) -> Tuple[np.ndarray, List[int]]:
    """Generate a realistic nanopore squiggle signal from a DNA sequence.

    The function composes modular steps implemented as helper functions to make
    the pipeline more readable and testable. Behavior and defaults are unchanged.
    """
    # Steps 1 & 2: map bases to levels and expand by dwell
    signal, dwell_times = _step_map_and_expand(dna_sequence)

    # Step 3: sliding-window filtering
    signal = _step_sliding_window_filter(signal)

    # Step 4: add drift and noise
    signal = _step_add_noise(signal, DRIFT_FACTOR, NOISE_STD)

    # Step 5: smoothing
    signal = _step_smooth(signal, WINDOW_SIZE)

    return signal, dwell_times


def standardize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Standardize a signal to zero mean and unit variance.

    Applies z-score normalization: (x - mean) / std

    Args:
        signal: Input signal array of shape (n_samples,)

    Returns:
        np.ndarray: Standardized signal with zero mean and unit variance
    """
    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        # Handle constant signal case (avoid division by zero)
        return signal - mean

    return (signal - mean) / std


def generate_random_dna_sequence(length: int) -> str:
    """
    Generate a random DNA sequence of specified length.

    Args:
        length: Length of the DNA sequence

    Returns:
        str: Random DNA sequence composed of A, C, G, T bases
    """
    bases = ["A", "C", "G", "T"]
    return "".join(np.random.choice(bases, length))


def save_dataset(
    all_signals: List[np.ndarray],
    all_sequences: List[str],
    all_dwell_times: List[List[int]],
    output_dir: Path,
) -> None:
    """
    Save generated dataset to disk.

    Args:
        all_signals: List of standardized signal arrays
        all_sequences: List of DNA sequences
        all_dwell_times: List of dwell times for each sequence
    """
    signal_lengths = [len(s) for s in all_signals]

    # Save signals as PyTorch tensors
    # Stack signals into a single tensor (signals may have variable lengths)
    torch.save(all_signals, output_dir / "signals.pt")
    print(f"\nSignals saved to {output_dir / 'signals.pt'}")

    # Save sequences as pickle for reference
    with open(output_dir / "sequences.pkl", "wb") as f:
        pickle.dump(all_sequences, f)
    print(f"Sequences saved to {output_dir / 'sequences.pkl'}")

    # Save dwell times for signal reconstruction
    with open(output_dir / "dwell_times.pkl", "wb") as f:
        pickle.dump(all_dwell_times, f)
    print(f"Dwell times saved to {output_dir / 'dwell_times.pkl'}")

    # Save metadata
    sequence_lengths = [len(s) for s in all_sequences]
    metadata = {
        "num_sequences": len(all_sequences),
        "min_sequence_length": min(sequence_lengths),
        "max_sequence_length": max(sequence_lengths),
        "min_signal_length": min(signal_lengths),
        "max_signal_length": max(signal_lengths),
        "mean_signal_length": np.mean(signal_lengths),
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {output_dir / 'metadata.pkl'}")


def main() -> None:
    """
    Main execution: Generate a dataset of simulated squiggle signals.

    Generates 1000 random DNA sequences (50-100 bases each),
    creates their corresponding squiggle signals, standardizes them,
    and saves the dataset for training as PyTorch tensors and pickle files.

    Output files:
    - data/signals.pt: PyTorch tensor of standardized signals
    - data/sequences.pkl: Pickle file of DNA sequences
    - data/metadata.pkl: Metadata about the dataset
    """
    # Allow overriding dataset generation params via src/input.json
    num_sequences = int(USER_CONFIG.get("num_sequences", NUM_SEQUENCES))
    min_length = int(USER_CONFIG.get("min_length", MIN_LENGTH))
    max_length = int(USER_CONFIG.get("max_length", MAX_LENGTH))

    print(f"Generating {num_sequences} DNA sequences and squiggle signals...")

    all_signals: List[np.ndarray] = []
    all_sequences: List[str] = []
    all_dwell_times: List[List[int]] = []

    for idx in range(num_sequences):
        # Generate random DNA sequence
        sequence_length = np.random.randint(min_length, max_length + 1)
        dna_sequence = generate_random_dna_sequence(sequence_length)

        # Generate squiggle signal
        signal, dwell_times = generate_squiggle(dna_sequence)

        # Standardize signal
        standardized_signal = standardize_signal(signal)

        all_signals.append(standardized_signal)
        all_sequences.append(dna_sequence)
        all_dwell_times.append(dwell_times)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{num_sequences} sequences")

    print(f"\nDataset generation complete!")
    print(f"Total sequences: {len(all_sequences)}")

    signal_lengths = [len(s) for s in all_signals]
    print(
        f"Signal lengths range: {min(signal_lengths)} - {max(signal_lengths)} samples"
    )

    # Create output directory
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Save dataset to disk
    save_dataset(all_signals, all_sequences, all_dwell_times, output_dir)


if __name__ == "__main__":
    main()
