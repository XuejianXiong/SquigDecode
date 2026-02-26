"""
SquigDecode: Data Simulator for Nanopore Signal Generation

This module provides functionality to simulate realistic nanopore squiggle signals
from DNA sequences, including noise, drift, and base-specific signal characteristics.
"""

import numpy as np
import pickle
import torch
from typing import List, Tuple
from pathlib import Path

from config import (
    BASE_PICOAMPERE_MAP,
    DWELL_TIME_MEAN,
    DWELL_TIME_STD,
    MIN_DWELL_TIME,
    NOISE_STD,
    DRIFT_FACTOR,
)


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


def generate_squiggle(dna_sequence: str) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a realistic nanopore squiggle signal from a DNA sequence.
    
    Process:
    1. Convert DNA bases to pA levels using BASE_PICOAMPERE_MAP
    2. Expand each level by its random dwell time
    3. Apply 3-base sliding window filter for context
    4. Add linear drift and Gaussian noise
    5. Smooth with weighted moving average
    
    The 3-base sliding window implements:
        output_i = 0.7 * level_i + 0.2 * level_{i-1} + 0.1 * level_{i+1}
    
    For boundary bases, the missing neighbor is replaced with the current base level.
    
    Args:
        dna_sequence: String of DNA bases (e.g., "ACGTACGT")
    
    Returns:
        Tuple[np.ndarray, List[int]]: Simulated squiggle signal and list of dwell times
    
    Raises:
        ValueError: If DNA sequence contains invalid bases
    """
    # Validate DNA sequence
    valid_bases = set(BASE_PICOAMPERE_MAP.keys()) - {'blank'}
    if not all(base in valid_bases for base in dna_sequence):
        raise ValueError(f"Invalid bases in sequence. Valid bases: {valid_bases}")
    
    # Step 1 & 2: Convert DNA to pA levels and apply dwell times
    # Create a long array where each base is repeated by its dwell time
    signal_list: List[float] = []
    dwell_times: List[int] = []
    for base in dna_sequence:
        base_level = BASE_PICOAMPERE_MAP[base]
        dwell_time = get_dwell_time()
        signal_list.extend([base_level] * dwell_time)
        dwell_times.append(dwell_time)
    
    signal = np.array(signal_list, dtype=np.float32)
    
    # Step 3: Apply 3-base sliding window filter
    # This creates context-dependent signal where each position is influenced by neighbors
    filtered_signal = np.zeros_like(signal)
    
    for i in range(len(signal)):
        # Determine neighbor indices (handle boundaries)
        prev_idx = max(0, i - 1)
        next_idx = min(len(signal) - 1, i + 1)
        
        # Apply weighted combination: 0.7 * current + 0.2 * previous + 0.1 * next
        filtered_signal[i] = (
            0.7 * signal[i] +
            0.2 * signal[prev_idx] +
            0.1 * signal[next_idx]
        )
    
    signal = filtered_signal
    
    # Step 4: Add linear drift over the sequence
    # Creates gradual baseline shift simulating electrical drift
    mean_signal = np.mean(signal)
    drift = np.linspace(0, DRIFT_FACTOR * mean_signal, len(signal))
    signal = signal + drift
    
    # Step 4 (continued): Add Gaussian noise
    noise = np.random.normal(0, NOISE_STD, len(signal))
    signal = signal + noise
    
    # Step 5: Smooth with weighted moving average (3-point window)
    # Reduces abrupt transitions while preserving signal features
    window_size = 3
    smoothed_signal = np.zeros_like(signal)
    
    for i in range(len(signal)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(signal), i + window_size // 2 + 1)
        window = signal[start_idx:end_idx]
        smoothed_signal[i] = np.mean(window)
    
    return smoothed_signal, dwell_times


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
    bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(bases, length))


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
    num_sequences = 1000
    min_length = 50
    max_length = 100
    
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
    print(f"Signal lengths range: {min(signal_lengths)} - {max(signal_lengths)} samples")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
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


if __name__ == "__main__":
    main()
