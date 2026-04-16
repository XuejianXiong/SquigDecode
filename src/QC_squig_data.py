"""
SquigDecode: Quality Control and Statistical Profiling for Squiggle Signals.

This module provides a comprehensive suite of visualization tools for inspecting 
simulated nanopore signals. It facilitates:
1.  Individual signal inspection with base-calling ground-truth overlays.
2.  Dataset-wide statistical profiling (Dwell times, Composition, Correlations).
3.  Validation of signal standardization and electronic lag modeling.

The plots generated here are critical for verifying that the 'data_simulator' 
is producing biophysically plausible inputs for the neural network.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import TRAIN_PATH, USER_CONFIG

# Configure Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalVisualizer:
    """
    Orchestrates the generation of QC plots for Nanopore signals.
    
    Attributes:
        base_colors (Dict[str, str]): Standardized color palette for DNA bases.
    """

    def __init__(self):
        self.base_colors = {
            "A": "#FF6B6B",  # Vibrant Red
            "C": "#4ECDC4",  # Teal
            "G": "#45B7D1",  # Sky Blue
            "T": "#FFA07A",  # Light Salmon
            "Unknown": "#7F8C8D" # Gray
        }

    def plot_single_read(
        self,
        signal: np.ndarray,
        sequence: str,
        dwell_times: List[int],
        title: str = "Squiggle Signal Inspection",
        figsize: Tuple[int, int] = (16, 6),
    ) -> plt.Figure:
        """
        Generates a high-fidelity plot of a signal with base annotations.

        Args:
            signal: The standardized 1D signal array.
            sequence: The corresponding DNA sequence string.
            dwell_times: Number of samples per base.
            title: Title for the figure.
            figsize: Dimensions of the resulting plot.

        Returns:
            plt.Figure: The Matplotlib figure containing the annotated squiggle.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot raw waveform
        ax.plot(signal, linewidth=1.2, color="#2C3E50", alpha=0.9, label="Standardized Signal")

        # Calculate boundaries
        base_end_positions = np.cumsum(dwell_times)
        base_start_positions = np.insert(base_end_positions[:-1], 0, 0)

        for base, start, end in zip(sequence, base_start_positions, base_end_positions):
            color = self.base_colors.get(base, self.base_colors["Unknown"])
            
            # Add vertical boundary and shaded region
            ax.axvline(x=end, color="#BDC3C7", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axvspan(start, end, alpha=0.1, color=color)

            # Label placement: find local signal peak for visibility
            local_segment = signal[int(start):int(end)]
            y_pos = np.max(local_segment) + 0.5 if len(local_segment) > 0 else 1.0
            
            ax.text(
                (start + end) / 2, y_pos, base,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
            )

        ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
        ax.set_xlabel("Sample Index (Time)", fontsize=10)
        ax.set_ylabel("Standardized Current (z-score)", fontsize=10)
        ax.grid(True, axis='y', alpha=0.2)
        ax.legend(frameon=False)
        
        plt.tight_layout()
        return fig

    def plot_dataset_stats(
        self,
        sequences: List[str],
        dwell_times_list: List[List[int]]
    ) -> plt.Figure:
        """
        Aggregates dataset metrics into a 2x2 diagnostic dashboard.

        Args:
            sequences: List of all simulated DNA sequences.
            dwell_times_list: Nested list of dwell times for all sequences.

        Returns:
            plt.Figure: A dashboard showing Dwell distribution, Composition, 
                Length Correlation, and Signal Density.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()

        # 1. Dwell Time Distribution
        all_dwells = [d for sublist in dwell_times_list for d in sublist]
        axes[0].hist(all_dwells, bins=40, color="#34495E", alpha=0.7, rwidth=0.85)
        axes[0].set_title("Distribution of Dwell Times", fontweight='bold')
        axes[0].axvline(np.mean(all_dwells), color='red', linestyle='--', label='Mean')

        # 2. Base Composition
        counts = {b: "".join(sequences).count(b) for b in "ACGT"}
        axes[1].bar(counts.keys(), counts.values(), color=[self.base_colors[b] for b in "ACGT"])
        axes[1].set_title("Base Composition", fontweight='bold')

        # 3. Correlation: Base Count vs Signal Length
        seq_lens = [len(s) for s in sequences]
        sig_lens = [sum(d) for d in dwell_times_list]
        axes[2].scatter(seq_lens, sig_lens, alpha=0.5, s=20, color="#8E44AD")
        axes[2].set_title("Base Count vs. Signal Duration", fontweight='bold')

        # 4. Total Signal Length Histogram
        axes[3].hist(sig_lens, bins=40, color="#27AE60", alpha=0.7)
        axes[3].set_title("Total Squiggle Lengths", fontweight='bold')

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        return fig


def load_dataset(data_dir: Optional[Path] = None) -> Tuple[List[np.ndarray], List[str], List[List[int]]]:
    """
    Utility to load generated tensors and metadata.

    Args:
        data_dir: Directory containing .pt and .pkl files.

    Returns:
        Standardized tuple of (signals, sequences, dwell_times).

    Raises:
        FileNotFoundError: If the data pipeline hasn't been run.
    """
    if data_dir is None:
        root = Path(__file__).parent.parent
        data_dir = root / Path(USER_CONFIG.get("data_dir", TRAIN_PATH))

    try:
        signals = torch.load(data_dir / "signals.pt", weights_only=False)
        with open(data_dir / "sequences.pkl", "rb") as f:
            sequences = pickle.load(f)
        with open(data_dir / "dwell_times.pkl", "rb") as f:
            dwell_times = pickle.load(f)
        return signals, sequences, dwell_times
    except FileNotFoundError as e:
        logger.error("Dataset not found at %s. Ensure data_simulator.py was run.", data_dir)
        raise e


def main():
    """Execution entry point for QC generation."""
    logger.info("Initializing SquigDecode QC Pipeline...")
    
    try:
        signals, sequences, dwells = load_dataset()
    except FileNotFoundError:
        return

    viz = SignalVisualizer()
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Inspect first read
    logger.info("Visualizing read 0: %s...", sequences[0][:20])
    fig1 = viz.plot_single_read(signals[0], sequences[0], dwells[0])
    fig1.savefig(results_dir / "signal_inspection.png", dpi=200)

    # Inspect dataset
    logger.info("Generating dataset-wide diagnostics...")
    fig2 = viz.plot_dataset_stats(sequences, dwells)
    fig2.savefig(results_dir / "dataset_stats.png", dpi=200)

    logger.info("QC Artifacts saved to %s", results_dir)
    plt.show()


if __name__ == "__main__":
    main()