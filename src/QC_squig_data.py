"""
SquigDecode: Quality Control Visualization for Squiggle Signals

This module provides functionality to load and visualize simulated squiggle signals
with base position annotations, allowing inspection of signal quality and transitions.
"""

try:
    import numpy as np
except ImportError as e:
    raise ImportError("numpy is required for QC scripts; install it with 'pip install numpy'.") from e

import pickle

try:
    import torch
except ImportError as e:
    raise ImportError("torch is required to load signal tensors; install with 'pip install torch'.") from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required for plotting; install with 'pip install matplotlib'.") from e
from pathlib import Path
from typing import List, Tuple, Optional

from config import BASE_PICOAMPERE_MAP



def load_data(
    data_dir: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[str], List[List[int]]]: 
    """
    Load signals, sequences, and dwell times from saved data files.
    
    Args:
        data_dir: Path to data directory. If None, uses ../data
            relative to script location.
    
    Returns:
        Tuple containing:
        - List of signal arrays
        - List of DNA sequences
        - List of dwell time lists
    
    Raises:
        FileNotFoundError: If required data files are not found
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    # Load signals
    signals_path = data_dir / "signals.pt"
    if not signals_path.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_path}")
    signals = torch.load(signals_path, weights_only=False)
    
    # Load sequences
    sequences_path = data_dir / "sequences.pkl"
    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
    with open(sequences_path, "rb") as f:
        sequences = pickle.load(f)
    
    # Load dwell times
    dwell_times_path = data_dir / "dwell_times.pkl"
    if not dwell_times_path.exists():
        raise FileNotFoundError(f"Dwell times file not found: {dwell_times_path}")
    with open(dwell_times_path, "rb") as f:
        dwell_times = pickle.load(f)
    
    return signals, sequences, dwell_times


def get_base_positions(
    dwell_times: List[int],
) -> Tuple[List[int], List[str], List[float]]:
    """
    Calculate sample index positions where each base ends in the signal.
    
    Args:
        dwell_times: List of dwell times for each base
    
    Returns:
        Tuple containing:
        - List of sample indices where each base ends
        - List of base letters (from sequence reconstruction)
        - List of expected pA levels for each base
    """
    # Calculate cumulative positions (where each base ends)
    base_end_positions = np.cumsum(dwell_times).tolist()
    
    return base_end_positions


def plot_signal_with_bases(
    signal: np.ndarray,
    sequence: str,
    dwell_times: List[int],
    title: str = "Squiggle Signal with Base Positions",
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Plot a standardized squiggle signal with base position annotations.
    
    Creates a visualization showing:
    - The standardized signal waveform
    - Vertical lines at base boundaries
    - Text labels for each DNA base
    - Shaded regions for expected pA levels
    
    Args:
        signal: Standardized signal array
        sequence: DNA sequence string
        dwell_times: Dwell times for each base
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the signal
    sample_indices = np.arange(len(signal))
    ax.plot(
        sample_indices,
        signal,
        linewidth=1.5,
        color='darkblue',
        alpha=0.8,
        label='Signal',
    )
    
    # Calculate base boundary positions
    base_end_positions = np.cumsum(dwell_times).tolist()
    base_start_positions = [0] + base_end_positions[:-1]
    
    # Color map for bases
    base_colors = {
        'A': '#FF6B6B',  # Red
        'C': '#4ECDC4',  # Teal
        'G': '#45B7D1',  # Blue
        'T': '#FFA07A',  # Light Salmon
    }
    
    # Add vertical lines and labels for each base
    for idx, (base, start_pos, end_pos) in enumerate(
        zip(sequence, base_start_positions, base_end_positions)
    ):
        # Vertical line at base boundary
        ax.axvline(x=end_pos, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        
        # Base label at mid-position
        mid_pos = (start_pos + end_pos) / 2
        # Get y position slightly above the signal for label placement
        y_max = np.max(signal[max(0, int(start_pos)):min(len(signal), int(end_pos))])
        ax.text(
            mid_pos,
            y_max + 0.3,
            base,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color=base_colors.get(base, 'black'),
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                alpha=0.7,
                edgecolor='none',
            ),
        )
        
        # Shade background by base
        ax.axvspan(
            start_pos,
            end_pos,
            alpha=0.08,
            color=base_colors.get(base, 'gray'),
        )
    
    # Labels and formatting
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Signal (pA)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add sequence info at top
    sequence_str = ''.join(sequence)
    ax.text(
        0.5,
        1.08,
        f"Sequence ({len(sequence)} bases): {sequence_str}",
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
    )
    
    plt.tight_layout()
    return fig


def plot_dwell_time_distribution(ax: plt.Axes, dwell_times: List[List[int]]) -> None:
    """Populate an axes with dwell time histogram."""
    all_dwell_times = []
    for dwell_list in dwell_times:
        all_dwell_times.extend(dwell_list)
    
    ax.hist(all_dwell_times, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_dwell_times), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_dwell_times):.1f}')
    ax.set_xlabel('Dwell Time (samples)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Dwell Times', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_base_composition(ax: plt.Axes, sequences: List[str]) -> None:
    """Populate an axes with base composition bar chart."""
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for sequence in sequences:
        for base in sequence:
            if base in base_counts:
                base_counts[base] += 1
    
    bases = list(base_counts.keys())
    counts = list(base_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    ax.bar(bases, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Base', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Base Composition in Dataset', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def plot_length_correlation(
    ax: plt.Axes,
    sequences: List[str],
    dwell_times: List[List[int]],
) -> None:
    """Populate an axes with signal length vs base count scatter."""
    signal_lengths = [sum(dw) for dw in dwell_times]
    base_lengths = [len(seq) for seq in sequences]
    ax.scatter(base_lengths, signal_lengths, alpha=0.6, color='purple', edgecolors='w')
    ax.set_xlabel('Number of Bases', fontsize=11, fontweight='bold')
    ax.set_ylabel('Signal Length (samples)', fontsize=11, fontweight='bold')
    ax.set_title('Signal Length vs. Base Count', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_squiggle_length_hist(ax: plt.Axes, dwell_times: List[List[int]]) -> None:
    """Populate an axes with histogram of total squiggle lengths."""
    signal_lengths = [sum(dw) for dw in dwell_times]
    ax.hist(signal_lengths, bins=60, color='olive', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(signal_lengths), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(signal_lengths):.1f}')
    ax.set_xlabel('Signal Length (samples)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Squiggle Lengths', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_signal_statistics(
    sequences: List[str],
    dwell_times: List[List[int]],
) -> plt.Figure:
    """
    Aggregate statistics figure composed of individual plot functions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    plot_dwell_time_distribution(axes[0], dwell_times)
    plot_base_composition(axes[1], sequences)
    plot_length_correlation(axes[2], sequences, dwell_times)
    plot_squiggle_length_hist(axes[3], dwell_times)

    plt.tight_layout()
    return fig


def main() -> None:
    """
    Main execution: Load and visualize the first sequence's squiggle signal.
    
    Creates two plots:
    1. The first sequence's signal with base annotations and boundaries
    2. Statistical summaries of dwell times, base composition, and signal lengths across the dataset
    """
    print("Loading squiggle data...")
    try:
        signals, sequences, dwell_times_list = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data_simulator.py first to generate the dataset.")
        return
    
    if len(signals) == 0:
        print("No signals found in dataset.")
        return
    
    print(f"Dataset loaded: {len(signals)} sequences")
    
    # Get first sequence and signal
    first_signal = signals[0]
    first_sequence = sequences[0]
    first_dwell_times = dwell_times_list[0]
    
    print(f"\nFirst sequence: {first_sequence}")
    print(f"Sequence length: {len(first_sequence)} bases")
    print(f"Signal length: {len(first_signal)} samples")
    print(f"Total dwell time: {sum(first_dwell_times)} samples")
    
    # Plot the first signal with base annotations
    fig1 = plot_signal_with_bases(
        first_signal,
        first_sequence,
        first_dwell_times,
        title="Squiggle Signal QC - First Sequence",
    )
    
    # Plot dataset statistics
    fig2 = plot_signal_statistics(sequences, dwell_times_list)
    
    # Save figures
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    fig1.savefig(output_dir / "signal_qc_plot.png", dpi=150, bbox_inches='tight')
    print(f"\nSignal QC plot saved to {output_dir / 'signal_qc_plot.png'}")
    
    fig2.savefig(output_dir / "dataset_statistics.png", dpi=150, bbox_inches='tight')
    print(f"Dataset statistics plot saved to {output_dir / 'dataset_statistics.png'}")
    
    plt.show()


if __name__ == "__main__":
    main()
