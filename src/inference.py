"""
SquigDecode: Inference Script for SquigNet Basecaller Evaluation

This module implements inference, decoding, and evaluation of the trained
SquigNet model on test signals.
"""

from difflib import SequenceMatcher
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from architecture import SquigNet
from config import INT_TO_BASE, MODEL_DIR, MODEL_FILE, NOISE_STD, TEST_PATH, USER_CONFIG
from data_simulator import (
    generate_random_dna_sequence,
    generate_squiggle,
    standardize_signal,
)


def greedy_decode(
    logits: torch.Tensor,
    blank_idx: int = 0,
) -> str:
    """
    Greedy decoder: convert logits to DNA sequence.

    Process:
    1. Take argmax to get predicted class indices
    2. Remove consecutive duplicates (CTC collapse)
    3. Remove blank tokens (idx 0)
    4. Convert to base characters

    Args:
        logits: Model output tensor of shape (batch, length, num_classes)
        blank_idx: Index of blank/padding token (default 0 for CTC)

    Returns:
        str: Decoded DNA sequence
    """
    # Get batch size and sequence length
    batch_size = logits.shape[0]

    decoded_sequences = []

    for b in range(batch_size):
        # Get predictions for this sample: (length,)
        predictions = torch.argmax(logits[b], dim=-1).cpu().numpy()

        # Remove consecutive duplicates (CTC collapse)
        collapsed = []
        last_idx = None
        for idx in predictions:
            if idx != last_idx:
                collapsed.append(idx)
                last_idx = idx

        # Remove blank tokens
        decoded = [INT_TO_BASE[int(idx)] for idx in collapsed if idx != blank_idx]

        # Join to form DNA string
        sequence = "".join(decoded)
        decoded_sequences.append(sequence)

    # Return first sequence (single sample inference)
    return decoded_sequences[0] if decoded_sequences else ""


def edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        int: Minimum edit distance
    """
    matcher = SequenceMatcher(None, s1, s2)
    matching_blocks = matcher.get_matching_blocks()
    return max(len(s1), len(s2)) - sum(block.size for block in matching_blocks)


def calculate_accuracy(predicted: str, target: str) -> float:
    """
    Calculate accuracy using normalized edit distance.

    Accuracy = 1 - (EditDistance / len(target))

    Args:
        predicted: Predicted DNA sequence
        target: Ground truth DNA sequence

    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    if len(target) == 0:
        return 1.0 if len(predicted) == 0 else 0.0

    dist = edit_distance(predicted, target)
    accuracy = 1.0 - (dist / len(target))
    return max(0.0, accuracy)  # Ensure non-negative


def load_dataset(
    data_dir: Path,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load signals and sequences from a dataset directory.

    Args:
        data_dir: Path to directory containing ``signals.pt`` and
            ``sequences.pkl``.

    Returns:
        Tuple of lists containing signals and DNA sequences.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    signals_path = data_dir / "signals.pt"
    sequences_path = data_dir / "sequences.pkl"

    missing: List[str] = []
    if not signals_path.exists():
        missing.append("signals.pt")
    if not sequences_path.exists():
        missing.append("sequences.pkl")

    if missing:
        raise FileNotFoundError(
            f"Test dataset incomplete: missing {', '.join(missing)} in {data_dir}. "
            "Both signals.pt and sequences.pkl are required. "
            "Run `python src/data_simulator.py` to generate the dataset."
        )

    signals: List[np.ndarray] = torch.load(signals_path, weights_only=False)

    with open(sequences_path, "rb") as f:
        sequences: List[str] = pickle.load(f)

    return signals, sequences


def load_model(
    model_path: Path = Path(USER_CONFIG.get("model_path", "models/squig_model.pt")),
    device: torch.device = None,
) -> SquigNet:
    """
    Load trained SquigNet model from checkpoint.

    Args:
        model_path: Path to model weights file
        device: torch.device to load model on. Auto-select if None.

    Returns:
        SquigNet: Model in eval mode

    Raises:
        FileNotFoundError: If model file does not exist
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = SquigNet().to(device)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def generate_test_sample(
    adv_noise: float = NOISE_STD*0.1,
    scale: float = 1.0,
    shift: float = 0.0,
) -> Tuple[np.ndarray, str]:
    """
    Generate a single test sample with adversarial signal conditions.

    This function tests model robustness by introducing higher noise
    and signal scaling/shifting that may not be present in training.

    Args:
        adv_noise: Standard deviation of additional Gaussian noise.
                   Default 3.0.
        scale: Scaling factor for signal amplitude. Default 1.0 (no change).
        shift: DC offset added to signal. Default 0.0 (no change).

    Returns:
        Tuple containing:
        - signal: Adversarial signal array with noise and scaling
        - sequence: Ground truth DNA sequence
    """
    # Generate random DNA sequence (50-100 bases)
    sequence_length = np.random.randint(50, 101)
    print(f"Generating test sample with sequence length: {sequence_length} bases")
    sequence = generate_random_dna_sequence(sequence_length)

    # Generate squiggle signal
    signal, _ = generate_squiggle(sequence)

    # Standardize signal
    standardized_signal = standardize_signal(signal)

    # Add additional Gaussian noise for adversarial testing
    additional_noise = np.random.normal(
        0,
        adv_noise,
        len(standardized_signal),
    )
    adversarial_signal = standardized_signal + additional_noise

    # Apply scale and shift transformations
    adversarial_signal = adversarial_signal * scale + shift

    return adversarial_signal, sequence


def validate_model_on_dataset(
    model: SquigNet,
    data_dir: Path,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
) -> float:
    """
    Validate a trained model on a saved test dataset.

    Args:
        model: Trained SquigNet model in evaluation mode.
        data_dir: Directory containing test ``signals.pt`` and
            ``sequences.pkl``.
        device: torch.device for computation. Auto-select if None.
        max_samples: Optional cap on the number of test samples to evaluate.

    Returns:
        float: Mean accuracy over the evaluated samples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure deterministic inference behavior (disable dropout, use running
    # batch-norm statistics) while preserving the caller's original mode.
    was_training = model.training
    model.eval()

    signals, sequences = load_dataset(data_dir)

    if max_samples is not None:
        signals = signals[:max_samples]
        sequences = sequences[:max_samples]

    accuracies: List[float] = []

    try:
        for idx, (signal, target_sequence) in enumerate(
            zip(signals, sequences),
            start=1,
        ):
            logits = run_inference(model, signal, device=device)
            predicted_sequence = greedy_decode(logits)
            accuracy = calculate_accuracy(predicted_sequence, target_sequence)
            accuracies.append(accuracy)

            print(
                f"Sample {idx}: length={len(target_sequence)} bases, "
                f"accuracy={accuracy:.2%}",
            )
    finally:
        # Restore original training/eval mode.
        if was_training:
            model.train()

    if not accuracies:
        return 0.0

    mean_accuracy = float(np.mean(accuracies))
    print(
        f"\nValidation complete on {len(accuracies)} samples. "
        f"Mean accuracy: {mean_accuracy:.2%}",
    )
    return mean_accuracy


def run_inference(
    model: SquigNet,
    signal: np.ndarray,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Run inference on a single signal.

    Args:
        model: SquigNet model (should be in eval mode)
        signal: Standardized signal array of shape (signal_length,)
        device: torch.device for computation

    Returns:
        torch.Tensor: Model output logits of shape (1, downsampled_length, 5)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert signal to tensor and add batch/channel dimensions
    signal_tensor = (
        torch.tensor(
            signal,
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )  # (1, 1, signal_length)

    # Forward pass
    with torch.no_grad():
        logits = model(signal_tensor)

    return logits


def plot_inference_results(
    signal: np.ndarray,
    predicted: str,
    target: str,
    accuracy: float,
    output_path: Path = Path("models/inference_result.png"),
) -> None:
    """
    Plot signal and display predicted vs. target sequences.

    Args:
        signal: Standardized signal array
        predicted: Predicted DNA sequence
        target: Ground truth DNA sequence
        accuracy: Accuracy score
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot signal
    sample_indices = np.arange(len(signal))
    ax.plot(
        sample_indices,
        signal,
        linewidth=1.5,
        color="darkblue",
        alpha=0.8,
        label="Signal",
    )

    ax.set_xlabel("Sample Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Standardized Signal (pA)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Inference Result: Signal with Predictions", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=10)

    # Add sequence info as text box
    success = accuracy >= 0.8
    color = "#90EE90" if success else "#FFB6C6"
    status = "PASS ✓" if success else "FAIL ✗"

    info_text = (
        f"Target:    {target}\n"
        f"Predicted: {predicted}\n"
        f"Accuracy:  {accuracy:.2%}\n"
        f"Status:    {status}"
    )

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        ),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Inference plot saved: {output_path}")
    plt.show()


def main(
    model_path: Path = Path(MODEL_DIR) / Path(MODEL_FILE),
    device: Optional[torch.device] = None,
) -> None:
    """
    Main inference pipeline.

    Args:
        model_path: Path to trained model weights
        device: torch.device for computation
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("SquigNet Inference & Evaluation")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    try:
        model = load_model(model_path, device)
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python3 src/train.py")
        return

    # Prefer dataset-based validation on the held-out test split.
    test_data_dir_str = USER_CONFIG.get("test_data_dir", TEST_PATH)
    test_data_dir = Path(test_data_dir_str)

    if (test_data_dir / "signals.pt").exists() and (
        test_data_dir / "sequences.pkl"
    ).exists():
        print(f"\nValidating on test dataset at: {test_data_dir}")
        validate_model_on_dataset(model, test_data_dir, device=device)
        print("\n✓ Dataset-based validation complete!")
        #return

    # Fallback: single adversarial sample if test dataset is unavailable.
    signals_pt = test_data_dir / "signals.pt"
    sequences_pkl = test_data_dir / "sequences.pkl"
    has_partial = signals_pt.exists() != sequences_pkl.exists()
    if test_data_dir.exists() and has_partial:
        print(
            f"\nTest dataset at {test_data_dir} is incomplete "
            "(both signals.pt and sequences.pkl required). "
            "Run `python src/data_simulator.py` to generate."
        )
    print(
        "\nFalling back to adversarial single-sample test...",
    )
    train_noise = USER_CONFIG.get("noise_std", NOISE_STD)
    adv_noise = USER_CONFIG.get("adversarial_noise", NOISE_STD*0.01)
    print(
        f"  Noise level (\u03c3): {adv_noise + train_noise} "
        f"(vs. training: {train_noise})",
    )
    signal, target_sequence = generate_test_sample(
        adv_noise=adv_noise,
        scale=1.0,
        shift=0.0,
    )
    print("✓ Generated adversarial test sample")
    print(f"  Target sequence length: {len(target_sequence)} bases")
    print(f"  Signal length: {len(signal)} samples")

    logits = run_inference(model, signal, device)
    predicted_sequence = greedy_decode(logits)
    accuracy = calculate_accuracy(predicted_sequence, target_sequence)

    print("\n" + "=" * 70)
    print("Adversarial Single-Sample Result")
    print("=" * 70)
    print(f"\nTarget DNA:    {target_sequence}")
    print(f"Predicted DNA: {predicted_sequence}")
    print(f"\nAccuracy: {accuracy:.2%}")

    dist = edit_distance(predicted_sequence, target_sequence)
    print(f"\nEdit Distance: {dist} operations")
    print(f"Sequence Length: {len(target_sequence)} bases")

    output_dir = Path(MODEL_DIR)
    output_dir.mkdir(exist_ok=True)
    plot_inference_results(
        signal,
        predicted_sequence,
        target_sequence,
        accuracy,
        output_path=output_dir / "inference_result.png",
    )

    print("\n✓ Inference complete!")


if __name__ == "__main__":

    model_dir=Path(USER_CONFIG.get("model_dir", MODEL_DIR))
    model_file=Path(USER_CONFIG.get("model_file", MODEL_FILE))

    main(
        model_path=model_dir / model_file,
    )
