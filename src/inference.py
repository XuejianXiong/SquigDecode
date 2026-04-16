"""
SquigDecode: Professional Inference & Evaluation Engine.

This module provides the `SquigBasecaller` class for translating raw Nanopore 
electrical signals into DNA sequences. It implements CTC greedy decoding, 
Levenshtein-based accuracy metrics, and visualization utilities.

The pipeline is optimized for MAD-standardized signals and supports both 
single-signal inference and bulk dataset validation.

Example:
    $ uv run src/inference.py
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from architecture import SquigNet
from config import (
    INT_TO_BASE, 
    MODEL_DIR, 
    MODEL_FILE,
    TEST_PATH, 
    USER_CONFIG
)

# Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SquigInference")

class SquigBasecaller:
    """
    High-level API for performing inference with trained SquigNet models.
    
    Attributes:
        device (torch.device): Hardware target (CPU/CUDA).
        model (nn.Module): The loaded SquigNet architecture.
        blank_idx (int): The CTC blank token index (default: 0).
    """
    def __init__(self, model_path: Path, device: torch.device):
        """
        Initializes the basecaller and loads weights.
        
        Args:
            model_path: Path to the .pt or .pth weight file.
            device: Torch device to host the model.
        """
        self.device = device
        self.blank_idx = 0
        self.model = self._load_model(model_path)
        logger.info(f"Basecaller initialized on {device}")

    def _load_model(self, path: Path) -> SquigNet:
        """Loads and verifies model state dictionary."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint missing: {path}")
        
        model = SquigNet().to(self.device)
        # Using weights_only=True for secure loading of untrusted checkpoints
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def basecall(self, signal: np.ndarray) -> str:
        """
        Translates a single raw signal into a DNA string.
        
        Args:
            signal: 1D NumPy array of standardized electrical current.
            
        Returns:
            The decoded DNA sequence string.
        """
        # (1, C, L) tensor format
        tensor = torch.as_tensor(signal, dtype=torch.float32).view(1, 1, -1).to(self.device)
        
        with torch.no_grad():
            log_probs = self.model(tensor)
            
        return self.decode_ctc(log_probs)[0]

    def decode_ctc(self, log_probs: torch.Tensor) -> List[str]:
        """
        Performs greedy CTC decoding (best-path) on model output.
        
        Args:
            log_probs: Tensor of shape (Batch, Time, Classes).
            
        Returns:
            List of decoded strings for the batch.
        """
        # Get most likely class indices: (Batch, Time)
        argmax_indices = torch.argmax(log_probs, dim=-1)
        
        decoded_batch = []
        for batch_idx in range(argmax_indices.size(0)):
            seq_indices = argmax_indices[batch_idx].cpu().numpy()
            
            collapsed = []
            prev_idx = -1
            for idx in seq_indices:
                # 1. Collapse repeated tokens
                # 2. Filter out blank tokens (self.blank_idx)
                if idx != prev_idx:
                    if idx != self.blank_idx:
                        collapsed.append(idx)
                prev_idx = idx
            
            dna_seq = "".join([INT_TO_BASE.get(int(i), 'N') for i in collapsed])
            decoded_batch.append(dna_seq)
            
        return decoded_batch

def calculate_levenshtein_accuracy(predicted: str, target: str) -> float:
    """
    Calculates sequence identity based on normalized Edit Distance.
    
    Formula: 1 - (LevenshteinDistance / LengthOfTarget)
    """
    if not target:
        return 0.0
    
    # Use a dynamic programming approach for edit distance
    rows = len(predicted) + 1
    cols = len(target) + 1
    dist = np.zeros((rows, cols), dtype=int)

    for i in range(1, rows): dist[i, 0] = i
    for i in range(1, cols): dist[0, i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if predicted[row-1] == target[col-1] else 1
            dist[row, col] = min(dist[row-1, col] + 1,      # deletion
                                 dist[row, col-1] + 1,      # insertion
                                 dist[row-1, col-1] + cost) # substitution

    edit_dist = dist[rows-1, cols-1]
    return max(0.0, 1.0 - (edit_dist / len(target)))

def plot_diagnostic(signal: np.ndarray, pred: str, target: str, acc: float, path: Path):
    """Generates a professional diagnostic plot of the basecalling event."""
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(signal, color='#2c3e50', linewidth=0.8, alpha=0.8)
    
    # Formatting
    plt.title(f"SquigNet Inference Diagnostic | Accuracy: {acc:.2%}", fontweight='bold')
    plt.xlabel("Time (Samples)")
    plt.ylabel("Standardized Current (z-score)")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Comparison box
    info_text = (f"REF: {target[:60]}...\n"
                 f"PRED: {pred[:60]}...")
    plt.annotate(info_text, xy=(0.02, 0.05), xycoords='axes fraction',
                 fontsize=10, family='monospace', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    """Execution entry point for test set evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).parent.parent.resolve()
    
    # Configuration Pathing
    model_path = root / Path(USER_CONFIG.get("model_dir", MODEL_DIR)) / Path(USER_CONFIG.get("model_file", MODEL_FILE))
    test_dir = root / Path(USER_CONFIG.get("test_data_dir", TEST_PATH))
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    try:
        basecaller = SquigBasecaller(model_path, device)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    # Load data
    try:
        signals = torch.load(test_dir / "signals.pt", weights_only=False)
        with open(test_dir / "sequences.pkl", "rb") as f:
            targets = pickle.load(f)
    except FileNotFoundError:
        logger.error("Test data not found. Ensure the simulator has run.")
        return

    # Evaluation Loop
    logger.info(f"Starting evaluation on {len(signals)} samples...")
    accuracies = []
    
    # Process samples (limit for quick validation)
    eval_limit = min(len(signals), 100)
    
    for i in tqdm(range(eval_limit), desc="Basecalling"):
        pred_seq = basecaller.basecall(signals[i])
        score = calculate_levenshtein_accuracy(pred_seq, targets[i])
        accuracies.append(score)

    mean_acc = np.mean(accuracies)
    logger.info("-" * 40)
    logger.info(f"FINAL MEAN ACCURACY: {mean_acc:.2%}")
    logger.info("-" * 40)

    # Save diagnostic plot for the last sample
    plot_diagnostic(signals[eval_limit-1], pred_seq, targets[eval_limit-1], score, results_dir / "last_inference.png")

if __name__ == "__main__":
    main()