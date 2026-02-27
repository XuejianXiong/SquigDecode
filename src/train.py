"""
SquigDecode: Training Script for SquigNet Basecaller Model

This module implements the full training pipeline for SquigNet, including
data loading, batch processing, CTC loss computation, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from architecture import SquigNet


# DNA-to-integer mapping for CTC loss
# 0 is reserved for CTC 'Blank' token
BASE_TO_INT = {
    'A': 1,
    'C': 2,
    'G': 3,
    'T': 4,
}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}


class SquigDataset(Dataset):
    """
    PyTorch Dataset for squiggle signals and DNA sequences.

    Loads pre-generated signals and corresponding DNA sequences,
    encodes bases to integers for CTC loss, and provides indexed access.
    """

    def __init__(
        self,
        signals: List[np.ndarray],
        sequences: List[str],
    ) -> None:
        """
        Initialize the SquigDataset.

        Args:
            signals: List of numpy arrays (squiggle signals)
            sequences: List of DNA sequence strings
        """
        if len(signals) != len(sequences):
            raise ValueError(
                f"Number of signals ({len(signals)}) must match "
                f"number of sequences ({len(sequences)})"
            )

        self.signals = signals
        self.sequences = sequences

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
            - signal: Tensor of shape (1, signal_length)
            - target: Tensor of base indices, shape (sequence_length,)
        """
        # Load signal and convert to tensor
        signal = torch.tensor(
            self.signals[idx],
            dtype=torch.float32,
        ).unsqueeze(0)  # Add channel dimension: (signal_length,) -> (1, signal_length)

        # Encode DNA sequence to integers
        sequence = self.sequences[idx]
        target = torch.tensor(
            [BASE_TO_INT[base] for base in sequence],
            dtype=torch.long,
        )

        return signal, target


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching variable-length sequences.

    Pads signals and targets to the maximum length in the batch,
    and returns the original lengths for CTC loss computation.

    Args:
        batch: List of (signal, target) tuples from the dataset

    Returns:
        Tuple containing:
        - signals: Padded signals of shape (batch, 1, max_signal_length)
        - targets: Padded targets of shape (batch, max_target_length)
        - input_lengths: Lengths of signals after CNN downsampling
                        (original_length // 4)
        - target_lengths: Actual number of bases in each sequence
    """
    signals, targets = zip(*batch)

    # Find maximum signal length
    max_signal_len = max(s.shape[1] for s in signals)

    # Pad signals: (batch, 1, max_signal_length)
    batch_signals = []
    for s in signals:
        # s is (1, signal_length)
        padded = torch.zeros(1, max_signal_len)
        padded[:, :s.shape[1]] = s
        batch_signals.append(padded)

    signals_padded = torch.stack(batch_signals)  # (batch, 1, max_signal_length)

    # Pad targets: (batch, max_target_length)
    targets_padded = pad_sequence(
        targets,
        batch_first=True,
        padding_value=0,
    )

    # Calculate input_lengths (after CNN downsampling by factor of 4)
    input_lengths = torch.tensor(
        [s.shape[1] // 4 for s in signals],
        dtype=torch.long,
    )

    # Calculate target_lengths (actual number of bases per sequence)
    target_lengths = torch.tensor(
        [len(t) for t in targets],
        dtype=torch.long,
    )

    return signals_padded, targets_padded, input_lengths, target_lengths


def load_training_data(
    data_dir: Path = Path("data"),
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load pre-generated training signals and sequences.

    Args:
        data_dir: Path to directory containing signals.pt and sequences.pkl

    Returns:
        Tuple containing:
        - signals: List of numpy arrays
        - sequences: List of DNA sequence strings

    Raises:
        FileNotFoundError: If data files do not exist
    """
    signals_path = data_dir / "signals.pt"
    sequences_path = data_dir / "sequences.pkl"

    if not signals_path.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_path}")
    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")

    # Load signals
    signals = torch.load(signals_path, weights_only=False)

    # Load sequences
    with open(sequences_path, "rb") as f:
        sequences = pickle.load(f)

    return signals, sequences


def create_checkpoint(
    model: SquigNet,
    optimizer: optim.Adam,
    epoch: int,
    losses: List[float],
    checkpoint_path: Path,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: SquigNet model instance
        optimizer: Adam optimizer instance
        epoch: Current epoch number
        losses: List of loss values
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[int, Dict, Dict, List[float]]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple containing:
        - epoch: Last trained epoch
        - model_state: Model state dictionary
        - optimizer_state: Optimizer state dictionary
        - losses: Previous loss history
    """
    checkpoint = torch.load(checkpoint_path)
    return (
        checkpoint['epoch'],
        checkpoint['model_state_dict'],
        checkpoint['optimizer_state_dict'],
        checkpoint['losses'],
    )


def train_epoch(
    model: SquigNet,
    dataloader: DataLoader,
    optimizer: optim.Adam,
    ctc_loss: nn.CTCLoss,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: SquigNet model instance
        dataloader: Training DataLoader
        optimizer: Adam optimizer
        ctc_loss: CTC loss function
        device: torch.device (cpu or cuda)

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for signals, targets, input_lengths, target_lengths in progress_bar:
        # Move to device
        signals = signals.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(signals)  # (batch, length, 5)

        # Transpose for CTC loss: (length, batch, num_classes)
        outputs = outputs.transpose(0, 1)

        # Compute CTC loss
        loss = ctc_loss(
            outputs,
            targets,
            input_lengths,
            target_lengths,
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def train(
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_dir: Path = Path("checkpoints"),
    model_dir: Path = Path("models"),
    data_dir: Path = Path("data"),
    device: Optional[torch.device] = None,
    resume_checkpoint: Optional[Path] = None,
) -> None:
    """
    Full training pipeline for SquigNet.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for DataLoader
        learning_rate: Learning rate for Adam optimizer
        checkpoint_dir: Directory to save training checkpoints
        model_dir: Directory to save final model
        data_dir: Directory containing training data
        device: torch.device (cpu or cuda). Auto-select if None.
        resume_checkpoint: Path to checkpoint file to resume training from
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    checkpoint_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading training data...")
    try:
        signals, sequences = load_training_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run data_simulator.py first to generate training data.")
        return

    print(f"Loaded {len(signals)} sequences")

    # Create dataset and dataloader
    dataset = SquigDataset(signals, sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    # Initialize model
    model = SquigNet().to(device)
    print(f"\nModel parameters: {SquigNet.count_parameters():,}")

    # Loss function and optimizer
    ctc_loss = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training state
    start_epoch = 0
    losses = []

    # Resume from checkpoint if provided
    if resume_checkpoint and resume_checkpoint.exists():
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        (
            start_epoch,
            model_state,
            optimizer_state,
            losses,
        ) = load_checkpoint(resume_checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        print(f"Resumed at epoch {start_epoch}")

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(start_epoch, num_epochs):
        avg_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            ctc_loss,
            device,
        )
        losses.append(avg_loss)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - "
                f"Avg Loss: {avg_loss:.4f}"
            )

        # Save checkpoint every epoch
        checkpoint_path = (
            checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        )
        create_checkpoint(
            model,
            optimizer,
            epoch + 1,
            losses,
            checkpoint_path,
        )

    # Save final model
    model_path = model_dir / "squig_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved: {model_path}")

    # Plot loss curve
    print("Generating loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2, color='steelblue')
    plt.axhline(y=np.mean(losses), color='red', linestyle='--',
                linewidth=2, label=f'Mean Loss: {np.mean(losses):.4f}')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('CTC Loss', fontsize=12, fontweight='bold')
    plt.title('SquigNet Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    loss_plot_path = model_dir / "loss_curve.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"Loss curve saved: {loss_plot_path}")

    plt.show()

    # Print summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Best Loss: {min(losses):.4f} (Epoch {np.argmin(losses) + 1})")
    print(f"Model saved to: {model_path}")
    print(f"Loss curve saved to: {loss_plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    # Default training configuration
    train(
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        checkpoint_dir=Path("checkpoints"),
        model_dir=Path("models"),
        data_dir=Path("data"),
        resume_checkpoint=Path("checkpoints/checkpoint.pt"),
    )
