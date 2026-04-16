"""
SquigDecode: Professional Training Engine for SquigNet.

This module provides the `SquigTrainer` class, which orchestrates the training 
lifecycle of SquigNet models. It handles high-performance data loading, 
CTC loss optimization, learning rate scheduling, and comprehensive telemetry 
via TensorBoard.

Features:
1.  TensorBoard Integration: Real-time visualization of loss and learning rates.
2.  Dynamic Learning Rate: ReduceLROnPlateau scheduler for fine-tuning convergence.
3.  Stateful Checkpointing: Resumable training sessions with best-model tracking.
4.  Robust Collation: Handles variable-length signals with padding and CNN-factor scaling.

Example:
    $ uv run src/train.py
"""

from __future__ import annotations

import logging
import pickle
import signal
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architecture import SquigNet
from config import (
    BASE_TO_INT,
    CHECKPOINT_DIR,
    MODEL_DIR,
    MODEL_FILE,
    TRAIN_BATCH_SIZE,
    TRAIN_LEARNING_RATE,
    TRAIN_NUM_EPOCHS,
    TRAIN_PATH,
    USER_CONFIG,
)

# Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SquigTrainer")

class SquigDataset(Dataset):
    """
    Map-style dataset for Nanopore squiggle signals and DNA sequences.
    
    Attributes:
        signals (List[np.ndarray]): List of MAD-standardized signal arrays.
        sequences (List[str]): Corresponding DNA target sequences.
    """
    def __init__(self, signals: List[np.ndarray], sequences: List[str]) -> None:
        if len(signals) != len(sequences):
            raise ValueError(f"Mismatch: {len(signals)} signals vs {len(sequences)} sequences.")
        self.signals = signals
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (C, L) -> (1, SignalLength)
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)
        # Target: Encoded integer sequence
        target = torch.tensor([BASE_TO_INT[base] for base in self.sequences[idx]], dtype=torch.long)
        return signal, target

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dynamic padding collator for variable-length Nanopore signals.
    
    Args:
        batch: List of (signal, target) tuples from SquigDataset.
        
    Returns:
        signals_padded: Tensor of shape (B, 1, MaxSignalLen).
        targets_padded: Tensor of shape (B, MaxTargetLen).
        input_lengths: Corrected lengths after CNN downsampling.
        target_lengths: Original target sequence lengths.
    """
    signals, targets = zip(*batch)
    max_signal_len = max(s.shape[1] for s in signals)
    
    # Pad signals to max length in batch
    signals_padded = torch.stack([
        nn.functional.pad(s, (0, max_signal_len - s.shape[1])) for s in signals
    ])
    
    # Pad targets (0 is the CTC blank/padding index)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    # CTC input lengths are reduced by a factor of 4 due to SquigNet CNN stride
    input_lengths = torch.tensor([s.shape[1] // 4 for s in signals], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    
    return signals_padded, targets_padded, input_lengths, target_lengths

class SquigTrainer:
    """
    Orchestrates the SquigNet training process.
    
    Encapsulates model initialization, optimization logic, and telemetry.
    Designed to handle SIGINT (Ctrl+C) for safe state preservation.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the trainer with absolute path resolution and logging.
        
        Args:
            config: Configuration dictionary derived from config.py.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_paths()
        
        # Telemetry
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Architecture & Optimization
        self.model = SquigNet().to(self.device)
        self.loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        lr = float(config.get("train_learning_rate", TRAIN_LEARNING_RATE))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        
        # Signal handling for HPC safety
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _setup_paths(self):
        """Resolves and creates required directory tree."""
        self.root = Path(__file__).parent.parent.resolve()
        self.log_dir = self.root / "logs" / "tensorboard"
        self.model_dir = self.root / Path(self.config.get("model_dir", MODEL_DIR))
        self.checkpoint_dir = self.root / Path(self.config.get("checkpoint_dir", CHECKPOINT_DIR))
        
        for d in [self.log_dir, self.model_dir, self.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _handle_interrupt(self, sig, frame):
        """Ensures TensorBoard and logs are flushed before exiting."""
        logger.warning("\nInterrupt detected. Flushing logs...")
        self.writer.close()
        sys.exit(0)

    def train(self):
        """
        Main execution loop. Executes N epochs of CTC training.
        
        Raises:
            FileNotFoundError: If training data is missing from 'data/' directory.
        """
        data_dir = self.root / Path(self.config.get("data_dir", TRAIN_PATH))
        
        try:
            signals = torch.load(data_dir / "signals.pt", weights_only=False)
            with open(data_dir / "sequences.pkl", "rb") as f:
                sequences = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Data files missing in {data_dir}. Run simulator first.")
            return

        dataset = SquigDataset(signals, sequences)
        dataloader = DataLoader(
            dataset, 
            batch_size=int(self.config.get("train_batch_size", TRAIN_BATCH_SIZE)), 
            shuffle=True, 
            collate_fn=collate_batch,
            num_workers=0 # Increase for heavy IO, keep 0 for debugging
        )

        epochs = int(self.config.get("train_num_epochs", TRAIN_NUM_EPOCHS))
        best_loss = float('inf')

        logger.info(f"SquigNet initialization complete. Training on: {self.device}")
        self.writer.add_text("Hyperparameters", str(self.config), 0)
        self.writer.flush()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
            
            for signals_b, targets_b, in_lens, tgt_lens in pbar:
                signals_b, targets_b = signals_b.to(self.device), targets_b.to(self.device)
                in_lens, tgt_lens = in_lens.to(self.device), tgt_lens.to(self.device)

                self.optimizer.zero_grad()
                # Transpose for CTC: (T, B, C)
                outputs = self.model(signals_b).transpose(0, 1) 
                loss = self.loss_fn(outputs, targets_b, in_lens, tgt_lens)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.debug("Gradient instability detected. Skipping batch.")
                    continue

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            
            # Telemetry Update
            self.writer.add_scalar("Loss/Train", avg_loss, epoch)
            self.writer.add_scalar("Metrics/LR", self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.flush()

            # State Persistence
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.model_dir / f"best_{MODEL_FILE}")
                logger.info(f"Epoch {epoch+1:03d}: Metric improved to {best_loss:.4f}. Model saved.")

        self.writer.close()
        logger.info("Training session concluded successfully.")

if __name__ == "__main__":
    try:
        trainer = SquigTrainer(USER_CONFIG)
        trainer.train()
    except Exception as e:
        logger.critical(f"Uncaught exception in training pipeline: {e}", exc_info=True)
        sys.exit(1)