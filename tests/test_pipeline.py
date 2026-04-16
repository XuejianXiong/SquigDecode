"""
SquigDecode: Unit Test Suite for Pipeline Components.

This module provides automated verification for model dimensionality, 
signal physics, and sequence decoding logic. It uses pytest fixtures 
and parameterization for high coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch

# Absolute path resolution for reliable test execution across environments
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architecture import SquigNet
from data_simulator import generate_squiggle
from inference import SquigBasecaller, calculate_levenshtein_accuracy

# --- Fixtures ---

@pytest.fixture
def model() -> SquigNet:
    """Provides a SquigNet instance in evaluation mode."""
    net = SquigNet()
    net.eval()
    return net

@pytest.fixture
def device() -> torch.device:
    """Detects available hardware for device-agnostic testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Architecture Tests ---

@pytest.mark.parametrize("signal_length, expected_timesteps", [
    (400, 100),   # 4x downsampling
    (1000, 250),
    (1600, 400),
])
def test_squignet_dimensions(model: SquigNet, signal_length: int, expected_timesteps: int):
    """Verifies that the CNN-BiLSTM preserves expected temporal downsampling."""
    dummy_input = torch.randn(1, 1, signal_length)
    with torch.no_grad():
        output = model(dummy_input)
    
    # Expected: (Batch, Downsampled_Time, Num_Classes)
    assert output.shape == (1, expected_timesteps, 5), \
        f"Output shape mismatch for input length {signal_length}"

def test_model_determinism(model: SquigNet):
    """Ensures eval mode disables dropout/batchnorm noise."""
    dummy = torch.randn(1, 1, 400)
    with torch.no_grad():
        out1 = model(dummy)
        out2 = model(dummy)
    assert torch.allclose(out1, out2), "Model output is non-deterministic in eval mode."

# --- Data Simulation Tests ---

def test_generate_squiggle_physics(monkeypatch: pytest.MonkeyPatch):
    """Verifies signal length matches forced dwell time geometry."""
    fixed_dwell = 20
    monkeypatch.setattr("data_simulator.get_dwell_time", lambda: fixed_dwell)
    
    test_dna = "ACGT"
    signal, _ = generate_squiggle(test_dna)
    
    assert len(signal) == len(test_dna) * fixed_dwell
    assert isinstance(signal, np.ndarray)
    assert not np.isnan(signal).any(), "Signal contains NaN values."

def test_empty_sequence_handling():
    """Edge case: Ensures simulator handles empty DNA strings gracefully."""
    with pytest.raises((ValueError, IndexError)):
        generate_squiggle("")

# --- Inference & Decoding Tests ---

def test_greedy_decoder_logic(model: SquigNet):
    """Tests CTC collapse logic: duplicates and blanks removal."""
    # 1. Create a "Mock" or Dummy Basecaller instance
    # We don't need to load a real model file for this logic test
    from unittest.mock import MagicMock
    mock_basecaller = MagicMock()
    mock_basecaller.blank_idx = 0  # Define the attribute the method looks for
    
    # 2. Setup indices: 1 (A), 1 (A), 0 (Blank), 2 (C), 2 (C), 0 (Blank)
    # Target collapse: [1, 2] -> "AC"
    logits = torch.zeros(1, 6, 5)
    indices = [1, 1, 0, 2, 2, 0]
    for t, idx in enumerate(indices):
        logits[0, t, idx] = 10.0
    
    # 3. Call the method using our mock instead of None
    # Use the actual class method logic
    decoded = SquigBasecaller.decode_ctc(mock_basecaller, logits)[0]
    
    assert decoded == "AC"
    
@pytest.mark.parametrize("pred, target, expected", [
    ("ACGT", "ACGT", 1.0),      # Exact match
    ("ACT", "ACGT", 0.75),      # Deletion
    ("ACGT", "", 0.0),          # Empty target
    ("ACGA", "ACGT", 0.75),     # Substitution
])
def test_accuracy_metrics(pred: str, target: str, expected: float):
    """Verifies Levenshtein distance calculations."""
    acc = calculate_levenshtein_accuracy(pred, target)
    assert pytest.approx(acc, 0.01) == expected

# --- Integration Tests ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gpu_compatibility(model: SquigNet):
    """Ensures model and tensors can move to CUDA without error."""
    model.cuda()
    dummy = torch.randn(1, 1, 400).cuda()
    with torch.no_grad():
        out = model(dummy)
    assert out.is_cuda