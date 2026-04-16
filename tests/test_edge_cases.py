"""
SquigDecode: Edge Case & Stress Testing Suite.
Validates model robustness against non-standard signal inputs, 
biological anomalies, and hardware-level numerical instability.
"""

from __future__ import annotations

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# --- Path Injection ---
# Resolve the project root and add the 'src' directory to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from architecture import SquigNet
from inference import SquigBasecaller, calculate_levenshtein_accuracy
from data_simulator import generate_squiggle

@pytest.fixture
def basecaller():
    """Provides a Basecaller instance using the production model weights."""
    model_path = Path("models/squig_model.pt")
    if not model_path.exists():
        pytest.skip("Production model weights not found. Skipping edge case tests.")
    return SquigBasecaller(model_path, device=torch.device("cpu"))

# --- 1. Biological Edge Cases (Homopolymers) ---

def test_homopolymer_resilience(basecaller: SquigBasecaller):
    """
    Tests if the CTC decoder correctly identifies long identical base stretches.
    Bioinformatics Challenge: Homopolymers are notorious for 'signal stalling'.
    """
    hp_dna = "AAAAA"
    # Manually simulate a very long stall for 'A'
    signal, _ = generate_squiggle(hp_dna)
    
    prediction = basecaller.basecall(signal)
    
    # We allow for slight errors, but it should not collapse 5 'A's into 1 'A'
    assert len(prediction) >= 3, f"Homopolymer collapse too aggressive: {prediction}"
    assert "A" in prediction

# --- 2. Signal Geometry Edge Cases ---

@pytest.mark.parametrize("length", [10, 50000])
def test_signal_length_extremes(basecaller: SquigBasecaller, length: int):
    """
    Verifies model stability across extreme temporal scales.
    - 10 samples: Extremely short fragments/adapters.
    - 50000 samples: Ultra-long reads (ULR).
    """
    dummy_signal = np.random.normal(0, 1, size=length).astype(np.float32)
    
    # Execution should not raise MemoryError or Shape Mismatch
    try:
        prediction = basecaller.basecall(dummy_signal)
        assert isinstance(prediction, str)
    except Exception as e:
        pytest.fail(f"Failed on signal length {length}: {e}")

# --- 3. Numerical Stability (Adversarial Inputs) ---

def test_nan_inf_handling(basecaller: SquigBasecaller):
    """
    Tests if the pipeline handles sensor glitches (NaNs or Infs) gracefully.
    Production data often contains dropped packets or hardware saturated signals.
    """
    signal, _ = generate_squiggle("ACGT")
    signal[10] = np.nan
    signal[20] = np.inf
    
    # In a professional pipeline, we expect the pre-processor to 
    # clean these or the model to handle them without a kernel panic.
    # Here we check if basecall() returns a string instead of crashing.
    prediction = basecaller.basecall(signal)
    assert isinstance(prediction, str)

def test_zero_variance_signal(basecaller: SquigBasecaller):
    """
    Tests a 'flatline' signal (e.g., pore is blocked or disconnected).
    MAD standardization should not divide by zero.
    """
    flat_signal = np.ones(400, dtype=np.float32) * 50.0 # Constant 50pA
    
    # This checks your MAD standardization: (x - med) / (mad + epsilon)
    prediction = basecaller.basecall(flat_signal)
    assert isinstance(prediction, str)

# --- 4. Accuracy Metric Sanity ---

def test_levenshtein_bounds():
    """Verifies that accuracy metrics stay within logical [0, 1] bounds."""
    # Complete mismatch
    acc_zero = calculate_levenshtein_accuracy("AAAA", "GGGG")
    assert acc_zero == 0.0
    
    # Empty prediction (Model failure mode)
    acc_empty = calculate_levenshtein_accuracy("", "ACTG")
    assert acc_empty == 0.0
    
    # Partial match
    acc_partial = calculate_levenshtein_accuracy("ACT", "ACGT")
    assert 0.0 < acc_partial < 1.0