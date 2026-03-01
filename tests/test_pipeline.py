import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure project's `src` directory is importable
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from architecture import SquigNet  # type: ignore
from data_simulator import generate_squiggle, get_dwell_time  # type: ignore
from inference import greedy_decode  # type: ignore


def test_squignet_forward_shape() -> None:
    """Initialize `SquigNet` and verify output shape for dummy input."""
    model = SquigNet()
    model.eval()

    dummy = torch.randn(1, 1, 1000, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 250, 5)


def test_generate_squiggle_is_tensor_and_length(monkeypatch: Any) -> None:
    """Verify generated squiggle can be converted to `torch.Tensor` and
    that its length equals 20x the DNA sequence length when dwell time is
    forced to 20 samples per base.
    """
    # Force deterministic dwell time of 20 for this test
    monkeypatch.setattr("data_simulator.get_dwell_time", lambda: 20)

    dna = "ACGTAC"
    signal, dwell_times = generate_squiggle(dna)

    tensor_signal = torch.tensor(signal)

    assert isinstance(tensor_signal, torch.Tensor)
    assert tensor_signal.numel() == 20 * len(dna)


def test_greedy_decoder_collapses_duplicates() -> None:
    """Test greedy decoder collapses repeated indices and removes blanks.

    Provide logits constructed so argmax produces indices:
    [1, 1, 0, 2, 2, 0] -> collapsed => [1,0,2,0] -> remove blanks => 'A' + 'C' => 'AC'
    """
    # Create logits tensor with shape (1, 6, 5)
    logits = torch.zeros(1, 6, 5, dtype=torch.float32)

    # Set high score for indices to force argmax outcome
    indices = [1, 1, 0, 2, 2, 0]
    for pos, idx in enumerate(indices):
        logits[0, pos, idx] = 10.0

    decoded = greedy_decode(logits)
    assert decoded == "AC"
