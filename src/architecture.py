"""
SquigDecode: Neural Architecture for Nanopore Basecalling

This module defines SquigNet, a hybrid CNN-RNN model optimized for 
robust nanopore signal decoding using MAD-standardized inputs.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

class SquigNet(nn.Module):
    """
    SquigNet: CNN-RNN hybrid model for nanopore signal basecalling.

    Architecture Upgrades:
    - Larger Conv1d kernels (7) to capture broader k-mer context.
    - Dilated Convolution in Block 2 to handle homopolymer plateaus.
    - LeakyReLU activation to prevent neuron death during noisy signal training.
    - LogSoftmax output for numerical stability with CTCLoss.
    """

    def __init__(self) -> None:
        """Initialize SquigNet with optimized CNN and BiLSTM layers."""
        super(SquigNet, self).__init__()

        # CNN Feature Extraction Block 1
        # Captures local electrical patterns with a receptive field of 7 samples.
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # CNN Feature Extraction Block 2
        # Uses Dilation (2) to expand the receptive field without adding parameters,
        # helping the model "see" across longer signal plateaus (homopolymers).
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Bidirectional LSTM for temporal sequence modeling
        # Input features: 128 (CNN filters)
        # Hidden size: 256 per direction -> 512 total
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3, 
        )

        # Output Classification Layer
        # Maps 512 features to 5 classes: 0(Blank), 1(A), 2(C), 3(G), 4(T)
        self.output_layer = nn.Linear(
            in_features=256 * 2,
            out_features=5,
        )
        
        # LogSoftmax is preferred for CTCLoss stability
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SquigNet.
        Input: (batch, 1, signal_length)
        Output: (batch, signal_length // 4, 5) -> Log-probabilities
        """
        # CNN Block 1
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))

        # CNN Block 2
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))

        # Transpose for LSTM: (batch, channels, length) -> (batch, length, channels)
        x = x.transpose(1, 2)

        # BiLSTM processing
        lstm_out, _ = self.lstm(x)

        # Output mapping to logits then log-probabilities
        logits = self.output_layer(lstm_out)
        return self.log_softmax(logits)

    @classmethod
    def count_parameters(cls) -> int:
        """Count total trainable parameters."""
        model = cls()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model metadata for reporting and publication."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": trainable_params,
            "architecture": "CNN-BiLSTM-Hybrid (Optimized)",
            "input_normalization": "MAD + Center-Clipping",
            "downsampling_factor": 4,
            "receptive_field_enhancements": ["Kernel 7", "Dilation 2"],
            "num_classes": 5,
            "classes": {0: "Blank", 1: "A", 2: "C", 3: "G", 4: "T"}
        }

def create_model() -> SquigNet:
    """Factory function to instantiate the model."""
    return SquigNet()

if __name__ == "__main__":
    # Test model initialization and forward pass
    print("=" * 60)
    print("SquigNet Architecture: Principal-Level Review")
    print("=" * 60)

    model = create_model()
    info = model.get_model_info()
    
    print(f"\nModel: {info['architecture']}")
    print(f"Trainable Parameters: {info['total_parameters']:,}")
    print(f"Features: {', '.join(info['receptive_field_enhancements'])}")

    # Dummy data test: Batch=8, Signal=1024
    test_signal = torch.randn(8, 1, 1024)
    with torch.no_grad():
        output = model(test_signal)

    print(f"\nForward Pass Test:")
    print(f"  Input:  {test_signal.shape}")
    print(f"  Output: {output.shape} (Log-probabilities)")
    
    assert output.shape == (8, 1024 // 4, 5)
    print("\n✓ Architecture check complete. Ready for training.")