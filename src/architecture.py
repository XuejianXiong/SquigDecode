"""
SquigDecode: Neural Architecture for Nanopore Basecalling

This module defines SquigNet, a hybrid CNN-RNN model designed to convert
raw nanopore signals (squiggles) into accurate base calls.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class SquigNet(nn.Module):
    """
    SquigNet: CNN-RNN hybrid model for nanopore signal basecalling.

    Architecture:
    - Two 1D-CNN blocks (64 and 128 filters) with MaxPooling for
      hierarchical feature extraction and downsampling
    - Bidirectional LSTM (hidden_size=256, num_layers=2) for temporal
      sequence modeling
    - Output layer mapping to 5 classes: 0 (Blank/CTC), 1 (A), 2 (C),
      3 (G), 4 (T)

    Input shape: (batch, 1, signal_length)
    Output shape: (batch, signal_length // 4, 5)

    The layer structure aligns signal length with CTC Loss requirements:
    downsampling by MaxPool1d(2) in each block reduces signal length
    by a factor of 4 total, which balances temporal resolution with
    computational efficiency.
    """

    def __init__(self) -> None:
        """Initialize SquigNet model with CNN and LSTM layers."""
        super(SquigNet, self).__init__()

        # CNN Feature Extraction Block 1
        # Input: (batch, 1, signal_length)
        # Output: (batch, 64, signal_length // 2)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # CNN Feature Extraction Block 2
        # Input: (batch, 64, signal_length // 2)
        # Output: (batch, 128, signal_length // 4)
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Bidirectional LSTM for sequence modeling
        # Input: (batch, signal_length // 4, 128)
        # Output: (batch, signal_length // 4, 512)  [256 * 2 for bidirectional]
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        # Output Classification Layer
        # Maps LSTM output (256 * 2) to 5 base classes
        self.output_layer = nn.Linear(
            in_features=256 * 2,
            out_features=5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SquigNet.

        Process:
        1. Two 1D-CNN blocks with batch norm, ReLU, and pooling
        2. Transpose for LSTM input (batch, length, channels)
        3. Bidirectional LSTM processing
        4. Linear output layer for classification

        Args:
            x: Input signal tensor of shape (batch, 1, signal_length)

        Returns:
            torch.Tensor: Output logits of shape
                (batch, signal_length // 4, num_classes=5)
        """
        # CNN Block 1: Convolution -> BatchNorm -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # CNN Block 2: Convolution -> BatchNorm -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Transpose for LSTM: (batch, channels, length) ->
        # (batch, length, channels)
        x = x.transpose(1, 2)

        # Bidirectional LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Output classification layer
        output = self.output_layer(lstm_out)

        return output

    @classmethod
    def count_parameters(cls) -> int:
        """
        Count the total number of trainable parameters in SquigNet.

        This is a class method that instantiates a model and counts
        all learnable parameters.

        Returns:
            int: Total number of trainable parameters
        """
        model = cls()
        total_params = sum(p.numel() for p in model.parameters())
        return total_params

    def get_model_info(self) -> Dict[str, any]:
        """
        Get detailed model information and statistics.

        Returns:
            Dict[str, any]: Dictionary containing:
                - total_parameters: Total number of parameters
                - trainable_parameters: Number of trainable parameters
                - num_classes: Number of output classes
                - architecture: Model description
                - input_shape: Expected input tensor shape
                - output_shape: Expected output tensor shape (depends on
                  input signal_length)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': 5,
            'architecture': 'CNN-BiLSTM Hybrid',
            'input_shape': '(batch, 1, signal_length)',
            'output_shape': '(batch, signal_length // 4, 5)',
            'base_classes': {
                '0': 'Blank (CTC)',
                '1': 'A',
                '2': 'C',
                '3': 'G',
                '4': 'T',
            },
        }


def create_model() -> SquigNet:
    """
    Factory function to create a SquigNet model instance.

    Returns:
        SquigNet: Initialized model ready for training or inference
    """
    return SquigNet()


if __name__ == "__main__":
    # Example usage and model testing
    print("=" * 70)
    print("SquigNet Architecture Summary")
    print("=" * 70)

    # Create model and display parameter count
    model = SquigNet()
    total_params = SquigNet.count_parameters()
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Display detailed model information
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        if key != 'base_classes':
            print(f"  {key}: {value}")

    print(f"\nBase Classes:")
    for idx, base_name in model_info['base_classes'].items():
        print(f"  {idx}: {base_name}")

    # Test forward pass with synthetic data
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)

    batch_size = 4
    signal_length = 1000

    test_input = torch.randn(batch_size, 1, signal_length)
    with torch.no_grad():
        output = model(test_input)

    print(f"\nInput shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    downsampled_length = signal_length // 4
    print(f"Expected:     torch.Size([{batch_size}, {downsampled_length}, 5])")

    # Verify output shape matches expectations
    expected_shape = (batch_size, downsampled_length, 5)
    assert output.shape == expected_shape, (
        f"Output shape {output.shape} does not match "
        f"expected {expected_shape}"
    )
    print("\n✓ Forward pass test successful!")
