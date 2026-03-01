# 🧬 SquigDecode: Deep Learning Basecaller
## A CNN-BiLSTM Hybrid for Nanopore Signal Transduction

SquigDecode is a sequence-to-sequence deep learning model designed to translate raw electrical "squiggles" from nanopores into DNA sequences (A, C, G, T). This project simulates the physics of DNA translocating through a pore and uses a CRNN (Convolutional Recurrent Neural Network) to decode the signal.


------------------------------------------------
## Technical Overview

The system consists of:
- **Signal Processing**: Raw signal preprocessing and normalization
- **Model Architecture**: Deep learning models optimized for signal classification
- **Base Calling**: Sequence inference from model predictions
- **Validation**: Comprehensive testing and signal analysis tools


------------------------------------------------
## 🏗️ Architecture: SquigNet
The model handles 1D signal data through a multi-stage feature extraction and decoding pipeline:

1. **Dual 1D-CNN Blocks**: Extract local features from the raw picoampere signal.

 - Layers: Conv1d → BatchNorm → ReLU → MaxPool.

 - Purpose: Noise reduction and temporal downsampling.

2. **2-Layer Bidirectional LSTM**: Models the long-term dependencies of the DNA sequence.

 - Logic: Uses forward and backward context to handle the 3-base sliding window physics of the pore.

3. **Linear Projection & CTC Loss**: Maps the hidden states to 5 classes (Blank, A, C, G, T) using Connectionist Temporal Classification to handle variable-length sequences.


------------------------------------------------
## 🚀 Performance & Results
 - Final Training Loss: 0.1293 (at Epoch 50).

 - Basecalling Accuracy (Clean Data): ~99%.

 - Robustness Test (High Noise): ~88% (Demonstrates the impact of distribution shift).

### Loss Curve
The model achieves rapid convergence, breaking the 0.2 loss threshold by Epoch 8, indicating high efficiency in learning the base-level step functions.


------------------------------------------------
## Project Structure

```
SquigDecode/
├── src/              # Model code and core algorithms
├── tests/            # Unit tests for signal generation and processing
├── data/             # Reference datasets and signal files
├── models/           # Trained model
├── results/          # Figures and results
├── notebooks/        # Jupyter notebooks for signal analysis
├── requirements.txt  # Python dependencies
└── README.md         
```

 - `src/architecture.py`: Definition of the SquigNet class.

 - `src/data_simulator.py`: Physics-based signal generator (Shift, Scale, and Gaussian noise).

 - `src/config.py`: Hardcoded internal physics constants (Sliding window weights, Model dims).

 - `input.json`: User-configurable parameters for inference and simulation.

 - `src/train.py`: Training engine utilizing CTC Loss.

 - `src/inference.py`: Evaluation script with a Greedy Decoder.


------------------------------------------------
## 🧪 Technical Insights
 - **Distribution Shift**: During the "Principal Engineer" stress test, increasing noise ($\sigma$) from 3.0 to 5.0 revealed model sensitivity. This highlights the need for Data Augmentation in production environments.

 - **Temporal Resolution**: The CNN downsamples the signal by 4x (800 → 200 samples), significantly reducing the computational load on the LSTM while maintaining enough data points to distinguish between similar current levels.


------------------------------------------------
## ⚙️ How to Run

1. Install Dependencies: pip install -r requirements.txt

2. Train the Model: python src/train.py

3. Run Inference: python src/inference.py


------------------------------------------------
## License

MIT License – feel free to use, adapt, and share.
