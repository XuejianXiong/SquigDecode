# ≣ SquigDecode: Deep Learning Basecaller
## A CNN-BiLSTM Hybrid for Nanopore Signal Transduction

SquigDecode is a professional-grade deep learning basecaller designed to translate raw electrical current signals ("squiggles") generated during nanopore sequencing into nucleotide sequences (A, C, G, T).

The project implements a Convolutional Recurrent Neural Network (CRNN) to decode complex signal patterns, mimicking the physics of DNA translocation through a R10.4.1 biological pore.

------------------------------------------------
## ◈ Project Overview

Nanopore sequencing measures electrical current changes as DNA molecules pass through a pore. These signals represent a "k-mer window" of 3–5 bases at once, requiring sophisticated sequence modeling to resolve.

SquigDecode provides an end-to-end pipeline for:

1. Signal Simulation: Physics-based generation of standardized current squiggles.

2. Feature Extraction: CNN-based local signal signature detection.

3. Sequence Modeling: Long-range dependency capture via Bidirectional LSTMs.

4. CTC Decoding: Probabilistic sequence alignment and decoding.

------------------------------------------------
## ◈ Architecture: SquigNet

SquigNet utilizes a hybrid architecture optimized for high-fidelity signal transduction.

1. Feature Extraction (CNN Blocks)
Two stacked 1D convolutional layers reduce signal noise and perform 4× temporal downsampling (e.g., 800 samples → 200 latent states). This significantly reduces the computational burden on the recurrent layers.

2. Sequence Modeling (Bi-LSTM)
Stacked Bidirectional LSTMs process the downsampled features. Because the pore senses multiple bases simultaneously, the bidirectional context is critical for inferring nucleotide identity from overlapping signal patterns.

3. Connectionist Temporal Classification (CTC)
The model is trained using CTC Loss, which allows for alignment-free training. During inference, a Greedy Decoder collapses repeated tokens and removes blanks to produce the final DNA string.

Architecture Pipeline:
<p align="center">
  <img src="images/squignet.png" width="300">
</p>


------------------------------------------------
## ◈ Performance & Telemetry

### Training Results

- Best Training Loss: 0.0862 (Converged at Epoch 50)

- Mean Basecalling Accuracy: ~96% on standardized test datasets.

- Robustness: Maintained ~87% accuracy under synthetic Gaussian noise (σ = 0.3), demonstrating the model's resilience to distribution shift.

### Experiment Tracking

The project integrates TensorBoard for real-time monitoring of training telemetry, including:

- Loss/Train: Tracking CTC convergence.

- Learning_Rate: Monitoring ReduceLROnPlateau scheduler steps.

- Config: Persistent logging of hyperparameters for experiment reproducibility.

------------------------------------------------
## ◈ Project Structure

```
SquigDecode/
├── src/
│   ├── architecture.py      # SquigNet (CNN-BiLSTM) definition
│   ├── train.py             # Professional training engine with signal handling
│   ├── inference.py         # SquigBasecaller & Levenshtein evaluation
│   ├── data_simulator.py    # Robust MAD-standardized signal generator
│   ├── QC_squig_data.py     # Signal profiling & QC modules
│   ├── config.py            # Global project configuration
│   └── input.json           # User-definable hyperparameters
├── logs/                    # TensorBoard event files
├── models/                  # Best model checkpoints (.pt)
├── results/                 # Diagnostic plots and inference reports
├── tests/                   # Automated unit tests for signal physics and tensor shapes
├── notebooks/               # Exploratory Data Analysis (EDA) and hyperparameter tuning├── pipeline.sh              # Orchestration script (Standardized Workflow)
├── pyproject.toml           # Modern dependency management (uv)
├── uv.lock                  # Deterministic lockfile for environment reproducibility
├── input_schema.json        # JSON Schema for validating user-defined hyperparameters
├├── LICENSE                  
└── README.md
```

------------------------------------------------
## ◈ Getting Started

1. **IEnvironment Setup** 
   
   This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.
   ```
   # Sync environment and install dependencies
   uv sync
   ```
2. **Execution**
   
   Use the provided pipeline.sh for standardized workflows:
   ```
   # Run entire pipeline (Simulate -> QC -> Train -> Infer)
   ./pipeline.sh all

   # Or run specific stages
   ./pipeline.sh train
   ./pipeline.sh infer
   ```
3. **Monitoring**:
   ```
   uv run python -m tensorboard.main --logdir logs/tensorboard
   ```

------------------------------------------------
## ◈ Key Technical Insights

- **Standardization**: Utilizes Median Absolute Deviation (MAD) standardization to ensure signal consistency regardless of pore-to-pore variations.

- **Resilience**: Implements a professional SquigTrainer class with SIGINT signal handling to ensure telemetry data is flushed to disk during unexpected shutdowns.

- **Accuracy Metrics**: Evaluation uses the Levenshtein Edit Distance, the gold standard in genomics for assessing read identity.

------------------------------------------------
## ◈ Future Directions

To transition SquigDecode from a research prototype to a production-scale basecaller, the following enhancements are proposed:

1. Advanced Architecture

- Transformer-Based Decoding: Integrate Multi-Head Attention mechanisms (e.g., Conformer) to better capture dependencies without the sequential bottleneck of LSTMs.

- Beam Search Decoding: Implement a Beam Search decoder to improve accuracy in difficult homopolymer regions (e.g., AAAAA).

2. Biological Fidelity

- Base Modification Detection: Train the model to distinguish canonical bases from modified bases like 5mC by analyzing subtle current deviations.

- Multi-Pore Simulation: Model signal drift and pore-stalling events common in real-world Nanopore sequencing.

3. Engineering & Scalability

- Quantization: Implement INT8 quantization to allow the basecaller to run in real-time on edge devices like the MinION.

- GPU-Accelerated Inference: Transition inference to vectorized GPU kernels for high-throughput processing.

------------------------------------------------
## ◈ License

MIT License – feel free to use, adapt, and share.
