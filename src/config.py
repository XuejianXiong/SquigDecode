"""
SquigDecode: Configuration for Nanopore Signal Simulation
Values represent standard R10.4.1 pore chemistry at 180mV bias.
"""

# Base-to-Picoampere (pA) Mapping
# Authentic levels where G is high and T is the deepest blockade.
BASE_PICOAMPERE_MAP = {
    'A': 95.0,
    'C': 110.0,
    'G': 125.0,
    'T': 80.0,
    'blank': 0.0  # Used for CTC Loss padding
}

# Simulation Parameters
DWELL_TIME_MEAN = 15 
DWELL_TIME_STD = 4   # The "spread" of how fast the DNA moves
MIN_DWELL_TIME = 5   # The minimum samples to detect a base

NOISE_STD = 3.5        # Gaussian noise in pA
DRIFT_FACTOR = 0.01    # Simulates slight electrical fluctuations over time