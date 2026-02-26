# SquigDecode Instructions

This document provides guidance for the SquigDecode project.

## Project Overview

SquigDecode is a Signal-to-Base decoder for nanopore or sequencing-by-expansion (SBX) reads, designed for high-SNR sequencing data.

## Project Structure

- **src/**: Core model code and algorithms for signal processing and basecalling
- **tests/**: Unit tests for signal generation and processing modules
- **data/**: Reference datasets and signal files
- **notebooks/**: Jupyter notebooks for signal analysis and visualization

## Development Guidelines

- Maintain clear separation of concerns between signal processing, model code, and testing
- All model code should reside in the `src/` directory with appropriate module organization
- Unit tests should mirror the structure of `src/` and be placed in the `tests/` directory
- Use descriptive variable names and include docstrings for all functions
- Follow PEP 8 style guidelines

## Testing

Unit tests should focus on:
- Signal generation accuracy
- Data processing correctness
- Model input/output validation
- Edge cases and error handling

Run tests with: `pytest tests/`

## Notes for Development

- When adding new modules, ensure corresponding unit tests are created
- Document signal processing assumptions and constraints
- Keep notebooks focused on analysis and exploration; production code goes in `src/`
