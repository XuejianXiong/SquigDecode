#!/bin/bash
# ==============================================================================
# SquigDecode: End-to-End Deep Learning Basecaller Pipeline
# 
# Description:
#   Orchestrates the lifecycle of SquigNet basecalling, from synthetic 
#   signal generation and QC to distributed training and inference.
#
# Usage:
#   ./pipeline.sh [all|simul|qc|train|infer|clean]
#
# Author: Xuejian Xiong
# Version: 1.1.0 | Date: 2026-04-15
# License: MIT
# ==============================================================================

# --- Strict Mode & Safety ---
set -euo pipefail
IFS=$'\n\t'

# --- Project Constants ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
RESULTS_DIR="$PROJECT_ROOT/results"
MODELS_DIR="$PROJECT_ROOT/models"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"

# --- ANSI Color Codes for CI/CD Visibility ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Initialization ---
mkdir -p "$RESULTS_DIR" "$MODELS_DIR" "$LOG_DIR"

# Timestamped Log setup with Process ID tracking
LOG_FILE="$LOG_DIR/squig_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Signal Handling (Trap) ---
cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}[ERROR] Pipeline failed at $(date) with exit code $exit_code.${NC}"
        echo -e "Check full logs at: $LOG_FILE"
    fi
}
trap cleanup_on_error EXIT

# --- Helper Functions ---
log_info()  { echo -e "${BLUE}[INFO]${NC} $(date +'%H:%M:%S') - $1"; }
log_stage() { echo -e "\n${GREEN}==> STAGE: $1${NC}"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

check_runtime_env() {
    log_info "Validating execution environment..."
    local tools=("uv" "python3" "git")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            echo -e "${RED}Error: Critical dependency '$tool' not found in PATH.${NC}"
            exit 1
        fi
    done
    log_info "Environment validated. Project Root: $PROJECT_ROOT"
}

show_metadata() {
    echo "------------------------------------------------------------"
    echo " SQUIGDECODE | Production Pipeline v1.1.0 "
    echo "------------------------------------------------------------"
    echo "User:        $(whoami)"
    echo "Host:        $(hostname)"
    echo "Hardware:    $(uname -sm)"
    echo "Start Time:  $(date)"
    echo "Log File:    $LOG_FILE"
    echo "------------------------------------------------------------"
}

# --- Pipeline Stage Implementations ---

run_simulation() {
    log_stage "1/5 - Data Simulation"
    log_info "Generating Nanopore signals with MAD-standardization..."
    uv run "$SRC_DIR/data_simulator.py"
}

run_qc() {
    log_stage "2/5 - Quality Control"
    log_info "Executing PeptideDeepDive QC modules and signal profiling..."
    uv run "$SRC_DIR/QC_squig_data.py"
}

run_architecture_check() {
    log_stage "3/5 - Architecture Validation"
    log_info "Verifying SquigNet Bi-LSTM/CNN graph and parameter count..."
    uv run "$SRC_DIR/architecture.py"
}

run_training() {
    log_stage "4/5 - Model Training"
    log_info "Optimizing with CTC Loss. Monitoring via TensorBoard at port 6006..."
    uv run "$SRC_DIR/train.py"
}

run_inference() {
    log_stage "5/5 - Inference & Performance Evaluation"
    log_info "Running held-out validation and generating accuracy metrics..."
    uv run "$SRC_DIR/inference.py"
}

# --- Command Logic ---

case "${1:-all}" in
    all)
        check_runtime_env
        show_metadata
        run_simulation
        run_qc
        run_architecture_check
        run_training
        run_inference
        ;;
    simul)
        run_simulation
        ;;
    qc)
        run_qc
        ;;
    train)
        run_training
        ;;
    infer)
        run_inference
        ;;
    clean)
        log_warn "Initiating project cleanup..."
        rm -rf "$DATA_DIR"
        log_info "Temporary data purged. Models and logs preserved in $MODELS_DIR and $LOG_DIR."
        ;;
    *)
        echo "Usage: $0 {all|simul|qc|train|infer|clean}"
        exit 1
        ;;
esac

log_info "Pipeline cycle completed successfully."