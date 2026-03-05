#!/usr/bin/env bash
# =============================================================================
# Stage 1: N-dimensional Kernel Synthesis
# =============================================================================
# Generates correlated multivariate time series from scratch using diverse
# kernel/GP-based synthesis methods (correlated_gp, lead_lag, VAR, causal_filter,
# hidden_regime, partial_mix, dynamic_mix, segment_mix, independent).
#
# Output layout (per-dimension folders):
#   <output-dir>/kernel_synth_1d_<min>-<max>_<N>samples/
#   <output-dir>/kernel_synth_2d_<min>-<max>_<N>samples/
#   ...
#   <output-dir>/kernel_synth_<D>d_<min>-<max>_<N>samples/
#
# Each folder contains a HuggingFace DatasetDict (train split) in Arrow format,
# plus statistics_summary.json, samples.png, correlation_matrices.png.
#
# Usage (defaults):
#   bash stage1_kernel_synth.sh
#
# Usage (custom):
#   bash stage1_kernel_synth.sh \
#       --num-samples 1000000 --max-variates 5 \
#       --min-len 64 --max-len 1024 \
#       --output-dir /group-volume/ts-dataset/chronos2_datasets \
#       --seed 42 --num-workers -1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/mv_synth/stage1_kernel_synth.py"

# ── Default parameters ────────────────────────────────────────────────────────
MIN_LEN=64
MAX_LEN=1024
NUM_SAMPLES=1000
MAX_VARIATES=5
OUTPUT_DIR="/group-volume/ts-dataset/chronos2_datasets"
SEED=42
NUM_WORKERS=-1   # -1 = use all available CPUs

# ── Parse arguments (pass-through to python) ─────────────────────────────────
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-len)       MIN_LEN="$2";      shift 2 ;;
        --max-len)       MAX_LEN="$2";      shift 2 ;;
        --num-samples)   NUM_SAMPLES="$2";  shift 2 ;;
        --max-variates)  MAX_VARIATES="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";   shift 2 ;;
        --seed)          SEED="$2";         shift 2 ;;
        --num-workers)   NUM_WORKERS="$2";  shift 2 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "======================================================================"
echo " Stage 1: Kernel Synthesis (multivariate)"
echo "======================================================================"
echo "  min-len      : ${MIN_LEN}"
echo "  max-len      : ${MAX_LEN}"
echo "  num-samples  : ${NUM_SAMPLES}"
echo "  max-variates : ${MAX_VARIATES}"
echo "  output-dir   : ${OUTPUT_DIR}"
echo "  seed         : ${SEED}"
echo "  num-workers  : ${NUM_WORKERS}"
echo "======================================================================"

# Activate conda environment if available
if command -v conda &> /dev/null && conda env list | grep -q "^tsm "; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tsm
fi

python "${PYTHON_SCRIPT}" \
    --min-len      "${MIN_LEN}" \
    --max-len      "${MAX_LEN}" \
    --num-samples  "${NUM_SAMPLES}" \
    --max-variates "${MAX_VARIATES}" \
    --output-dir   "${OUTPUT_DIR}" \
    --seed         "${SEED}" \
    --num-workers  "${NUM_WORKERS}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "======================================================================"
echo " Stage 1 complete. Outputs in: ${OUTPUT_DIR}"
echo "======================================================================"
