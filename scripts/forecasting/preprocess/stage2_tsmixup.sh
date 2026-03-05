#!/usr/bin/env bash
# =============================================================================
# Stage 2: TSMixup — Multivariate Synthesis from Real + Synthetic Data
# =============================================================================
# Reads from m input paths (HuggingFace datasets or raw .arrow files),
# lazily samples series, and applies mixing operations (weighted_sum,
# negative_weighted_sum, nonlinear_mix, lag_lead, piecewise_mix,
# time_warp, causal_filter, independent) to produce correlated multivariate
# output datasets.
#
# Output layout (per-dimension folders):
#   <output-dir>/tsmixup_1d_<min>-<max>_<N>samples/
#   <output-dir>/tsmixup_2d_<min>-<max>_<N>samples/
#   ...
#   <output-dir>/tsmixup_<D>d_<min>-<max>_<N>samples/
#
# Each folder contains a HuggingFace DatasetDict (train split) in Arrow format,
# plus statistics_summary.json, samples.png, correlation_matrices.png.
#
# Usage (defaults — uses kernel_synth_1m + Stage1 outputs as source):
#   bash stage2_tsmixup.sh
#
# Usage (custom):
#   bash stage2_tsmixup.sh \
#       --num-samples 1000000 --max-variates 5 \
#       --output-dir /group-volume/ts-dataset/chronos2_datasets \
#       --seed 42 --num-workers -1 \
#       --input-paths /path/to/source1 /path/to/source2
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/mv_synth/stage2_tsmixup.py"

# ── Default parameters ────────────────────────────────────────────────────────
MIN_LEN=64
MAX_LEN=1024
NUM_SAMPLES=1000
MAX_VARIATES=5
OUTPUT_DIR="/group-volume/ts-dataset/chronos2_datasets"
SEED=42
NUM_WORKERS=-1   # -1 = use all available CPUs

# Default input paths (Stage1 kernel_synth outputs + chronos1 training data)
DEFAULT_INPUT_PATHS=(
    "/group-volume/ts-dataset/chronos_datasets/training_corpus_kernel_synth_1m"
    "/group-volume/ts-dataset/chronos_datasets/training_corpus_tsmixup_10m"
    "/group-volume/ts-dataset/chronos2_datasets"   # Stage1 outputs
)

# ── Parse arguments ───────────────────────────────────────────────────────────
INPUT_PATHS=()
EXTRA_ARGS=()
PARSING_INPUTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-len)       MIN_LEN="$2";      shift 2; PARSING_INPUTS=false ;;
        --max-len)       MAX_LEN="$2";      shift 2; PARSING_INPUTS=false ;;
        --num-samples)   NUM_SAMPLES="$2";  shift 2; PARSING_INPUTS=false ;;
        --max-variates)  MAX_VARIATES="$2"; shift 2; PARSING_INPUTS=false ;;
        --output-dir)    OUTPUT_DIR="$2";   shift 2; PARSING_INPUTS=false ;;
        --seed)          SEED="$2";         shift 2; PARSING_INPUTS=false ;;
        --num-workers)   NUM_WORKERS="$2";  shift 2; PARSING_INPUTS=false ;;
        --input-paths)   PARSING_INPUTS=true; shift ;;
        *)
            if $PARSING_INPUTS; then
                INPUT_PATHS+=("$1")
            else
                EXTRA_ARGS+=("$1")
            fi
            shift ;;
    esac
done

# Fall back to defaults if no input paths provided
if [[ ${#INPUT_PATHS[@]} -eq 0 ]]; then
    INPUT_PATHS=("${DEFAULT_INPUT_PATHS[@]}")
fi

echo "======================================================================"
echo " Stage 2: TSMixup (multivariate synthesis from real + synthetic data)"
echo "======================================================================"
echo "  min-len      : ${MIN_LEN}"
echo "  max-len      : ${MAX_LEN}"
echo "  num-samples  : ${NUM_SAMPLES}"
echo "  max-variates : ${MAX_VARIATES}"
echo "  output-dir   : ${OUTPUT_DIR}"
echo "  seed         : ${SEED}"
echo "  num-workers  : ${NUM_WORKERS}"
echo "  input-paths  :"
for p in "${INPUT_PATHS[@]}"; do
    echo "    ${p}"
done
echo "======================================================================"

# Activate conda environment if available
if command -v conda &> /dev/null && conda env list | grep -q "^tsm "; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tsm
fi

python "${PYTHON_SCRIPT}" \
    --input-paths  "${INPUT_PATHS[@]}" \
    --min-len      "${MIN_LEN}" \
    --max-len      "${MAX_LEN}" \
    --num-samples  "${NUM_SAMPLES}" \
    --max-variates "${MAX_VARIATES}" \
    --output-dir   "${OUTPUT_DIR}" \
    --seed         "${SEED}" \
    --num-workers  "${NUM_WORKERS}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "======================================================================"
echo " Stage 2 complete. Outputs in: ${OUTPUT_DIR}"
echo "======================================================================"
