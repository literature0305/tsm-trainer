#!/usr/bin/env bash
# =============================================================================
# run_synth_pipeline.sh  — End-to-end synthetic data pipeline
# =============================================================================
#
# Generates datasets across 4 length ranges:
#
#   Range 1:  64 –  1024  → kernel_synth_3d_64-1024
#                           composite_synth_{2..MAX_N_DIM}d_64-1024
#   Range 2: 1024 – 2048  → kernel_synth_3d_1024-2048
#                           composite_synth_{2..MAX_N_DIM}d_1024-2048
#   Range 3: 2048 – 4096  → kernel_synth_3d_2048-4096
#                           composite_synth_{2..MAX_N_DIM}d_2048-4096
#   Range 4: 4096 – 8192  → kernel_synth_3d_4096-8192
#                           composite_synth_{2..MAX_N_DIM}d_4096-8192
#
# Kernel synthesis (3D) runs first; composite then sources from it.
# Composite generates separate folders per dimension (2D … MAX_N_DIM).
#
# Python scripts live in:  scripts/forecasting/preprocess/data_synth/
#
# USAGE
# -----
#   bash scripts/forecasting/preprocess/run_synth_pipeline.sh [OPTIONS]
#
#   Options (all optional, defaults shown):
#     --base_dir PATH              /group-volume/ts-dataset/chronos2_datasets
#     --n_per_range N N N N        1000 1000 500 250  (one per range, in order)
#     --max_n_dim INT              10   (max composite dimensionality)
#     --max_tokens INT             40000 (dim × length budget per sample)
#     --uncorrelated_ratio FLOAT   0.05
#     --noise_scale FLOAT          0.35
#     --hidden_regime_ratio FLOAT  0.05  (kernel synth only)
#     --n_workers INT              (auto = cpu_count)
#     --seed INT                   42
#     --cleanup_tmp                remove tmp dirs after Arrow conversion
#     --skip_kernel                skip all kernel synthesis steps
#     --skip_composite             skip all composite synthesis steps
#     --skip_ranges N...           skip specific range indices (1-based, e.g. 3 4)
#     --real_data_paths PATH...    extra data paths appended to every composite step
#
# EXAMPLE
# -------
#   # Full run with custom counts and workers (from repo root):
#   bash scripts/forecasting/preprocess/run_synth_pipeline.sh \
#       --n_per_range 5000 5000 2500 1000 \
#       --max_n_dim 8 \
#       --n_workers 16 \
#       --cleanup_tmp
#
#   # Regenerate only ranges 1 and 2 (skip 3 and 4):
#   bash scripts/forecasting/preprocess/run_synth_pipeline.sh --skip_ranges 3 4
#
#   # Only composite step, add extra real data:
#   bash scripts/forecasting/preprocess/run_synth_pipeline.sh \
#       --skip_kernel \
#       --real_data_paths \
#           /group-volume/ts-dataset/chronos_datasets/training_corpus_tsmixup_10m
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_SYNTH_DIR="${SCRIPT_DIR}/data_synth"
PYTHON="${PYTHON:-/root/miniconda3/envs/tsm/bin/python}"
PYTHONPATH_PREFIX="${SCRIPT_DIR%/scripts/*}"   # repo root

# ── Range definitions (fixed) ─────────────────────────────────────────────────
RANGE_NAMES=("64-1024" "1024-2048" "2048-4096" "4096-8192")
RANGE_MINS=(64 1024 2048 4096)
RANGE_MAXS=(1024 2048 4096 8192)

# ── Defaults ─────────────────────────────────────────────────────────────────
BASE_DIR="/group-volume/ts-dataset/chronos2_datasets"
N_PER_RANGE=(1000 1000 500 250)   # sample count per range / per dim
MAX_N_DIM=10                       # max composite dimensionality (2 .. MAX_N_DIM)
MAX_TOKENS=999999999               # dim × length budget for composite samples (deprecated) # TODO: remove this option
UNCORRELATED_RATIO=0.05
NOISE_SCALE=0.35
NEG_CORR_RATIO=0.35
HIDDEN_REGIME_RATIO=0.05
N_WORKERS=""                       # empty → cpu_count inside scripts
SEED=42
CLEANUP_TMP=""
SKIP_KERNEL=""
SKIP_COMPOSITE=""
SKIP_RANGES=()
REAL_DATA_PATHS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_dir)             BASE_DIR="$2";              shift 2 ;;
    --n_per_range)
      shift
      N_PER_RANGE=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        N_PER_RANGE+=("$1"); shift
      done
      ;;
    --max_n_dim)            MAX_N_DIM="$2";             shift 2 ;;
    --max_tokens)           MAX_TOKENS="$2";            shift 2 ;;
    --uncorrelated_ratio)   UNCORRELATED_RATIO="$2";   shift 2 ;;
    --noise_scale)          NOISE_SCALE="$2";           shift 2 ;;
    --neg_corr_ratio)       NEG_CORR_RATIO="$2";       shift 2 ;;
    --hidden_regime_ratio)  HIDDEN_REGIME_RATIO="$2";  shift 2 ;;
    --n_workers)            N_WORKERS="$2";             shift 2 ;;
    --seed)                 SEED="$2";                  shift 2 ;;
    --cleanup_tmp)          CLEANUP_TMP="--cleanup_tmp"; shift ;;
    --skip_kernel)          SKIP_KERNEL="1";            shift ;;
    --skip_composite)       SKIP_COMPOSITE="1";         shift ;;
    --skip_ranges)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SKIP_RANGES+=("$1"); shift
      done
      ;;
    --real_data_paths)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        REAL_DATA_PATHS+=("$1"); shift
      done
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Validate n_per_range ──────────────────────────────────────────────────────
if [[ ${#N_PER_RANGE[@]} -ne 4 ]]; then
  echo "[ERROR] --n_per_range requires exactly 4 values (one per length range)." >&2
  echo "  Example: --n_per_range 1000 1000 500 250" >&2
  exit 1
fi

WORKERS_ARG=""
[[ -n "$N_WORKERS" ]] && WORKERS_ARG="--n_workers $N_WORKERS"

# ── Helper: check whether a range index (1-based) is in SKIP_RANGES ──────────
is_skipped() {
  local idx="$1"
  for s in "${SKIP_RANGES[@]+"${SKIP_RANGES[@]}"}"; do
    [[ "$s" == "$idx" ]] && return 0
  done
  return 1
}

# ── Print banner ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  TSM-Trainer Synthetic Data Pipeline (ND Composite)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo "  Base dir         : $BASE_DIR"
echo "  Ranges           : ${RANGE_NAMES[*]}"
echo "  Samples/range    : ${N_PER_RANGE[*]}  (per dim for composite)"
echo "  Composite dims   : 2 – $MAX_N_DIM"
echo "  max_tokens       : $MAX_TOKENS"
echo "  uncorrelated     : $UNCORRELATED_RATIO"
echo "  noise_scale      : $NOISE_SCALE"
echo "  hidden_regime    : $HIDDEN_REGIME_RATIO  (kernel only)"
echo "  Seed             : $SEED"
[[ -n "$N_WORKERS" ]] && echo "  Workers          : $N_WORKERS" \
                       || echo "  Workers          : auto"
[[ -n "$SKIP_KERNEL" ]]    && echo "  skip_kernel      : YES"
[[ -n "$SKIP_COMPOSITE" ]] && echo "  skip_composite   : YES"
[[ ${#SKIP_RANGES[@]} -gt 0 ]] \
  && echo "  skip_ranges      : ${SKIP_RANGES[*]}"
echo "============================================================"
echo ""
echo "  Output datasets (will be created):"
for i in 0 1 2 3; do
  echo "    ${BASE_DIR}/kernel_synth_3d_${RANGE_NAMES[$i]}  (n=${N_PER_RANGE[$i]})"
  echo "    ${BASE_DIR}/composite_synth_2d..${MAX_N_DIM}d_${RANGE_NAMES[$i]}"
  echo "        (total=${N_PER_RANGE[$i]}, ~$((N_PER_RANGE[$i] / (MAX_N_DIM - 1)))/dim, $(( MAX_N_DIM - 1 )) folders)"
done
echo "============================================================"

# ── Main loop: one pass per length range ──────────────────────────────────────
for i in 0 1 2 3; do
  RANGE_IDX=$((i + 1))
  RANGE_NAME="${RANGE_NAMES[$i]}"
  MIN_LEN="${RANGE_MINS[$i]}"
  MAX_LEN="${RANGE_MAXS[$i]}"
  N="${N_PER_RANGE[$i]}"
  KERNEL_OUTPUT="${BASE_DIR}/kernel_synth_3d_${RANGE_NAME}"

  if is_skipped "$RANGE_IDX"; then
    echo ""
    echo ">>> RANGE $RANGE_IDX [${RANGE_NAME}]  — skipped (--skip_ranges)"
    continue
  fi

  echo ""
  echo "============================================================"
  echo "  RANGE $RANGE_IDX / 4  [${RANGE_NAME}]  length ${MIN_LEN}–${MAX_LEN}  n=${N}"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  # ── Step 1: Kernel synthesis (3D, unchanged) ───────────────────────────────
  if [[ -z "$SKIP_KERNEL" ]]; then
    echo ""
    echo "  >> [${RANGE_NAME}] Step 1: Kernel synthesis (3D)"
    echo "     $(date '+%H:%M:%S')"
    PYTHONPATH="${PYTHONPATH_PREFIX}/src:${PYTHONPATH:-}" \
    "$PYTHON" "${DATA_SYNTH_DIR}/generate_kernel_synth_3d.py" \
      --output_path         "$KERNEL_OUTPUT" \
      --length              "$MAX_LEN" \
      --n_datasets          "$N" \
      --uncorrelated_ratio  "$UNCORRELATED_RATIO" \
      --hidden_regime_ratio "$HIDDEN_REGIME_RATIO" \
      --min_length          "$MIN_LEN" \
      --seed                "$SEED" \
      $WORKERS_ARG \
      $CLEANUP_TMP
    echo "     Step 1 done: $(date '+%H:%M:%S')"
  else
    echo "  >> [${RANGE_NAME}] Step 1 skipped (--skip_kernel)"
  fi

  # ── Step 2: Composite synthesis (variable dim) ─────────────────────────────
  if [[ -z "$SKIP_COMPOSITE" ]]; then
    echo ""
    echo "  >> [${RANGE_NAME}] Step 2: Composite synthesis (dim 2–${MAX_N_DIM})"
    echo "     $(date '+%H:%M:%S')"

    COMPOSITE_SOURCES=()
    if [[ -d "$KERNEL_OUTPUT" ]]; then
      COMPOSITE_SOURCES+=("$KERNEL_OUTPUT")
    else
      echo "  [WARN] Kernel output not found: $KERNEL_OUTPUT" >&2
      echo "         Run without --skip_kernel first, or the composite may be empty." >&2
    fi
    [[ ${#REAL_DATA_PATHS[@]} -gt 0 ]] && COMPOSITE_SOURCES+=("${REAL_DATA_PATHS[@]}")

    if [[ ${#COMPOSITE_SOURCES[@]} -eq 0 ]]; then
      echo "  [WARN] No data sources for composite [${RANGE_NAME}] — skipping." >&2
    else
      PYTHONPATH="${PYTHONPATH_PREFIX}/src:${PYTHONPATH:-}" \
      "$PYTHON" "${DATA_SYNTH_DIR}/generate_composite_synth_nd.py" \
        --output_dir         "$BASE_DIR" \
        --range_name         "$RANGE_NAME" \
        --data_paths         "${COMPOSITE_SOURCES[@]}" \
        --n_datasets         "$N" \
        --min_dim            2 \
        --max_dim            "$MAX_N_DIM" \
        --max_tokens         "$MAX_TOKENS" \
        --uncorrelated_ratio "$UNCORRELATED_RATIO" \
        --noise_scale        "$NOISE_SCALE" \
        --neg_corr_ratio     "$NEG_CORR_RATIO" \
        --min_length         "$MIN_LEN" \
        --max_length         "$MAX_LEN" \
        --seed               "$SEED" \
        $WORKERS_ARG \
        $CLEANUP_TMP
      echo "     Step 2 done: $(date '+%H:%M:%S')"
    fi
  else
    echo "  >> [${RANGE_NAME}] Step 2 skipped (--skip_composite)"
  fi

done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Pipeline DONE  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Generated datasets:"
for i in 0 1 2 3; do
  RANGE_NAME="${RANGE_NAMES[$i]}"
  N="${N_PER_RANGE[$i]}"
  KOUT="${BASE_DIR}/kernel_synth_3d_${RANGE_NAME}"
  K_DONE=""
  [[ -d "$KOUT" ]] && K_DONE="✓" || K_DONE="✗ (missing)"
  echo "    ${K_DONE} kernel_synth_3d_${RANGE_NAME}  (n=${N})"

  # Check per-dim composite folders
  COMPOSITE_OK=0
  COMPOSITE_MISS=0
  for d in $(seq 2 "$MAX_N_DIM"); do
    COUT="${BASE_DIR}/composite_synth_${d}d_${RANGE_NAME}"
    if [[ -d "$COUT" ]]; then
      COMPOSITE_OK=$(( COMPOSITE_OK + 1 ))
    else
      COMPOSITE_MISS=$(( COMPOSITE_MISS + 1 ))
    fi
  done
  TOTAL_DIMS=$(( MAX_N_DIM - 1 ))
  if [[ $COMPOSITE_MISS -eq 0 ]]; then
    echo "    ✓ composite_synth_2d..${MAX_N_DIM}d_${RANGE_NAME}  (${TOTAL_DIMS} folders, n=${N}/dim)"
  elif [[ $COMPOSITE_OK -eq 0 ]]; then
    echo "    ✗ composite_synth_*d_${RANGE_NAME}  (all ${TOTAL_DIMS} missing)"
  else
    echo "    ~ composite_synth_*d_${RANGE_NAME}  (${COMPOSITE_OK}/${TOTAL_DIMS} done)"
  fi
done
echo "============================================================"
