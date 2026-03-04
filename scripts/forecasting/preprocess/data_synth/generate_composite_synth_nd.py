#!/usr/bin/env python3
"""
generate_composite_synth_nd.py — Variable-Dimension Composite Synthesizer
==========================================================================

Generates N-dimensional multivariate time series (N ∈ [min_dim, max_dim]) by
combining real/synthetic source series through higher-order mixing, producing
one HuggingFace Arrow dataset per target dimension.

OUTPUT
------
For each dimension d in [min_dim, max_dim]:
  {output_dir}/composite_synth_{d}d_{range_name}/   ← HF Arrow DatasetDict

Aggregate statistics (written to output_dir):
  composite_{range_name}_stats.csv
  composite_{range_name}_stats.png    ← length/std/corr distributions by dim
  composite_{range_name}_samples.png  ← 10 random sample time-series plots

ALGORITHM (per sample, target dimension = n)
--------------------------------------------
1.  k, h = _pick_k_h(n):
      k = # direct variates (1 ≤ k < n), included verbatim in output.
      h = # latent variates (h ≥ 1, k+h ≥ n), drive synthesis only.
2.  Fetch k+h series from the variate pool; align each to target_L (a
    multiple of 64) via random crop or tile/interpolation.
3.  For each of the (n−k) synthesized dimensions, apply one of:
      lag_lead      — α·roll(s1, τ) + (1−α)·s2
      weighted_sum  — Σwᵢ·sᵢ over 2–4 sources
      causal_filter — exponential FIR on s1, mixed with s2
      piecewise_mix — segment-by-segment alternation (s1 ↔ s2)
      time_warp     — monotonic time-axis warp of s1, mixed with s2
      nonlinear_mix — tanh / abs-power / clip nonlinearity on weighted mix
    All synthesis is in z-score space. Latent sources are weighted 2× over
    direct sources to reduce cross-variate correlation in the output.
    Result is denormalised to the mean scale of the k direct variates.
4.  target_L (multiple of 64) satisfies:
      n × L < max_tokens  (default 40 000)
      min_length ≤ L ≤ max_length
    L is chosen stochastically: 40% near min input length, 40% near max,
    20% uniform in the valid range.

USAGE
-----
python generate_composite_synth_nd.py \\
    --data_paths /path/to/source1 /path/to/source2 \\
    --output_dir /group-volume/ts-dataset/chronos2_datasets \\
    --range_name 1024-2048 \\
    --n_datasets 1000 \\
    --min_dim 2 --max_dim 10 \\
    --max_tokens 40000 \\
    --min_length 64 --max_length 8192 \\
    --uncorrelated_ratio 0.05 \\
    --n_workers 8 --seed 42 [--cleanup_tmp]
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import time
from collections import Counter
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SYNTH_METHODS = [
    "lag_lead", "weighted_sum", "causal_filter",
    "piecewise_mix", "time_warp", "nonlinear_mix",
]


# ──────────────────────────────────────────────────────────────────────────────
# Variate Pool
# ──────────────────────────────────────────────────────────────────────────────

class VariatePool:
    """Random-access pool of 1-D time series drawn from multiple HF Arrow datasets."""

    def __init__(self, data_paths: list[str], min_length: int) -> None:
        import datasets as hf_datasets

        self._datasets: list = []
        self._n_variates: list[int] = []
        self._valid_rows: list[np.ndarray] = []   # pre-filtered row indices per dataset
        self._cum: list[int] = [0]
        self._min_length = min_length

        for path in data_paths:
            try:
                raw = hf_datasets.load_from_disk(str(path))
                ds = raw.get("train", next(iter(raw.values()))) if hasattr(raw, "keys") else raw
            except Exception as exc:
                logger.warning("Cannot load %s: %s — skipping.", path, exc)
                continue

            n_var = _detect_n_variates(ds)
            valid = _build_length_filter(ds, min_length, n_var)
            if len(valid) == 0:
                logger.warning(
                    "  Pool SKIP %-50s  (no rows with length >= %d)",
                    Path(path).name, min_length,
                )
                continue

            pool_size = len(valid) * n_var
            self._datasets.append(ds)
            self._n_variates.append(n_var)
            self._valid_rows.append(valid)
            self._cum.append(self._cum[-1] + pool_size)
            logger.info("  Pool +  %-55s  valid=%d/%d  variates/row=%d",
                        Path(path).name, len(valid), len(ds), n_var)

        if self.total == 0:
            raise RuntimeError("Variate pool is empty — check --data_paths.")

    @property
    def total(self) -> int:
        return self._cum[-1]

    def get_variate(self, global_idx: int) -> np.ndarray | None:
        global_idx = int(global_idx) % self.total
        ds_idx = next(
            (i for i in range(len(self._datasets))
             if self._cum[i] <= global_idx < self._cum[i + 1]),
            None,
        )
        if ds_idx is None:
            return None
        local = global_idx - self._cum[ds_idx]
        n_var = self._n_variates[ds_idx]
        valid_rows = self._valid_rows[ds_idx]
        local_row_rank, var_idx = divmod(local, n_var)
        row_idx = int(valid_rows[local_row_rank % len(valid_rows)])
        try:
            row = self._datasets[ds_idx][row_idx]
            tgt = np.array(row["target"], dtype=np.float64)
            series = tgt if tgt.ndim == 1 else (tgt[var_idx] if tgt.ndim == 2 else None)
            if series is None or len(series) < self._min_length:
                return None
            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                return None
            return series
        except Exception:
            return None


def _detect_n_variates(ds) -> int:
    if len(ds) == 0:
        return 1
    sample = ds[0]["target"]
    if isinstance(sample, (list, np.ndarray)) and len(sample) > 0:
        first = sample[0]
        if isinstance(first, (list, np.ndarray)):
            return len(sample)
    return 1


def _build_length_filter(ds, min_length: int, n_var: int) -> np.ndarray:
    """Return row indices where series length >= min_length.

    Fast path: PyArrow vectorized list_size for univariate datasets (O(N) in C).
    Fallback:  sample up to 50 K rows for multivariate or on PyArrow error.
    If ALL sampled rows pass, all rows are assumed valid → returns np.arange(len(ds)).
    """
    n_total = len(ds)

    if n_var == 1:
        try:
            import pyarrow as pa
            # Read list sizes from offset buffers only — zero data materialisation.
            # Arrow ListArray layout: buffers = [validity, offsets(int32), ...]
            # LargeListArray: offsets are int64.
            tgt_col = ds.data.column("target")   # ChunkedArray[ListArray]
            sizes_parts: list[np.ndarray] = []
            for chunk in tgt_col.chunks:
                bufs = chunk.buffers()
                # bufs[0] = validity bitmap (may be None), bufs[1] = offsets
                offsets_buf = bufs[1]
                if offsets_buf is None:
                    raise ValueError("no offsets buffer")
                is_large = pa.types.is_large_list(chunk.type)
                dtype = np.int64 if is_large else np.int32
                offs = np.frombuffer(offsets_buf, dtype=dtype)
                sizes_parts.append(np.diff(offs).astype(np.int64))
            sizes_np = np.concatenate(sizes_parts) if sizes_parts else np.array([], np.int64)
            return np.where(sizes_np >= min_length)[0].astype(np.int64)
        except Exception:
            pass  # fall through to sampling

    # Sampling fallback (multivariate or PyArrow unavailable)
    n_scan = min(n_total, 50_000)
    rng0 = np.random.default_rng(0)
    sample_idx = (
        np.sort(rng0.choice(n_total, n_scan, replace=False))
        if n_scan < n_total else np.arange(n_total)
    )
    valid: list[int] = []
    for idx in sample_idx:
        try:
            tgt = np.array(ds[int(idx)]["target"], dtype=np.float64)
            if tgt.shape[-1] >= min_length:
                valid.append(int(idx))
        except Exception:
            pass
    # If every sampled row is valid, conservatively assume all rows are valid
    if len(valid) == len(sample_idx):
        return np.arange(n_total, dtype=np.int64)
    return np.array(valid, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Time-series transformations
# ──────────────────────────────────────────────────────────────────────────────

def _apply_fir(src: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import lfilter
    fl = int(rng.integers(5, 50))
    alpha = rng.uniform(0.50, 0.95)
    h = alpha ** np.arange(fl)
    h /= h.sum()
    return lfilter(h, [1.0], src)


def _time_warp(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Monotonic piecewise-linear time-axis warp."""
    L = len(arr)
    n_ctrl = int(rng.integers(4, 12))
    x_ctrl = np.sort(rng.uniform(0, 1, n_ctrl))
    y_ctrl = np.sort(rng.uniform(0, 1, n_ctrl))
    x_ctrl = np.concatenate([[0.0], x_ctrl, [1.0]])
    y_ctrl = np.concatenate([[0.0], y_ctrl, [1.0]])
    t_norm = np.linspace(0, 1, L)
    t_src = np.interp(t_norm, x_ctrl, y_ctrl) * (L - 1)
    return np.interp(t_src, np.arange(L), arr)


def _piecewise_mix(s1: np.ndarray, s2: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """Segment-by-segment alternation between two series."""
    L = len(s1)
    n_breaks = min(int(rng.integers(3, 12)), max(2, L // 16))
    bp = np.sort(rng.choice(np.arange(1, L), size=min(n_breaks, L - 1), replace=False))
    segs = np.concatenate([[0], bp, [L]])
    out = np.empty(L)
    use_s1 = bool(rng.random() < 0.5)
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        out[a:b] = s1[a:b] if use_s1 else s2[a:b]
        use_s1 = not use_s1
    return out


def _align_to_64(arr: np.ndarray, target_L: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Crop or stretch 1-D array to exactly target_L timesteps (multiple of 64)."""
    L = len(arr)
    if L == target_L:
        return arr.copy()
    if L > target_L:
        start = int(rng.integers(0, L - target_L + 1))
        return arr[start: start + target_L].copy()
    # Too short: tile+noise or linear interpolation
    if rng.random() < 0.5:
        n_tile = target_L // L + 1
        out = np.tile(arr, n_tile)[:target_L]
        out += rng.standard_normal(target_L) * max(np.std(arr) * 0.02, 1e-8)
        return out
    old_idx = np.linspace(0, L - 1, target_L)
    return np.interp(old_idx, np.arange(L), arr)


# ──────────────────────────────────────────────────────────────────────────────
# Length & dimension helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pick_target_length(
    raw_lengths: list[int],
    n_dim: int,
    min_length: int,
    max_length: int,
    max_tokens: int,
    rng: np.random.Generator,
) -> int | None:
    """Return a target length that is a positive multiple of 64."""
    L_budget = (max_tokens - 1) // n_dim
    L_hard_max = min(max_length, L_budget)
    L_floor_max = (L_hard_max // 64) * 64
    L_ceil_min = ((min_length + 63) // 64) * 64
    if L_floor_max < L_ceil_min or L_floor_max < 64:
        return None

    min_raw = min(raw_lengths)
    max_raw = max(raw_lengths)
    choice = rng.random()
    if choice < 0.40:
        L = (min(min_raw, L_floor_max) // 64) * 64
    elif choice < 0.80:
        L = (min(max_raw, L_floor_max) // 64) * 64
    else:
        n_opts = (L_floor_max - L_ceil_min) // 64 + 1
        L = L_ceil_min + int(rng.integers(0, n_opts)) * 64

    L = max(L_ceil_min, min(L_floor_max, (L // 64) * 64))
    return L if (L_ceil_min <= L <= L_floor_max and L % 64 == 0) else None


def _pick_k_h(n_dim: int, rng: np.random.Generator) -> tuple[int, int]:
    """Pick k (direct) and h (latent) variates with k<n, h≥1, k+h≥n."""
    k = int(rng.integers(1, n_dim))   # [1, n_dim-1]
    n_synth = n_dim - k
    h_min = max(1, n_synth)
    h_max = max(h_min, n_dim - 1)
    h = int(rng.integers(h_min, h_max + 1))
    return k, h


# ──────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ──────────────────────────────────────────────────────────────────────────────

_POOL: VariatePool | None = None
_WORKER_CFG: dict = {}


def _worker_init(data_paths: list[str], cfg: dict) -> None:
    global _POOL, _WORKER_CFG
    _WORKER_CFG = cfg
    _POOL = VariatePool(data_paths, cfg["min_length"])


def _fetch_variate(rng: np.random.Generator, max_attempts: int = 40) -> np.ndarray | None:
    for _ in range(max_attempts):
        idx = int(rng.integers(0, _POOL.total))
        v = _POOL.get_variate(idx)
        if v is not None:
            return v
    return None


def _norm(s: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (z-scored array, mean, std)."""
    mu, sigma = float(np.mean(s)), float(np.std(s))
    sigma = max(sigma, 1e-6)
    return (s - mu) / sigma, mu, sigma


def _synthesize_variate(
    direct_norm: list[np.ndarray],
    latent_norm: list[np.ndarray],
    rng: np.random.Generator,
    uncorrelated_ratio: float,
    noise_scale: float,
    neg_corr_ratio: float = 0.35,
) -> tuple[np.ndarray, str]:
    """Generate one synthesized variate in z-score space.

    Latent sources are weighted 2× to reduce correlation with direct outputs.
    neg_corr_ratio controls the fraction of variates that are negatively
    correlated with their sources (sign-flipped after synthesis).
    Returns (normalized_series, method_name).
    """
    L = len(direct_norm[0])
    # Build source pool: latent weighted 2× over direct
    source_pool = latent_norm * 2 + direct_norm

    def _pick() -> np.ndarray:
        return source_pool[int(rng.integers(0, len(source_pool)))]

    if rng.random() < uncorrelated_ratio:
        # Fully independent — use only a latent source + large noise
        src = latent_norm[int(rng.integers(0, len(latent_norm)))]
        noise_std = rng.uniform(0.20, 0.50)
        v = src + rng.standard_normal(L) * noise_std
        return v, "independent"

    method = SYNTH_METHODS[int(rng.integers(0, len(SYNTH_METHODS)))]
    s1, s2 = _pick(), _pick()

    if method == "lag_lead":
        lag = int(rng.integers(1, max(2, L // 8)))
        if rng.random() < 0.5:
            lag = -lag
        # Allow negative alpha → inverted lag-lead (negatively correlated)
        alpha = rng.uniform(-0.8, 0.8)
        v = alpha * np.roll(s1, lag) + (1.0 - abs(alpha)) * s2

    elif method == "weighted_sum":
        n_srcs = min(len(source_pool), int(rng.integers(2, 5)))
        srcs = [_pick() for _ in range(n_srcs)]
        # Allow mixed signs: ~35% of weights are negative
        weights = rng.dirichlet(np.ones(n_srcs))
        signs = rng.choice([-1.0, 1.0], size=n_srcs,
                           p=[neg_corr_ratio, 1.0 - neg_corr_ratio])
        weights = weights * signs
        v = sum(w * s for w, s in zip(weights, srcs))

    elif method == "causal_filter":
        filtered = _apply_fir(s1, rng)
        v = 0.6 * filtered + 0.4 * s2

    elif method == "piecewise_mix":
        v = _piecewise_mix(s1, s2, rng)

    elif method == "time_warp":
        s1_warped = _time_warp(s1, rng)
        alpha = rng.uniform(0.3, 0.7)
        v = alpha * s1_warped + (1.0 - alpha) * s2

    else:  # nonlinear_mix
        alpha = rng.uniform(0.3, 0.7)
        mixed = alpha * s1 + (1.0 - alpha) * s2
        nl = rng.choice(["tanh", "abs_power", "clip"])
        if nl == "tanh":
            v = np.tanh(mixed * rng.uniform(0.5, 2.0))
        elif nl == "abs_power":
            p = rng.uniform(0.5, 2.0)
            v = np.sign(mixed) * (np.abs(mixed) ** p)
            v_sigma = max(float(np.std(v)), 1e-6)
            v = (v - float(np.mean(v))) / v_sigma
        else:  # clip
            v = np.clip(mixed, -1.5, 1.5)

    # Add noise and re-normalize to unit variance
    noise_std = rng.uniform(0.05, noise_scale)
    v = v + rng.standard_normal(L) * noise_std
    v_sigma = max(float(np.std(v)), 1e-6)
    v = (v - float(np.mean(v))) / v_sigma

    # Sign flip: neg_corr_ratio fraction of variates become negatively correlated.
    # Applied AFTER noise/normalisation so the magnitude is unchanged.
    # Excludes causal_filter, piecewise_mix, time_warp which already mix two
    # independent sources (their sign relationship is less deterministic).
    if method in ("lag_lead", "weighted_sum", "nonlinear_mix"):
        pass   # handled per-method above or below
    if rng.random() < neg_corr_ratio and method in ("causal_filter",
                                                      "piecewise_mix",
                                                      "time_warp"):
        v = -v

    return v, method


def generate_composite_sample_nd(task: tuple) -> tuple:
    """Worker: generate one composite sample for the given n_dim.

    Returns ((sample_id, n_dim), data_array | None, status_string).
    """
    sample_id, n_dim = task
    cfg = _WORKER_CFG
    rng = np.random.default_rng(cfg["base_seed"] + sample_id * 7_919 + n_dim * 97)

    min_length        = cfg["min_length"]
    max_length        = cfg["max_length"]
    max_tokens        = cfg["max_tokens"]
    uncorrelated_ratio = cfg["uncorrelated_ratio"]
    noise_scale       = cfg.get("noise_scale", 0.35)
    neg_corr_ratio    = cfg.get("neg_corr_ratio", 0.35)

    try:
        # ── Step 1: pick k (direct) and h (latent) ─────────────────────────
        k, h = _pick_k_h(n_dim, rng)
        n_synth = n_dim - k

        # ── Step 2: fetch k+h series ────────────────────────────────────────
        wanted = k + h
        fetched: list[np.ndarray] = []
        for _ in range(wanted * 4):
            if len(fetched) >= wanted:
                break
            v = _fetch_variate(rng)
            if v is not None:
                fetched.append(v)
        if len(fetched) < wanted:
            return (sample_id, n_dim), None, "pool_miss"

        direct_raw = fetched[:k]
        latent_raw = fetched[k: k + h]

        # ── Step 3: pick target length (multiple of 64) ─────────────────────
        all_lens = [len(s) for s in fetched]
        target_L = _pick_target_length(
            all_lens, n_dim, min_length, max_length, max_tokens, rng
        )
        if target_L is None:
            return (sample_id, n_dim), None, "length_constraint"

        # ── Step 4: align all series to target_L ────────────────────────────
        direct_aligned = [_align_to_64(s, target_L, rng) for s in direct_raw]
        latent_aligned = [_align_to_64(s, target_L, rng) for s in latent_raw]

        # Compute mean scale of direct variates for denormalization
        direct_stats = [_norm(s) for s in direct_aligned]
        direct_norm  = [x[0] for x in direct_stats]
        direct_mus   = [x[1] for x in direct_stats]
        direct_sigs  = [x[2] for x in direct_stats]
        latent_norm  = [_norm(s)[0] for s in latent_aligned]

        denorm_std  = float(np.mean(direct_sigs))
        denorm_mean = float(np.mean(direct_mus))

        # ── Step 5: synthesize (n_dim − k) variates ─────────────────────────
        synthesized: list[np.ndarray] = []
        last_method = "direct"
        for _ in range(n_synth):
            v_norm, method = _synthesize_variate(
                direct_norm, latent_norm, rng, uncorrelated_ratio, noise_scale,
                neg_corr_ratio=neg_corr_ratio,
            )
            # Denormalize to direct variate scale
            synthesized.append(v_norm * denorm_std + denorm_mean)
            last_method = method

        # ── Step 6: stack and validate ──────────────────────────────────────
        Y = np.stack(direct_aligned + synthesized).astype(np.float64)  # (n_dim, L)

        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            return (sample_id, n_dim), None, "nan_or_inf"
        if np.any(np.var(Y, axis=1) < 1e-12):
            return (sample_id, n_dim), None, "zero_variance"

        return (sample_id, n_dim), Y, last_method

    except Exception as exc:
        return (sample_id, n_dim), None, f"error:{exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Arrow conversion
# ──────────────────────────────────────────────────────────────────────────────

def _build_timestamps(length: int) -> list[datetime]:
    epoch = datetime(1970, 1, 1)
    h1 = timedelta(hours=1)
    return [epoch + h1 * i for i in range(length)]


def convert_to_arrow(tmp_dir: Path, output_path: Path, dim: int) -> None:
    import datasets as hf_datasets

    tmp_files = sorted(tmp_dir.glob("sample_*.npz"))
    if not tmp_files:
        logger.warning("No samples in %s — skipping Arrow conversion.", tmp_dir)
        return
    logger.info("Converting %d samples (dim=%d) to HF Arrow …", len(tmp_files), dim)

    targets_list, ids_list, timestamps_list = [], [], []
    _ts_cache: dict[int, list] = {}

    for f in tmp_files:
        Y = np.load(f)["target"]   # (dim, L)
        L = Y.shape[1]
        sid = int(f.stem.split("_")[1])
        targets_list.append(Y.tolist())
        ids_list.append(f"{dim}DC_{sid:08d}")
        if L not in _ts_cache:
            _ts_cache[L] = _build_timestamps(L)
        timestamps_list.append(_ts_cache[L])

    features = hf_datasets.Features({
        "target":    hf_datasets.Sequence(hf_datasets.Sequence(hf_datasets.Value("float64"))),
        "id":        hf_datasets.Value("string"),
        "timestamp": hf_datasets.Sequence(hf_datasets.Value("timestamp[ms]")),
    })
    ds = hf_datasets.Dataset.from_dict(
        {"target": targets_list, "id": ids_list, "timestamp": timestamps_list},
        features=features,
    )
    hf_datasets.DatasetDict({"train": ds}).save_to_disk(str(output_path))
    logger.info("Saved → %s  (%d rows)", output_path, len(ds))


# ──────────────────────────────────────────────────────────────────────────────
# Statistics collection
# ──────────────────────────────────────────────────────────────────────────────

def _collect_dim_stats(tmp_dir: Path, dim: int) -> dict:
    """Collect statistics from all .npz files for one dimension."""
    files = sorted(tmp_dir.glob("sample_*.npz"))
    if not files:
        return {}

    lengths, all_stds, all_corrs = [], [], []
    for f in files:
        Y = np.load(f)["target"].astype(np.float64)   # (dim, L)
        lengths.append(Y.shape[1])
        all_stds.extend(float(s) for s in np.std(Y, axis=1))
        if dim > 1:
            C = np.corrcoef(Y)
            n = C.shape[0]
            off = [abs(C[i, j]) for i in range(n) for j in range(n) if i != j]
            all_corrs.append(float(np.mean(off)))

    la = np.array(lengths)
    sa = np.array(all_stds)
    return {
        "dim":         dim,
        "n":           len(lengths),
        "lengths":     la,
        "stds":        sa,
        "corrs":       np.array(all_corrs) if all_corrs else np.array([]),
        "len_min":     int(la.min()),
        "len_mean":    float(la.mean()),
        "len_median":  float(np.median(la)),
        "len_max":     int(la.max()),
        "std_median":  float(np.median(sa)),
        "corr_mean":   float(np.mean(all_corrs)) if all_corrs else float("nan"),
    }


def save_statistics_csv(
    stats_by_dim: dict[int, dict],
    method_counts_by_dim: dict[int, Counter],
    out_path: Path,
) -> None:
    rows = []
    all_methods = sorted({m for cnt in method_counts_by_dim.values() for m in cnt})
    for dim, st in sorted(stats_by_dim.items()):
        if not st:
            continue
        row = {
            "dim":        dim,
            "n_samples":  st["n"],
            "len_min":    st["len_min"],
            "len_mean":   f"{st['len_mean']:.1f}",
            "len_median": f"{st['len_median']:.0f}",
            "len_max":    st["len_max"],
            "std_median": f"{st['std_median']:.4f}",
            "corr_mean":  f"{st['corr_mean']:.4f}" if not np.isnan(st["corr_mean"]) else "nan",
        }
        cnt = method_counts_by_dim.get(dim, Counter())
        total = max(sum(cnt.values()), 1)
        for m in all_methods:
            row[f"pct_{m}"] = f"{100 * cnt[m] / total:.1f}"
        rows.append(row)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Stats CSV → %s", out_path)


def save_statistics_png(stats_by_dim: dict[int, dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = sorted(d for d, st in stats_by_dim.items() if st)
    cmap = plt.cm.tab10
    colors = {d: cmap(i % 10) for i, d in enumerate(dims)}

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Composite Synth ND — Statistics by Dimension", fontsize=13)
    ax_len, ax_std, ax_corr = axes[0]
    ax_len_box, ax_std_box, ax_summary = axes[1]

    # Row 0: histograms
    for d in dims:
        st = stats_by_dim[d]
        c = colors[d]
        lbl = f"{d}D (n={st['n']})"
        ax_len.hist(st["lengths"], bins=30, density=True, alpha=0.5, color=c, label=lbl)
        ax_std.hist(st["stds"], bins=30, density=True, alpha=0.5, color=c)
        if len(st["corrs"]) > 0:
            ax_corr.hist(st["corrs"], bins=20, density=True, alpha=0.5, color=c, label=lbl)

    ax_len.set_title("Length Distribution by Dim", fontsize=9)
    ax_len.set_xlabel("series length"); ax_len.legend(fontsize=6)
    ax_std.set_title("Std Dev Distribution (all variates)", fontsize=9)
    ax_std.set_xlabel("std dev")
    ax_corr.set_title("Mean |Cross-variate Corr|", fontsize=9)
    ax_corr.set_xlabel("|correlation|"); ax_corr.legend(fontsize=6)

    # Row 1: boxplots
    ax_len_box.boxplot(
        [stats_by_dim[d]["lengths"] for d in dims],
        tick_labels=[str(d) for d in dims], showfliers=False,
    )
    ax_len_box.set_title("Length by Dim (boxplot)", fontsize=9)
    ax_len_box.set_xlabel("n_dim")

    ax_std_box.boxplot(
        [stats_by_dim[d]["stds"] for d in dims],
        tick_labels=[str(d) for d in dims], showfliers=False,
    )
    ax_std_box.set_title("Std Dev by Dim (boxplot)", fontsize=9)
    ax_std_box.set_xlabel("n_dim")

    # Summary text
    ax_summary.axis("off")
    lines = ["Dimension Summary\n"]
    for d in dims:
        st = stats_by_dim[d]
        corr_str = f"{st['corr_mean']:.3f}" if not np.isnan(st["corr_mean"]) else "n/a"
        lines += [
            f"[{d}D]  n={st['n']}",
            f"  L: {st['len_min']}–{st['len_median']:.0f}–{st['len_max']}",
            f"  std_med={st['std_median']:.3f}  |corr|={corr_str}",
            "",
        ]
    ax_summary.text(
        0.04, 0.98, "\n".join(lines),
        transform=ax_summary.transAxes, fontsize=7, va="top", ha="left",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F5F5", "edgecolor": "#BDBDBD"},
    )

    for ax in axes.flat:
        ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Stats PNG → %s", out_path)
    plt.close(fig)


def save_samples_png(
    stats_by_dim: dict[int, dict],
    all_tmp_dirs: dict[int, Path],
    out_path: Path,
    n_samples: int = 10,
    rng_seed: int = 0,
) -> None:
    """Plot n_samples random time series (spread across available dims)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(rng_seed)
    dims_with_data = [d for d, st in stats_by_dim.items() if st and st["n"] > 0]
    if not dims_with_data:
        return

    # Pick samples round-robin across dims
    chosen: list[tuple[int, Path]] = []
    pool_by_dim = {d: sorted(all_tmp_dirs[d].glob("sample_*.npz")) for d in dims_with_data}
    while len(chosen) < n_samples:
        for d in dims_with_data:
            files = pool_by_dim[d]
            if files:
                f = files[int(rng.integers(0, len(files)))]
                chosen.append((d, f))
            if len(chosen) >= n_samples:
                break

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle("Composite Synth ND — Random Samples", fontsize=12)

    cmap = plt.cm.tab10
    for ax, (d, f) in zip(axes, chosen):
        Y = np.load(f)["target"].astype(np.float64)   # (d, L)
        L = Y.shape[1]
        for j in range(Y.shape[0]):
            ax.plot(Y[j], lw=0.8, alpha=0.85, color=cmap(j % 10),
                    label=f"v{j+1}")
        ax.set_title(f"dim={d}  L={L}", fontsize=8, loc="left")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right", ncol=min(d, 5))
        ax.grid(True, lw=0.2, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    logger.info("Samples PNG → %s", out_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate variable-dimension composite multivariate time series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data_paths", nargs="+", required=True,
                   help="Source HF Arrow dataset paths")
    p.add_argument("--output_dir", required=True,
                   help="Base directory; per-dim folders created inside")
    p.add_argument("--range_name", required=True,
                   help="Range tag used in folder names, e.g. '1024-2048'")
    p.add_argument("--n_datasets",  type=int,   default=1_000,
                   help="Total samples across all dimensions; each dim gets "
                        "n_datasets // (max_dim - min_dim + 1) samples (default: 1000)")
    p.add_argument("--min_dim",     type=int,   default=2,
                   help="Minimum output dimensionality (default: 2)")
    p.add_argument("--max_dim",     type=int,   default=10,
                   help="Maximum output dimensionality (default: 10)")
    p.add_argument("--max_tokens",  type=int,   default=40_000,
                   help="Max dim × length budget (default: 40000)")
    p.add_argument("--uncorrelated_ratio", type=float, default=0.05,
                   help="Fraction of independent (uncorrelated) variates (default: 0.05)")
    p.add_argument("--noise_scale", type=float, default=0.35,
                   help="Max synthesis noise std in z-score space (default: 0.35)")
    p.add_argument("--neg_corr_ratio", type=float, default=0.35,
                   help="Fraction of synthesized variates that are negatively correlated "
                        "with their sources (default: 0.35)")
    p.add_argument("--n_workers",   type=int,   default=None,
                   help="Worker processes (default: cpu_count)")
    p.add_argument("--min_length",  type=int,   default=64,
                   help="Minimum output series length (default: 64)")
    p.add_argument("--max_length",  type=int,   default=8_192,
                   help="Maximum output length before token budget (default: 8192)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--cleanup_tmp", action="store_true",
                   help="Remove tmp directories after Arrow conversion")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers or cpu_count()

    cfg = {
        "min_length":          args.min_length,
        "max_length":          args.max_length,
        "max_tokens":          args.max_tokens,
        "uncorrelated_ratio":  args.uncorrelated_ratio,
        "noise_scale":         args.noise_scale,
        "neg_corr_ratio":      args.neg_corr_ratio,
        "base_seed":           args.seed,
    }

    # ── Compute per-dim sample count ────────────────────────────────────────
    dims = list(range(args.min_dim, args.max_dim + 1))
    n_total_dims = len(dims)
    n_per_dim = max(1, args.n_datasets // n_total_dims)

    logger.info("=" * 60)
    logger.info("Composite Synth ND  |  range=%s  dim=%d–%d",
                args.range_name, args.min_dim, args.max_dim)
    logger.info("  n_datasets(total)=%d  n_dims=%d  n_per_dim=%d",
                args.n_datasets, n_total_dims, n_per_dim)
    logger.info("  max_tokens=%d  noise_scale=%.2f  uncorr_ratio=%.3f",
                args.max_tokens, args.noise_scale, args.uncorrelated_ratio)

    # Log pool info once before launching per-dim pools
    logger.info("Scanning input datasets:")
    _dummy = VariatePool(args.data_paths, args.min_length)
    logger.info("Total variate pool size: %d", _dummy.total)
    del _dummy
    tmp_dirs: dict[int, Path] = {}
    out_paths: dict[int, Path] = {}
    for d in dims:
        tag = f"composite_synth_{d}d_{args.range_name}"
        out_paths[d] = output_dir / tag
        tmp_dirs[d] = output_dir / (tag + "_tmp")
        tmp_dirs[d].mkdir(parents=True, exist_ok=True)

    stats_by_dim: dict[int, dict] = {}
    method_counts_by_dim: dict[int, Counter] = {}
    t_total = time.time()

    # ── Determine which dims still need generation ───────────────────────────
    dims_todo: list[int] = []
    for d in dims:
        out_path = out_paths[d]
        if out_path.exists():
            try:
                import datasets as hf_datasets
                ex = hf_datasets.load_from_disk(str(out_path))
                n_ex = len(ex.get("train", ex))
                if n_ex >= n_per_dim:
                    logger.info("[dim=%d] Already done (%d samples). Skipping.", d, n_ex)
                    stats_by_dim[d] = _collect_dim_stats(tmp_dirs[d], d)
                    method_counts_by_dim[d] = Counter()
                    continue
            except Exception:
                pass
        dims_todo.append(d)

    if dims_todo:
        # ── Per-dim accounting (resume support) ──────────────────────────────
        n_valid    : dict[int, int]     = {}
        next_id    : dict[int, int]     = {}
        method_cnts: dict[int, Counter] = {d: Counter() for d in dims_todo}
        skip_cnts  : dict[int, Counter] = {d: Counter() for d in dims_todo}
        t_start    : dict[int, float]   = {}

        for d in dims_todo:
            existing_ids = {int(f.stem.split("_")[1])
                            for f in tmp_dirs[d].glob("sample_*.npz")}
            n_valid[d] = len(existing_ids)
            next_id[d] = max(existing_ids, default=-1) + 1
            t_start[d] = time.time()
            logger.info("=" * 60)
            logger.info("[dim=%d] Need %d samples  (resume: %d done)",
                        d, n_per_dim, n_valid[d])

        # ── Single Pool shared across ALL dims ───────────────────────────────
        # Workers are initialised ONCE: each worker loads the datasets a single
        # time regardless of how many dims are being generated.  The old code
        # recreated the pool per dim, paying the full dataset-loading cost
        # (10 M-row tsmixup scan → 1–2 min per dim) for every dimension.
        with Pool(n_workers, initializer=_worker_init,
                  initargs=(args.data_paths, cfg)) as pool:
            while any(n_valid[d] < n_per_dim for d in dims_todo):
                # One combined batch across all unfinished dims
                batch: list[tuple[int, int]] = []
                for d in dims_todo:
                    if n_valid[d] < n_per_dim:
                        need = n_per_dim - n_valid[d]
                        bs = max(need + n_workers, int(need * 1.05) + 16)
                        batch.extend(
                            [(sid, d) for sid in range(next_id[d], next_id[d] + bs)]
                        )
                        next_id[d] += bs

                chunksize = max(1, len(batch) // (n_workers * 4))
                for (sid, d), data, status in pool.imap_unordered(
                    generate_composite_sample_nd, batch, chunksize=chunksize
                ):
                    if data is not None and n_valid[d] < n_per_dim:
                        np.savez_compressed(
                            tmp_dirs[d] / f"sample_{sid:08d}.npz",
                            target=data,
                        )
                        n_valid[d] += 1
                        method_cnts[d][status] += 1
                        if n_valid[d] % max(1, n_per_dim // 20) == 0:
                            elapsed = time.time() - t_start[d]
                            rate    = n_valid[d] / max(elapsed, 1e-9)
                            eta     = (n_per_dim - n_valid[d]) / max(rate, 1e-9)
                            total   = sum(n_valid.values())
                            logger.info(
                                "  [dim=%d] %d/%d  (%.1f s/s  ETA %.0fs)"
                                "  [total %d/%d]",
                                d, n_valid[d], n_per_dim, rate, eta,
                                total, n_per_dim * len(dims_todo),
                            )
                    elif data is None:
                        skip_cnts[d][status] += 1

        logger.info("Generation done in %.1fs", time.time() - t_total)

        # ── Post-process per dim: Arrow conversion + stats ───────────────────
        for d in dims_todo:
            logger.info("[dim=%d] Done  skipped=%d", d, sum(skip_cnts[d].values()))
            for reason, cnt in skip_cnts[d].most_common(5):
                logger.info("  skip %-22s: %d", reason, cnt)
            for method, cnt in method_cnts[d].most_common():
                logger.info("  method %-20s: %d (%.1f%%)",
                            method, cnt, 100 * cnt / max(n_valid[d], 1))
            convert_to_arrow(tmp_dirs[d], out_paths[d], d)
            stats_by_dim[d] = _collect_dim_stats(tmp_dirs[d], d)
            method_counts_by_dim[d] = method_cnts[d]

    # ── Save aggregate statistics ────────────────────────────────────────────
    stats_prefix = output_dir / f"composite_{args.range_name}"
    save_statistics_csv(stats_by_dim, method_counts_by_dim, stats_prefix.with_suffix(".csv"))
    save_statistics_png(stats_by_dim, stats_prefix.with_suffix(".png"))
    save_samples_png(stats_by_dim, tmp_dirs, output_dir / f"composite_{args.range_name}_samples.png",
                     n_samples=10, rng_seed=args.seed)

    # ── Optional cleanup ─────────────────────────────────────────────────────
    if args.cleanup_tmp:
        for d in dims:
            shutil.rmtree(tmp_dirs[d], ignore_errors=True)
            logger.info("Removed tmp dir: %s", tmp_dirs[d])

    logger.info("=" * 60)
    logger.info("ALL DONE in %.1fs — %d dims × %d samples = %d total",
                time.time() - t_total, n_total_dims, n_per_dim, n_total_dims * n_per_dim)
    logger.info("Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
