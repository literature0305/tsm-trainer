#!/usr/bin/env python3
"""
Stage 2: TSMixup — Combine real + synthetic time series to create
correlated multivariate datasets.

Reads from m input paths (HuggingFace DatasetDict directories or raw .arrow files),
lazily samples series, and applies mixing operations to produce multivariate outputs.

Usage:
    python stage2_tsmixup.py \\
        --input-paths /path/to/chronos_datasets/training_corpus_kernel_synth_1m \\
                      /path/to/benchmarks/gift_eval/* \\
                      /path/to/chronos2_datasets/kernel_synth_1d_64-1024_1000samples \\
        --min-len 64 --max-len 1024 \\
        --num-samples 1000 --max-variates 5 \\
        --output-dir /group-volume/ts-dataset/chronos2_datasets \\
        --seed 42 --num-workers -1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    add_events,
    add_trend,
    compute_dataset_statistics,
    generate_timestamps,
    plot_correlation_matrices,
    plot_samples,
    print_statistics,
    rff_sample,
    ar_sample,
    sample_length,
    save_as_hf_dataset,
    save_statistics_summary,
    scale_to_realistic,
    discover_arrow_files,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy Arrow index: efficient random access over potentially billions of rows
# ─────────────────────────────────────────────────────────────────────────────

class ArrowFileIndex:
    """
    Build a global index over multiple Arrow IPC stream files
    for O(log N) random access without loading all data.

    Uses disk-based row-count cache (내용4) and an in-process
    per-file table LRU cache to avoid repeated full-file reads.
    """

    # Per-process in-memory table cache  (file_path → pa.Table)
    # Shared across all ArrowFileIndex instances in the same process.
    _table_cache: dict[Path, pa.Table] = {}
    _TABLE_CACHE_MAX = 8  # max files to keep fully in memory

    def __init__(self, arrow_files: list[Path], cache_dir: Optional[Path] = None):
        self._files = list(arrow_files)
        self._cache_dir = cache_dir or Path("/tmp/arrow_index_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._row_counts: list[int] = []
        self._cumulative: np.ndarray = np.array([], dtype=np.int64)
        self._build_index()

    def _get_cache_path(self) -> Path:
        # Use a hash of file paths as cache key
        import hashlib
        key = hashlib.md5("|".join(str(f) for f in self._files).encode()).hexdigest()
        return self._cache_dir / f"index_{key}.json"

    def _build_index(self) -> None:
        cache_path = self._get_cache_path()

        # Try loading from cache
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                if cached.get("n_files") == len(self._files):
                    self._row_counts = cached["row_counts"]
                    self._cumulative = np.array([0] + list(
                        np.cumsum(self._row_counts)), dtype=np.int64)
                    logger.info(f"ArrowFileIndex: loaded from cache ({self.total_rows:,} rows "
                                f"across {len(self._files)} files)")
                    return
            except Exception:
                pass

        # Scan files to get row counts
        logger.info(f"ArrowFileIndex: scanning {len(self._files)} arrow files ...")
        row_counts = []
        for fpath in self._files:
            try:
                n = self._count_rows(fpath)
            except Exception as e:
                warnings.warn(f"Skipping {fpath}: {e}")
                n = 0
            row_counts.append(n)

        self._row_counts = row_counts
        self._cumulative = np.array([0] + list(np.cumsum(row_counts)), dtype=np.int64)

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump({"n_files": len(self._files), "row_counts": row_counts}, f)
        except Exception:
            pass

        logger.info(f"ArrowFileIndex: {self.total_rows:,} rows across {len(self._files)} files")

    @staticmethod
    def _count_rows(fpath: Path) -> int:
        """Count rows in an arrow file without loading all data."""
        with open(fpath, "rb") as f:
            try:
                reader = pa_ipc.open_stream(f)
                n = sum(batch.num_rows for batch in reader)
                return n
            except pa.lib.ArrowInvalid:
                pass
        # Try as arrow IPC file format
        with open(fpath, "rb") as f:
            try:
                reader = pa_ipc.open_file(f)
                return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
            except Exception:
                pass
        return 0

    @property
    def total_rows(self) -> int:
        return int(self._cumulative[-1]) if len(self._cumulative) > 0 else 0

    def __len__(self) -> int:
        return self.total_rows

    def get_row(self, global_idx: int) -> dict:
        """Read a single row by global index."""
        if global_idx < 0 or global_idx >= self.total_rows:
            raise IndexError(f"Index {global_idx} out of range [0, {self.total_rows})")

        # Binary search for file
        file_idx = int(np.searchsorted(self._cumulative, global_idx + 1, side="right")) - 1
        local_idx = global_idx - int(self._cumulative[file_idx])
        return self._read_local(self._files[file_idx], local_idx)

    @classmethod
    def _load_table(cls, fpath: Path) -> pa.Table:
        """Load an arrow file as a PyArrow Table, with LRU in-memory cache (내용4)."""
        if fpath in cls._table_cache:
            return cls._table_cache[fpath]

        # Load from disk
        with open(fpath, "rb") as f:
            try:
                table = pa_ipc.open_stream(f).read_all()
            except pa.lib.ArrowInvalid:
                table = pa_ipc.open_file(fpath).read_all()

        # LRU eviction: remove oldest entry if at capacity
        if len(cls._table_cache) >= cls._TABLE_CACHE_MAX:
            oldest_key = next(iter(cls._table_cache))
            del cls._table_cache[oldest_key]
        cls._table_cache[fpath] = table
        return table

    @classmethod
    def _read_local(cls, fpath: Path, local_idx: int) -> dict:
        """Read local_idx-th row from an arrow IPC stream file (cache-backed)."""
        table = cls._load_table(fpath)
        row = {col: table.column(col)[local_idx].as_py()
               for col in table.schema.names}
        return row

    def sample_target_arrays(self, n: int, rng: np.random.Generator,
                              min_len: int, max_len: int,
                              max_retries: int = 20) -> list[np.ndarray]:
        """
        Sample n 1-D target arrays of length in [min_len, max_len].
        Returns flat float32 arrays of valid length.
        """
        results = []
        attempts = 0
        while len(results) < n and attempts < n * max_retries:
            global_idx = int(rng.integers(0, self.total_rows))
            try:
                row = self.get_row(global_idx)
                target_raw = row.get("target", None)
                if target_raw is None:
                    attempts += 1
                    continue
                target = np.array(target_raw, dtype=np.float32)
                # Flatten if 2D (multivariate stored as nested list)
                if target.ndim == 2:
                    # Pick one variate at random
                    v = int(rng.integers(0, target.shape[0]))
                    target = target[v]
                if target.ndim != 1 or len(target) < min_len:
                    attempts += 1
                    continue
                # Clip to max_len from a random start
                if len(target) > max_len:
                    start = int(rng.integers(0, len(target) - max_len + 1))
                    target = target[start: start + max_len]
                results.append(target)
            except Exception:
                attempts += 1
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Mixing methods (공통5, 내용5, 내용6)
# ─────────────────────────────────────────────────────────────────────────────

def _align_lengths(arrays: list[np.ndarray], T: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Crop/pad all arrays to exactly length T."""
    out = []
    for a in arrays:
        if len(a) >= T:
            start = int(rng.integers(0, len(a) - T + 1))
            out.append(a[start: start + T].astype(np.float64))
        else:
            # Tile
            n_reps = T // len(a) + 2
            tiled = np.tile(a, n_reps)[:T].astype(np.float64)
            out.append(tiled)
    return out


def _z_score(a: np.ndarray) -> np.ndarray:
    std = np.std(a)
    if std < 1e-8:
        return a - np.mean(a)
    return (a - np.mean(a)) / std


def mix_weighted_sum(sources: list[np.ndarray], n_out: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Linear positive weighted sum."""
    T = sources[0].shape[0]
    result = np.zeros((n_out, T))
    for i in range(n_out):
        weights = rng.exponential(1.0, len(sources))
        weights /= weights.sum()
        s = sum(w * _z_score(src) for w, src in zip(weights, sources))
        noise = rng.standard_normal(T) * rng.uniform(0.02, 0.15)
        result[i] = s + noise
    return result


def mix_negative_weighted_sum(sources: list[np.ndarray], n_out: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Weighted sum with some negative weights → anti-correlation."""
    T = sources[0].shape[0]
    result = np.zeros((n_out, T))
    for i in range(n_out):
        weights = rng.normal(0.0, 1.0, len(sources))
        # Normalize
        norm = np.linalg.norm(weights)
        weights = weights / max(norm, 1e-8)
        s = sum(w * _z_score(src) for w, src in zip(weights, sources))
        noise = rng.standard_normal(T) * rng.uniform(0.02, 0.15)
        result[i] = s + noise
    return result


def mix_nonlinear(sources: list[np.ndarray], n_out: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Non-linear mixing via tanh/sigmoid nonlinearities."""
    T = sources[0].shape[0]
    result = np.zeros((n_out, T))
    for i in range(n_out):
        weights = rng.normal(0.0, 1.0, len(sources))
        linear = sum(w * _z_score(src) for w, src in zip(weights, sources))
        nonlin_type = rng.choice(["tanh", "sigmoid", "square", "abs"])
        if nonlin_type == "tanh":
            out_i = np.tanh(linear)
        elif nonlin_type == "sigmoid":
            out_i = 2.0 / (1.0 + np.exp(-linear)) - 1.0
        elif nonlin_type == "square":
            sign_flip = rng.choice([-1, 1])
            out_i = sign_flip * linear ** 2 / max(np.std(linear ** 2), 1e-6)
        else:
            out_i = np.abs(linear) * rng.choice([-1, 1])
        noise = rng.standard_normal(T) * rng.uniform(0.02, 0.15)
        result[i] = out_i + noise
    return result


def mix_lag_lead(sources: list[np.ndarray], n_out: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Each output variate is a lagged/leaded version of a source."""
    T = sources[0].shape[0]
    max_lag = max(1, T // 6)
    result = np.zeros((n_out, T))

    # Shared base series
    base_idx = int(rng.integers(len(sources)))
    base = _z_score(sources[base_idx])

    for i in range(n_out):
        lag = int(rng.integers(-max_lag, max_lag + 1))
        shifted = np.roll(base, lag)
        if lag > 0:
            shifted[:lag] = np.nan
        elif lag < 0:
            shifted[lag:] = np.nan
        # Replace NaN with noise
        nan_mask = np.isnan(shifted)
        if nan_mask.any():
            shifted[nan_mask] = rng.standard_normal(nan_mask.sum()) * 0.1

        # Mix with another source
        alpha = rng.uniform(0.1, 0.5)
        other_idx = int(rng.integers(len(sources)))
        other = _z_score(sources[other_idx])
        result[i] = (1 - alpha) * shifted + alpha * other
        result[i] += rng.standard_normal(T) * rng.uniform(0.02, 0.15)

    return result


def mix_piecewise(sources: list[np.ndarray], n_out: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Piecewise mixing: different linear combinations in different time segments."""
    T = sources[0].shape[0]
    n_segments = int(rng.integers(2, 5))
    boundaries = sorted(rng.choice(T - 2, size=n_segments - 1, replace=False) + 1)
    boundaries = [0] + list(boundaries) + [T]

    result = np.zeros((n_out, T))
    for seg_i in range(n_segments):
        t_start, t_end = boundaries[seg_i], boundaries[seg_i + 1]
        if t_end <= t_start:
            continue
        seg_len = t_end - t_start
        # Random weights for this segment
        for i in range(n_out):
            weights = rng.normal(0.0, 1.0, len(sources))
            seg_out = sum(w * _z_score(src[t_start:t_end])
                          for w, src in zip(weights, sources))
            result[i, t_start:t_end] = seg_out

    return result


def mix_causal_filter(sources: list[np.ndarray], n_out: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Apply random AR filter to source series to produce correlated output."""
    T = sources[0].shape[0]
    result = np.zeros((n_out, T))

    for i in range(n_out):
        src_idx = int(rng.integers(len(sources)))
        x = _z_score(sources[src_idx]).copy()

        order = int(rng.integers(1, 6))
        phi = rng.normal(0.0, 0.3 / order, order)
        # Ensure stability
        if order > 1:
            rho = max(abs(np.roots(np.concatenate([[1], -phi]))))
        else:
            rho = abs(phi[0])
        if rho >= 0.98:
            phi *= 0.9 / (rho + 1e-9)

        # Apply filter
        y = x.copy()
        for t in range(order, T):
            y[t] = phi @ y[t - order:t][::-1] + x[t]

        noise = rng.standard_normal(T) * rng.uniform(0.02, 0.15)
        result[i] = y + noise

    return result


def mix_time_warp(sources: list[np.ndarray], n_out: int,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Apply time warping: stretch/compress segments of source series.
    Creates non-stationary temporal relationships.
    """
    T = sources[0].shape[0]
    result = np.zeros((n_out, T))

    for i in range(n_out):
        src_idx = int(rng.integers(len(sources)))
        src = _z_score(sources[src_idx])

        # Random piecewise-linear time warp
        n_knots = int(rng.integers(3, 8))
        knots_x = np.sort(rng.uniform(0, T, n_knots))
        knots_y = np.sort(rng.uniform(0, T, n_knots))
        # Ensure endpoints are fixed
        knots_x = np.concatenate([[0], knots_x, [T - 1]])
        knots_y = np.concatenate([[0], knots_y, [T - 1]])

        # Warp the time axis
        new_t = np.interp(np.arange(T), knots_x, knots_y)
        new_t = np.clip(new_t, 0, T - 1)
        warped = np.interp(new_t, np.arange(T), src)

        noise = rng.standard_normal(T) * rng.uniform(0.02, 0.15)
        result[i] = warped + noise

    return result


def mix_independent(sources: list[np.ndarray], n_out: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Each output variate is independently sampled from a different source."""
    T = sources[0].shape[0]
    result = np.zeros((n_out, T))
    for i in range(n_out):
        idx = int(rng.integers(len(sources)))
        result[i] = _z_score(sources[idx])
        result[i] += rng.standard_normal(T) * rng.uniform(0.02, 0.15)
    return result


# Registry
_MIX_METHODS = [
    (mix_weighted_sum,          0.18),
    (mix_negative_weighted_sum, 0.12),
    (mix_nonlinear,             0.15),
    (mix_lag_lead,              0.15),
    (mix_piecewise,             0.12),
    (mix_causal_filter,         0.12),
    (mix_time_warp,             0.08),
    (mix_independent,           0.08),
]
_MIX_FUNCS = [m[0] for m in _MIX_METHODS]
_MIX_WEIGHTS = np.array([m[1] for m in _MIX_METHODS])
_MIX_WEIGHTS /= _MIX_WEIGHTS.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (must be picklable — no class methods)
# ─────────────────────────────────────────────────────────────────────────────

# Global shared index (set in worker initializer or before Pool creation)
_GLOBAL_INDEX: Optional[ArrowFileIndex] = None


def _worker_init(files_list: list[Path], cache_dir: Path) -> None:
    """Initialize global arrow index in each worker process."""
    global _GLOBAL_INDEX
    _GLOBAL_INDEX = ArrowFileIndex(files_list, cache_dir=cache_dir)


def _generate_one_sample(args: tuple) -> Optional[dict]:
    """
    Generate one mixed multivariate sample.
    Args = (idx, n_variates, min_len, max_len, base_seed)
    """
    global _GLOBAL_INDEX
    idx, n_variates, min_len, max_len, base_seed = args
    rng = np.random.default_rng(base_seed + idx * 97 + 31)

    T = sample_length(min_len, max_len, rng)

    # ── Hidden variable scenario (내용6): sample extra sources ──────────────
    hidden_extra = 0
    if n_variates > 1 and rng.random() < 0.3:
        hidden_extra = int(rng.integers(1, min(n_variates + 2, 5)))

    n_sources_needed = n_variates + hidden_extra

    # ── Sample source series ─────────────────────────────────────────────────
    if _GLOBAL_INDEX is None or _GLOBAL_INDEX.total_rows == 0:
        return None

    source_arrays = _GLOBAL_INDEX.sample_target_arrays(
        n_sources_needed, rng, min_len=max(32, T // 2), max_len=T * 4
    )
    if len(source_arrays) < max(1, n_sources_needed // 2):
        return None  # Not enough data

    # Align all sources to length T
    source_arrays = _align_lengths(source_arrays, T, rng)

    # ── Apply mixing ─────────────────────────────────────────────────────────
    method_idx = int(rng.choice(len(_MIX_FUNCS), p=_MIX_WEIGHTS))
    mix_fn = _MIX_FUNCS[method_idx]

    try:
        data = mix_fn(source_arrays, n_variates, rng)  # (n_variates, T)
    except Exception:
        data = mix_independent(source_arrays, n_variates, rng)

    # ── Post-processing ──────────────────────────────────────────────────────
    if rng.random() < 0.10:  # 공통3
        data = add_trend(data, rng)

    if rng.random() < 0.35:  # 공통4
        data = add_events(data, rng)

    data = scale_to_realistic(data, rng)  # 공통7

    # ── Format output ────────────────────────────────────────────────────────
    if n_variates == 1:
        target = data[0].tolist()
    else:
        target = [data[v].tolist() for v in range(n_variates)]

    timestamps = generate_timestamps(T, rng)

    return {
        "target": target,
        "id": f"mixup_{n_variates}d_{idx:010d}",
        "timestamp": timestamps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: TSMixup — multivariate synthesis from real + synthetic data."
    )
    parser.add_argument("--input-paths", nargs="+", required=True,
                        help="Input paths (HuggingFace dirs or directories with .arrow files). "
                             "Supports glob expansion by the shell.")
    parser.add_argument("--min-len", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-variates", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default="/group-volume/ts-dataset/chronos2_datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    n_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    logger.info(f"Using {n_workers} workers")

    # ── Discover all arrow files ─────────────────────────────────────────────
    logger.info("Discovering arrow files from input paths ...")
    arrow_files = discover_arrow_files(args.input_paths)
    if not arrow_files:
        raise ValueError(f"No arrow files found in: {args.input_paths}")
    logger.info(f"Found {len(arrow_files)} arrow files")

    # Build global index (with disk cache)
    cache_dir = output_root / ".index_cache"
    index = ArrowFileIndex(arrow_files, cache_dir=cache_dir)

    if index.total_rows == 0:
        raise ValueError("All arrow files appear to be empty or unreadable.")

    logger.info(f"Total rows available: {index.total_rows:,}")

    # ── Distribute samples across dimensions ─────────────────────────────────
    n_dims = args.max_variates
    base_per_dim = args.num_samples // n_dims
    remainder = args.num_samples - base_per_dim * n_dims
    samples_per_dim = [base_per_dim + (1 if i < remainder else 0)
                       for i in range(n_dims)]

    total_generated = 0
    for dim_idx, n_variates in enumerate(range(1, n_dims + 1)):
        n_samples = samples_per_dim[dim_idx]
        if n_samples == 0:
            continue

        folder_name = (f"tsmixup_{n_variates}d_"
                       f"{args.min_len}-{args.max_len}_{n_samples}samples")
        output_dir = output_root / folder_name

        logger.info(f"[dim={n_variates}] Generating {n_samples} samples → {output_dir}")
        t0 = time.time()

        base_seed = args.seed + dim_idx * 1_000_000
        task_args = [
            (i, n_variates, args.min_len, args.max_len, base_seed)
            for i in range(n_samples)
        ]

        if n_workers == 1:
            # Single-process: initialize global index directly
            _worker_init(arrow_files, cache_dir)
            raw_results = [_generate_one_sample(a) for a in task_args]
        else:
            chunk = max(1, n_samples // (n_workers * 8))
            with Pool(processes=n_workers,
                      initializer=_worker_init,
                      initargs=(arrow_files, cache_dir)) as pool:
                raw_results = list(pool.imap(_generate_one_sample, task_args,
                                             chunksize=chunk))

        records = [r for r in raw_results if r is not None]
        elapsed = time.time() - t0

        if not records:
            logger.warning(f"  No valid samples generated for dim={n_variates}. Skipping.")
            continue

        logger.info(f"  Generated {len(records)}/{n_samples} samples in {elapsed:.1f}s "
                    f"({len(records)/elapsed:.0f} samples/s)")

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        save_as_hf_dataset(records, output_dir, n_variates)

        # Stats + plots
        stats = compute_dataset_statistics(records, n_variates)
        print_statistics(stats)
        save_statistics_summary(stats, output_dir)
        n_plot = min(10, len(records))
        plot_samples(records, n_variates, output_dir, n_plot=n_plot)
        plot_correlation_matrices(records, n_variates, output_dir, n_plot=n_plot)

        total_generated += len(records)
        logger.info(f"  Saved. Total so far: {total_generated}")

    logger.info(f"Done. Total samples generated: {total_generated}")


if __name__ == "__main__":
    main()
