#!/usr/bin/env python3
"""
Stage 1: N-dimensional Kernel Synthesis for Multivariate Time Series Generation.

Generates correlated multivariate time series using diverse synthesis methods.
Output is saved as HuggingFace DatasetDict in arrow format, one folder per variate-dim.

Usage:
    python stage1_kernel_synth.py \\
        --min-len 64 --max-len 1024 \\
        --num-samples 1000 --max-variates 5 \\
        --output-dir /group-volume/ts-dataset/chronos2_datasets \\
        --seed 42 --num-workers -1
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np

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
    periodic_component,
    sample_length,
    save_as_hf_dataset,
    save_statistics_summary,
    scale_to_realistic,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generation methods
# ─────────────────────────────────────────────────────────────────────────────

def _base_signal(T: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a rich base signal (GP-like + optional seasonality)."""
    ls = float(np.exp(rng.uniform(-2.5, -0.5)))  # length scale in [0.08, 0.6]
    sig = rff_sample(T, length_scale=ls, n_features=128, rng=rng)

    # Optionally add periodic component
    if rng.random() < 0.6:
        period_choices = [T // 4, T // 7, T // 12, T // 24, T // 52]
        period_choices = [p for p in period_choices if p >= 2]
        if period_choices:
            period = float(rng.choice(period_choices))
            amp = rng.uniform(0.2, 1.0) * np.std(sig)
            sig = sig + periodic_component(T, period, amplitude=amp, rng=rng)

    return sig  # (T,) zero-mean


def gen_correlated_gp(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Correlated GP: x_i = w_i * z_base + sqrt(1-w_i^2) * z_i + noise_i
    All variates share a common latent process z_base.
    Not all variates need full correlation (공통1).
    """
    z_base = _base_signal(T, rng)
    z_base_std = max(float(np.std(z_base)), 1e-6)
    z_base_n = z_base / z_base_std

    result = np.zeros((n, T))
    for i in range(n):
        w = rng.uniform(0.2, 0.95)  # correlation with base
        z_ind = rff_sample(T, length_scale=float(np.exp(rng.uniform(-2.5, -0.5))),
                           n_features=128, rng=rng)
        z_ind_std = max(float(np.std(z_ind)), 1e-6)
        z_ind_n = z_ind / z_ind_std

        noise_level = rng.uniform(0.02, 0.2)  # SNR > 1
        noise = rng.standard_normal(T) * noise_level
        result[i] = w * z_base_n + math.sqrt(max(1 - w ** 2, 0.0)) * z_ind_n + noise

    return result


def gen_lead_lag(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Lead-lag: x_i(t) = z(t - lag_i) + noise_i.
    Some variates may have no lag (independent component mixed in).
    """
    max_lag = max(1, T // 6)
    z_base = _base_signal(T + max_lag * 2, rng)

    result = np.zeros((n, T))
    for i in range(n):
        lag = int(rng.integers(0, max_lag + 1))
        start = max_lag - lag
        result[i] = z_base[start: start + T]

        # Random individual component
        if rng.random() < 0.4:
            ind = _base_signal(T, rng)
            alpha = rng.uniform(0.1, 0.5)
            result[i] = (1.0 - alpha) * result[i] + alpha * ind

        noise = rng.standard_normal(T) * rng.uniform(0.02, 0.15)
        result[i] += noise

    return result


def gen_var(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Vector Autoregression VAR(p): X(t) = A_1 X(t-1) + ... + A_p X(t-p) + noise.
    A is sampled with controlled spectral radius < 0.95.
    """
    p = int(rng.integers(1, 4))  # AR order 1-3
    burn_in = 200

    # Build stable A matrix (block companion form)
    # Simple approach: random A with normalized spectral radius
    A_list = []
    for _ in range(p):
        A = rng.normal(0.0, 0.5 / (n * p), (n, n))
        A_list.append(A)

    # Scale to ensure stability via spectral radius of companion matrix
    if n == 1 and p == 1:
        rho = float(np.abs(A_list[0][0, 0]))
    elif p == 1:
        rho = float(np.max(np.abs(np.linalg.eigvals(A_list[0]))))
    else:
        companion = np.zeros((n * p, n * p))
        for k, A in enumerate(A_list):
            companion[:n, k * n:(k + 1) * n] = A
        companion[n:, :-n] = np.eye(n * (p - 1))
        rho = float(np.max(np.abs(np.linalg.eigvals(companion))))

    if rho >= 0.95:
        scale_factor = 0.9 / (rho + 1e-9)
        A_list = [A * scale_factor for A in A_list]

    # Generate time series
    noise_cov = _random_covariance(n, rng)
    noise_all = rng.multivariate_normal(np.zeros(n), noise_cov, T + burn_in)

    X = np.zeros((T + burn_in, n))
    for t in range(p, T + burn_in):
        for k, A in enumerate(A_list):
            X[t] += A @ X[t - 1 - k]
        X[t] += noise_all[t]

    return X[burn_in:].T  # (n, T)


def _random_covariance(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random positive-definite covariance matrix."""
    L = rng.standard_normal((n, n)) * 0.5
    return L @ L.T + np.eye(n) * 0.1


def gen_causal_filter(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Causal filter: apply different AR filters to shared + individual noise sources.
    Creates complex temporal dependencies.
    """
    # Shared noise source (creates cross-variate correlation)
    shared_noise = rng.standard_normal(T + 100)
    result = np.zeros((n, T))

    for i in range(n):
        order = int(rng.integers(2, 8))
        # Stable AR coefficients
        phi = rng.normal(0.0, 0.4 / order, order)
        # Apply filter
        w_shared = rng.uniform(0.3, 0.9)
        ind_noise = rng.standard_normal(T + 100)
        combined = w_shared * shared_noise + math.sqrt(1 - w_shared**2) * ind_noise

        x = np.zeros(T + 100)
        for t in range(order, T + 100):
            x[t] = phi @ x[t - order:t][::-1] + combined[t]

        result[i] = x[100:]

    return result


def gen_independent(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Independent GP samples — no correlation between variates (공통1)."""
    result = np.zeros((n, T))
    for i in range(n):
        result[i] = _base_signal(T, rng)
    return result


def gen_hidden_regime(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Hidden regime model: K regimes with shared Markov chain.
    Each variate has regime-specific mean/variance.
    """
    K = int(rng.integers(2, 5))  # 2-4 regimes

    # Transition matrix (each row sums to 1)
    trans = rng.dirichlet(np.ones(K) * 2.0, size=K)

    # Regime means and variances per variate
    regime_means = rng.normal(0.0, 2.0, (n, K))
    regime_stds = np.exp(rng.normal(0.0, 0.5, (n, K)))

    # Generate regime sequence via Markov chain
    regime_seq = np.zeros(T, dtype=int)
    regime_seq[0] = rng.integers(K)
    for t in range(1, T):
        regime_seq[t] = rng.choice(K, p=trans[regime_seq[t - 1]])

    # Generate observations
    result = np.zeros((n, T))
    for t in range(T):
        r = regime_seq[t]
        result[:, t] = rng.normal(regime_means[:, r], regime_stds[:, r])

    return result


def gen_partial_mix(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Partial mixing: variates divided into groups; within-group correlated, across independent.
    Not all variates are mutually dependent (공통1).
    """
    if n <= 1:
        return gen_correlated_gp(n, T, rng)

    # Split into 2 to min(n,3) groups
    n_groups = int(rng.integers(2, min(n, 3) + 1))
    group_ids = rng.integers(0, n_groups, size=n)

    result = np.zeros((n, T))
    for g in range(n_groups):
        members = np.where(group_ids == g)[0]
        if len(members) == 0:
            # assign to random member
            members = np.array([rng.integers(n)])
        if len(members) == 1:
            result[members[0]] = _base_signal(T, rng)
        else:
            block = gen_correlated_gp(len(members), T, rng)
            for mi, m in enumerate(members):
                result[m] = block[mi]

    return result


def gen_dynamic_mix(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Dynamic mixing: mixing weights evolve over time (non-stationary correlations).
    """
    n_sources = min(n + 1, 4)
    sources = np.array([_base_signal(T, rng) for _ in range(n_sources)])

    # Time-varying weights (each source gets sigmoid-shaped weight curve)
    t_norm = np.linspace(0.0, 1.0, T)
    result = np.zeros((n, T))

    for i in range(n):
        # Random mixing weights that vary over time
        n_transitions = int(rng.integers(1, 4))
        w = np.zeros((n_sources, T))
        for s in range(n_sources):
            # Build weight via sum of logistic curves
            base_w = rng.uniform(0.0, 1.0)
            w[s] = base_w
            for _ in range(n_transitions):
                center = rng.uniform(0.1, 0.9)
                rate = rng.uniform(5.0, 20.0) * rng.choice([-1, 1])
                delta_w = rng.uniform(-0.5, 0.5)
                w[s] += delta_w / (1 + np.exp(-rate * (t_norm - center)))
        # Softmax normalize
        w = np.exp(w - w.max(axis=0, keepdims=True))
        w = w / (w.sum(axis=0, keepdims=True) + 1e-8)

        result[i] = (w * sources).sum(axis=0)
        noise = rng.standard_normal(T) * rng.uniform(0.02, 0.15)
        result[i] += noise

    return result


def gen_mixture_segments(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Piecewise mixing: different correlation structures in different time segments.
    """
    n_segments = int(rng.integers(2, 5))
    # Random segment boundaries
    boundaries = sorted(rng.choice(T - 2, size=n_segments - 1, replace=False) + 1)
    boundaries = [0] + list(boundaries) + [T]

    result = np.zeros((n, T))
    for seg_i in range(n_segments):
        t_start, t_end = boundaries[seg_i], boundaries[seg_i + 1]
        seg_len = t_end - t_start
        if seg_len < 2:
            continue
        # Choose random method for this segment
        seg_method = rng.choice([gen_correlated_gp, gen_lead_lag, gen_independent])
        seg_data = seg_method(n, seg_len, rng)
        result[:, t_start:t_end] = seg_data

    return result


# Registry of all methods (공통1: not all variates need full dependence)
_GENERATION_METHODS = [
    (gen_correlated_gp,     0.25),
    (gen_lead_lag,          0.15),
    (gen_var,               0.15),
    (gen_causal_filter,     0.12),
    (gen_hidden_regime,     0.12),
    (gen_partial_mix,       0.08),
    (gen_dynamic_mix,       0.06),
    (gen_mixture_segments,  0.04),
    (gen_independent,       0.03),
]
_METHOD_FUNCS = [m[0] for m in _GENERATION_METHODS]
_METHOD_WEIGHTS = np.array([m[1] for m in _GENERATION_METHODS])
_METHOD_WEIGHTS /= _METHOD_WEIGHTS.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Single sample generation (for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_one_sample(args: tuple) -> dict:
    """Generate one multivariate sample. Args = (idx, n_variates, min_len, max_len, base_seed)."""
    idx, n_variates, min_len, max_len, base_seed = args
    rng = np.random.default_rng(base_seed + idx * 97 + 13)

    T = sample_length(min_len, max_len, rng)

    # Select generation method
    method_idx = rng.choice(len(_METHOD_FUNCS), p=_METHOD_WEIGHTS)
    method = _METHOD_FUNCS[method_idx]

    try:
        data = method(n_variates, T, rng)  # (n_variates, T)
    except Exception:
        # Fallback to simple GP on error
        data = gen_correlated_gp(n_variates, T, rng)

    # Apply trend (공통3: ~10% of samples)
    if rng.random() < 0.10:
        data = add_trend(data, rng)

    # Apply events (공통4)
    if rng.random() < 0.35:
        data = add_events(data, rng)

    # Scale to realistic values (공통7)
    data = scale_to_realistic(data, rng)

    # Build target
    if n_variates == 1:
        target = data[0].tolist()
    else:
        target = [data[v].tolist() for v in range(n_variates)]

    timestamps = generate_timestamps(T, rng)

    return {
        "target": target,
        "id": f"ks_{n_variates}d_{idx:010d}",
        "timestamp": timestamps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: N-dimensional kernel synthesis of multivariate time series."
    )
    parser.add_argument("--min-len", type=int, default=64,
                        help="Minimum series length (default: 64)")
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Maximum series length (default: 1024)")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Total number of samples to generate (default: 1000)")
    parser.add_argument("--max-variates", type=int, default=5,
                        help="Maximum number of variates (default: 5)")
    parser.add_argument("--output-dir", type=str,
                        default="/group-volume/ts-dataset/chronos2_datasets",
                        help="Root output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="Number of parallel workers (-1 = all CPUs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    n_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    logger.info(f"Using {n_workers} workers")

    # Distribute samples across dimensions
    n_dims = args.max_variates
    base_per_dim = args.num_samples // n_dims
    remainder = args.num_samples - base_per_dim * n_dims
    # Give remainder to lowest dimensions
    samples_per_dim = [base_per_dim + (1 if i < remainder else 0)
                       for i in range(n_dims)]

    total_generated = 0
    for dim_idx, n_variates in enumerate(range(1, n_dims + 1)):
        n_samples = samples_per_dim[dim_idx]
        if n_samples == 0:
            continue

        folder_name = (f"kernel_synth_{n_variates}d_"
                       f"{args.min_len}-{args.max_len}_{n_samples}samples")
        output_dir = output_root / folder_name

        logger.info(f"[dim={n_variates}] Generating {n_samples} samples → {output_dir}")
        t0 = time.time()

        # Seed offset per dimension so samples don't repeat
        base_seed = args.seed + dim_idx * 1_000_000

        task_args = [
            (i, n_variates, args.min_len, args.max_len, base_seed)
            for i in range(n_samples)
        ]

        if n_workers == 1:
            records = [_generate_one_sample(a) for a in task_args]
        else:
            chunk = max(1, n_samples // (n_workers * 8))
            with Pool(processes=n_workers) as pool:
                records = list(pool.imap(_generate_one_sample, task_args,
                                         chunksize=chunk))

        elapsed = time.time() - t0
        logger.info(f"  Generated {len(records)} samples in {elapsed:.1f}s "
                    f"({len(records)/elapsed:.0f} samples/s)")

        # Save as HuggingFace DatasetDict
        logger.info(f"  Saving to {output_dir} ...")
        save_as_hf_dataset(records, output_dir, n_variates)

        # Statistics and plots
        stats = compute_dataset_statistics(records, n_variates)
        print_statistics(stats)
        save_statistics_summary(stats, output_dir)

        n_plot = min(10, n_samples)
        plot_samples(records, n_variates, output_dir, n_plot=n_plot)
        plot_correlation_matrices(records, n_variates, output_dir, n_plot=n_plot)

        total_generated += len(records)
        logger.info(f"  Saved. Total so far: {total_generated}")

    logger.info(f"Done. Total samples generated: {total_generated}")


if __name__ == "__main__":
    main()
