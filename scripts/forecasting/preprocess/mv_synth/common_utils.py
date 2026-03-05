"""
Common utilities for multivariate time series synthesis.

Shared by stage1_kernel_synth.py and stage2_tsmixup.py.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

# ─────────────────────────────────────────────────────────────────────────────
# Random signal primitives
# ─────────────────────────────────────────────────────────────────────────────

def rff_sample(T: int, length_scale: float = 0.15, n_features: int = 128,
               rng: np.random.Generator = None) -> np.ndarray:
    """Random Fourier Feature GP approximation — O(D·T), smooth signal."""
    rng = rng or np.random.default_rng()
    t = np.linspace(0.0, 1.0, T)
    omegas = rng.normal(0.0, 1.0 / max(length_scale, 1e-3), n_features)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_features)
    # shape (D, T)
    Z = np.sqrt(2.0 / n_features) * np.cos(omegas[:, None] * t[None, :] + phases[:, None])
    w = rng.standard_normal(n_features)
    return w @ Z  # (T,)


def ar_sample(T: int, order: int = 2, rng: np.random.Generator = None) -> np.ndarray:
    """Sample from a stationary AR(p) process — O(T·p)."""
    rng = rng or np.random.default_rng()
    # Draw AR coefficients with spectral radius < 0.95
    for _ in range(50):
        phi = rng.normal(0.0, 0.5 / order, order)
        # Companion matrix roots
        companion = np.zeros((order, order))
        companion[0] = phi
        companion[1:, :-1] = np.eye(order - 1)
        rho = np.max(np.abs(np.linalg.eigvals(companion))) if order > 1 else abs(phi[0])
        if rho < 0.95:
            break
        phi *= 0.9 / (rho + 1e-9)

    x = rng.standard_normal(T + 100)
    noise = rng.standard_normal(T + 100) * 0.3
    for t in range(order, T + 100):
        x[t] = phi @ x[t - order:t][::-1] + noise[t]
    return x[100:]  # burn-in


def periodic_component(T: int, period: float, amplitude: float = 1.0,
                        rng: np.random.Generator = None) -> np.ndarray:
    t = np.linspace(0.0, 1.0, T)
    phase = (rng or np.random.default_rng()).uniform(0.0, 2.0 * np.pi)
    return amplitude * np.sin(2.0 * np.pi * t / (period / T) + phase)


# ─────────────────────────────────────────────────────────────────────────────
# Trend augmentation (공통3)
# ─────────────────────────────────────────────────────────────────────────────

def add_trend(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add linear/polynomial/exponential trend to ~10% of samples. Overflow-safe."""
    T = series.shape[-1]
    t_norm = np.linspace(0.0, 1.0, T)
    trend_type = rng.choice(["linear", "polynomial", "exponential"])

    # Estimate series scale to keep trend moderate
    scale = max(float(np.std(series)), 1e-3)

    if trend_type == "linear":
        slope = rng.uniform(-3.0, 3.0) * scale
        trend = slope * t_norm
    elif trend_type == "polynomial":
        coeff = rng.uniform(-2.0, 2.0) * scale
        trend = coeff * (t_norm ** 2)
    else:  # exponential
        # rate chosen so exp(rate) ≤ 10 → rate ≤ ln(10) ≈ 2.3
        rate = rng.uniform(-1.5, 1.5)
        trend = scale * (np.exp(rate * t_norm) - 1.0)
        # Clip to avoid extreme values
        trend = np.clip(trend, -10 * scale, 10 * scale)

    if series.ndim == 2:
        trend = trend[None, :]
    return series + trend


# ─────────────────────────────────────────────────────────────────────────────
# Event injection (공통4)
# ─────────────────────────────────────────────────────────────────────────────

def add_events(series: np.ndarray, rng: np.random.Generator,
               n_events: int = None) -> np.ndarray:
    """Inject sparse, aperiodic events: point anomalies, level shifts, bursts."""
    T = series.shape[-1]
    if n_events is None:
        n_events = rng.integers(1, 4)

    scale = max(float(np.std(series)), 1e-3)
    out = series.copy()

    for _ in range(n_events):
        event_type = rng.choice(["spike", "level_shift", "burst"])
        t0 = int(rng.integers(0, T))

        if event_type == "spike":
            # Random spike (1-3 time steps)
            width = int(rng.integers(1, 4))
            amplitude = rng.uniform(2.0, 5.0) * scale * rng.choice([-1, 1])
            t_end = min(t0 + width, T)
            if out.ndim == 2:
                variate = rng.integers(0, out.shape[0])
                out[variate, t0:t_end] += amplitude
            else:
                out[t0:t_end] += amplitude

        elif event_type == "level_shift":
            # Sudden mean shift from t0 onward
            shift = rng.uniform(1.0, 3.0) * scale * rng.choice([-1, 1])
            if out.ndim == 2:
                variate = rng.integers(0, out.shape[0])
                out[variate, t0:] += shift
            else:
                out[t0:] += shift

        else:  # burst
            # Short high-frequency oscillation
            duration = int(rng.integers(T // 10, max(T // 5, T // 10 + 1)))
            t_end = min(t0 + duration, T)
            freq = rng.uniform(5.0, 20.0)
            amplitude = rng.uniform(0.5, 2.0) * scale
            t_arr = np.arange(t_end - t0)
            burst = amplitude * np.sin(2.0 * np.pi * freq * t_arr / (t_end - t0))
            if out.ndim == 2:
                variate = rng.integers(0, out.shape[0])
                out[variate, t0:t_end] += burst
            else:
                out[t0:t_end] += burst

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Realistic scaling (공통7: match benchmark statistics)
# ─────────────────────────────────────────────────────────────────────────────

def scale_to_realistic(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random but realistic scaling and offset to match benchmark value distributions."""
    # Normalize first
    std = np.std(series)
    if std < 1e-8:
        series = series + rng.standard_normal(series.shape) * 0.01
        std = max(np.std(series), 1e-8)
    series = (series - np.mean(series)) / std

    # Random scale drawn from log-normal (covers orders of magnitude like real data)
    log_scale = rng.uniform(-2.0, 4.0)  # scale in [0.01, 10000] range
    target_std = np.exp(log_scale)

    # Random offset (many real series are positive)
    positive = rng.random() < 0.5  # 50% chance of positive-only series
    if positive:
        offset = abs(rng.normal(0, target_std))
    else:
        offset = rng.normal(0, target_std * 0.5)

    series = series * target_std + offset

    # Final NaN/Inf safety
    series = np.nan_to_num(series, nan=0.0, posinf=1e6, neginf=-1e6)
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Sample length sampling (match benchmark length distributions)
# ─────────────────────────────────────────────────────────────────────────────

def sample_length(min_len: int, max_len: int, rng: np.random.Generator) -> int:
    """Log-uniform length sampling to match benchmark length distributions."""
    log_min = np.log(min_len)
    log_max = np.log(max_len)
    return int(np.exp(rng.uniform(log_min, log_max)))


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp generation
# ─────────────────────────────────────────────────────────────────────────────

_FREQ_CHOICES = ["H", "D", "15min", "30min", "W", "M"]
_FREQ_OFFSETS = {
    "H":    np.timedelta64(1, "h"),
    "D":    np.timedelta64(1, "D"),
    "15min": np.timedelta64(15, "m"),
    "30min": np.timedelta64(30, "m"),
    "W":    np.timedelta64(7, "D"),
    "M":    np.timedelta64(30, "D"),
}
_COMMON_STARTS = [
    np.datetime64("2000-01-01", "ms"),
    np.datetime64("2010-01-01", "ms"),
    np.datetime64("2015-06-01", "ms"),
    np.datetime64("2018-01-01", "ms"),
]


def generate_timestamps(T: int, rng: np.random.Generator) -> list:
    """Generate a realistic timestamp sequence of length T."""
    freq = rng.choice(_FREQ_CHOICES)
    start = _COMMON_STARTS[rng.integers(len(_COMMON_STARTS))]
    delta = _FREQ_OFFSETS[freq]
    ts = [start + i * delta for i in range(T)]
    return [t.astype("datetime64[ms]").item() for t in ts]


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace dataset I/O
# ─────────────────────────────────────────────────────────────────────────────

def make_hf_features(n_variates: int) -> datasets.Features:
    if n_variates == 1:
        target_feature = datasets.Sequence(datasets.Value("float64"))
    else:
        target_feature = datasets.Sequence(datasets.Sequence(datasets.Value("float64")))
    return datasets.Features({
        "target": target_feature,
        "id": datasets.Value("string"),
        "timestamp": datasets.Sequence(datasets.Value("timestamp[ms]")),
    })


def save_as_hf_dataset(records: list[dict], output_dir: Path, n_variates: int) -> None:
    """Save list of dicts as HuggingFace DatasetDict in arrow format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features = make_hf_features(n_variates)
    ds = datasets.Dataset.from_list(records, features=features)
    ds_dict = datasets.DatasetDict({"train": ds})
    ds_dict.save_to_disk(str(output_dir))


# ─────────────────────────────────────────────────────────────────────────────
# Statistics and plotting
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_statistics(records: list[dict], n_variates: int) -> dict:
    """Compute summary statistics for a list of sample records."""
    all_lengths = []
    all_means = []
    all_stds = []
    all_min = []
    all_max = []

    for r in records:
        target = np.array(r["target"], dtype=np.float64)
        if target.ndim == 1:
            lengths = [len(target)]
            means = [float(np.nanmean(target))]
            stds = [float(np.nanstd(target))]
            mins = [float(np.nanmin(target))]
            maxs = [float(np.nanmax(target))]
        else:
            lengths = [target.shape[1]] * target.shape[0]
            means = [float(np.nanmean(target[i])) for i in range(target.shape[0])]
            stds = [float(np.nanstd(target[i])) for i in range(target.shape[0])]
            mins = [float(np.nanmin(target[i])) for i in range(target.shape[0])]
            maxs = [float(np.nanmax(target[i])) for i in range(target.shape[0])]
        all_lengths.extend(lengths)
        all_means.extend(means)
        all_stds.extend(stds)
        all_min.extend(mins)
        all_max.extend(maxs)

    stats = {
        "n_samples": len(records),
        "n_variates": n_variates,
        "length": {
            "mean": float(np.mean(all_lengths)),
            "median": float(np.median(all_lengths)),
            "min": int(np.min(all_lengths)),
            "max": int(np.max(all_lengths)),
            "std": float(np.std(all_lengths)),
        },
        "values": {
            "mean_of_means": float(np.mean(all_means)),
            "mean_of_stds": float(np.mean(all_stds)),
            "global_min": float(np.min(all_min)),
            "global_max": float(np.max(all_max)),
        },
    }
    return stats


def print_statistics(stats: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Samples : {stats['n_samples']}")
    print(f"  Variates: {stats['n_variates']}")
    print(f"  Length  : mean={stats['length']['mean']:.1f}, "
          f"median={stats['length']['median']:.1f}, "
          f"min={stats['length']['min']}, max={stats['length']['max']}")
    print(f"  Values  : mean(mean)={stats['values']['mean_of_means']:.4f}, "
          f"mean(std)={stats['values']['mean_of_stds']:.4f}, "
          f"range=[{stats['values']['global_min']:.4f}, {stats['values']['global_max']:.4f}]")
    print(f"{'='*60}\n")


def save_statistics_summary(stats: dict, output_dir: Path) -> None:
    import json
    with open(output_dir / "statistics_summary.json", "w") as f:
        json.dump(stats, f, indent=2)


def plot_samples(records: list[dict], n_variates: int, output_dir: Path,
                 n_plot: int = 10) -> None:
    """Plot random sample time series."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping plots.")
        return

    n_plot = min(n_plot, len(records))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(records), size=n_plot, replace=False)

    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 2.5 * n_plot))
    if n_plot == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax_i, idx in enumerate(indices):
        target = np.array(records[idx]["target"], dtype=np.float64)
        if target.ndim == 1:
            ax = axes[ax_i]
            ax.plot(target, color=colors[0], linewidth=0.8, label="v0")
            ax.set_title(f"Sample {idx} (T={len(target)})", fontsize=8)
        else:
            ax = axes[ax_i]
            n_v = target.shape[0]
            for v in range(n_v):
                ax.plot(target[v], color=colors[v % len(colors)],
                        linewidth=0.8, alpha=0.8, label=f"v{v}")
            ax.set_title(f"Sample {idx} ({n_v}D, T={target.shape[1]})", fontsize=8)
            ax.legend(loc="upper right", fontsize=6, ncol=n_v)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "samples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrices(records: list[dict], n_variates: int,
                               output_dir: Path, n_plot: int = 10) -> None:
    """Plot variable correlation matrices for sample multivariate records."""
    if n_variates <= 1:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_plot = min(n_plot, len(records))
    rng = np.random.default_rng(99)
    indices = rng.choice(len(records), size=n_plot, replace=False)

    ncols = min(5, n_plot)
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n_plot > 1 else [axes]

    for ax_i, idx in enumerate(indices):
        target = np.array(records[idx]["target"], dtype=np.float64)
        if target.ndim == 2 and target.shape[0] > 1:
            corr = np.corrcoef(target)
            im = axes_flat[ax_i].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
            axes_flat[ax_i].set_title(f"Sample {idx}", fontsize=8)
            plt.colorbar(im, ax=axes_flat[ax_i], fraction=0.046, pad=0.04)

    # Hide unused axes
    for ax_i in range(n_plot, len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    plt.suptitle(f"Correlation Matrices (n_variates={n_variates})", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrices.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Arrow file discovery for Stage 2
# ─────────────────────────────────────────────────────────────────────────────

def discover_arrow_files(paths: list[str]) -> list[Path]:
    """
    Recursively find all .arrow files under the given paths.
    Also handles HuggingFace DatasetDict directories (has dataset_dict.json).
    """
    arrow_files: list[Path] = []
    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            warnings.warn(f"Path does not exist: {p}")
            continue
        if p.is_file() and p.suffix == ".arrow":
            arrow_files.append(p)
        elif p.is_dir():
            found = sorted(p.rglob("*.arrow"))
            if found:
                arrow_files.extend(found)
            else:
                # Maybe it's a HF DatasetDict root without arrow in root
                warnings.warn(f"No .arrow files found under {p}")
    return arrow_files
