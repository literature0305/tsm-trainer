#!/usr/bin/env python3
"""
generate_kernel_synth_3d.py — Correlated 3D Multivariate Time Series Synthesizer
==================================================================================

Generates 3-dimensional multivariate time series with controllable cross-variate
correlation structure using kernel/GP-based methods.  Output is saved as a
HuggingFace Arrow dataset (DatasetDict with a "train" split) that the TSM-Trainer
training pipeline can read directly via LazyHFTaskSource.

OUTPUT FORMAT
-------------
target : Sequence(Sequence(float64))   shape (3, length) per row
         ← read by training code as np.array → (3, L), treated as 3 targets
         with the same group_id → GroupSelfAttention learns cross-variate relations
id     : string
timestamp : Sequence(timestamp[ms])    hourly from Unix epoch

GENERATION METHODS
------------------
1. Correlated GP – Factor Model                [~40 % of correlated samples]
   Draw K ∈ {1, 2} independent GP factors z_k with diverse kernels.
   Y = A @ Z  where A is a 3×K random loading matrix.
   Cross-variate correlation ≈ A A^T (normalised).
   K = 1 → nearly perfect correlation; K = 2 → partial correlation.

2. Lead-Lag                                    [~25 % of correlated samples]
   Base GP z(t).  y₁ = z(t),  y₂ = z(t−lag₂) + ε₂,
   y₃ = α·y₁(t−lag₃) + (1−α)·y₂ + ε₃.
   Models realistic lead-lag / Granger-causality between variates.

3. VAR(1) – Vector AutoRegression             [~20 % of correlated samples]
   y[t] = A·y[t−1] + ε[t].
   A is a random 3×3 matrix scaled to spectral radius ρ ~ U(0.70, 0.97).
   Encodes direct Granger-causal structure.

4. Causal Chain Filter                         [~15 % of correlated samples]
   y₁ ~ GP.  y₂ = (h₂ * y₁)(t) + ε₂.  y₃ = (h₃ * y₂)(t) + ε₃.
   h_k = exponential-decay FIR filter → causal propagation y₁ → y₂ → y₃.

5. Hidden Regime (unobserved Markov state)     [hidden_regime_ratio fraction]
   A latent Markov state h(t) ∈ {0,…,K−1}  (K ∈ {2, 3}) switches the
   cross-variate coupling strength.  h(t) is NEVER included in the output.
     Y[j,t] = c[h(t), j] · z(t) + √(1 − c²) · ε_j(t)
   where z(t) is a shared GP factor and ε_j are per-variate private GPs.
   Each regime r has a (3,) coupling vector c[r] ∈ [0,1]:
     • at least one "high-coupling" regime  c ∈ [0.7, 1.0] per variate
     • at least one "low-coupling" regime   c ∈ [0.0, 0.2] per variate
   Regime durations follow a geometric distribution with mean U(50, 500).
   The model must infer h from cross-variate co-movement — exactly the
   scenario that makes GroupSelfAttention non-trivially useful.

6. Independent (uncorrelated)                  [uncorrelated_ratio fraction]
   Three fully independent GPs drawn from different kernels.

KERNELS (used inside methods 1–5)
----------------------------------
• Ornstein-Uhlenbeck (OU):  k(τ) = σ² exp(−θ|τ|)
  Simulated via exact AR(1): y[t] = e^{-θ} y[t-1] + σ√(1−e^{-2θ}) ε[t]
• RBF via Random Fourier Features (RFF):  k(τ) = σ² exp(−τ²/2l²)
  Approximated with D=1500 random features → O(D·L) per sample.
• Periodic:  sum of random harmonics with random period and phases.
• Composite:  0.4·OU + 0.4·RBF + 0.2·Periodic

LENGTH SAMPLING
---------------
• actual_length is sampled uniformly from multiples of 128 in [min_length, max_length]
  e.g. --min_length 128 --length 8192  →  128, 256, 384, …, 8192  (64 choices)
• All generation methods receive the sampled length directly.

VARIANCE DIVERSITY
------------------
• scale ~ U(0.5, 20) per sample
• Optional random linear trend  (40 % of variates)
• Optional random seasonal overlay (30 % of variates)
• SNR is NOT controlled – no normalization applied (model handles it)

RESUME SUPPORT
--------------
Generated samples are first written to  <output_path>_tmp/sample_NNNNNNN.npz.
On re-run the existing .npz files are counted; only the remaining samples are
generated.  After reaching n_datasets valid samples the tmp files are bulk-
converted to Arrow format and saved to output_path.

USAGE
-----
python generate_kernel_synth_3d.py \\
    --output_path /group-volume/ts-dataset/chronos2_datasets/kernel_synth_3d \\
    --length 8192 \\
    --n_datasets 1000 \\
    --uncorrelated_ratio 0.01 \\
    --n_workers 8 \\
    --min_length 128 \\
    --seed 42 \\
    [--cleanup_tmp]

DEPENDENCIES
------------
numpy, datasets (huggingface)
"""

import argparse
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

# ──────────────────────────────────────────────────────────────────────────────
# Low-level GP simulators  (top-level so multiprocessing can pickle them)
# ──────────────────────────────────────────────────────────────────────────────

def _sim_ou(length: int, theta: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Exact discrete Ornstein-Uhlenbeck AR(1) via scipy lfilter (no Python loop)."""
    from scipy.signal import lfilter
    phi = np.exp(-theta)
    sig_eps = sigma * np.sqrt(max(1.0 - phi ** 2, 0.0))
    y0 = rng.normal(0.0, sigma)
    noise = rng.standard_normal(length - 1) * sig_eps
    # AR(1): y[t] = phi*y[t-1] + u[t], u[0]=y0, u[t>=1]=noise[t-1]
    u = np.empty(length)
    u[0] = y0
    u[1:] = noise
    return lfilter([1.0], [1.0, -phi], u)


def _sim_rff_rbf(
    length: int, l: float, sigma: float, rng: np.random.Generator, n_feat: int = 1500
) -> np.ndarray:
    """RBF GP via Random Fourier Features.  k(τ) ≈ σ² exp(−τ²/2l²).

    Features are computed in chunks to keep peak memory ≤ 8 MB regardless of length.
    """
    t = np.arange(length, dtype=np.float64)
    omegas = rng.normal(0.0, 1.0 / l, size=n_feat)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_feat)
    coeffs = rng.standard_normal(n_feat)
    # chunk size: target ≤ 8 MB per slice → 8MB / (8 bytes × L) features
    chunk = max(1, (8 * 1024 * 1024) // (8 * length))
    y = np.zeros(length)
    for start in range(0, n_feat, chunk):
        end = min(start + chunk, n_feat)
        feat = np.cos(np.outer(t, omegas[start:end]) + phases[start:end])  # (L, chunk)
        y += feat @ coeffs[start:end]
    return sigma * np.sqrt(2.0 / n_feat) * y


def _sim_periodic(
    length: int, period: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Sum of random harmonics: periodic GP."""
    t = np.arange(length, dtype=np.float64)
    n_harmonics = rng.integers(1, 6)
    y = np.zeros(length)
    for h in range(1, n_harmonics + 1):
        amp = rng.uniform(0.2, 1.0) * sigma / np.sqrt(n_harmonics)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        y += amp * np.cos(2.0 * np.pi * h / period * t + phase)
    return y


def _sim_one_gp(
    length: int, scale: float, rng: np.random.Generator, kernel: str | None = None
) -> np.ndarray:
    """Sample one GP from a randomly chosen kernel."""
    if kernel is None:
        kernel = rng.choice(["ou", "rbf", "periodic", "composite"])

    if kernel == "ou":
        theta = rng.uniform(0.001, 0.15)
        return _sim_ou(length, theta, scale, rng)
    elif kernel == "rbf":
        l = rng.uniform(30.0, 1500.0)
        return _sim_rff_rbf(length, l, scale, rng)
    elif kernel == "periodic":
        period = rng.uniform(12.0, 800.0)
        return _sim_periodic(length, period, scale, rng)
    else:  # composite
        theta = rng.uniform(0.002, 0.05)
        l = rng.uniform(100.0, 3000.0)
        period = rng.uniform(20.0, 600.0)
        y = (
            0.4 * _sim_ou(length, theta, scale, rng)
            + 0.4 * _sim_rff_rbf(length, l, scale, rng)
            + 0.2 * _sim_periodic(length, period, scale, rng)
        )
        return y


# ──────────────────────────────────────────────────────────────────────────────
# High-level generation methods
# ──────────────────────────────────────────────────────────────────────────────

def _method_correlated_gp(
    length: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Factor model: Y[3×L] = A[3×K] @ Z[K×L] + noise, K ∈ {1, 2}."""
    K = rng.integers(1, 3)
    kernels = rng.choice(["ou", "rbf", "periodic", "composite"], size=K)
    Z = np.stack([_sim_one_gp(length, 1.0, rng, k) for k in kernels])   # (K, L)
    A = rng.standard_normal((3, K))
    Y = A @ Z  # (3, L)
    noise_std = rng.uniform(0.02, 0.15) * scale
    Y += rng.standard_normal((3, length)) * noise_std
    # Rescale each variate to target variance
    for j in range(3):
        std = np.std(Y[j])
        if std > 1e-10:
            Y[j] = Y[j] / std * scale * rng.uniform(0.7, 1.5)
    return Y


def _method_lead_lag(
    length: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Lead-lag: y₁=z, y₂=shift(z,lag₂)+ε, y₃=mix(y₁,y₂,lag₃)+ε."""
    base = _sim_one_gp(length, scale, rng)
    lag2 = int(rng.integers(1, max(2, length // 10)))
    lag3 = int(rng.integers(1, max(2, length // 10)))
    noise_std = rng.uniform(0.05, 0.30) * scale

    y1 = base.copy()

    y2 = np.roll(base, lag2) + rng.standard_normal(length) * noise_std
    y2[:lag2] = rng.standard_normal(lag2) * scale

    alpha = rng.uniform(0.3, 0.9)
    y3 = (
        alpha * np.roll(y1, lag3)
        + (1.0 - alpha) * y2
        + rng.standard_normal(length) * noise_std
    )
    y3[:lag3] = rng.standard_normal(lag3) * scale

    return np.stack([y1, y2, y3])   # (3, L)


def _method_var(
    length: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """VAR(1): y[t] = A·y[t-1] + ε[t] via eigendecomposition + lfilter (no Python loop).

    Diagonalises A = V @ diag(λ) @ V⁻¹, solves 3 independent AR(1) processes in
    modal coordinates z[t] = λ·z[t-1] + w[t] using lfilter, then transforms back.
    Supports complex eigenvalues; the final result is taken as real.
    """
    from scipy.signal import lfilter
    rho = rng.uniform(0.70, 0.97)
    A_raw = rng.standard_normal((3, 3))
    sr = np.max(np.abs(np.linalg.eigvals(A_raw)))
    A = A_raw * (rho / (sr + 1e-8))

    noise_std = scale * np.sqrt(max(1.0 - rho ** 2, 1e-8))
    y0  = rng.standard_normal(3) * scale           # (3,)
    eps = rng.standard_normal((3, length - 1)) * noise_std  # (3, L-1)

    # Eigendecomposition: A = V @ diag(eigvals) @ V_inv
    eigvals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)

    # Transform initial state and innovations to modal coordinates
    z0 = (V_inv @ y0).astype(complex)   # (3,)
    w  = (V_inv @ eps).astype(complex)  # (3, L-1)

    # Solve 3 decoupled AR(1): z[j,t] = λ[j]*z[j,t-1] + u[j,t]
    Z = np.empty((3, length), dtype=complex)
    for j in range(3):
        u = np.empty(length, dtype=complex)
        u[0]  = z0[j]
        u[1:] = w[j]
        phi = complex(eigvals[j])
        Z[j] = lfilter([1.0 + 0j], [1.0 + 0j, -phi], u)

    # Transform back to original space; imaginary residuals vanish for real A
    return (V @ Z).real


def _method_causal_filter(
    length: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Causal chain via exponential FIR: y₁ → y₂ → y₃."""
    from scipy.signal import lfilter
    y1 = _sim_one_gp(length, scale, rng)
    noise_std = rng.uniform(0.05, 0.25) * scale

    def _apply_fir(src):
        fl = int(rng.integers(5, 60))
        alpha = rng.uniform(0.50, 0.95)
        h = alpha ** np.arange(fl)
        h = h / h.sum()
        return lfilter(h, [1.0], src)

    y2 = _apply_fir(y1) + rng.standard_normal(length) * noise_std
    y3 = _apply_fir(y2) + rng.standard_normal(length) * noise_std
    return np.stack([y1, y2, y3])   # (3, L)


def _method_independent(
    length: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Three independent GPs with different kernels."""
    kernel_pool = ["ou", "rbf", "periodic", "composite"]
    kernels = rng.choice(kernel_pool, size=3, replace=False)
    Y = np.stack([
        _sim_one_gp(length, scale * rng.uniform(0.5, 2.0), rng, k)
        for k in kernels
    ])
    return Y


def _generate_regime_sequence(
    length: int, n_regimes: int, mean_dur: float, rng: np.random.Generator
) -> np.ndarray:
    """Markov chain regime sequence using geometric holding times.

    Each regime persists for Geometric(1/mean_dur) steps then transitions
    uniformly to one of the other n_regimes-1 regimes.  Expected number of
    transitions ≈ length / mean_dur (e.g. ~16–163 for length=8192, mean_dur=50–500).
    """
    h = np.zeros(length, dtype=np.int32)
    t = 0
    current = int(rng.integers(0, n_regimes))
    others = list(range(n_regimes))
    while t < length:
        dur = int(np.clip(rng.geometric(1.0 / mean_dur), 1, length - t))
        h[t : t + dur] = current
        t += dur
        if t < length:
            choices = [r for r in others if r != current]
            current = int(rng.choice(choices))
    return h


def _method_hidden_regime(
    length: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Hidden Markov regime: unobserved state h(t) switches cross-variate coupling.

    Structure
    ---------
    • K ∈ {2, 3} regimes, holding times ~ Geometric(1/mean_dur), mean_dur ~ U(50, 500).
    • Per-regime coupling matrix  c[regime, variate] ∈ [0, 1].
      – One "high-coupling" regime:  c ≈ U(0.7, 1.0) for each variate.
      – One "low-coupling" regime:   c ≈ U(0.0, 0.2) for each variate.
      – Additional regimes (if K=3): c ≈ U(0.2, 0.7).
    • Shared factor GP  z(t) and per-variate private GPs  ε_j(t):
        Y[j, t] = c[h(t), j] · z(t) + √(1 − c²[h(t), j]) · ε_j(t)
    • h(t) is NEVER in the output — the model must infer regime context
      from the co-movement of all three observed variates.

    Learning signal for GroupSelfAttention
    ---------------------------------------
    In a high-coupling regime y₁, y₂, y₃ all track z(t) closely.
    In a low-coupling regime they diverge to private noise.
    Detecting the current regime from any single variate is ambiguous;
    cross-variate agreement (or divergence) is the decisive cue.
    """
    n_regimes = int(rng.integers(2, 4))
    mean_dur = rng.uniform(50.0, 500.0)

    # ── Latent regime sequence ───────────────────────────────────────────────
    h = _generate_regime_sequence(length, n_regimes, mean_dur, rng)   # (length,)

    # ── Per-regime coupling profiles ─────────────────────────────────────────
    regime_coupling = rng.uniform(0.2, 0.8, size=(n_regimes, 3))
    hi_r = int(rng.integers(0, n_regimes))
    lo_r = (hi_r + 1) % n_regimes
    regime_coupling[hi_r] = rng.uniform(0.70, 1.00, size=3)
    regime_coupling[lo_r] = rng.uniform(0.00, 0.20, size=3)
    if n_regimes == 3:
        mid_r = (lo_r + 1) % n_regimes
        regime_coupling[mid_r] = rng.uniform(0.30, 0.70, size=3)

    # ── Common factor GP  z(t) ───────────────────────────────────────────────
    z = _sim_one_gp(length, 1.0, rng)   # (length,)

    # ── Per-variate private noise GPs  ε_j(t) ───────────────────────────────
    private = np.stack([_sim_one_gp(length, 1.0, rng) for _ in range(3)])  # (3, L)

    # ── Mix: Y[j,t] = c[h[t],j]·z[t] + √(1−c²)·private[j,t] ───────────────
    c = regime_coupling[h].T            # (3, length)  — time-varying coupling
    Y = c * z[np.newaxis, :] + np.sqrt(np.maximum(1.0 - c ** 2, 0.0)) * private

    # ── Rescale each variate to target scale ─────────────────────────────────
    for j in range(3):
        std = np.std(Y[j])
        if std > 1e-10:
            Y[j] = Y[j] / std * scale * rng.uniform(0.7, 1.5)

    return Y


def _add_trend_seasonality(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Optionally add random linear trend and/or seasonal component per variate."""
    length = Y.shape[1]
    t = np.arange(length, dtype=np.float64)
    for j in range(3):
        if rng.random() < 0.40:
            slope = rng.normal(0.0, np.std(Y[j]) * 2.0 / length)
            Y[j] += slope * t
        if rng.random() < 0.30:
            period = rng.uniform(12.0, 500.0)
            amp = rng.uniform(0.05, 0.30) * np.std(Y[j])
            Y[j] += amp * np.sin(2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi))
    return Y


# ──────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ──────────────────────────────────────────────────────────────────────────────

_WORKER_CFG: dict = {}

def _worker_init(cfg: dict) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def generate_one_sample(sample_id: int) -> tuple[int, bool | None, str]:
    """Generate one 3D sample and write it to the tmp directory.

    Workers write directly to NFS instead of returning large arrays through
    IPC.  This parallelises disk I/O across all workers and eliminates the
    single-threaded savez_compressed bottleneck in the main process.

    Returns (sample_id, True | None, method/status):
      True  — sample written to <tmp_dir>/sample_<id>.npz
      None  — sample rejected; status contains the reason
    """
    cfg = _WORKER_CFG
    rng = np.random.default_rng(cfg["base_seed"] + sample_id * 7_919)
    max_length = cfg["max_length"]
    min_length = cfg["min_length"]
    uncorrelated_ratio   = cfg["uncorrelated_ratio"]
    hidden_regime_ratio  = cfg["hidden_regime_ratio"]

    try:
        # ── Sample length from multiples of 128 in [min_length, max_length] ──
        lo = (min_length + 127) // 128   # first multiple of 128 >= min_length
        hi = max_length // 128           # last multiple of 128 <= max_length
        length = int(rng.integers(lo, hi + 1)) * 128

        scale = rng.uniform(0.5, 20.0)

        # ── Method selection ─────────────────────────────────────────────────
        r = rng.random()
        if r < uncorrelated_ratio:
            method = "independent"
        elif r < uncorrelated_ratio + hidden_regime_ratio:
            method = "hidden_regime"
        else:
            method = rng.choice(
                ["correlated_gp", "lead_lag", "var", "causal_filter"],
                p=[0.40, 0.25, 0.20, 0.15],
            )

        if method == "correlated_gp":
            Y = _method_correlated_gp(length, scale, rng)
        elif method == "lead_lag":
            Y = _method_lead_lag(length, scale, rng)
        elif method == "var":
            Y = _method_var(length, scale, rng)
        elif method == "causal_filter":
            Y = _method_causal_filter(length, scale, rng)
        elif method == "hidden_regime":
            Y = _method_hidden_regime(length, scale, rng)
        else:
            Y = _method_independent(length, scale, rng)

        Y = _add_trend_seasonality(Y, rng)

        # ── Validation ──────────────────────────────────────────────────────
        if Y.shape != (3, length):
            return sample_id, None, "shape_mismatch"
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            return sample_id, None, "nan_or_inf"
        if np.all(Y == 0.0):
            return sample_id, None, "all_zero"
        if np.any(np.var(Y, axis=1) < 1e-12):
            return sample_id, None, "zero_variance"

        # ── Write directly from worker (parallel NFS writes) ────────────────
        np.savez_compressed(
            Path(cfg["tmp_dir"]) / f"sample_{sample_id:08d}.npz",
            target=Y.astype(np.float64),
        )
        return sample_id, True, method

    except Exception as exc:   # noqa: BLE001
        return sample_id, None, f"error:{exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Arrow conversion
# ──────────────────────────────────────────────────────────────────────────────

def _build_timestamps(length: int) -> list[datetime]:
    epoch = datetime(1970, 1, 1)
    h1 = timedelta(hours=1)
    return [epoch + h1 * i for i in range(length)]


def convert_to_arrow(tmp_dir: Path, output_path: Path) -> None:
    """Bulk-convert .npz temp files → HF Arrow DatasetDict."""
    import datasets as hf_datasets

    tmp_files = sorted(tmp_dir.glob("sample_*.npz"))
    logger.info("Converting %d samples to HF Arrow format …", len(tmp_files))

    targets_list: list = []
    ids_list: list[str] = []
    timestamps_list: list = []
    _ts_cache: dict[int, list] = {}

    for f in tmp_files:
        Y = np.load(f)["target"]   # (3, L)
        L = Y.shape[1]
        sid = int(f.stem.split("_")[1])
        targets_list.append(Y.tolist())       # list of 3 lists
        ids_list.append(f"3DK_{sid:08d}")
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
    logger.info("Saved Arrow dataset → %s  (%d rows)", output_path, len(ds))


# ──────────────────────────────────────────────────────────────────────────────
# Statistics report
# ──────────────────────────────────────────────────────────────────────────────

def print_statistics(tmp_dir: Path, skipped: Counter, method_counts: Counter) -> None:
    files = sorted(tmp_dir.glob("sample_*.npz"))
    if not files:
        logger.warning("No samples found for statistics.")
        return

    all_vars = []
    all_lengths = []
    all_corrs = []
    for f in files:
        Y = np.load(f)["target"].astype(np.float64)   # (3, L)
        all_lengths.append(Y.shape[1])
        all_vars.extend(np.var(Y, axis=1).tolist())
        # Mean absolute pairwise correlation
        corr = np.corrcoef(Y)
        off_diag = [corr[0, 1], corr[0, 2], corr[1, 2]]
        all_corrs.append(np.mean(np.abs(off_diag)))

    n_valid = len(files)
    n_skip = sum(skipped.values())
    logger.info("=" * 60)
    logger.info("GENERATION STATISTICS")
    logger.info("  Valid samples  : %d", n_valid)
    logger.info("  Skipped        : %d", n_skip)
    if skipped:
        for reason, cnt in skipped.most_common():
            logger.info("    %-20s : %d", reason, cnt)
    logger.info("  Method distribution:")
    for m, c in method_counts.most_common():
        logger.info("    %-20s : %d  (%.1f %%)", m, c, 100 * c / max(n_valid, 1))
    logger.info("  Length  min/mean/max : %d / %.1f / %d",
                min(all_lengths), np.mean(all_lengths), max(all_lengths))
    logger.info("  Variance  min/mean/max : %.4f / %.4f / %.4f",
                min(all_vars), np.mean(all_vars), max(all_vars))
    logger.info("  Mean |cross-variate correlation|  : %.4f ± %.4f",
                np.mean(all_corrs), np.std(all_corrs))
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate correlated 3D multivariate time series (kernel/GP methods)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output_path", default="/group-volume/ts-dataset/chronos2_datasets/kernel_synth_3d")
    p.add_argument("--length",      type=int,   default=8192,  help="Time series length")
    p.add_argument("--n_datasets",  type=int,   default=1_000, help="Target number of samples")
    p.add_argument("--uncorrelated_ratio", type=float, default=0.01,
                   help="Fraction of samples with no cross-variate correlation (default 0.01)")
    p.add_argument("--hidden_regime_ratio", type=float, default=0.05,
                   help="Fraction of samples using hidden Markov regime switching "
                        "(0.0 to disable; default 0.05)")
    p.add_argument("--n_workers",   type=int,   default=None,  help="Worker processes (default: cpu_count)")
    p.add_argument("--min_length",  type=int,   default=128,   help="Minimum series length (must be multiple of 128; default 128)")
    p.add_argument("--seed",        type=int,   default=42,    help="Base random seed")
    p.add_argument("--cleanup_tmp", action="store_true",       help="Remove tmp dir after conversion")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    tmp_dir = output_path.parent / (output_path.name + "_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_workers = args.n_workers or cpu_count()

    # ── Check if already finished ────────────────────────────────────────────
    if output_path.exists():
        try:
            import datasets as hf_datasets
            existing_ds = hf_datasets.load_from_disk(str(output_path))
            n_existing = len(existing_ds.get("train", existing_ds))
            if n_existing >= args.n_datasets:
                logger.info("Output already contains %d samples (≥ %d). Done.",
                            n_existing, args.n_datasets)
                return
        except Exception:
            pass

    # ── Resume: count already generated tmp files ────────────────────────────
    existing_files = list(tmp_dir.glob("sample_*.npz"))
    existing_ids: set[int] = {int(f.stem.split("_")[1]) for f in existing_files}
    n_valid = len(existing_ids)
    next_id = max(existing_ids, default=-1) + 1

    logger.info("Resuming: %d / %d samples already in tmp dir %s",
                n_valid, args.n_datasets, tmp_dir)
    logger.info(
        "Workers: %d | Max length: %d | Min length: %d | "
        "Uncorrelated ratio: %.3f | Hidden regime ratio: %.3f%s",
        n_workers, args.length, args.min_length,
        args.uncorrelated_ratio, args.hidden_regime_ratio,
        "" if args.hidden_regime_ratio > 0 else "  (disabled)",
    )
    logger.info("Length sampling: multiples of 128 in [%d, %d]  (%d choices)",
                args.min_length, args.length,
                args.length // 128 - (args.min_length + 127) // 128 + 1)

    cfg = {
        "max_length":          args.length,
        "min_length":          args.min_length,
        "uncorrelated_ratio":  args.uncorrelated_ratio,
        "hidden_regime_ratio": args.hidden_regime_ratio,
        "base_seed":           args.seed,
        "tmp_dir":             str(tmp_dir),  # workers write directly (parallel I/O)
    }

    skipped = Counter()
    method_counts = Counter()

    t0 = time.time()

    with Pool(n_workers, initializer=_worker_init, initargs=(cfg,)) as pool:
        while n_valid < args.n_datasets:
            need = args.n_datasets - n_valid
            # Overshoot to absorb expected failures (< 1 % in practice)
            batch_size = max(need + n_workers * 2, int(need * 1.05) + 64)
            batch_ids = list(range(next_id, next_id + batch_size))
            next_id += batch_size

            chunksize = max(1, batch_size // (n_workers * 4))
            for sample_id, wrote, status in pool.imap_unordered(
                generate_one_sample, batch_ids, chunksize=chunksize
            ):
                if wrote is True and n_valid < args.n_datasets:
                    n_valid += 1
                    method_counts[status] += 1
                    if n_valid % max(1, args.n_datasets // 20) == 0:
                        elapsed = time.time() - t0
                        rate = n_valid / elapsed
                        eta = (args.n_datasets - n_valid) / max(rate, 1e-9)
                        logger.info("  %d / %d  (%.0f samples/s, ETA %.0fs)",
                                    n_valid, args.n_datasets, rate, eta)
                elif wrote is None:
                    skipped[status] += 1

    logger.info("Generation complete in %.1f s", time.time() - t0)

    # ── Convert tmp → Arrow ──────────────────────────────────────────────────
    convert_to_arrow(tmp_dir, output_path)

    # ── Statistics ───────────────────────────────────────────────────────────
    print_statistics(tmp_dir, skipped, method_counts)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if args.cleanup_tmp:
        shutil.rmtree(tmp_dir)
        logger.info("Removed tmp dir: %s", tmp_dir)


if __name__ == "__main__":
    main()
