#!/usr/bin/env python3
"""
generate_composite_synth_3d.py — Composite 3D Multivariate Synthesizer
========================================================================

Builds 3-dimensional multivariate time series by randomly pairing real and/or
synthetic time series and combining them into novel composite samples.  Accepts
any mixture of univariate (1D target) and multivariate (nD target) HuggingFace
Arrow datasets as input.

OUTPUT FORMAT
-------------
Same as generate_kernel_synth_3d.py:
  target    : Sequence(Sequence(float64))  shape (3, L) per row
  id        : string
  timestamp : Sequence(timestamp[ms])

COMPOSITE GENERATION ALGORITHM
--------------------------------
For each output sample:

1. SAMPLE TWO VARIATES from the unified variate pool.
   The pool is the union of all input datasets.  For univariate datasets each
   row contributes 1 variate; for k-dimensional datasets each row contributes k
   variates (each accessed as target[j]).  A global index maps to
   (dataset, row, variate_index).

2. TRUNCATE TO SHORTER LENGTH.
   L = min(len(v₁), len(v₂)), capped at --max_length.
   If L < --min_length the pair is discarded.

3. GENERATE VARIATE 3.
   With probability (1 − uncorrelated_ratio):
     a. Lead-lag mix  [1/3]:  v₃ = α·shift(v₁, lag) + (1−α)·v₂ + ε
     b. Weighted sum  [1/3]:  v₃ = w₁·v₁ + w₂·v₂ + ε   (w ~ Dirichlet)
     c. Causal filter [1/3]:  v₃ = (h * v₂)(t) + ε    (exponential FIR)
   With probability uncorrelated_ratio (independent):
     v₃ = randomly sampled third variate from the pool, truncated to L.

4. STACK to shape (3, L).

NORMALIZATION & SCALE
---------------------
• v₁ and v₂ are individually z-score normalized (zero-mean, unit-variance) before
  synthesis to prevent scale amplification and biased correlation.
• v₃ is generated entirely in the normalized (unit-variance) space.
• v₃ is then denormalized to the average scale of v₁ and v₂:
    denorm_std  = (σ₁ + σ₂) / 2
    denorm_mean = (μ₁ + μ₂) / 2
  so v₃ ≈ v₃_norm × denorm_std + denorm_mean
• v₁ and v₂ are returned in their original scale.
• noise_std ~ U(0.05, 0.30) is relative to unit-variance (not inflated by raw scale).
• A minimum-variance check rejects samples where any variate has σ² < 1e-12.

RESUME SUPPORT
--------------
Same tmp-directory approach as generate_kernel_synth_3d.py.

USAGE
-----
python generate_composite_synth_3d.py \\
    --data_paths \\
        /group-volume/ts-dataset/chronos_datasets/training_corpus_tsmixup_10m \\
        /group-volume/ts-dataset/chronos_datasets/training_corpus_kernel_synth_1m \\
        /group-volume/ts-dataset/chronos2_datasets/kernel_synth_3d \\
    --output_path /group-volume/ts-dataset/chronos2_datasets/composite_synth_3d \\
    --n_datasets 1000 \\
    --uncorrelated_ratio 0.01 \\
    --n_workers 8 \\
    --min_length 256 \\
    --max_length 8192 \\
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
# Variate Pool
# ──────────────────────────────────────────────────────────────────────────────

class VariatePool:
    """Lazy random-access pool of 1-D time series drawn from multiple datasets.

    Univariate datasets:      each row → 1 entry  (variate_idx=0)
    k-dimensional datasets:   each row → k entries (variate_idx ∈ 0..k-1)

    All datasets are opened via memory-mapped Arrow (no data copied to RAM).
    """

    def __init__(self, data_paths: list[str], min_length: int) -> None:
        import datasets as hf_datasets

        self._datasets: list = []         # HF Dataset objects
        self._n_variates: list[int] = []  # n_variates per dataset
        self._n_rows: list[int] = []      # n_rows per dataset
        self._cum: list[int] = [0]        # cumulative pool sizes (len = n_ds + 1)
        self._min_length = min_length

        for path in data_paths:
            path = str(path)
            try:
                raw = hf_datasets.load_from_disk(path)
                if hasattr(raw, "keys"):       # DatasetDict
                    ds = raw.get("train", next(iter(raw.values())))
                else:
                    ds = raw
            except Exception as exc:
                logger.warning("Cannot load dataset at %s: %s — skipping.", path, exc)
                continue

            n_rows = len(ds)
            n_var = _detect_n_variates(ds)
            pool_size = n_rows * n_var

            self._datasets.append(ds)
            self._n_variates.append(n_var)
            self._n_rows.append(n_rows)
            self._cum.append(self._cum[-1] + pool_size)

            logger.info("  Pool +  %-55s  rows=%d  variates/row=%d",
                        Path(path).name, n_rows, n_var)

        if self.total == 0:
            raise RuntimeError("Variate pool is empty — check --data_paths.")

    @property
    def total(self) -> int:
        return self._cum[-1]

    def get_variate(self, global_idx: int) -> np.ndarray | None:
        """Return a 1-D float64 array for the given pool index, or None on error."""
        global_idx = int(global_idx) % self.total   # safety wrap
        # Find dataset
        ds_idx = None
        for i in range(len(self._datasets)):
            if self._cum[i] <= global_idx < self._cum[i + 1]:
                ds_idx = i
                break
        if ds_idx is None:
            return None

        local_idx = global_idx - self._cum[ds_idx]
        row_idx = local_idx // self._n_variates[ds_idx]
        var_idx = local_idx % self._n_variates[ds_idx]

        try:
            row = self._datasets[ds_idx][int(row_idx)]
            target = np.array(row["target"], dtype=np.float64)
            if target.ndim == 1:
                series = target
            elif target.ndim == 2:
                series = target[var_idx]
            else:
                return None

            if len(series) < self._min_length:
                return None
            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                return None
            return series
        except Exception:
            return None


def _detect_n_variates(ds) -> int:
    """Detect how many variates are in the 'target' column."""
    if len(ds) == 0:
        return 1
    sample = ds[0]["target"]
    if isinstance(sample, (list, np.ndarray)) and len(sample) > 0:
        first = sample[0]
        if isinstance(first, (list, np.ndarray, bytes)):
            return len(sample)     # 2-D: outer = variates
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ──────────────────────────────────────────────────────────────────────────────

_POOL: VariatePool | None = None
_WORKER_CFG: dict = {}


def _worker_init(data_paths: list[str], cfg: dict) -> None:
    global _POOL, _WORKER_CFG
    _WORKER_CFG = cfg
    # Re-open datasets inside each worker (safe with fork; also works with spawn)
    _POOL = VariatePool(data_paths, cfg["min_length"])


def _apply_fir(src: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fl = int(rng.integers(5, 50))
    alpha = rng.uniform(0.50, 0.95)
    h = alpha ** np.arange(fl)
    h = h / h.sum()
    return np.convolve(src, h, mode="full")[: len(src)]


def _fetch_variate(rng: np.random.Generator, max_attempts: int = 30) -> np.ndarray | None:
    """Sample a valid variate from the pool, retrying if needed."""
    for _ in range(max_attempts):
        idx = int(rng.integers(0, _POOL.total))
        v = _POOL.get_variate(idx)
        if v is not None:
            return v
    return None


def generate_composite_sample(
    sample_id: int,
) -> tuple[int, np.ndarray | None, str]:
    """Generate one composite 3D sample.  Returns (sample_id, data|None, status)."""
    cfg = _WORKER_CFG
    rng = np.random.default_rng(cfg["base_seed"] + sample_id * 7_919)
    min_length = cfg["min_length"]
    max_length = cfg["max_length"]
    uncorrelated_ratio = cfg["uncorrelated_ratio"]

    try:
        # ── Step 1: sample two variates ────────────────────────────────────
        v1 = _fetch_variate(rng)
        v2 = _fetch_variate(rng)
        if v1 is None or v2 is None:
            return sample_id, None, "pool_miss"

        # ── Step 2: truncate to common length ──────────────────────────────
        L = min(len(v1), len(v2), max_length)
        if L < min_length:
            return sample_id, None, "too_short"

        # Random start to add diversity when series are long
        if len(v1) > L:
            start1 = int(rng.integers(0, len(v1) - L + 1))
            v1 = v1[start1 : start1 + L]
        else:
            v1 = v1[:L]

        if len(v2) > L:
            start2 = int(rng.integers(0, len(v2) - L + 1))
            v2 = v2[start2 : start2 + L]
        else:
            v2 = v2[:L]

        # ── Step 2.5: normalize v1, v2 to zero-mean / unit-variance ──────────
        mean1, std1 = float(np.mean(v1)), float(np.std(v1))
        mean2, std2 = float(np.mean(v2)), float(np.std(v2))
        std1 = max(std1, 1e-6)
        std2 = max(std2, 1e-6)
        v1_n = (v1 - mean1) / std1
        v2_n = (v2 - mean2) / std2

        # Target scale for denormalizing v3: average of v1/v2 statistics
        denorm_std  = (std1 + std2) / 2.0
        denorm_mean = (mean1 + mean2) / 2.0

        # ── Step 3: generate variate 3 in normalized space ─────────────────
        noise_std = rng.uniform(0.05, 0.30)  # relative to unit variance

        if rng.random() < uncorrelated_ratio:
            # Independent third variate: normalize to unit-variance then rescale
            v3_raw = _fetch_variate(rng)
            if v3_raw is None or len(v3_raw) < min_length:
                return sample_id, None, "pool_miss_v3"
            if len(v3_raw) > L:
                s3 = int(rng.integers(0, len(v3_raw) - L + 1))
                v3_raw = v3_raw[s3 : s3 + L]
            else:
                v3_raw = v3_raw[:L]
            std3 = max(float(np.std(v3_raw)), 1e-6)
            v3_n = (v3_raw - float(np.mean(v3_raw))) / std3
            method = "independent"
        else:
            mix_method = rng.choice(["lead_lag_mix", "weighted_sum", "causal_filter"])
            if mix_method == "lead_lag_mix":
                lag = int(rng.integers(1, max(2, L // 10)))
                alpha = rng.uniform(0.20, 0.80)
                v3_n = (
                    alpha * np.roll(v1_n, lag)
                    + (1.0 - alpha) * v2_n
                    + rng.standard_normal(L) * noise_std
                )
            elif mix_method == "weighted_sum":
                w1, w2 = rng.dirichlet([1.0, 1.0])
                v3_n = w1 * v1_n + w2 * v2_n + rng.standard_normal(L) * noise_std
            else:  # causal_filter
                v3_n = _apply_fir(v2_n, rng) + rng.standard_normal(L) * noise_std
            method = mix_method

        # ── Denormalize v3 to average scale of v1 / v2 ────────────────────
        v3 = v3_n * denorm_std + denorm_mean

        # ── Step 4: stack (v1, v2 original scale; v3 denormalized) ────────
        Y = np.stack([v1, v2, v3]).astype(np.float64)    # (3, L)

        # ── Validation ─────────────────────────────────────────────────────
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            return sample_id, None, "nan_or_inf"
        if Y.shape[1] < min_length:
            return sample_id, None, "too_short_post"
        if np.any(np.var(Y, axis=1) < 1e-12):
            return sample_id, None, "zero_variance"

        return sample_id, Y, method

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
    import datasets as hf_datasets

    tmp_files = sorted(tmp_dir.glob("sample_*.npz"))
    logger.info("Converting %d samples to HF Arrow format …", len(tmp_files))

    targets_list: list = []
    ids_list: list[str] = []
    timestamps_list: list = []
    _ts_cache: dict[int, list] = {}

    for f in tmp_files:
        Y = np.load(f)["target"]      # (3, L)
        L = Y.shape[1]
        sid = int(f.stem.split("_")[1])
        targets_list.append(Y.tolist())
        ids_list.append(f"3DC_{sid:08d}")
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

    all_vars: list[float] = []
    all_lengths: list[int] = []
    all_corrs: list[float] = []

    for f in files:
        Y = np.load(f)["target"].astype(np.float64)
        all_lengths.append(Y.shape[1])
        all_vars.extend(np.var(Y, axis=1).tolist())
        corr = np.corrcoef(Y)
        off_diag = [corr[0, 1], corr[0, 2], corr[1, 2]]
        all_corrs.append(float(np.mean(np.abs(off_diag))))

    n_valid = len(files)
    logger.info("=" * 60)
    logger.info("COMPOSITE GENERATION STATISTICS")
    logger.info("  Valid samples  : %d", n_valid)
    logger.info("  Skipped        : %d", sum(skipped.values()))
    for reason, cnt in skipped.most_common():
        logger.info("    %-22s : %d", reason, cnt)
    logger.info("  Method distribution:")
    for m, c in method_counts.most_common():
        logger.info("    %-22s : %d  (%.1f %%)", m, c, 100 * c / max(n_valid, 1))
    logger.info("  Length  min/mean/max : %d / %.1f / %d",
                min(all_lengths), np.mean(all_lengths), max(all_lengths))
    logger.info("  Variance  min/mean/max : %.4f / %.4f / %.4f",
                min(all_vars), np.mean(all_vars), max(all_vars))
    logger.info("  Mean |cross-variate corr| : %.4f ± %.4f",
                np.mean(all_corrs), np.std(all_corrs))
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate composite 3D multivariate time series from real+synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data_paths", nargs="+", required=True,
        help="One or more HF Arrow dataset paths (univariate or multivariate)",
    )
    p.add_argument("--output_path", default="/group-volume/ts-dataset/chronos2_datasets/composite_synth_3d")
    p.add_argument("--n_datasets",  type=int,   default=1_000, help="Target number of samples")
    p.add_argument("--uncorrelated_ratio", type=float, default=0.01,
                   help="Fraction of samples where v3 is independent (default 0.01)")
    p.add_argument("--n_workers",   type=int,   default=None,  help="Worker processes (default: cpu_count)")
    p.add_argument("--min_length",  type=int,   default=256,   help="Minimum output length")
    p.add_argument("--max_length",  type=int,   default=8192,  help="Maximum output length (truncation cap)")
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

    # ── Already finished? ────────────────────────────────────────────────────
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

    # ── Resume: count existing tmp files ────────────────────────────────────
    existing_files = list(tmp_dir.glob("sample_*.npz"))
    existing_ids: set[int] = {int(f.stem.split("_")[1]) for f in existing_files}
    n_valid = len(existing_ids)
    next_id = max(existing_ids, default=-1) + 1

    logger.info("Resuming: %d / %d samples already generated.", n_valid, args.n_datasets)
    logger.info("Workers: %d | min_length: %d | max_length: %d | uncorrelated_ratio: %.2f",
                n_workers, args.min_length, args.max_length, args.uncorrelated_ratio)

    # ── Build pool metadata in parent to log info ────────────────────────────
    logger.info("Scanning input datasets:")
    _dummy_pool = VariatePool(args.data_paths, args.min_length)
    logger.info("Total variate pool size: %d", _dummy_pool.total)
    del _dummy_pool

    cfg = {
        "min_length":         args.min_length,
        "max_length":         args.max_length,
        "uncorrelated_ratio": args.uncorrelated_ratio,
        "base_seed":          args.seed,
    }

    skipped = Counter()
    method_counts = Counter()
    t0 = time.time()

    with Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(args.data_paths, cfg),
    ) as pool:
        while n_valid < args.n_datasets:
            need = args.n_datasets - n_valid
            batch_size = max(need + n_workers * 2, int(need * 1.05) + 64)
            batch_ids = list(range(next_id, next_id + batch_size))
            next_id += batch_size

            for sample_id, data, status in pool.imap_unordered(
                generate_composite_sample,
                batch_ids,
                chunksize=max(1, batch_size // (n_workers * 4)),
            ):
                if data is not None and n_valid < args.n_datasets:
                    np.savez_compressed(
                        tmp_dir / f"sample_{sample_id:08d}.npz",
                        target=data,
                    )
                    n_valid += 1
                    method_counts[status] += 1
                    if n_valid % max(1, args.n_datasets // 20) == 0:
                        elapsed = time.time() - t0
                        rate = n_valid / elapsed
                        eta = (args.n_datasets - n_valid) / max(rate, 1e-9)
                        logger.info("  %d / %d  (%.0f samples/s, ETA %.0fs)",
                                    n_valid, args.n_datasets, rate, eta)
                elif data is None:
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
