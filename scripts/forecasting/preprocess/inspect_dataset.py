#!/usr/bin/env python3
"""
inspect_dataset.py — Dataset Statistics & Visualizer
=====================================================

Accepts a folder containing either:
  • A HuggingFace Arrow dataset  (DatasetDict with "train" split, or a bare Dataset)
  • A directory of .npz temp files  (sample_XXXXXXXX.npz, target shape (k, L) or (L,))

Outputs a printed statistics report and two PNG figures:
  • <output_dir>/samples.png   — time-series plots of 3 randomly chosen samples
  • <output_dir>/stats.png     — aggregate distribution plots

Works for both univariate (1D) and multivariate (nD) targets.

USAGE
-----
  python inspect_dataset.py  PATH  [OPTIONS]

  python inspect_dataset.py /group-volume/ts-dataset/chronos2_datasets/kernel_synth_3d
  python inspect_dataset.py /tmp/kernel_synth_3d_tmp              # .npz dir
  python inspect_dataset.py /group-volume/ts-dataset/chronos_datasets/training_corpus_tsmixup_10m \\
      --max_samples 2000 --seed 7

OPTIONS
  PATH                  Dataset folder (required, positional)
  --max_samples  N      Max rows to analyse for statistics (default: 1000)
  --seed         N      Random seed for sample selection (default: 42)
  --output_dir   PATH   Where to save PNGs (default: same folder as dataset)
  --show                Display the figures interactively instead of saving
  --split        STR    HF Dataset split to use (default: train)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_arrow(folder: Path, split: str):
    """Return (HF Dataset, format_label).  Handles DatasetDict and bare Dataset."""
    import datasets as hf
    raw = hf.load_from_disk(str(folder))
    if isinstance(raw, hf.DatasetDict):
        if split in raw:
            return raw[split], f"arrow_datasetdict[{split}]"
        first = next(iter(raw))
        log.warning("Split '%s' not found; using '%s'.", split, first)
        return raw[first], f"arrow_datasetdict[{first}]"
    return raw, "arrow_dataset"


def load_dataset(folder: Path, split: str):
    """
    Returns:
        samples  – list of np.ndarray, each shape (k, L) or (L,)
        fmt      – human-readable format string
        total    – total rows in the underlying dataset (before --max_samples)
    """
    # ── .npz directory ────────────────────────────────────────────────────────
    npz_files = sorted(folder.glob("sample_*.npz"))
    if npz_files:
        log.info("Detected .npz directory: %d files", len(npz_files))
        samples = []
        for f in npz_files:
            arr = np.load(f)["target"]
            samples.append(arr)
        return samples, "npz_directory", len(npz_files)

    # ── HF Arrow ───────────────────────────────────────────────────────────────
    if (folder / "dataset_dict.json").exists() or (folder / "dataset_info.json").exists():
        ds, fmt = _load_arrow(folder, split)
        return ds, fmt, len(ds)

    raise ValueError(
        f"Cannot determine dataset format for {folder}.\n"
        "Expected either sample_*.npz files or an HF Arrow dataset directory."
    )


def _get_target(ds, idx: int) -> np.ndarray:
    """Extract target array (1-D or 2-D) from an HF Dataset row."""
    row = ds[int(idx)]
    arr = np.array(row["target"], dtype=np.float64)
    return arr   # shape (k, L) if 2D, (L,) if 1D


def collect_samples(ds, total: int, max_samples: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Draw up to max_samples rows from an HF Dataset."""
    n = min(total, max_samples)
    indices = rng.choice(total, size=n, replace=False)
    samples = []
    for idx in indices:
        try:
            arr = _get_target(ds, idx)
            samples.append(arr)
        except Exception as exc:
            log.debug("Row %d failed: %s", idx, exc)
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Statistics computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(samples: list[np.ndarray]) -> dict:
    """Compute per-sample and aggregate statistics."""
    lengths, n_variates_list = [], []
    per_var_stds: list[list[float]] = []
    per_var_means: list[list[float]] = []
    cross_corrs: list[float] = []      # mean |off-diagonal corr| per multivariate sample
    nan_inf_count = 0
    zero_var_count = 0
    empty_count = 0

    for arr in samples:
        if arr is None or arr.size == 0:
            empty_count += 1
            continue
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            nan_inf_count += 1
            continue

        arr2d = arr.reshape(-1, arr.shape[-1])   # (k, L)
        k, L = arr2d.shape
        lengths.append(L)
        n_variates_list.append(k)

        stds = np.std(arr2d, axis=1)
        means = np.mean(arr2d, axis=1)

        if np.any(stds < 1e-12):
            zero_var_count += 1

        per_var_stds.append(stds.tolist())
        per_var_means.append(means.tolist())

        if k > 1:
            corr = np.corrcoef(arr2d)
            n = corr.shape[0]
            off = [corr[i, j] for i in range(n) for j in range(n) if i != j]
            cross_corrs.append(float(np.mean(np.abs(off))))

    lengths_arr = np.array(lengths)
    all_stds = np.concatenate(per_var_stds) if per_var_stds else np.array([])

    n_valid = len(lengths)
    mode_variates = int(np.bincount(n_variates_list).argmax()) if n_variates_list else 0

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    return {
        "n_valid":          n_valid,
        "n_empty":          empty_count,
        "n_nan_inf":        nan_inf_count,
        "n_zero_variance":  zero_var_count,
        "mode_variates":    mode_variates,
        "n_variates_list":  n_variates_list,
        "lengths": {
            "min":  int(lengths_arr.min()) if len(lengths_arr) else 0,
            "max":  int(lengths_arr.max()) if len(lengths_arr) else 0,
            "mean": float(lengths_arr.mean()) if len(lengths_arr) else 0,
            "std":  float(lengths_arr.std())  if len(lengths_arr) else 0,
            "p25":  pct(lengths_arr, 25),
            "p50":  pct(lengths_arr, 50),
            "p75":  pct(lengths_arr, 75),
        },
        "std_per_variate": {
            "min":  float(all_stds.min())  if len(all_stds) else 0,
            "max":  float(all_stds.max())  if len(all_stds) else 0,
            "mean": float(all_stds.mean()) if len(all_stds) else 0,
            "p50":  pct(all_stds, 50),
        },
        "cross_corr": {
            "mean": float(np.mean(cross_corrs)) if cross_corrs else float("nan"),
            "std":  float(np.std(cross_corrs))  if cross_corrs else float("nan"),
            "p10":  pct(cross_corrs, 10),
            "p50":  pct(cross_corrs, 50),
            "p90":  pct(cross_corrs, 90),
        } if cross_corrs else None,
        "per_var_stds":  per_var_stds,
        "per_var_means": per_var_means,
        "cross_corrs":   cross_corrs,
        "lengths_raw":   lengths,
    }


def print_stats(stats: dict, folder: Path, fmt: str, total: int, max_samples: int) -> None:
    n = stats["n_valid"]
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Dataset Statistics")
    print(f"  Path   : {folder}")
    print(f"  Format : {fmt}")
    print(sep)
    print(f"  Total rows in dataset  : {total:,}")
    print(f"  Analysed (random sample): {n + stats['n_nan_inf'] + stats['n_empty']:,}"
          f"  (max_samples={max_samples})")
    print(f"  Valid samples          : {n:,}")
    print(f"  Skipped — NaN/Inf      : {stats['n_nan_inf']}")
    print(f"  Skipped — Empty        : {stats['n_empty']}")
    print(f"  Samples w/ zero var    : {stats['n_zero_variance']}")
    print()
    print(f"  Dimensionality (mode)  : {stats['mode_variates']}D")
    if len(set(stats["n_variates_list"])) > 1:
        from collections import Counter
        print(f"  Variate counts         : {dict(Counter(stats['n_variates_list']))}")
    print()
    L = stats["lengths"]
    print("  Time-series length")
    print(f"    min / mean ± std / max : {L['min']:,} / {L['mean']:,.1f} ± {L['std']:,.1f} / {L['max']:,}")
    print(f"    p25 / p50 / p75        : {L['p25']:,.0f} / {L['p50']:,.0f} / {L['p75']:,.0f}")
    print()
    S = stats["std_per_variate"]
    print("  Std dev per variate  (across all samples and variates)")
    print(f"    min / mean / p50 / max : {S['min']:.4f} / {S['mean']:.4f} / {S['p50']:.4f} / {S['max']:.4f}")
    print()
    if stats["cross_corr"] is not None:
        C = stats["cross_corr"]
        print("  Mean |cross-variate correlation|  (per sample, then aggregated)")
        print(f"    mean ± std : {C['mean']:.4f} ± {C['std']:.4f}")
        print(f"    p10 / p50 / p90 : {C['p10']:.4f} / {C['p50']:.4f} / {C['p90']:.4f}")
    else:
        print("  Cross-variate correlation : N/A  (univariate data)")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting — sample viewer
# ──────────────────────────────────────────────────────────────────────────────

_VARIATE_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
_VARIATE_LABELS = ["v₁", "v₂", "v₃", "v₄", "v₅"]


def _vlabels(k: int) -> list[str]:
    """Return k variate label strings, extending beyond the preset list if needed."""
    return [_VARIATE_LABELS[j] if j < len(_VARIATE_LABELS) else f"v{j+1}" for j in range(k)]


def _plot_one_sample(ax_ts, ax_corr, arr: np.ndarray, title: str) -> None:
    """Plot one sample's time series and (optionally) correlation heatmap."""
    import matplotlib.pyplot as plt

    arr2d = arr.reshape(-1, arr.shape[-1])    # (k, L)
    k, L = arr2d.shape
    t = np.arange(L)

    # ── time-series panel ────────────────────────────────────────────────────
    for j in range(k):
        label = _vlabels(k)[j] if k > 1 else "target"
        color = _VARIATE_COLORS[j % len(_VARIATE_COLORS)]
        ax_ts.plot(t, arr2d[j], lw=0.8, alpha=0.85, color=color, label=label)

    ax_ts.set_title(title, fontsize=9, pad=3)
    ax_ts.set_xlabel("time step", fontsize=7)
    ax_ts.set_ylabel("value", fontsize=7)
    ax_ts.tick_params(labelsize=6)
    if k > 1:
        ax_ts.legend(fontsize=6, loc="upper right", framealpha=0.6)
    ax_ts.grid(True, lw=0.3, alpha=0.4)

    # ── correlation heatmap ──────────────────────────────────────────────────
    if ax_corr is None:
        return
    if k > 1:
        corr = np.corrcoef(arr2d)
        im = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
        ax_corr.set_xticks(range(k))
        ax_corr.set_yticks(range(k))
        ax_corr.set_xticklabels(_vlabels(k), fontsize=7)
        ax_corr.set_yticklabels(_vlabels(k), fontsize=7)
        ax_corr.set_title("corr", fontsize=8, pad=2)
        for i in range(k):
            for j in range(k):
                ax_corr.text(j, i, f"{corr[i, j]:.2f}",
                             ha="center", va="center", fontsize=6,
                             color="white" if abs(corr[i, j]) > 0.5 else "black")
    else:
        ax_corr.axis("off")
        ax_corr.text(0.5, 0.5, "univariate", ha="center", va="center",
                     fontsize=8, transform=ax_corr.transAxes)


def plot_samples(
    chosen: list[np.ndarray],
    folder: Path,
    output_path: Path,
    show: bool,
) -> None:
    """Figure 1: 3-column grid showing time series + correlation heatmaps."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = len(chosen)
    multivariate = any(a.ndim == 2 and a.shape[0] > 1 for a in chosen)

    n_rows = 2 if multivariate else 1
    fig, axes = plt.subplots(
        n_rows, n_samples,
        figsize=(5 * n_samples, 3.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"Random Sample Viewer — {folder.name}", fontsize=11, y=1.01)

    for col, arr in enumerate(chosen):
        arr2d = arr.reshape(-1, arr.shape[-1])
        k, L = arr2d.shape
        title = f"sample {col + 1}  (shape {k}×{L})"
        ax_ts   = axes[0][col]
        ax_corr = axes[1][col] if multivariate else None
        _plot_one_sample(ax_ts, ax_corr, arr, title)

    fig.tight_layout()

    if show:
        plt.show()
    else:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Saved sample plot → %s", output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting — aggregate statistics
# ──────────────────────────────────────────────────────────────────────────────

def plot_stats(
    samples: list[np.ndarray],
    stats: dict,
    folder: Path,
    output_path: Path,
    show: bool,
) -> None:
    """Figure 2: distribution plots (length, std, correlation, value)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    multivariate = stats["mode_variates"] > 1
    n_cols = 4 if multivariate else 3
    fig = plt.figure(figsize=(4.5 * n_cols, 7))
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"Dataset Statistics — {folder.name}", fontsize=12)

    lengths = np.array(stats["lengths_raw"])
    all_stds = np.concatenate(stats["per_var_stds"]) if stats["per_var_stds"] else np.array([])
    all_means = np.concatenate(stats["per_var_means"]) if stats["per_var_means"] else np.array([])

    # ── Row 0, Col 0: Length distribution ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(lengths, bins=min(40, max(10, len(lengths) // 10)),
            color="#2196F3", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(np.mean(lengths), color="red", lw=1.2, linestyle="--", label=f"mean={np.mean(lengths):.0f}")
    ax.set_title("Series Length Distribution", fontsize=9)
    ax.set_xlabel("length (timesteps)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── Row 0, Col 1: Std distribution ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(all_stds, bins=min(40, max(10, len(all_stds) // 5)),
            color="#FF5722", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(np.median(all_stds), color="darkred", lw=1.2, linestyle="--",
               label=f"median={np.median(all_stds):.3f}")
    ax.set_title("Std Dev Distribution (all variates)", fontsize=9)
    ax.set_xlabel("std dev", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── Row 0, Col 2: Mean distribution ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(all_means, bins=min(40, max(10, len(all_means) // 5)),
            color="#4CAF50", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(np.mean(all_means), color="darkgreen", lw=1.2, linestyle="--",
               label=f"mean={np.mean(all_means):.3f}")
    ax.set_title("Mean Distribution (all variates)", fontsize=9)
    ax.set_xlabel("mean value", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── Row 0, Col 3 (multivariate only): Cross-corr distribution ────────────
    if multivariate and n_cols == 4:
        ax = fig.add_subplot(gs[0, 3])
        corrs = stats["cross_corrs"]
        if corrs:
            ax.hist(corrs, bins=min(30, max(5, len(corrs) // 5)),
                    color="#9C27B0", edgecolor="white", linewidth=0.4, alpha=0.85)
            ax.axvline(np.mean(corrs), color="purple", lw=1.2, linestyle="--",
                       label=f"mean={np.mean(corrs):.3f}")
            ax.set_title("Mean |Cross-Variate Corr|", fontsize=9)
            ax.set_xlabel("|correlation|", fontsize=8)
            ax.set_ylabel("count (samples)", fontsize=8)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no multivariate samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title("Mean |Cross-Variate Corr|", fontsize=9)
        ax.tick_params(labelsize=7)

    # ── Row 1, Col 0: Per-variate std box plot ───────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    n_var = stats["mode_variates"]
    if n_var > 1 and stats["per_var_stds"]:
        by_var = [[] for _ in range(n_var)]
        for row_stds in stats["per_var_stds"]:
            for j, s in enumerate(row_stds[:n_var]):
                by_var[j].append(s)
        bp = ax.boxplot(by_var, labels=_vlabels(n_var),
                        patch_artist=True, medianprops={"color": "white", "lw": 1.5})
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_VARIATE_COLORS[j % len(_VARIATE_COLORS)])
        ax.set_title("Std Dev per Variate", fontsize=9)
        ax.set_ylabel("std dev", fontsize=8)
    else:
        ax.hist(all_stds, bins=20, color="#FF5722", alpha=0.8, edgecolor="white")
        ax.set_title("Std Dev Distribution", fontsize=9)
        ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── Row 1, Col 1: Value range summary (violin or summary stats) ───────────
    ax = fig.add_subplot(gs[1, 1])
    # Sample a subset of values to build value distribution
    subset_vals = []
    for arr in samples[:200]:
        arr2d = arr.reshape(-1, arr.shape[-1])
        subset_vals.extend(arr2d.flatten().tolist())
    subset_vals = np.array(subset_vals)
    if len(subset_vals) > 50000:
        subset_vals = np.random.choice(subset_vals, size=50000, replace=False)
    ax.hist(np.clip(subset_vals, np.percentile(subset_vals, 1), np.percentile(subset_vals, 99)),
            bins=60, color="#607D8B", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_title("Value Distribution (clipped p1–p99)", fontsize=9)
    ax.set_xlabel("value", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── Row 1, Col 2: Dimensionality / quick summary text ─────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    L = stats["lengths"]
    S = stats["std_per_variate"]
    lines = [
        f"Dataset: {folder.name}",
        "",
        f"Total rows      : {stats['n_valid'] + stats['n_nan_inf'] + stats['n_empty']:,}  (analysed)",
        f"Valid samples   : {stats['n_valid']:,}",
        f"Skipped NaN/Inf : {stats['n_nan_inf']}",
        f"Skipped empty   : {stats['n_empty']}",
        f"Zero variance   : {stats['n_zero_variance']}",
        "",
        f"Dimensionality  : {stats['mode_variates']}D",
        "",
        "Length (timesteps)",
        f"  min  : {L['min']:,}",
        f"  mean : {L['mean']:,.1f}",
        f"  max  : {L['max']:,}",
        "",
        "Std Dev (all variates)",
        f"  min  : {S['min']:.4f}",
        f"  mean : {S['mean']:.4f}",
        f"  max  : {S['max']:.4f}",
    ]
    if stats["cross_corr"]:
        C = stats["cross_corr"]
        lines += [
            "",
            "Cross-Variate |corr|",
            f"  mean : {C['mean']:.4f}",
            f"  p50  : {C['p50']:.4f}",
        ]
    ax.text(0.05, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left", family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F5F5F5", "edgecolor": "#BDBDBD"})

    # ── Row 1, Col 3 (multivariate): mean per-variate correlation heatmap ─────
    if multivariate and n_cols == 4:
        ax = fig.add_subplot(gs[1, 3])
        multivar_samples = [a for a in samples if a.ndim == 2 and a.shape[0] > 1]
        if multivar_samples:
            k_ref = stats["mode_variates"]
            corr_sum = np.zeros((k_ref, k_ref))
            count = 0
            for arr in multivar_samples[:500]:
                arr2d = arr.reshape(-1, arr.shape[-1])
                if arr2d.shape[0] == k_ref:
                    corr_sum += np.corrcoef(arr2d)
                    count += 1
            if count > 0:
                mean_corr = corr_sum / count
                im = ax.imshow(mean_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(k_ref))
                ax.set_yticks(range(k_ref))
                ax.set_xticklabels(_vlabels(k_ref), fontsize=8)
                ax.set_yticklabels(_vlabels(k_ref), fontsize=8)
                ax.set_title(f"Mean Corr Matrix\n(n={count} samples)", fontsize=9)
                for i in range(k_ref):
                    for j in range(k_ref):
                        ax.text(j, i, f"{mean_corr[i, j]:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="white" if abs(mean_corr[i, j]) > 0.5 else "black")

    if show:
        plt.show()
    else:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Saved stats plot  → %s", output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a time-series dataset: print statistics and plot samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", help="Dataset folder (Arrow or .npz directory)")
    p.add_argument("--max_samples", type=int, default=1000,
                   help="Max rows to analyse (default: 1000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for PNGs (default: dataset folder)")
    p.add_argument("--show", action="store_true",
                   help="Display figures interactively instead of saving")
    p.add_argument("--split", default="train",
                   help="HF Dataset split to use (default: train)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).resolve()
    if not folder.exists():
        log.error("Path does not exist: %s", folder)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else folder
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading dataset from: %s", folder)
    raw, fmt, total = load_dataset(folder, args.split)
    log.info("Format: %s | Total rows: %d", fmt, total)

    # ── Collect sample arrays ─────────────────────────────────────────────────
    if isinstance(raw, list):   # already a list of np.ndarray (npz mode)
        all_arrays = raw
        n_to_analyse = min(len(all_arrays), args.max_samples)
        chosen_indices = rng.choice(len(all_arrays), size=n_to_analyse, replace=False)
        samples = [all_arrays[i] for i in chosen_indices]
    else:
        log.info("Sampling up to %d rows for analysis …", args.max_samples)
        samples = collect_samples(raw, total, args.max_samples, rng)

    log.info("Collected %d samples for analysis.", len(samples))
    if not samples:
        log.error("No valid samples found.")
        sys.exit(1)

    # ── Statistics ────────────────────────────────────────────────────────────
    stats = compute_stats(samples)
    print_stats(stats, folder, fmt, total, args.max_samples)

    # ── Choose 3 random samples for plotting ──────────────────────────────────
    n_plot = min(3, len(samples))
    plot_indices = rng.choice(len(samples), size=n_plot, replace=False)
    chosen = [samples[i] for i in plot_indices]

    # ── Plot 1: sample viewer ─────────────────────────────────────────────────
    plot_samples(
        chosen,
        folder,
        output_dir / "samples.png",
        show=args.show,
    )

    # ── Plot 2: aggregate stats ───────────────────────────────────────────────
    plot_stats(
        samples,
        stats,
        folder,
        output_dir / "stats.png",
        show=args.show,
    )

    if not args.show:
        log.info("Done.  Figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
