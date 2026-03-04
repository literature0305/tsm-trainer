#!/usr/bin/env python3
"""
inspect_benchmarks.py — Aggregate statistics across all unified benchmark datasets
===================================================================================

Collects samples from every dataset under benchmarks_unified/ and produces:
  • Printed summary table (one row per benchmark type + grand total)
  • benchmarks_stats.png  — aggregate distribution plots split by univariate / multivariate
  • benchmarks_length_by_source.png  — length distributions per benchmark type

Optionally compares against training datasets (kernel_synth_3d, composite_synth_3d).

USAGE
-----
  python inspect_benchmarks.py \\
      --benchmarks_root /group-volume/ts-dataset/benchmarks_unified \\
      [--train_paths PATH ...]          # optional training sets for comparison
      [--max_per_dataset N]             # samples per dataset (default: 200)
      [--output_dir PATH]               # default: benchmarks_root
      [--seed N]

EXAMPLE
-------
  python inspect_benchmarks.py \\
      --benchmarks_root /group-volume/ts-dataset/benchmarks_unified \\
      --train_paths \\
          /group-volume/ts-dataset/chronos2_datasets/kernel_synth_3d \\
          /group-volume/ts-dataset/chronos2_datasets/composite_synth_3d \\
      --max_per_dataset 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_hf_dataset_dir(p: Path) -> bool:
    return p.is_dir() and (
        (p / "dataset_dict.json").exists() or
        (p / "dataset_info.json").exists()
    )


def _load_samples(ds_dir: Path, max_n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Load up to max_n target arrays from an HF Arrow dataset."""
    import datasets as hf
    try:
        raw = hf.load_from_disk(str(ds_dir))
        ds = raw["train"] if isinstance(raw, hf.DatasetDict) else raw
        total = len(ds)
        n = min(total, max_n)
        indices = rng.choice(total, size=n, replace=False)
        out = []
        for idx in indices:
            try:
                arr = np.array(ds[int(idx)]["target"], dtype=np.float64)
                if arr.size > 0 and not (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
                    out.append(arr)
            except Exception:
                pass
        return out
    except Exception as exc:
        log.debug("  [skip] %s: %s", ds_dir.name, exc)
        return []


def _collect_group(root: Path, max_per: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Collect samples from all HF Arrow datasets directly under root."""
    samples = []
    dirs = sorted(d for d in root.iterdir() if _is_hf_dataset_dir(d))
    for d in dirs:
        s = _load_samples(d, max_per, rng)
        samples.extend(s)
    log.info("  %s: %d datasets → %d samples", root.name, len(dirs), len(samples))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_stats(samples: list[np.ndarray]) -> dict:
    lengths, stds, means, n_vars, cross_corrs = [], [], [], [], []
    for arr in samples:
        arr2d = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr[np.newaxis, :]
        k, L = arr2d.shape
        lengths.append(L)
        n_vars.append(k)
        s = np.std(arr2d, axis=1)
        m = np.mean(arr2d, axis=1)
        stds.extend(s.tolist())
        means.extend(m.tolist())
        if k > 1:
            corr = np.corrcoef(arr2d)
            n = corr.shape[0]
            off = [abs(corr[i, j]) for i in range(n) for j in range(n) if i != j]
            cross_corrs.append(float(np.mean(off)))

    if not lengths:
        return {}

    la = np.array(lengths)
    sa = np.array(stds)
    ma = np.array(means)
    return {
        "n":          len(lengths),
        "lengths":    la,
        "stds":       sa,
        "means":      ma,
        "n_vars":     np.array(n_vars),
        "cross_corrs": np.array(cross_corrs) if cross_corrs else None,
        # scalar summaries
        "len_min":   int(la.min()),
        "len_mean":  float(la.mean()),
        "len_median":float(np.median(la)),
        "len_max":   int(la.max()),
        "std_median":float(np.median(sa)),
        "std_max":   float(sa.max()),
        "pct_multi": 100.0 * np.mean(np.array(n_vars) > 1),
        "corr_mean": float(np.mean(cross_corrs)) if cross_corrs else float("nan"),
    }


def _print_table(groups: dict[str, dict]) -> None:
    header = f"{'Group':<22} {'N':>7} {'%Multi':>7} {'L_min':>7} {'L_med':>8} "
    header += f"{'L_max':>8} {'std_med':>9} {'|corr|_mean':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  Benchmark Statistics Summary")
    print(sep)
    print(header)
    print(sep)
    for name, st in groups.items():
        if not st:
            print(f"  {name:<20}  (no samples)")
            continue
        corr_str = f"{st['corr_mean']:.3f}" if not np.isnan(st['corr_mean']) else "  n/a "
        print(
            f"  {name:<20} {st['n']:>7,} {st['pct_multi']:>7.1f}"
            f" {st['len_min']:>7,} {st['len_median']:>8,.0f}"
            f" {st['len_max']:>8,} {st['std_median']:>9.3f} {corr_str:>12}"
        )
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "chronos":          "#2196F3",
    "fev_bench":        "#FF5722",
    "gift_eval":        "#4CAF50",
    "ltsf":             "#9C27B0",
    "kernel_synth_3d":  "#FF9800",
    "composite_synth_3d": "#795548",
}


def _plot_aggregate(groups: dict[str, dict], output_path: Path) -> None:
    """Single-figure 3×2 grid: lengths, stds, means, cross_corr, n_vars, summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Benchmark + Training Data: Aggregate Statistics", fontsize=13)

    ax_len, ax_std, ax_mean = axes[0]
    ax_corr, ax_nvar, ax_summary = axes[1]

    def _hist(ax, key, title, xlabel, log_scale=False, xlim_pct=None):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        for name, st in groups.items():
            if not st or key not in st or st[key] is None:
                continue
            data = st[key]
            if len(data) == 0:
                continue
            color = _COLORS.get(name, "#607D8B")
            if xlim_pct:
                lo, hi = np.percentile(data, xlim_pct[0]), np.percentile(data, xlim_pct[1])
                data = data[(data >= lo) & (data <= hi)]
            ax.hist(data, bins=50, density=True, alpha=0.55,
                    color=color, edgecolor="none", label=name)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)

    _hist(ax_len, "lengths", "Series Length Distribution", "length (timesteps)")
    _hist(ax_std, "stds",    "Std Dev Distribution (all variates)", "std dev", xlim_pct=(1, 99))
    _hist(ax_mean, "means",  "Mean Distribution (all variates)", "mean value", xlim_pct=(1, 99))

    # Cross-variate correlation
    ax_corr.set_title("Mean |Cross-Variate Corr| (multivariate only)", fontsize=9)
    ax_corr.set_xlabel("|correlation|", fontsize=8)
    ax_corr.set_ylabel("density", fontsize=8)
    ax_corr.tick_params(labelsize=7)
    has_corr = False
    for name, st in groups.items():
        if not st or st.get("cross_corrs") is None or len(st["cross_corrs"]) == 0:
            continue
        color = _COLORS.get(name, "#607D8B")
        ax_corr.hist(st["cross_corrs"], bins=30, density=True, alpha=0.55,
                     color=color, edgecolor="none", label=name)
        has_corr = True
    if not has_corr:
        ax_corr.text(0.5, 0.5, "no multivariate data", ha="center", va="center",
                     transform=ax_corr.transAxes)
    ax_corr.legend(fontsize=6)
    ax_corr.grid(True, lw=0.3, alpha=0.4)

    # Variate count distribution
    ax_nvar.set_title("Variate Count Distribution", fontsize=9)
    ax_nvar.set_xlabel("n_variates", fontsize=8)
    ax_nvar.set_ylabel("fraction", fontsize=8)
    ax_nvar.tick_params(labelsize=7)
    all_n_vars = []
    for name, st in groups.items():
        if st and "n_vars" in st:
            all_n_vars.extend(st["n_vars"].tolist())
    if all_n_vars:
        from collections import Counter
        cnt = Counter(all_n_vars)
        k_vals = sorted(cnt)
        totals = sum(cnt.values())
        ax_nvar.bar(k_vals, [cnt[k] / totals for k in k_vals], color="#607D8B", alpha=0.8)

    # Summary text panel
    ax_summary.axis("off")
    lines = ["Scalar Summary\n"]
    for name, st in groups.items():
        if not st:
            continue
        lines.append(f"[{name}]")
        lines.append(f"  N={st['n']:,}  %multi={st['pct_multi']:.0f}%")
        lines.append(f"  L: {st['len_min']:,}–{st['len_median']:.0f}–{st['len_max']:,}")
        lines.append(f"  std_median={st['std_median']:.3f}")
        if not np.isnan(st['corr_mean']):
            lines.append(f"  |corr|_mean={st['corr_mean']:.3f}")
        lines.append("")
    ax_summary.text(0.04, 0.98, "\n".join(lines), transform=ax_summary.transAxes,
                    fontsize=7, va="top", ha="left", family="monospace",
                    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F5F5F5",
                          "edgecolor": "#BDBDBD"})

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", output_path)
    plt.close(fig)


def _plot_length_by_source(groups: dict[str, dict], output_path: Path) -> None:
    """CDF of series lengths, one line per group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Series Length Distribution by Source", fontsize=12)

    for ax, log_scale in zip(axes, (False, True)):
        for name, st in groups.items():
            if not st or "lengths" not in st or len(st["lengths"]) == 0:
                continue
            data = np.sort(st["lengths"])
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, label=f"{name} (n={len(data):,})",
                    color=_COLORS.get(name, "#607D8B"), lw=1.6)
        ax.set_ylabel("CDF", fontsize=9)
        ax.set_xlabel("series length (timesteps)", fontsize=9)
        if log_scale:
            ax.set_xscale("log")
            ax.set_title("CDF (log x-axis)", fontsize=9)
        else:
            ax.set_title("CDF (linear)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
        # Mark common training lengths
        for L, lbl in [(128, "128"), (512, "512"), (2048, "2k"), (4096, "4k"), (8192, "8k")]:
            ax.axvline(L, lw=0.8, linestyle="--", color="#BDBDBD", alpha=0.7)
            ax.text(L, 0.02, lbl, fontsize=6, color="#888888", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate benchmark statistics for train/eval comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--benchmarks_root",
                   default="/group-volume/ts-dataset/benchmarks_unified",
                   help="Root of unified benchmark datasets")
    p.add_argument("--train_paths", nargs="*", default=[],
                   help="Optional training dataset paths for comparison")
    p.add_argument("--max_per_dataset", type=int, default=200,
                   help="Max samples per individual dataset (default: 200)")
    p.add_argument("--output_dir", default=None,
                   help="Where to save PNGs (default: benchmarks_root)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    broot = Path(args.benchmarks_root)
    output_dir = Path(args.output_dir) if args.output_dir else broot
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    groups: dict[str, dict] = {}

    # ── Benchmarks ────────────────────────────────────────────────────────────
    for bench_name in ["chronos", "fev_bench", "gift_eval", "ltsf"]:
        src = broot / bench_name
        if not src.exists():
            log.warning("[skip] %s not found", src)
            continue
        log.info("Loading %s …", bench_name)
        samples = _collect_group(src, args.max_per_dataset, rng)
        groups[bench_name] = _compute_stats(samples) if samples else {}

    # ── Training datasets (optional) ──────────────────────────────────────────
    for train_path in args.train_paths:
        tp = Path(train_path)
        name = tp.name
        log.info("Loading training: %s …", name)
        samples = _load_samples(tp, args.max_per_dataset * 5, rng)  # more samples for training
        groups[name] = _compute_stats(samples) if samples else {}
        log.info("  %s: %d samples", name, groups[name].get("n", 0))

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_table(groups)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_aggregate(groups, output_dir / "benchmarks_stats.png")
    _plot_length_by_source(groups, output_dir / "benchmarks_length_cdf.png")

    log.info("Done. Output at: %s", output_dir)


if __name__ == "__main__":
    main()
