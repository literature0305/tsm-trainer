#!/usr/bin/env python3
"""Final Comprehensive Analysis for APC Occupancy Classification.

Generates 12+ publication-quality figures covering:
- Cross-phase experiment comparison (CSV-only, no GPU)
- Embedding space analysis with t-SNE (requires GPU + MantisV2)

Input: 15 experiment CSV files from Groups A-O (Phases 1-3).
Output: 12+ figures saved as PNG + PDF.

Usage:
    cd examples/classification/apc_occupancy

    # CSV-only analysis (no GPU needed)
    python analysis/run_final_analysis.py \
        --csv-dir results/phase3_additional \
        --output-dir results/final_analysis/plots \
        --csv-only

    # Full analysis including embeddings (requires GPU)
    python analysis/run_final_analysis.py \
        --config training/configs/occupancy-phase2.yaml \
        --csv-dir results/phase3_additional \
        --output-dir results/final_analysis/plots \
        --device cuda

    # Select specific figures
    python analysis/run_final_analysis.py \
        --figures 1 2 3 \
        --csv-dir results/phase3_additional

    # With custom Phase 1 and Phase 2 CSV directories
    python analysis/run_final_analysis.py \
        --csv-dir results/phase3_additional \
        --csv-dir-p1 results/phase1_sweep \
        --csv-dir-p2 results/phase2_sweep \
        --output-dir results/final_analysis/plots
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib as mpl

# Headless backend for GPU servers without DISPLAY
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path setup — run from apc_occupancy/ directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent  # apc_occupancy/
sys.path.insert(0, str(_PROJECT_DIR))

from visualization.style import (
    setup_style,
    save_figure,
    configure_output,
    CLASS_COLORS,
    CLASS_NAMES,
    ACCENT_COLOR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DPI = 300

# CSV filename map: group letter -> expected filename stem
CSV_FILENAMES = {
    "A": "group_a_context_sweep",
    "B": "group_b_channel_sweep",
    "C": "group_c_classifier_layer",
    "D": "group_d_aug_head",
    "E": "group_e_context",
    "F": "group_f_layer_fusion",
    "G": "group_g_training_hp",
    "H": "group_h_aug_deep",
    "I": "group_i_tta_ensemble",
    "J": "group_j_training_fix",
    "K": "group_k_layer_classifier",
    "L": "group_l_context_tuning",
    "M": "group_m_mlp_recipe",
    "N": "group_n_svm_grid",
    "O": "group_o_final",
}

# Classifier type -> color mapping for consistent visuals across figures
CLF_PALETTE = {
    "SVM": "#0173B2",
    "SVM_rbf": "#0173B2",
    "MLP": "#CC3311",
    "LogReg": "#029E73",
    "RF": "#7F7F7F",
    "ExtraTrees": "#9467BD",
    "Linear": "#DE8F05",
    "GBT": "#8C564B",
    "NearestCentroid": "#E377C2",
    # fallback
    "other": "#BCBD22",
}

# Phase boundaries: which groups belong to which phase
PHASE_GROUPS = {
    1: ["A", "B", "C"],
    2: ["D", "E", "F", "G", "H", "I"],
    3: ["J", "K", "L", "M", "N", "O"],
}


# ============================================================================
# CSV Loading
# ============================================================================

def _find_csv(group: str, search_dirs: list[Path]) -> Path | None:
    """Search for a group CSV file across multiple directories.

    Looks in each directory's ``tables/`` subdirectory first, then the
    directory root.
    """
    stem = CSV_FILENAMES.get(group)
    if stem is None:
        return None

    for base in search_dirs:
        for subdir in [base / "tables", base]:
            candidate = subdir / f"{stem}.csv"
            if candidate.exists():
                return candidate
    return None


def load_all_csvs(
    csv_dirs: list[Path],
) -> dict[str, pd.DataFrame]:
    """Load all available experiment CSV files from the search directories.

    Returns a dict mapping group letter to its DataFrame.
    """
    loaded = {}
    for group in CSV_FILENAMES:
        path = _find_csv(group, csv_dirs)
        if path is not None:
            try:
                df = pd.read_csv(path)
                loaded[group] = df
                logger.info("Loaded group %s: %s (%d rows)", group, path, len(df))
            except Exception as e:
                logger.warning("Failed to load group %s from %s: %s", group, path, e)
        else:
            logger.debug("Group %s CSV not found in search paths", group)

    logger.info(
        "Loaded %d/%d group CSVs: %s",
        len(loaded), len(CSV_FILENAMES), sorted(loaded.keys()),
    )
    return loaded


def _classify_clf_type(name: str) -> str:
    """Map a classifier name to a high-level type for coloring."""
    n = str(name).upper()
    if "SVM" in n:
        return "SVM"
    if "MLP" in n:
        return "MLP"
    if "LOGREG" in n or "LOGISTIC" in n:
        return "LogReg"
    if "RF" in n or "RANDOM_FOREST" in n:
        return "RF"
    if "EXTRA" in n:
        return "ExtraTrees"
    if "LINEAR" in n:
        return "Linear"
    if "GBT" in n or "GRADIENT" in n:
        return "GBT"
    if "NEAREST" in n:
        return "NearestCentroid"
    return "other"


def _get_clf_color(name: str) -> str:
    """Get color for a classifier name."""
    clf_type = _classify_clf_type(name)
    return CLF_PALETTE.get(clf_type, CLF_PALETTE["other"])


def _combine_all_results(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all group DataFrames into a single ranked DataFrame."""
    frames = []
    for group_key, df in groups.items():
        df_copy = df.copy()
        if "group" not in df_copy.columns:
            df_copy["group"] = group_key
        frames.append(df_copy)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


# ============================================================================
# Publication-quality style setup
# ============================================================================

def setup_pub_style():
    """Apply enhanced publication-quality matplotlib rcParams."""
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "grid.color": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "white",
    })


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as PNG + PDF, then close."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=DPI, format=fmt, bbox_inches="tight")
    logger.info("Saved: %s.{{png,pdf}}", output_dir / name)
    plt.close(fig)


# ============================================================================
# Part A: Cross-Phase Comparison (CSV-only, no GPU)
# ============================================================================

def fig01_grand_overview(groups: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Fig 1: Top-30 experiments across all groups by AUC."""
    logger.info("Generating Fig 1: Grand Overview (Top-30 by AUC)")

    combined = _combine_all_results(groups)
    if combined.empty or "auc" not in combined.columns:
        logger.warning("Fig 1: No AUC data available, skipping")
        return

    valid = combined.dropna(subset=["auc"]).copy()
    if valid.empty:
        logger.warning("Fig 1: All AUC values are NaN, skipping")
        return

    top30 = valid.sort_values("auc", ascending=True).tail(30).reset_index(drop=True)

    # Build labels
    labels = []
    for _, row in top30.iterrows():
        group = row.get("group", "?")
        name = str(row.get("name", ""))
        # Truncate long names
        if len(name) > 50:
            name = name[:47] + "..."
        labels.append(f"[{group}] {name}")

    # Determine classifier type for coloring
    clf_types = []
    for _, row in top30.iterrows():
        name = str(row.get("name", "")) + str(row.get("classifier", ""))
        clf_types.append(_classify_clf_type(name))

    colors = [CLF_PALETTE.get(ct, CLF_PALETTE["other"]) for ct in clf_types]

    fig_height = max(6, 0.35 * len(top30))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(top30))
    bars = ax.barh(y_pos, top30["auc"].values, color=colors, height=0.7,
                   edgecolor="white", linewidth=0.5)

    # Annotate values
    for i, (val, bar) in enumerate(zip(top30["auc"].values, bars)):
        ax.text(val + 0.0008, i, f"{val:.4f}", va="center", fontsize=7,
                fontweight="bold" if i == len(top30) - 1 else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("AUC", fontsize=10)
    ax.set_title("Top-30 Experiments Across All Groups (A-O)", fontsize=12,
                 fontweight="bold")

    # Mark the #1 result
    best_row = top30.iloc[-1]
    ax.annotate(
        f"BEST: AUC={best_row['auc']:.4f}",
        xy=(best_row["auc"], len(top30) - 1),
        xytext=(-80, 15), textcoords="offset points",
        fontsize=9, fontweight="bold", color="#CC3311",
        arrowprops=dict(arrowstyle="->", color="#CC3311", lw=1.5),
    )

    # Set x-axis to start near the minimum for readability
    auc_min = top30["auc"].min()
    x_start = max(0, auc_min - 0.01)
    ax.set_xlim(x_start, 1.002)

    # Legend for classifier types
    unique_types = sorted(set(clf_types))
    legend_elements = [
        Patch(facecolor=CLF_PALETTE.get(ct, CLF_PALETTE["other"]), label=ct)
        for ct in unique_types
    ]
    ax.legend(handles=legend_elements, loc="lower right", title="Classifier Type",
              fontsize=7, title_fontsize=8)

    save_fig(fig, output_dir, "fig01_grand_overview_top30")


def fig02_phase_progression(groups: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Fig 2: AUC improvement over experiment phases, split by classifier family."""
    logger.info("Generating Fig 2: Phase Progression")

    # Collect best AUC per phase, separately for neural and sklearn
    phase_data = {}
    for phase_num, group_letters in PHASE_GROUPS.items():
        neural_aucs = []
        sklearn_aucs = []
        for g in group_letters:
            if g not in groups:
                continue
            df = groups[g].dropna(subset=["auc"]) if "auc" in groups[g].columns else pd.DataFrame()
            for _, row in df.iterrows():
                auc_val = row["auc"]
                name = str(row.get("name", "")) + str(row.get("classifier", ""))
                clf_type = _classify_clf_type(name)
                if clf_type == "MLP" or clf_type == "Linear":
                    neural_aucs.append(auc_val)
                else:
                    sklearn_aucs.append(auc_val)

        phase_data[phase_num] = {
            "neural_best": max(neural_aucs) if neural_aucs else None,
            "sklearn_best": max(sklearn_aucs) if sklearn_aucs else None,
            "neural_count": len(neural_aucs),
            "sklearn_count": len(sklearn_aucs),
        }

    phases = sorted(phase_data.keys())
    neural_vals = [phase_data[p]["neural_best"] for p in phases]
    sklearn_vals = [phase_data[p]["sklearn_best"] for p in phases]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(phases))
    width = 0.35

    # Plot bars
    neural_bars = ax.bar(
        x - width / 2,
        [v if v is not None else 0 for v in neural_vals],
        width, label="Neural (MLP/Linear)", color="#CC3311", alpha=0.85,
    )
    sklearn_bars = ax.bar(
        x + width / 2,
        [v if v is not None else 0 for v in sklearn_vals],
        width, label="Sklearn (SVM/RF/LogReg)", color="#0173B2", alpha=0.85,
    )

    # Annotate
    for bars, vals in [(neural_bars, neural_vals), (sklearn_bars, sklearn_vals)]:
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold",
                )

    # Phase group annotations
    phase_labels = []
    for p in phases:
        groups_str = ",".join(PHASE_GROUPS[p])
        n = phase_data[p]["neural_count"] + phase_data[p]["sklearn_count"]
        phase_labels.append(f"Phase {p}\n({groups_str})\nn={n}")

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=9)
    ax.set_ylabel("Best AUC", fontsize=10)
    ax.set_title("AUC Improvement Across Experiment Phases", fontsize=12,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    # Set y-axis to zoom in on the competitive range
    all_vals = [v for v in neural_vals + sklearn_vals if v is not None]
    if all_vals:
        y_min = max(0, min(all_vals) - 0.02)
        ax.set_ylim(y_min, 1.005)

    save_fig(fig, output_dir, "fig02_phase_progression")


def fig03_layer_classifier_heatmap(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 3: Layer x Classifier AUC heatmap from Group K."""
    logger.info("Generating Fig 3: Layer x Classifier Heatmap (Group K)")

    if "K" not in groups:
        logger.warning("Fig 3: Group K CSV not found, skipping")
        return

    df = groups["K"].copy()
    if "auc" not in df.columns or "layer" not in df.columns or "classifier" not in df.columns:
        logger.warning("Fig 3: Required columns missing in Group K, skipping")
        return

    # Also check "name" column as alternative to "classifier"
    if df["classifier"].isna().all() and "name" in df.columns:
        # Try extracting classifier from name (format: "L{layer}|{classifier}")
        df["classifier"] = df["name"].str.split("|").str[-1].str.strip()

    df = df.dropna(subset=["auc"])
    if df.empty:
        logger.warning("Fig 3: No valid AUC data in Group K, skipping")
        return

    # Pivot: layers as rows, classifiers as columns
    pivot = df.pivot_table(
        values="auc", index="layer", columns="classifier", aggfunc="max"
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), 5))

    # Use a diverging colormap centered on the overall mean
    vmin = pivot.min().min()
    vmax = pivot.max().max()

    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "AUC", "shrink": 0.8},
        ax=ax,
    )

    # Highlight the global best cell
    best_val = pivot.max().max()
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if not pd.isna(pivot.iloc[i, j]) and pivot.iloc[i, j] == best_val:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="#CC3311",
                    linewidth=3, clip_on=False,
                ))

    ax.set_xlabel("Classifier", fontsize=10)
    ax.set_ylabel("Layer (L0-L5)", fontsize=10)
    ax.set_title(
        f"Layer x Classifier AUC (Group K, M+C+T1, 251min)\nBest: {best_val:.4f}",
        fontsize=12, fontweight="bold",
    )
    # Use subplots_adjust instead of tight_layout (colorbar conflict)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.15)

    save_fig(fig, output_dir, "fig03_layer_classifier_heatmap")


def fig04_context_window_curves(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 4: AUC vs context window length from Groups A and L."""
    logger.info("Generating Fig 4: Context Window Performance Curves")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # --- Panel 1: Group A (Phase 1) ---
    ax1 = axes[0]
    if "A" in groups:
        df_a = groups["A"].copy()
        if "total_context_min" in df_a.columns and "auc" in df_a.columns:
            df_a = df_a.dropna(subset=["auc"])
            df_a = df_a.sort_values("total_context_min")

            # Group by subgroup if available
            if "subgroup" in df_a.columns:
                subgroup_colors = {
                    "symmetric": "#0173B2",
                    "backward": "#CC3311",
                    "asym_past": "#029E73",
                    "future_sensitivity": "#DE8F05",
                    "asym_future": "#9467BD",
                }
                for sg, color in subgroup_colors.items():
                    mask = df_a["subgroup"] == sg
                    if mask.any():
                        sub = df_a[mask].sort_values("total_context_min")
                        ax1.plot(
                            sub["total_context_min"], sub["auc"],
                            marker="o", markersize=4, lw=1.5, color=color,
                            label=sg, alpha=0.8,
                        )
            else:
                ax1.plot(
                    df_a["total_context_min"], df_a["auc"],
                    marker="o", markersize=4, lw=2, color="#0173B2",
                )

            ax1.legend(fontsize=7, loc="lower right")
        else:
            ax1.text(0.5, 0.5, "No context data in Group A",
                     ha="center", va="center", transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, "Group A not loaded",
                 ha="center", va="center", transform=ax1.transAxes)

    ax1.set_xlabel("Context Window (minutes)", fontsize=10)
    ax1.set_ylabel("AUC", fontsize=10)
    ax1.set_title("Phase 1 (Group A): Context Deep Exploration", fontsize=11)

    # --- Panel 2: Group L (Phase 3) ---
    ax2 = axes[1]
    if "L" in groups:
        df_l = groups["L"].copy()
        ctx_col = None
        for candidate in ["context_min", "total_context_min", "context_before"]:
            if candidate in df_l.columns:
                ctx_col = candidate
                break

        if ctx_col and "auc" in df_l.columns:
            df_l = df_l.dropna(subset=["auc"])

            # Derive total context if needed
            if ctx_col == "context_before" and "context_after" in df_l.columns:
                df_l["_total_ctx"] = df_l["context_before"] + 1 + df_l["context_after"]
                ctx_col = "_total_ctx"

            # Group by layer and classifier
            group_cols = []
            if "layer" in df_l.columns:
                group_cols.append("layer")
            if "classifier" in df_l.columns:
                group_cols.append("classifier")

            if group_cols:
                line_colors = [
                    "#0173B2", "#CC3311", "#029E73", "#DE8F05",
                    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
                ]
                color_idx = 0
                for key, sub in df_l.groupby(group_cols):
                    label = str(key) if isinstance(key, str) else "|".join(str(k) for k in key)
                    sub = sub.sort_values(ctx_col)
                    ax2.plot(
                        sub[ctx_col], sub["auc"],
                        marker="o", markersize=3, lw=1.5,
                        color=line_colors[color_idx % len(line_colors)],
                        label=label, alpha=0.8,
                    )
                    color_idx += 1
                ax2.legend(fontsize=6, loc="lower right", ncol=2)
            else:
                df_l = df_l.sort_values(ctx_col)
                ax2.plot(df_l[ctx_col], df_l["auc"],
                         marker="o", markersize=4, lw=2, color="#0173B2")
        else:
            ax2.text(0.5, 0.5, "Missing context/AUC columns in Group L",
                     ha="center", va="center", transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "Group L not loaded",
                 ha="center", va="center", transform=ax2.transAxes)

    ax2.set_xlabel("Context Window (minutes)", fontsize=10)
    ax2.set_title("Phase 3 (Group L): Context Fine-Tuning", fontsize=11)

    # Add golden zone shading (221-261 min) if relevant
    for ax in axes:
        ylim = ax.get_ylim()
        ax.axvspan(221, 261, alpha=0.12, color="#FFD700", label="_golden zone")
        ax.set_ylim(ylim)

    fig.suptitle("Context Window vs Classification Performance", fontsize=13,
                 fontweight="bold", y=1.02)

    save_fig(fig, output_dir, "fig04_context_window_curves")


def fig05_training_fix_impact(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 5: Training fix impact — paired comparison from Group J."""
    logger.info("Generating Fig 5: Training Fix Impact (Group J)")

    if "J" not in groups:
        logger.warning("Fig 5: Group J CSV not found, skipping")
        return

    df = groups["J"].copy()
    if "auc" not in df.columns:
        logger.warning("Fig 5: No AUC column in Group J, skipping")
        return
    df = df.dropna(subset=["auc"])
    if df.empty:
        logger.warning("Fig 5: No valid AUC data in Group J, skipping")
        return

    # Parse fix and head from name or dedicated columns
    if "fix" not in df.columns and "name" in df.columns:
        df["fix"] = df["name"].str.split("|").str[0].str.strip()
    if "head" not in df.columns and "name" in df.columns:
        df["head"] = df["name"].str.split("|").str[-1].str.strip()

    if "fix" not in df.columns or "head" not in df.columns:
        logger.warning("Fig 5: Cannot parse fix/head columns, skipping")
        return

    # Pivot for grouped bar chart
    pivot = df.pivot_table(values="auc", index="head", columns="fix", aggfunc="max")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), 5))

    fix_colors = [
        "#7F7F7F",  # old_baseline
        "#0173B2",  # fix_valES
        "#029E73",  # fix_std
        "#DE8F05",  # fix_both
        "#CC3311",  # fix_both+FroFA
        "#9467BD",  # fix_both+AdaptN
    ]

    n_heads = len(pivot.index)
    n_fixes = len(pivot.columns)
    bar_width = 0.8 / max(n_fixes, 1)
    x = np.arange(n_heads)

    for i, fix_name in enumerate(pivot.columns):
        vals = pivot[fix_name].values
        offset = (i - (n_fixes - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width * 0.9,
            label=fix_name, color=fix_colors[i % len(fix_colors)],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if not pd.isna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_title("Training Fix Impact (Group J): Old vs Fixed Training",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", ncol=2, title="Fix Variant",
              title_fontsize=8)

    # Zoom y-axis
    all_vals = pivot.values.flatten()
    valid_vals = all_vals[~np.isnan(all_vals)]
    if len(valid_vals) > 0:
        y_min = max(0, valid_vals.min() - 0.02)
        ax.set_ylim(y_min, min(1.005, valid_vals.max() + 0.02))

    save_fig(fig, output_dir, "fig05_training_fix_impact")


def fig06_mlp_recipe_heatmap(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 6: MLP recipe heatmap from Group M."""
    logger.info("Generating Fig 6: MLP Recipe Heatmap (Group M)")

    if "M" not in groups:
        logger.warning("Fig 6: Group M CSV not found, skipping")
        return

    df = groups["M"].copy()
    if "auc" not in df.columns:
        logger.warning("Fig 6: No AUC column in Group M, skipping")
        return
    df = df.dropna(subset=["auc"])
    if df.empty:
        logger.warning("Fig 6: No valid data in Group M, skipping")
        return

    # Try to parse hyperparameter columns from "name" or dedicated cols
    hp_cols_available = all(c in df.columns for c in ["lr", "dropout"])

    if not hp_cols_available and "name" in df.columns:
        # Attempt to extract from name patterns
        logger.info("Fig 6: Parsing hyperparameters from experiment names")

    # Try to determine panels
    if "hidden_dims" in df.columns or "hidden" in df.columns:
        hidden_col = "hidden_dims" if "hidden_dims" in df.columns else "hidden"
        unique_hidden = df[hidden_col].unique()

        if len(unique_hidden) > 1 and "lr" in df.columns and "dropout" in df.columns:
            n_panels = min(len(unique_hidden), 4)
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
            if n_panels == 1:
                axes = [axes]

            for idx, (hd, ax) in enumerate(zip(sorted(unique_hidden)[:n_panels], axes)):
                sub = df[df[hidden_col] == hd]
                pivot = sub.pivot_table(values="auc", index="lr", columns="dropout",
                                        aggfunc="max")
                if pivot.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

                sns.heatmap(
                    pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                    linewidths=0.5, linecolor="white", cbar=False, ax=ax,
                )
                ax.set_title(f"hidden={hd}", fontsize=10)
                ax.set_xlabel("Dropout" if idx == len(axes) // 2 else "")
                ax.set_ylabel("Learning Rate" if idx == 0 else "")

            fig.suptitle("MLP Recipe Grid (Group M): LR x Dropout x Hidden Dims",
                         fontsize=12, fontweight="bold", y=1.02)
            fig.subplots_adjust(wspace=0.3)
            save_fig(fig, output_dir, "fig06_mlp_recipe_heatmap")
            return

    # Fallback: simple bar chart of top experiments
    top = df.sort_values("auc", ascending=False).head(20)
    label_col = "name" if "name" in top.columns else top.columns[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["auc"].values, color="#CC3311", height=0.7,
            edgecolor="white", linewidth=0.5)

    labels = top[label_col].astype(str).tolist()
    for i, val in enumerate(top["auc"].values):
        ax.text(val + 0.0005, i, f"{val:.4f}", va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l[:60] for l in labels], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("AUC", fontsize=10)
    ax.set_title("MLP Recipe Sweep (Group M): Top-20 by AUC", fontsize=12,
                 fontweight="bold")

    save_fig(fig, output_dir, "fig06_mlp_recipe_heatmap")


def fig07_svm_gamma_sensitivity(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 7: SVM gamma sensitivity from Group N."""
    logger.info("Generating Fig 7: SVM Gamma Sensitivity (Group N)")

    if "N" not in groups:
        logger.warning("Fig 7: Group N CSV not found, skipping")
        return

    df = groups["N"].copy()
    if "auc" not in df.columns:
        logger.warning("Fig 7: No AUC column in Group N, skipping")
        return
    df = df.dropna(subset=["auc"])

    # Try to find C, gamma, layer columns
    has_params = "C" in df.columns or "svm_C" in df.columns
    has_gamma = "gamma" in df.columns or "svm_gamma" in df.columns

    c_col = "C" if "C" in df.columns else ("svm_C" if "svm_C" in df.columns else None)
    g_col = "gamma" if "gamma" in df.columns else ("svm_gamma" if "svm_gamma" in df.columns else None)

    if not (has_params and has_gamma):
        # Try to parse from name
        if "name" in df.columns:
            logger.info("Fig 7: Parsing SVM params from experiment names")
            # Names like "L2|SVM_C=1.0_gamma=0.01" or similar
            # Fall back to simple bar chart
            top = df.sort_values("auc", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(top))
            ax.barh(y_pos, top["auc"].values, color="#0173B2", height=0.7)
            labels = top["name"].astype(str).tolist()
            for i, val in enumerate(top["auc"].values):
                ax.text(val + 0.0005, i, f"{val:.4f}", va="center", fontsize=7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([l[:50] for l in labels], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("AUC")
            ax.set_title("SVM Grid (Group N): Top-20 by AUC", fontsize=12,
                         fontweight="bold")
            save_fig(fig, output_dir, "fig07_svm_gamma_sensitivity")
            return

    # Build heatmap(s) — one per layer if available
    layer_col = "layer" if "layer" in df.columns else None

    if layer_col:
        layers = sorted(df[layer_col].unique())
    else:
        layers = [None]

    n_panels = min(len(layers), 3)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for idx, (layer, ax) in enumerate(zip(layers[:n_panels], axes)):
        sub = df[df[layer_col] == layer] if layer is not None else df

        pivot = sub.pivot_table(values="auc", index=c_col, columns=g_col,
                                aggfunc="max")
        if pivot.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Sort by parameter value
        pivot = pivot.sort_index()
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        sns.heatmap(
            pivot, annot=True, fmt=".4f", cmap="RdYlBu",
            linewidths=0.5, linecolor="white", cbar=False, ax=ax,
            vmin=0.5, vmax=1.0,
        )
        layer_label = f"L{layer}" if layer is not None else "All"
        ax.set_title(f"{layer_label}", fontsize=10)
        ax.set_xlabel("gamma")
        ax.set_ylabel("C" if idx == 0 else "")

    fig.suptitle("SVM C x gamma Grid (Group N)", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.3)

    save_fig(fig, output_dir, "fig07_svm_gamma_sensitivity")


def fig08_fusion_tta_summary(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 8: Fusion and TTA summary from Group O."""
    logger.info("Generating Fig 8: Fusion & TTA Summary (Group O)")

    if "O" not in groups:
        logger.warning("Fig 8: Group O CSV not found, skipping")
        return

    df = groups["O"].copy()
    if "auc" not in df.columns:
        logger.warning("Fig 8: No AUC column in Group O, skipping")
        return
    df = df.dropna(subset=["auc"])
    if df.empty:
        logger.warning("Fig 8: No valid data in Group O, skipping")
        return

    # Sort by AUC
    df = df.sort_values("auc", ascending=True).reset_index(drop=True)
    label_col = "name" if "name" in df.columns else df.columns[0]
    labels = df[label_col].astype(str).tolist()

    # Color by method type
    colors = []
    for lbl in labels:
        lbl_upper = lbl.upper()
        if "TTA" in lbl_upper:
            colors.append("#9467BD")
        elif "FUSION" in lbl_upper or "CONCAT" in lbl_upper or "MULTI" in lbl_upper:
            colors.append("#DE8F05")
        elif "ENSEMBLE" in lbl_upper or "SEED" in lbl_upper:
            colors.append("#029E73")
        elif "STABILITY" in lbl_upper:
            colors.append("#E377C2")
        else:
            colors.append("#7F7F7F")

    fig_height = max(5, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["auc"].values, color=colors, height=0.7,
            edgecolor="white", linewidth=0.5)

    # Annotate values
    for i, val in enumerate(df["auc"].values):
        ax.text(val + 0.0005, i, f"{val:.4f}", va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l[:55] for l in labels], fontsize=7)
    ax.set_xlabel("AUC", fontsize=10)
    ax.set_title("Fusion & TTA Methods (Group O)", fontsize=12,
                 fontweight="bold")

    # Reference line for single-best (e.g., 0.9895)
    # Find from Group K if available
    ref_auc = None
    if "K" in groups:
        k_valid = groups["K"].dropna(subset=["auc"]) if "auc" in groups["K"].columns else pd.DataFrame()
        if not k_valid.empty:
            ref_auc = k_valid["auc"].max()

    if ref_auc is not None:
        ax.axvline(ref_auc, color="#CC3311", ls="--", lw=1.5, alpha=0.8)
        ax.text(ref_auc, len(df) - 0.5, f"  Single L3 best\n  ({ref_auc:.4f})",
                fontsize=7, color="#CC3311", va="top")

    # Zoom x-axis
    auc_min = df["auc"].min()
    ax.set_xlim(max(0, auc_min - 0.01), 1.002)

    # Legend
    legend_elements = [
        Patch(facecolor="#7F7F7F", label="Single"),
        Patch(facecolor="#DE8F05", label="Fusion"),
        Patch(facecolor="#9467BD", label="TTA"),
        Patch(facecolor="#029E73", label="Ensemble"),
        Patch(facecolor="#E377C2", label="Stability"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    save_fig(fig, output_dir, "fig08_fusion_tta_summary")


def fig09_multiseed_stability(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 9: Multi-seed stability box plot from Group O."""
    logger.info("Generating Fig 9: Multi-Seed Stability Box Plot (Group O)")

    if "O" not in groups:
        logger.warning("Fig 9: Group O CSV not found, skipping")
        return

    df = groups["O"].copy()
    if "auc" not in df.columns:
        logger.warning("Fig 9: No AUC column in Group O, skipping")
        return

    # Look for stability experiments (multi-seed runs with same config)
    # These typically have "seed" in the name or a "seed" column
    stability_mask = pd.Series([False] * len(df))
    if "name" in df.columns:
        stability_mask = df["name"].str.contains("seed|stability|multi", case=False, na=False)
    if "subgroup" in df.columns:
        stability_mask = stability_mask | (df["subgroup"] == "stability")

    if stability_mask.sum() < 2:
        # Fall back: look for repeated config patterns
        logger.info("Fig 9: No explicit stability experiments, attempting config grouping")
        # Group by classifier/head type and look for multi-run configs
        # Try using all Group O data grouped by base config
        if "name" in df.columns:
            # Strip seed suffix to find base configs
            df["_base_config"] = df["name"].str.replace(r"\|seed=\d+", "", regex=True)
            config_counts = df["_base_config"].value_counts()
            multi_seed_configs = config_counts[config_counts >= 2].index.tolist()

            if not multi_seed_configs:
                logger.warning("Fig 9: No multi-seed experiments found, skipping")
                return

            stability_df = df[df["_base_config"].isin(multi_seed_configs)].copy()
        else:
            logger.warning("Fig 9: Cannot identify stability experiments, skipping")
            return
    else:
        stability_df = df[stability_mask].copy()
        if "name" in stability_df.columns:
            stability_df["_base_config"] = stability_df["name"].str.replace(
                r"\|seed=\d+", "", regex=True
            )
        else:
            stability_df["_base_config"] = "config"

    stability_df = stability_df.dropna(subset=["auc"])
    if stability_df.empty:
        logger.warning("Fig 9: No valid stability data, skipping")
        return

    # Group by base config and collect AUC values
    config_groups = {}
    for config_name, sub in stability_df.groupby("_base_config"):
        if len(sub) >= 2:
            config_groups[config_name] = sub["auc"].values

    if not config_groups:
        logger.warning("Fig 9: No multi-seed groups found, skipping")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(config_groups) * 1.5), 5))

    labels = list(config_groups.keys())
    data = [config_groups[k] for k in labels]

    bp = ax.boxplot(
        data, labels=[l[:35] for l in labels],
        patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#CC3311", markersize=6),
        medianprops=dict(color="#CC3311", linewidth=2),
    )

    # Color boxes
    box_colors = ["#0173B2", "#029E73", "#DE8F05", "#9467BD", "#CC3311",
                  "#8C564B", "#E377C2", "#7F7F7F"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(box_colors[i % len(box_colors)])
        patch.set_alpha(0.5)

    # Overlay individual points
    for i, vals in enumerate(data):
        x_jitter = np.random.default_rng(42).normal(i + 1, 0.05, size=len(vals))
        ax.scatter(x_jitter, vals, color=box_colors[i % len(box_colors)],
                   s=40, zorder=5, alpha=0.8, edgecolors="white", linewidth=0.5)

    ax.set_ylabel("AUC", fontsize=10)
    ax.set_title("Multi-Seed Stability (Group O)", fontsize=12,
                 fontweight="bold")
    if len(labels) > 4:
        ax.set_xticklabels([l[:30] for l in labels], rotation=30, ha="right",
                           fontsize=7)

    save_fig(fig, output_dir, "fig09_multiseed_stability")


def fig10_neural_vs_sklearn(
    groups: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fig 10: Neural vs sklearn head-to-head multi-metric comparison."""
    logger.info("Generating Fig 10: Neural vs Sklearn Head-to-Head")

    combined = _combine_all_results(groups)
    if combined.empty or "auc" not in combined.columns:
        logger.warning("Fig 10: No data available, skipping")
        return

    valid = combined.dropna(subset=["auc"]).copy()
    if valid.empty:
        logger.warning("Fig 10: No valid AUC data, skipping")
        return

    # Classify each experiment
    valid["_clf_type"] = valid.apply(
        lambda r: _classify_clf_type(
            str(r.get("name", "")) + str(r.get("classifier", ""))
        ), axis=1,
    )

    # Find top experiment for each major type
    metrics_to_compare = ["auc", "accuracy", "f1", "precision", "recall"]
    # Add EER (inverted for radar: lower is better, so use 1-EER)
    if "eer" in valid.columns:
        valid["1-eer"] = 1 - valid["eer"]
        metrics_to_compare.append("1-eer")

    clf_types_to_show = ["SVM", "MLP", "LogReg", "RF"]
    best_per_type = {}
    for ct in clf_types_to_show:
        sub = valid[valid["_clf_type"] == ct]
        if not sub.empty:
            best_idx = sub["auc"].idxmax()
            best_per_type[ct] = sub.loc[best_idx]

    if len(best_per_type) < 2:
        logger.warning("Fig 10: Need at least 2 classifier types, only found %d",
                       len(best_per_type))
        return

    # Grouped bar chart (more readable than radar for publication)
    metrics_available = [m for m in metrics_to_compare
                         if all(m in row.index and not pd.isna(row.get(m, np.nan))
                                for row in best_per_type.values())]

    if not metrics_available:
        logger.warning("Fig 10: No common metrics found, skipping")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(metrics_available) * 1.8), 5))

    x = np.arange(len(metrics_available))
    n_types = len(best_per_type)
    bar_width = 0.8 / max(n_types, 1)

    type_colors = {
        "SVM": "#0173B2", "MLP": "#CC3311",
        "LogReg": "#029E73", "RF": "#7F7F7F",
    }

    for i, (ct, row) in enumerate(best_per_type.items()):
        vals = [float(row.get(m, 0)) for m in metrics_available]
        offset = (i - (n_types - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width * 0.9,
            label=ct, color=type_colors.get(ct, "#BCBD22"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                rotation=45,
            )

    # Pretty metric names
    metric_names = {
        "auc": "AUC", "accuracy": "Accuracy", "f1": "F1",
        "precision": "Precision", "recall": "Recall", "1-eer": "1-EER",
        "f1_macro": "F1-macro",
    }

    ax.set_xticks(x)
    ax.set_xticklabels([metric_names.get(m, m) for m in metrics_available],
                       fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Neural vs Sklearn: Multi-Metric Head-to-Head", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9, title="Classifier", title_fontsize=10)
    ax.set_ylim(0, 1.08)

    save_fig(fig, output_dir, "fig10_neural_vs_sklearn")


# ============================================================================
# Part B: Embedding Analysis (requires GPU + model)
# ============================================================================

def _load_mantis_model(pretrained: str, layer: int, output_token: str, device: str):
    """Load MantisV2 model + MantisTrainer wrapper."""
    import torch
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    net = MantisV2(
        device=device,
        return_transf_layer=layer,
        output_token=output_token,
    )
    net = net.from_pretrained(pretrained)
    trainer = MantisTrainer(device=device, network=net)
    return trainer


def _extract_embeddings(model, dataset, device: str) -> np.ndarray:
    """Extract frozen MantisV2 embeddings (per-channel concatenated)."""
    X, _ = dataset.get_numpy_arrays()  # (N, C, L)
    n_samples, n_channels, seq_len = X.shape

    all_emb = []
    for ch in range(n_channels):
        X_ch = X[:, [ch], :]  # (N, 1, L)
        Z_ch = model.transform(X_ch)  # (N, D)
        all_emb.append(Z_ch)

    Z = np.concatenate(all_emb, axis=-1)  # (N, C*D)
    if np.isnan(Z).any():
        n_nan = np.isnan(Z).sum()
        logger.warning("Found %d NaN in embeddings, replacing with 0", n_nan)
        Z = np.nan_to_num(Z, nan=0.0)
    return Z


def _tsne_joint(*arrays, seed=42, perplexity=30, n_iter=1000, pca_pre=50):
    """Joint t-SNE for multiple arrays in shared coordinate space."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    sizes = [len(a) for a in arrays]
    combined = np.concatenate(arrays, axis=0)
    n, d = combined.shape

    X = StandardScaler().fit_transform(combined)
    n_pre = min(pca_pre, d, n)
    if d > n_pre:
        X = PCA(n_components=n_pre, random_state=seed).fit_transform(X)

    perp = min(perplexity, max(2, n - 1))
    reduced = TSNE(
        n_components=2, perplexity=perp, max_iter=n_iter, random_state=seed,
    ).fit_transform(X)

    result, offset = [], 0
    for s in sizes:
        result.append(reduced[offset:offset + s])
        offset += s
    return result


def fig11_tsne_grid(
    config: dict, output_dir: Path, device: str = "cuda",
) -> None:
    """Fig 11: 3x3 t-SNE grid for different layer/context/channel combos."""
    logger.info("Generating Fig 11: t-SNE Embedding Grid (3x3)")

    import torch
    from data.preprocess import load_unified_split
    from data.dataset import DatasetConfig, OccupancyDataset

    data_cfg = config.get("data", {})
    train_csv = data_cfg["train_csv"]
    test_csv = data_cfg["test_csv"]
    label_col = data_cfg.get("label_column", "occupancy_label")
    split_date = config.get("split_date")
    pretrained = config["model"]["pretrained_name"]
    output_token = config["model"].get("output_token", "combined")
    stride = config.get("stride", 1)

    CHANNEL_MAP = {
        "M": "d620900d_motionSensor",
        "C": "408981c2_contactSensor",
        "T1": "d620900d_temperatureMeasurement",
    }

    # --- Define the 3x3 grid configurations ---
    # Row 1: Layer variation (L2, L3, L5), fixed M+C+T1, 251min
    # Row 2: Channel variation (M+C, M+C+T1, M+T1), fixed L3, 251min
    # Row 3: Context variation (181min, 251min, 401min), fixed L3, M+C+T1

    grid_configs = [
        # Row 1: Layer sweep
        {"label": "L2 | M+C+T1 | 251min", "layer": 2, "channels": ["M", "C", "T1"],
         "ctx_before": 150, "ctx_after": 100},
        {"label": "L3 | M+C+T1 | 251min", "layer": 3, "channels": ["M", "C", "T1"],
         "ctx_before": 150, "ctx_after": 100},
        {"label": "L5 | M+C+T1 | 251min", "layer": 5, "channels": ["M", "C", "T1"],
         "ctx_before": 150, "ctx_after": 100},
        # Row 2: Channel sweep
        {"label": "L3 | M+C | 251min", "layer": 3, "channels": ["M", "C"],
         "ctx_before": 150, "ctx_after": 100},
        {"label": "L3 | M+C+T1 | 251min", "layer": 3, "channels": ["M", "C", "T1"],
         "ctx_before": 150, "ctx_after": 100},
        {"label": "L3 | M+T1 | 251min", "layer": 3, "channels": ["M", "T1"],
         "ctx_before": 150, "ctx_after": 100},
        # Row 3: Context sweep
        {"label": "L3 | M+C+T1 | 181min", "layer": 3, "channels": ["M", "C", "T1"],
         "ctx_before": 90, "ctx_after": 90},
        {"label": "L3 | M+C+T1 | 251min", "layer": 3, "channels": ["M", "C", "T1"],
         "ctx_before": 150, "ctx_after": 100},
        {"label": "L3 | M+C+T1 | 401min", "layer": 3, "channels": ["M", "C", "T1"],
         "ctx_before": 300, "ctx_after": 100},
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    row_titles = [
        "Row 1: Layer Variation (fixed M+C+T1, 251min)",
        "Row 2: Channel Variation (fixed L3, 251min)",
        "Row 3: Context Variation (fixed L3, M+C+T1)",
    ]

    for idx, (gcfg, ax) in enumerate(zip(grid_configs, axes.flat)):
        row = idx // 3
        col = idx % 3

        logger.info("  [%d/9] %s", idx + 1, gcfg["label"])

        try:
            channels = [CHANNEL_MAP[k] for k in gcfg["channels"]]

            # Load unified data
            sensor, train_labels, test_labels, ch_names, timestamps = load_unified_split(
                train_csv, test_csv,
                label_column=label_col, channels=channels,
                split_date=split_date,
            )

            # Build datasets
            ds_cfg = DatasetConfig(
                context_mode="bidirectional",
                context_before=gcfg["ctx_before"],
                context_after=gcfg["ctx_after"],
                stride=stride,
            )
            train_ds = OccupancyDataset(sensor, train_labels, timestamps, ds_cfg)
            test_ds = OccupancyDataset(sensor, test_labels, timestamps, ds_cfg)

            # Extract embeddings
            model = _load_mantis_model(pretrained, gcfg["layer"], output_token, device)
            Z_train = _extract_embeddings(model, train_ds, device)
            Z_test = _extract_embeddings(model, test_ds, device)
            y_train = train_ds.labels
            y_test = test_ds.labels

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Joint t-SNE
            Z_train_2d, Z_test_2d = _tsne_joint(
                Z_train, Z_test,
                seed=42, perplexity=30, n_iter=1000,
            )

            # Plot train (transparent) + test (solid)
            for cls in [0, 1]:
                # Train points — transparent
                mask_tr = y_train == cls
                if mask_tr.any():
                    ax.scatter(
                        Z_train_2d[mask_tr, 0], Z_train_2d[mask_tr, 1],
                        c=CLASS_COLORS[cls], s=10, alpha=0.15,
                        edgecolors="none",
                    )
                # Test points — solid
                mask_te = y_test == cls
                if mask_te.any():
                    ax.scatter(
                        Z_test_2d[mask_te, 0], Z_test_2d[mask_te, 1],
                        c=CLASS_COLORS[cls], s=20, alpha=0.7,
                        edgecolors="white", linewidth=0.3,
                        label=f"{CLASS_NAMES[cls]} (test, n={mask_te.sum()})",
                    )

            ax.set_title(gcfg["label"], fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            n_info = f"Train={len(y_train)}, Test={len(y_test)}"
            ax.text(
                0.02, 0.02, n_info, transform=ax.transAxes,
                fontsize=6, color="gray", va="bottom",
            )

            if col == 0:
                ax.legend(fontsize=6, loc="upper left", markerscale=1.5)

        except Exception as e:
            logger.error("  Failed %s: %s", gcfg["label"], e)
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8, color="red")
            ax.set_title(gcfg["label"], fontsize=9)

    # Row titles
    for row_idx, title in enumerate(row_titles):
        fig.text(
            0.5, 1.0 - row_idx * 0.33 - 0.005, title,
            ha="center", fontsize=10, fontstyle="italic",
            transform=fig.transFigure,
        )

    fig.suptitle(
        "t-SNE Embedding Space: Layer / Channel / Context Ablation",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    save_fig(fig, output_dir, "fig11_tsne_embedding_grid")


def fig12_tsne_classifier_decisions(
    config: dict, output_dir: Path, device: str = "cuda",
) -> None:
    """Fig 12: t-SNE with SVM vs MLP classifier decisions overlaid."""
    logger.info("Generating Fig 12: t-SNE with Classifier Decisions")

    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from data.preprocess import load_unified_split
    from data.dataset import DatasetConfig, OccupancyDataset
    from training.heads import MLPHead

    data_cfg = config.get("data", {})
    train_csv = data_cfg["train_csv"]
    test_csv = data_cfg["test_csv"]
    label_col = data_cfg.get("label_column", "occupancy_label")
    split_date = config.get("split_date")
    pretrained = config["model"]["pretrained_name"]
    output_token = config["model"].get("output_token", "combined")
    stride = config.get("stride", 1)

    CHANNEL_MAP = {
        "M": "d620900d_motionSensor",
        "C": "408981c2_contactSensor",
        "T1": "d620900d_temperatureMeasurement",
    }

    # Best config: L3, M+C+T1, 251min
    channels = [CHANNEL_MAP[k] for k in ["M", "C", "T1"]]

    try:
        sensor, train_labels, test_labels, ch_names, timestamps = load_unified_split(
            train_csv, test_csv,
            label_column=label_col, channels=channels,
            split_date=split_date,
        )

        ds_cfg = DatasetConfig(
            context_mode="bidirectional",
            context_before=150, context_after=100,
            stride=stride,
        )
        train_ds = OccupancyDataset(sensor, train_labels, timestamps, ds_cfg)
        test_ds = OccupancyDataset(sensor, test_labels, timestamps, ds_cfg)

        # Extract embeddings at L3
        model = _load_mantis_model(pretrained, 3, output_token, device)
        Z_train = _extract_embeddings(model, train_ds, device)
        Z_test = _extract_embeddings(model, test_ds, device)
        y_train = train_ds.labels
        y_test = test_ds.labels

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        embed_dim = Z_train.shape[1]

        # --- Train classifiers ---
        scaler = StandardScaler()
        Ztr_s = scaler.fit_transform(Z_train)
        Zte_s = scaler.transform(Z_test)

        # SVM
        logger.info("  Training SVM_rbf classifier...")
        svm_clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        svm_clf.fit(Ztr_s, y_train)
        y_pred_svm = svm_clf.predict(Zte_s)

        # MLP
        logger.info("  Training MLP[128,64] classifier...")
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        mlp = MLPHead(embed_dim, 2, hidden_dims=[128, 64], dropout=0.3,
                      use_batchnorm=True).to(dev)

        Ztr_t = torch.from_numpy(Ztr_s).float().to(dev)
        ytr_t = torch.from_numpy(y_train).long().to(dev)
        Zte_t = torch.from_numpy(Zte_s).float().to(dev)

        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        mlp.train()
        for epoch in range(200):
            opt.zero_grad()
            logits = mlp(Ztr_t)
            loss = criterion(logits, ytr_t)
            loss.backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            logits_test = mlp(Zte_t)
            y_pred_mlp = logits_test.argmax(dim=1).cpu().numpy()

        del mlp, Ztr_t, ytr_t, Zte_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Joint t-SNE ---
        logger.info("  Computing joint t-SNE...")
        Z_train_2d, Z_test_2d = _tsne_joint(
            Z_train, Z_test,
            seed=42, perplexity=30, n_iter=1000,
        )

        # --- Plot 1x3 grid: SVM | MLP | Ground Truth ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        panels = [
            ("SVM_rbf Predictions", y_pred_svm),
            ("MLP[128,64] Predictions", y_pred_mlp),
            ("Ground Truth", y_test),
        ]

        for ax, (title, y_panel) in zip(axes, panels):
            # Determine correct/incorrect
            if title != "Ground Truth":
                correct_mask = y_panel == y_test
                incorrect_mask = ~correct_mask
                n_correct = correct_mask.sum()
                n_incorrect = incorrect_mask.sum()
                acc = n_correct / len(y_test) * 100

                # Plot correct (circles)
                for cls in [0, 1]:
                    mask = (y_panel == cls) & correct_mask
                    if mask.any():
                        ax.scatter(
                            Z_test_2d[mask, 0], Z_test_2d[mask, 1],
                            c=CLASS_COLORS[cls], s=25, alpha=0.7,
                            marker="o", edgecolors="none",
                            label=f"{CLASS_NAMES[cls]} correct",
                        )

                # Plot incorrect (X marks)
                for cls in [0, 1]:
                    mask = (y_panel == cls) & incorrect_mask
                    if mask.any():
                        ax.scatter(
                            Z_test_2d[mask, 0], Z_test_2d[mask, 1],
                            c=CLASS_COLORS[cls], s=60, alpha=0.9,
                            marker="X", edgecolors="black", linewidth=0.5,
                            label=f"{CLASS_NAMES[cls]} ERROR",
                        )

                ax.set_title(f"{title}\nAcc={acc:.1f}% ({n_incorrect} errors)",
                             fontsize=10, fontweight="bold")
            else:
                # Ground truth: simple scatter
                for cls in [0, 1]:
                    mask = y_test == cls
                    if mask.any():
                        ax.scatter(
                            Z_test_2d[mask, 0], Z_test_2d[mask, 1],
                            c=CLASS_COLORS[cls], s=25, alpha=0.7,
                            marker="o", edgecolors="none",
                            label=f"{CLASS_NAMES[cls]} (n={mask.sum()})",
                        )
                ax.set_title(title, fontsize=10, fontweight="bold")

            ax.legend(fontsize=7, loc="upper left", markerscale=1.2)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            "t-SNE: Classifier Decisions at Best Config (L3, M+C+T1, 251min)",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.subplots_adjust(wspace=0.1)

        save_fig(fig, output_dir, "fig12_tsne_classifier_decisions")

    except Exception as e:
        logger.error("Fig 12 failed: %s", e, exc_info=True)


# ============================================================================
# Main execution
# ============================================================================

ALL_FIGURES = {
    1: ("Grand Overview: Top-30 by AUC", fig01_grand_overview, "csv"),
    2: ("Phase Progression", fig02_phase_progression, "csv"),
    3: ("Layer x Classifier Heatmap", fig03_layer_classifier_heatmap, "csv"),
    4: ("Context Window Curves", fig04_context_window_curves, "csv"),
    5: ("Training Fix Impact", fig05_training_fix_impact, "csv"),
    6: ("MLP Recipe Heatmap", fig06_mlp_recipe_heatmap, "csv"),
    7: ("SVM Gamma Sensitivity", fig07_svm_gamma_sensitivity, "csv"),
    8: ("Fusion & TTA Summary", fig08_fusion_tta_summary, "csv"),
    9: ("Multi-Seed Stability", fig09_multiseed_stability, "csv"),
    10: ("Neural vs Sklearn", fig10_neural_vs_sklearn, "csv"),
    11: ("t-SNE Embedding Grid", fig11_tsne_grid, "gpu"),
    12: ("t-SNE Classifier Decisions", fig12_tsne_classifier_decisions, "gpu"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Final Comprehensive Analysis for APC Occupancy Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv-dir", type=str, default="results/phase3_additional",
        help="Primary directory containing experiment CSV files (default: results/phase3_additional)",
    )
    parser.add_argument(
        "--csv-dir-p1", type=str, default=None,
        help="Phase 1 results directory (default: search in csv-dir)",
    )
    parser.add_argument(
        "--csv-dir-p2", type=str, default=None,
        help="Phase 2 results directory (default: search in csv-dir)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/final_analysis/plots",
        help="Output directory for figures (default: results/final_analysis/plots)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config for embedding extraction (required for Figs 11-12)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for embedding extraction (default: cuda)",
    )
    parser.add_argument(
        "--figures", type=int, nargs="+", default=None,
        help="Specific figure numbers to generate (default: all)",
    )
    parser.add_argument(
        "--csv-only", action="store_true",
        help="Skip GPU-dependent figures (11-12)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for t-SNE (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Style
    setup_pub_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build search paths ---
    csv_dirs = [Path(args.csv_dir)]
    if args.csv_dir_p1:
        csv_dirs.append(Path(args.csv_dir_p1))
    if args.csv_dir_p2:
        csv_dirs.append(Path(args.csv_dir_p2))

    # Also try parent directory's phase1_sweep and phase2_sweep
    primary = Path(args.csv_dir)
    parent = primary.parent
    for sibling in ["phase1_sweep", "phase2_sweep", "phase3_additional"]:
        candidate = parent / sibling
        if candidate.exists() and candidate not in csv_dirs:
            csv_dirs.append(candidate)

    logger.info("=" * 70)
    logger.info("FINAL ANALYSIS: APC Occupancy Classification")
    logger.info("=" * 70)
    logger.info("CSV search paths: %s", [str(p) for p in csv_dirs])
    logger.info("Output directory: %s", output_dir)

    # --- Load CSVs ---
    groups = load_all_csvs(csv_dirs)
    if not groups:
        logger.error("No experiment CSV files found. Check --csv-dir path.")
        logger.error("Expected files in tables/ subdirectory: %s",
                      list(CSV_FILENAMES.values()))
        sys.exit(1)

    # --- Determine which figures to generate ---
    if args.figures:
        figure_nums = args.figures
    elif args.csv_only:
        figure_nums = list(range(1, 11))  # Figs 1-10 only
    else:
        figure_nums = list(range(1, 13))  # All 12

    # Load YAML config if needed for GPU figures
    yaml_config = None
    if any(ALL_FIGURES[n][2] == "gpu" for n in figure_nums if n in ALL_FIGURES):
        if args.config is None:
            logger.error(
                "Figures 11-12 require --config (YAML config for model/data). "
                "Use --csv-only to skip GPU figures."
            )
            sys.exit(1)
        import yaml
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        logger.info("Loaded config: %s", args.config)

    # --- Generate figures ---
    total_start = time.time()
    generated = 0
    failed = 0

    for num in figure_nums:
        if num not in ALL_FIGURES:
            logger.warning("Unknown figure number: %d (valid: 1-12)", num)
            continue

        desc, func, mode = ALL_FIGURES[num]

        if mode == "gpu" and args.csv_only:
            logger.info("Skipping Fig %d (%s) — csv-only mode", num, desc)
            continue

        logger.info("")
        logger.info("-" * 50)
        logger.info("Fig %d: %s", num, desc)
        logger.info("-" * 50)

        t0 = time.time()
        try:
            if mode == "csv":
                func(groups, output_dir)
            elif mode == "gpu":
                func(yaml_config, output_dir, device=args.device)
            elapsed = time.time() - t0
            logger.info("Fig %d completed in %.1fs", num, elapsed)
            generated += 1
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("Fig %d FAILED after %.1fs: %s", num, elapsed, e,
                         exc_info=True)
            failed += 1

    total_elapsed = time.time() - total_start

    # --- Summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info("Generated: %d figures", generated)
    if failed > 0:
        logger.info("Failed:    %d figures", failed)
    logger.info("Output:    %s", output_dir)
    logger.info("Total time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("")
    logger.info("Generated files:")
    for p in sorted(output_dir.glob("fig*")):
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
