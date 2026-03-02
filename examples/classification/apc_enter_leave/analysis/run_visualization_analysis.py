#!/usr/bin/env python3
"""Comprehensive Visualization Analysis for Enter/Leave Event Detection.

Generates publication-quality figures analyzing MantisV2 embeddings for the
Enter/Leave classification task across multiple analysis dimensions.

Figures produced (saved as PNG + PDF):
  Fig 1 — N=109 vs N=105: Embedding space comparison (t-SNE + UMAP)
  Fig 2 — Collision analysis: 4 indistinguishable samples highlighted
  Fig 3 — Classification overlay: RF vs MLP correct/incorrect markers
  Fig 4 — Uncertainty analysis: Per-sample entropy (RF vs MLP)
  Fig 5 — Context window effect: Class separation across 5 window sizes
  Fig 6 — Layer progression: Embedding evolution across L0-L5

Usage:
    cd examples/classification/apc_enter_leave
    python analysis/run_visualization_analysis.py \\
        --config training/configs/enter-leave-phase1.yaml \\
        --device cuda --output-dir results/visualization_analysis

    # Quick mode (skip slow experiments like context sweep & layer sweep)
    python analysis/run_visualization_analysis.py \\
        --config training/configs/enter-leave-phase1.yaml \\
        --device cuda --quick

Dependencies:
    pip install umap-learn  # Optional, falls back to t-SNE if missing
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent  # apc_enter_leave/
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import EventPreprocessConfig, load_sensor_and_events
from data.dataset import EventDatasetConfig, EventDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional UMAP
# ---------------------------------------------------------------------------
_UMAP_AVAILABLE = False
try:
    from umap import UMAP
    _UMAP_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Style constants (publication quality)
# ---------------------------------------------------------------------------
CLASS_COLORS = {0: "#009E73", 1: "#CC3311", 2: "#999999"}
CLASS_NAMES = {0: "Enter", 1: "Leave", 2: "None"}
ACCENT_BLUE = "#0173B2"
ACCENT_ORANGE = "#E69F00"
DPI = 300


def setup_style():
    """Apply clean publication rcParams."""
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, output_dir: Path, name: str):
    """Save figure as PNG + PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=DPI, format=fmt, bbox_inches="tight")
    logger.info("Saved: %s.{png,pdf}", output_dir / name)
    plt.close(fig)


# ============================================================================
# Dimensionality reduction helpers
# ============================================================================

def reduce_2d(
    Z: np.ndarray,
    method: str = "tsne",
    pca_pre: int = 50,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Reduce high-dim embeddings to 2D. Pipeline: Scale → PCA → method."""
    n, d = Z.shape
    X = StandardScaler().fit_transform(Z)

    if method == "pca":
        pca = PCA(n_components=min(2, d, n), random_state=seed)
        out = pca.fit_transform(X)
        if out.shape[1] < 2:
            out = np.hstack([out, np.zeros((n, 1))])
        return out

    n_pre = min(pca_pre, d, n)
    if d > n_pre:
        X = PCA(n_components=n_pre, random_state=seed).fit_transform(X)

    if method == "tsne":
        perp = kwargs.pop("perplexity", min(30, max(2, n - 1)))
        return TSNE(
            n_components=2, perplexity=perp, max_iter=1500,
            random_state=seed, **kwargs,
        ).fit_transform(X)

    if method == "umap":
        if not _UMAP_AVAILABLE:
            logger.warning("umap-learn not installed, falling back to t-SNE")
            return reduce_2d(Z, "tsne", pca_pre, seed)
        nn = kwargs.pop("n_neighbors", min(15, max(2, n - 1)))
        return UMAP(
            n_components=2, n_neighbors=nn, min_dist=0.1,
            metric="cosine", random_state=seed, **kwargs,
        ).fit_transform(X)

    raise ValueError(f"Unknown method: {method}")


def reduce_2d_joint(*arrays, method="tsne", seed=42, **kwargs):
    """Reduce multiple arrays in shared coordinate space."""
    sizes = [len(a) for a in arrays]
    combined = np.concatenate(arrays, axis=0)
    reduced = reduce_2d(combined, method=method, seed=seed, **kwargs)
    result, offset = [], 0
    for s in sizes:
        result.append(reduced[offset:offset + s])
        offset += s
    return result


# ============================================================================
# Data & model helpers
# ============================================================================

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(raw_cfg: dict, include_none: bool = True, events_csv_override: str | None = None):
    """Load sensor + events. Returns (sensor_array, sensor_ts, event_ts, labels, ch_names, cls_names)."""
    data_cfg = raw_cfg.get("data", {})
    cfg = EventPreprocessConfig(
        sensor_csv=data_cfg.get("sensor_csv", ""),
        events_csv=events_csv_override or data_cfg.get("events_csv", ""),
        column_names_csv=data_cfg.get("column_names_csv"),
        column_names=data_cfg.get("column_names"),
        channels=data_cfg.get("channels"),
        exclude_channels=data_cfg.get("exclude_channels", []),
        nan_threshold=data_cfg.get("nan_threshold", 0.3),
        include_none=include_none,
        add_time_features=data_cfg.get("add_time_features", False),
    )
    return load_sensor_and_events(cfg)


def load_mantis(pretrained: str, layer: int, output_token: str, device: str):
    """Load MantisV2 + MantisTrainer for embedding extraction."""
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net = MantisV2(device=device, return_transf_layer=layer, output_token=output_token)
    net = net.from_pretrained(pretrained)
    trainer = MantisTrainer(device=device, network=net)
    return trainer


def extract_embeddings(
    model, sensor_array, sensor_ts, event_ts, event_labels,
    channels, ctx_before=2, ctx_after=2,
):
    """Build dataset and extract embeddings. Returns (Z, y)."""
    ds_cfg = EventDatasetConfig(
        context_mode="bidirectional",
        context_before=ctx_before,
        context_after=ctx_after,
    )
    ds = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)
    X, y = ds.get_numpy_arrays()
    Z = model.transform(X)
    nan_count = int(np.isnan(Z).sum())
    if nan_count > 0:
        Z = np.nan_to_num(Z, nan=0.0)
    return Z, y


def detect_collision_indices(event_ts, event_labels):
    """Find indices of same-timestamp collision events.

    Returns list of (idx_a, idx_b) tuples where both events share a timestamp.
    """
    import pandas as pd
    ts_series = pd.Series(event_labels, index=pd.DatetimeIndex(event_ts))
    dup_mask = ts_series.index.duplicated(keep=False)
    dup_indices = np.where(dup_mask)[0]

    # Group by timestamp
    pairs = []
    seen = set()
    for i in dup_indices:
        if i in seen:
            continue
        t = event_ts[i]
        group = [j for j in dup_indices if event_ts[j] == t and j not in seen]
        for j in group:
            seen.add(j)
        if len(group) == 2:
            pairs.append((group[0], group[1]))
        elif len(group) > 2:
            for k in range(0, len(group), 2):
                if k + 1 < len(group):
                    pairs.append((group[k], group[k + 1]))
    return pairs, dup_indices


# ============================================================================
# LOOCV helpers (sklearn + neural)
# ============================================================================

def run_rf_loocv(Z, y, seed=42):
    """Run LOOCV with RF. Returns (y_pred, y_prob) where y_prob is (N, n_classes)."""
    n = len(y)
    n_cls = len(np.unique(y))
    y_pred = np.zeros(n, dtype=np.int64)
    y_prob = np.zeros((n, n_cls), dtype=np.float64)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scaler = StandardScaler()
        Ztr = scaler.fit_transform(Z[mask])
        Zte = scaler.transform(Z[i:i+1])
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        clf.fit(Ztr, y[mask])
        y_pred[i] = clf.predict(Zte)[0]
        proba = clf.predict_proba(Zte)[0]
        # Handle missing classes
        full = np.zeros(n_cls, dtype=np.float64)
        for ci, c in enumerate(clf.classes_):
            full[int(c)] = proba[ci]
        y_prob[i] = full

    return y_pred, y_prob


def run_mlp_loocv(Z, y, embed_dim, device="cpu", seed=42):
    """Run LOOCV with MLP[64]-d0.5 head. Returns (y_pred, y_prob)."""
    from training.heads import MLPHead

    n = len(y)
    n_cls = len(np.unique(y))
    y_pred = np.zeros(n, dtype=np.int64)
    y_prob = np.zeros((n, n_cls), dtype=np.float64)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scaler = StandardScaler()
        Ztr_np = scaler.fit_transform(Z[mask])
        Zte_np = scaler.transform(Z[i:i+1])

        Ztr = torch.from_numpy(Ztr_np).float()
        ytr = torch.from_numpy(y[mask]).long()
        Zte = torch.from_numpy(Zte_np).float().to(dev)

        torch.manual_seed(seed)
        head = MLPHead(embed_dim, n_cls, hidden_dims=[64], dropout=0.5, use_batchnorm=False)
        head = head.to(dev)
        head.train()

        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        loss_fn = torch.nn.CrossEntropyLoss()

        Ztr_d = Ztr.to(dev)
        ytr_d = ytr.to(dev)

        best_loss, patience = float("inf"), 0
        for epoch in range(200):
            logits = head(Ztr_d)
            loss = loss_fn(logits, ytr_d)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            lv = loss.item()
            if lv < best_loss - 1e-5:
                best_loss = lv
                patience = 0
            else:
                patience += 1
            if patience >= 30 or lv < 0.01:
                break

        head.eval()
        with torch.no_grad():
            logits = head(Zte)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        y_pred[i] = int(probs.argmax())
        y_prob[i] = probs

    return y_pred, y_prob


# ============================================================================
# Figure generators
# ============================================================================

def fig1_n109_vs_n105(Z_109, y_109, Z_105, y_105, output_dir: Path):
    """Fig 1: N=109 vs N=105 embedding comparison via t-SNE and UMAP."""
    setup_style()
    methods = ["tsne", "umap"] if _UMAP_AVAILABLE else ["tsne", "pca"]
    n_methods = len(methods)

    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 9))

    for col, method in enumerate(methods):
        method_label = "t-SNE" if method == "tsne" else method.upper()

        # Row 0: N=109
        emb109 = reduce_2d(Z_109, method=method)
        ax = axes[0, col]
        for cls in sorted(CLASS_COLORS):
            m = y_109 == cls
            if m.any():
                ax.scatter(emb109[m, 0], emb109[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=30, alpha=0.7, edgecolors="none")
        ax.set_title(f"N=109 — {method_label}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=1.5)

        # Row 1: N=105
        emb105 = reduce_2d(Z_105, method=method)
        ax = axes[1, col]
        for cls in sorted(CLASS_COLORS):
            m = y_105 == cls
            if m.any():
                ax.scatter(emb105[m, 0], emb105[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=30, alpha=0.7, edgecolors="none")
        ax.set_title(f"N=105 — {method_label}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=1.5)

    fig.suptitle("Enter/Leave Embedding Space: N=109 (with collisions) vs N=105 (cleaned)",
                 fontsize=13, fontweight="bold", y=1.01)
    save_fig(fig, output_dir, "fig1_n109_vs_n105")


def fig2_collision_analysis(Z_109, y_109, event_ts_109, output_dir: Path):
    """Fig 2: Highlight collision samples — identical embeddings, different labels."""
    setup_style()
    pairs, dup_indices = detect_collision_indices(event_ts_109, y_109)

    if len(dup_indices) == 0:
        logger.warning("No collision events detected; skipping Fig 2")
        return

    methods = ["tsne", "umap"] if _UMAP_AVAILABLE else ["tsne", "pca"]
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5.5))
    if len(methods) == 1:
        axes = [axes]

    for col, method in enumerate(methods):
        method_label = "t-SNE" if method == "tsne" else method.upper()
        emb = reduce_2d(Z_109, method=method)
        ax = axes[col]

        # Background: all samples (faded)
        non_dup = np.ones(len(y_109), dtype=bool)
        non_dup[dup_indices] = False
        for cls in sorted(CLASS_COLORS):
            m = (y_109 == cls) & non_dup
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=20, alpha=0.3, edgecolors="none")

        # Highlight collision samples with large markers + black edge
        for cls in sorted(CLASS_COLORS):
            m = np.zeros(len(y_109), dtype=bool)
            for idx in dup_indices:
                if y_109[idx] == cls:
                    m[idx] = True
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           s=120, alpha=1.0, edgecolors="black", linewidths=1.5,
                           zorder=10, marker="D",
                           label=f"{CLASS_NAMES[cls]} (collision)")

        # Draw lines between collision pairs
        for a, b in pairs:
            ax.plot([emb[a, 0], emb[b, 0]], [emb[a, 1], emb[b, 1]],
                    color="black", linewidth=1.5, linestyle="--", alpha=0.8, zorder=9)
            # Annotate with labels
            la = CLASS_NAMES[y_109[a]]
            lb = CLASS_NAMES[y_109[b]]
            mid_x = (emb[a, 0] + emb[b, 0]) / 2
            mid_y = (emb[a, 1] + emb[b, 1]) / 2
            ax.annotate(f"{la}/{lb}", (mid_x, mid_y), fontsize=7, fontweight="bold",
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8))

        ax.set_title(f"Collision Pairs — {method_label}", fontsize=11)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(fontsize=7, loc="best", markerscale=0.8)

    fig.suptitle(
        f"Timestamp Collision Analysis: {len(dup_indices)} events at "
        f"{len(pairs)} shared timestamps\n"
        "Collision pairs share identical sensor input → indistinguishable embeddings",
        fontsize=11, fontweight="bold", y=1.03,
    )
    save_fig(fig, output_dir, "fig2_collision_analysis")


def fig3_classification_overlay(Z, y, y_pred_rf, y_pred_mlp, output_dir: Path):
    """Fig 3: RF vs MLP classification results on N=105 (correct/incorrect overlay)."""
    setup_style()
    methods = ["tsne", "umap"] if _UMAP_AVAILABLE else ["tsne", "pca"]

    fig, axes = plt.subplots(2, len(methods), figsize=(6 * len(methods), 10))

    for col, method in enumerate(methods):
        method_label = "t-SNE" if method == "tsne" else method.upper()
        emb = reduce_2d(Z, method=method)

        for row, (name, y_pred) in enumerate([("RF (sklearn)", y_pred_rf),
                                                ("MLP[64]-d0.5 (neural)", y_pred_mlp)]):
            ax = axes[row, col]
            correct = y == y_pred
            incorrect = ~correct

            # Correct samples: colored by true class, small markers
            for cls in sorted(CLASS_COLORS):
                m = (y == cls) & correct
                if m.any():
                    ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                               label=f"{CLASS_NAMES[cls]} (correct)",
                               s=25, alpha=0.6, edgecolors="none")

            # Incorrect samples: large red-border X markers
            if incorrect.any():
                ax.scatter(emb[incorrect, 0], emb[incorrect, 1],
                           c=[CLASS_COLORS[int(c)] for c in y[incorrect]],
                           s=100, alpha=1.0, edgecolors="red", linewidths=2.0,
                           marker="X", zorder=10, label="Misclassified")
                # Add arrows showing predicted class
                for idx in np.where(incorrect)[0]:
                    true_cls = CLASS_NAMES[y[idx]]
                    pred_cls = CLASS_NAMES[y_pred[idx]]
                    ax.annotate(
                        f"{true_cls}→{pred_cls}", (emb[idx, 0], emb[idx, 1]),
                        fontsize=6, fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", alpha=0.9),
                    )

            n_err = incorrect.sum()
            acc = 100 * correct.mean()
            ax.set_title(f"{name} — {method_label}\nAcc={acc:.2f}% ({n_err} errors)", fontsize=10)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(fontsize=7, loc="best", markerscale=0.8)

    fig.suptitle("N=105 Classification Results: RF (sklearn) vs MLP (neural)",
                 fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, output_dir, "fig3_classification_overlay")


def fig4_uncertainty_analysis(Z, y, y_prob_rf, y_prob_mlp, y_pred_rf, y_pred_mlp,
                               output_dir: Path):
    """Fig 4: Per-sample uncertainty (entropy) comparison between RF and MLP."""
    setup_style()

    # Compute entropy per sample
    ent_rf = scipy_entropy(y_prob_rf, axis=1)
    ent_mlp = scipy_entropy(y_prob_mlp, axis=1)

    # Normalize for color mapping
    max_ent = np.log(y_prob_rf.shape[1])  # max entropy = log(n_classes)

    emb = reduce_2d(Z, method="tsne")

    fig = plt.figure(figsize=(16, 10))

    # Panel (a): t-SNE colored by RF entropy
    ax1 = fig.add_subplot(2, 3, 1)
    sc1 = ax1.scatter(emb[:, 0], emb[:, 1], c=ent_rf, cmap="YlOrRd",
                       s=30, alpha=0.8, vmin=0, vmax=max_ent)
    plt.colorbar(sc1, ax=ax1, label="Entropy", shrink=0.8)
    ax1.set_title("RF — Entropy Map (t-SNE)")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    # Panel (b): t-SNE colored by MLP entropy
    ax2 = fig.add_subplot(2, 3, 2)
    sc2 = ax2.scatter(emb[:, 0], emb[:, 1], c=ent_mlp, cmap="YlOrRd",
                       s=30, alpha=0.8, vmin=0, vmax=max_ent)
    plt.colorbar(sc2, ax=ax2, label="Entropy", shrink=0.8)
    ax2.set_title("MLP — Entropy Map (t-SNE)")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")

    # Panel (c): RF entropy vs MLP entropy scatter
    ax3 = fig.add_subplot(2, 3, 3)
    correct_both = (y == y_pred_rf) & (y == y_pred_mlp)
    wrong_both = (y != y_pred_rf) & (y != y_pred_mlp)
    rf_only_wrong = (y != y_pred_rf) & (y == y_pred_mlp)
    mlp_only_wrong = (y == y_pred_rf) & (y != y_pred_mlp)

    if correct_both.any():
        ax3.scatter(ent_rf[correct_both], ent_mlp[correct_both],
                    c="#009E73", s=15, alpha=0.5, label="Both correct")
    if wrong_both.any():
        ax3.scatter(ent_rf[wrong_both], ent_mlp[wrong_both],
                    c="#CC3311", s=80, alpha=1.0, marker="X", label="Both wrong")
    if rf_only_wrong.any():
        ax3.scatter(ent_rf[rf_only_wrong], ent_mlp[rf_only_wrong],
                    c=ACCENT_BLUE, s=60, alpha=0.9, marker="s", label="RF wrong only")
    if mlp_only_wrong.any():
        ax3.scatter(ent_rf[mlp_only_wrong], ent_mlp[mlp_only_wrong],
                    c=ACCENT_ORANGE, s=60, alpha=0.9, marker="^", label="MLP wrong only")
    ax3.plot([0, max_ent], [0, max_ent], "k--", alpha=0.3, label="y=x")
    ax3.set_xlabel("RF Entropy")
    ax3.set_ylabel("MLP Entropy")
    ax3.set_title("RF vs MLP Uncertainty")
    ax3.legend(fontsize=7)

    # Panel (d): Entropy histogram comparison
    ax4 = fig.add_subplot(2, 3, 4)
    bins = np.linspace(0, max_ent, 25)
    ax4.hist(ent_rf, bins=bins, alpha=0.5, color=ACCENT_BLUE, label="RF", density=True)
    ax4.hist(ent_mlp, bins=bins, alpha=0.5, color=ACCENT_ORANGE, label="MLP", density=True)
    ax4.axvline(np.median(ent_rf), color=ACCENT_BLUE, linestyle="--", linewidth=1.5)
    ax4.axvline(np.median(ent_mlp), color=ACCENT_ORANGE, linestyle="--", linewidth=1.5)
    ax4.set_xlabel("Entropy")
    ax4.set_ylabel("Density")
    ax4.set_title("Entropy Distribution")
    ax4.legend()

    # Panel (e): Per-class entropy boxplot
    ax5 = fig.add_subplot(2, 3, 5)
    data_rf = [ent_rf[y == c] for c in sorted(CLASS_COLORS) if (y == c).any()]
    data_mlp = [ent_mlp[y == c] for c in sorted(CLASS_COLORS) if (y == c).any()]
    labels_box = [CLASS_NAMES[c] for c in sorted(CLASS_COLORS) if (y == c).any()]
    n_cls = len(labels_box)
    pos_rf = np.arange(n_cls) - 0.18
    pos_mlp = np.arange(n_cls) + 0.18
    bp1 = ax5.boxplot(data_rf, positions=pos_rf, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor=ACCENT_BLUE, alpha=0.4))
    bp2 = ax5.boxplot(data_mlp, positions=pos_mlp, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor=ACCENT_ORANGE, alpha=0.4))
    ax5.set_xticks(np.arange(n_cls))
    ax5.set_xticklabels(labels_box)
    ax5.set_ylabel("Entropy")
    ax5.set_title("Per-Class Entropy")
    ax5.legend([bp1["boxes"][0], bp2["boxes"][0]], ["RF", "MLP"], fontsize=7)

    # Panel (f): Confidence (max prob) comparison
    ax6 = fig.add_subplot(2, 3, 6)
    conf_rf = y_prob_rf.max(axis=1)
    conf_mlp = y_prob_mlp.max(axis=1)
    ax6.scatter(conf_rf, conf_mlp, c=["#009E73" if y[i] == y_pred_rf[i] else "#CC3311"
                                       for i in range(len(y))],
                s=20, alpha=0.6)
    ax6.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax6.set_xlabel("RF Max Probability")
    ax6.set_ylabel("MLP Max Probability")
    ax6.set_title("Prediction Confidence")
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#009E73",
               markersize=6, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CC3311",
               markersize=6, label="Incorrect"),
    ]
    ax6.legend(handles=legend_elems, fontsize=7)

    fig.suptitle("Uncertainty & Confidence Analysis: RF vs MLP (N=105, L3, M+C, 2+1+2)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    save_fig(fig, output_dir, "fig4_uncertainty_analysis")


def fig5_context_window_effect(
    raw_cfg, sensor_array, sensor_ts, event_ts, labels, model_loader,
    device, output_dir: Path,
):
    """Fig 5: Class separation across different context window sizes."""
    setup_style()
    contexts = [
        (1, 1, "1+1+1 (3min)"),
        (2, 2, "2+1+2 (5min)"),
        (3, 3, "3+1+3 (7min)"),
        (4, 4, "4+1+4 (9min)"),
        (5, 5, "5+1+5 (11min)"),
    ]
    layer = 3  # Optimal layer

    fig, axes = plt.subplots(1, len(contexts), figsize=(4.5 * len(contexts), 4))
    accuracies = []

    for col, (cb, ca, label) in enumerate(contexts):
        logger.info("  Context %s: extracting embeddings...", label)
        model = model_loader(layer)
        Z, y = extract_embeddings(
            model, sensor_array, sensor_ts, event_ts, labels,
            channels=None, ctx_before=cb, ctx_after=ca,
        )

        # Quick RF accuracy
        y_pred, _ = run_rf_loocv(Z, y)
        acc = 100 * (y == y_pred).mean()
        accuracies.append(acc)

        emb = reduce_2d(Z, method="tsne")
        ax = axes[col]
        for cls in sorted(CLASS_COLORS):
            m = y == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=20, alpha=0.6, edgecolors="none")
        ax.set_title(f"{label}\nAcc={acc:.1f}%", fontsize=10)
        ax.set_xlabel("Comp 1")
        if col == 0:
            ax.set_ylabel("Comp 2")
        ax.legend(fontsize=6, markerscale=1.2)

    fig.suptitle("Context Window Effect on Class Separation (L3, M+C, RF LOOCV)",
                 fontsize=12, fontweight="bold", y=1.02)
    save_fig(fig, output_dir, "fig5_context_window_effect")

    # Save accuracy summary
    summary = output_dir / "fig5_context_accuracies.txt"
    with open(summary, "w") as f:
        f.write("Context Window vs Accuracy (RF, L3, M+C, N=105 LOOCV)\n")
        f.write("=" * 50 + "\n")
        for (cb, ca, label), acc in zip(contexts, accuracies):
            f.write(f"  {label:20s}  Acc = {acc:.2f}%\n")
    logger.info("Context summary saved: %s", summary)


def fig6_layer_progression(
    raw_cfg, sensor_array, sensor_ts, event_ts, labels, model_loader,
    output_dir: Path,
):
    """Fig 6: Embedding evolution across L0-L5."""
    setup_style()
    layers = [0, 1, 2, 3, 4, 5]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    accuracies = {}

    for idx, layer in enumerate(layers):
        logger.info("  Layer L%d: extracting embeddings...", layer)
        model = model_loader(layer)
        Z, y = extract_embeddings(
            model, sensor_array, sensor_ts, event_ts, labels,
            channels=None, ctx_before=2, ctx_after=2,
        )

        y_pred, _ = run_rf_loocv(Z, y)
        acc = 100 * (y == y_pred).mean()
        accuracies[layer] = acc

        emb = reduce_2d(Z, method="tsne")
        ax = axes[idx]
        for cls in sorted(CLASS_COLORS):
            m = y == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=20, alpha=0.6, edgecolors="none")
        ax.set_title(f"Layer {layer} — Acc={acc:.1f}%", fontsize=10)
        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")
        if idx == 0:
            ax.legend(fontsize=7, markerscale=1.2)

    fig.suptitle("MantisV2 Layer Progression (t-SNE, M+C, 2+1+2, RF LOOCV, N=105)",
                 fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, output_dir, "fig6_layer_progression")

    # Save summary
    summary = output_dir / "fig6_layer_accuracies.txt"
    with open(summary, "w") as f:
        f.write("Layer vs Accuracy (RF, M+C, 2+1+2, N=105 LOOCV)\n")
        f.write("=" * 40 + "\n")
        for l, acc in sorted(accuracies.items()):
            f.write(f"  L{l}: {acc:.2f}%\n")
    logger.info("Layer summary saved: %s", summary)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive visualization analysis for Enter/Leave detection",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g., enter-leave-phase1.yaml)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/visualization_analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow experiments (context sweep, layer sweep)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    setup_style()

    output_dir = Path(args.output_dir)
    raw_cfg = load_config(args.config)
    device = args.device
    pretrained = raw_cfg.get("model", {}).get("pretrained_name", "paris-noah/MantisV2")
    output_token = raw_cfg.get("model", {}).get("output_token", "combined")

    # Determine data paths
    data_cfg = raw_cfg.get("data", {})
    events_csv_cleaned = data_cfg.get("events_csv", "")  # N=105 (cleaned)
    # N=109: replace '_with_none_cleaned' with '' to get original
    events_csv_original = events_csv_cleaned.replace("_with_none_cleaned", "")
    if events_csv_original == events_csv_cleaned:
        # Fallback: try replacing '_cleaned'
        events_csv_original = events_csv_cleaned.replace("_cleaned", "")

    # Restrict channels to M+C (optimal)
    cfg_mc = copy.deepcopy(raw_cfg)
    cfg_mc["data"]["channels"] = [
        "d620900d_motionSensor",
        "408981c2_contactSensor",
    ]

    # Model loader factory (caches models per layer)
    _model_cache = {}
    def model_loader(layer):
        if layer not in _model_cache:
            _model_cache[layer] = load_mantis(pretrained, layer, output_token, device)
        return _model_cache[layer]

    t0 = time.time()

    # ==== Load N=109 data (original CSV with collisions) ====
    logger.info("=" * 60)
    logger.info("Loading N=109 data (with collisions)...")
    try:
        sensor_arr, sensor_ts, event_ts_109, labels_109, ch_names, cls_names = \
            load_data(cfg_mc, include_none=True, events_csv_override=events_csv_original)
        logger.info("N=109: %d events, %d channels", len(labels_109), len(ch_names))
        has_109 = True
    except Exception as e:
        logger.warning("Could not load N=109 data (%s). Skipping Fig 1 & 2.", e)
        has_109 = False

    # ==== Load N=105 data (cleaned CSV) ====
    logger.info("Loading N=105 data (cleaned)...")
    sensor_arr_105, sensor_ts_105, event_ts_105, labels_105, ch_names_105, cls_names_105 = \
        load_data(cfg_mc, include_none=True)
    logger.info("N=105: %d events, %d channels", len(labels_105), len(ch_names_105))

    # ==== Extract embeddings at L3 (optimal layer) ====
    logger.info("Extracting L3 embeddings for N=105 (M+C, 2+1+2)...")
    model_l3 = model_loader(3)
    Z_105, y_105 = extract_embeddings(
        model_l3, sensor_arr_105, sensor_ts_105, event_ts_105, labels_105,
        channels=ch_names_105, ctx_before=2, ctx_after=2,
    )
    embed_dim = Z_105.shape[1]
    logger.info("N=105 embeddings: shape=%s", Z_105.shape)

    if has_109:
        logger.info("Extracting L3 embeddings for N=109...")
        Z_109, y_109 = extract_embeddings(
            model_l3, sensor_arr, sensor_ts, event_ts_109, labels_109,
            channels=ch_names, ctx_before=2, ctx_after=2,
        )
        logger.info("N=109 embeddings: shape=%s", Z_109.shape)

    # ==== Fig 1: N=109 vs N=105 ====
    if has_109:
        logger.info("=" * 60)
        logger.info("Generating Fig 1: N=109 vs N=105...")
        fig1_n109_vs_n105(Z_109, y_109, Z_105, y_105, output_dir)

    # ==== Fig 2: Collision analysis ====
    if has_109:
        logger.info("Generating Fig 2: Collision analysis...")
        fig2_collision_analysis(Z_109, y_109, event_ts_109, output_dir)

    # ==== Run LOOCV for N=105 (RF + MLP) ====
    logger.info("=" * 60)
    logger.info("Running RF LOOCV on N=105 (this takes ~2s)...")
    y_pred_rf, y_prob_rf = run_rf_loocv(Z_105, y_105, seed=args.seed)
    acc_rf = 100 * (y_105 == y_pred_rf).mean()
    logger.info("RF Accuracy: %.2f%% (%d/%d)", acc_rf, (y_105 == y_pred_rf).sum(), len(y_105))

    logger.info("Running MLP LOOCV on N=105 (this takes ~30-60s)...")
    y_pred_mlp, y_prob_mlp = run_mlp_loocv(
        Z_105, y_105, embed_dim, device=device, seed=args.seed,
    )
    acc_mlp = 100 * (y_105 == y_pred_mlp).mean()
    logger.info("MLP Accuracy: %.2f%% (%d/%d)", acc_mlp, (y_105 == y_pred_mlp).sum(), len(y_105))

    # ==== Fig 3: Classification overlay ====
    logger.info("Generating Fig 3: Classification overlay...")
    fig3_classification_overlay(Z_105, y_105, y_pred_rf, y_pred_mlp, output_dir)

    # ==== Fig 4: Uncertainty analysis ====
    logger.info("Generating Fig 4: Uncertainty analysis...")
    fig4_uncertainty_analysis(Z_105, y_105, y_prob_rf, y_prob_mlp,
                              y_pred_rf, y_pred_mlp, output_dir)

    # ==== Fig 5: Context window effect (slow) ====
    if not args.quick:
        logger.info("=" * 60)
        logger.info("Generating Fig 5: Context window effect (slow — 5 extractions)...")
        fig5_context_window_effect(
            cfg_mc, sensor_arr_105, sensor_ts_105, event_ts_105, labels_105,
            model_loader, device, output_dir,
        )
    else:
        logger.info("Skipping Fig 5 (--quick mode)")

    # ==== Fig 6: Layer progression (slow) ====
    if not args.quick:
        logger.info("Generating Fig 6: Layer progression (slow — 6 extractions)...")
        fig6_layer_progression(
            cfg_mc, sensor_arr_105, sensor_ts_105, event_ts_105, labels_105,
            model_loader, output_dir,
        )
    else:
        logger.info("Skipping Fig 6 (--quick mode)")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All figures saved to: %s", output_dir.resolve())
    logger.info("Total time: %.1fs", elapsed)
    logger.info(
        "Summary — RF: %.2f%% (%d errors) | MLP: %.2f%% (%d errors)",
        acc_rf, (y_105 != y_pred_rf).sum(),
        acc_mlp, (y_105 != y_pred_mlp).sum(),
    )


if __name__ == "__main__":
    main()
