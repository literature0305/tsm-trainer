#!/usr/bin/env python3
"""Visualization Analysis for Enter/Leave Event Detection.

Generates publication-quality t-SNE figures analyzing MantisV2 embeddings
for the 3-class (Enter/Leave/None) classification task.

Representative model:
    MantisV2 L3, M+C (2 channels, 1024-d), 2+1+2 bidirectional context.

Figures (saved as PNG + PDF):
  Fig 1 — N=109 vs N=105: Side-by-side t-SNE showing collision cleanup impact
  Fig 2 — Collision proof: Indistinguishable collision pairs highlighted
  Fig 3 — Classification overlay: RF vs MLP on N=105 (correct/incorrect)
  Fig 4 — Uncertainty analysis: Per-sample entropy & confidence comparison
  Fig 5 — Layer ablation: L0-L5 embedding quality comparison
  Fig 6 — Context window ablation: Impact of temporal context size
  Fig 7 — Channel ablation: Contribution of each sensor channel

Usage:
    cd examples/classification/apc_enter_leave
    python analysis/run_visualization_analysis.py \
        --config training/configs/enter-leave-phase1.yaml \
        --device cuda --output-dir results/visualization_analysis
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import sys
import time
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
# Color palette (matplotlib tab10 — universally recognized, high contrast)
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    0: "#1f77b4",  # Enter — standard blue
    1: "#ff7f0e",  # Leave — standard orange
    2: "#7f7f7f",  # None  — neutral gray
}
CLASS_NAMES = {0: "Enter", 1: "Leave", 2: "None"}
ACCENT_GREEN = "#2ca02c"  # correct / positive
ACCENT_RED = "#d62728"    # error / negative
DPI = 300


def setup_style():
    """Apply publication-quality rcParams with refined typography."""
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 8,
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


def save_fig(fig, output_dir: Path, name: str):
    """Save figure as PNG + PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=DPI, format=fmt, bbox_inches="tight")
    logger.info("Saved: %s.{png,pdf}", output_dir / name)
    plt.close(fig)


# ============================================================================
# t-SNE reduction
# ============================================================================

def tsne_2d(Z: np.ndarray, seed: int = 42, pca_pre: int = 50) -> np.ndarray:
    """Scale -> PCA pre-reduction -> t-SNE to 2D."""
    n, d = Z.shape
    X = StandardScaler().fit_transform(Z)
    n_pre = min(pca_pre, d, n)
    if d > n_pre:
        X = PCA(n_components=n_pre, random_state=seed).fit_transform(X)
    perp = min(30, max(2, n - 1))
    return TSNE(
        n_components=2, perplexity=perp, max_iter=1500, random_state=seed,
    ).fit_transform(X)


# ============================================================================
# Data & model helpers
# ============================================================================

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(raw_cfg: dict, include_none: bool = True,
              events_csv_override: str | None = None):
    """Load sensor + events."""
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
    net = MantisV2(device=device, return_transf_layer=layer,
                   output_token=output_token)
    net = net.from_pretrained(pretrained)
    trainer = MantisTrainer(device=device, network=net)
    return trainer


def extract_embeddings(
    model, sensor_array, sensor_ts, event_ts, event_labels,
    ctx_before=2, ctx_after=2,
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
    if int(np.isnan(Z).sum()) > 0:
        Z = np.nan_to_num(Z, nan=0.0)
    return Z, y


def detect_collision_pairs(event_ts, event_labels):
    """Find collision pairs (same-timestamp, different labels)."""
    import pandas as pd
    ts_series = pd.Series(event_labels, index=pd.DatetimeIndex(event_ts))
    dup_mask = ts_series.index.duplicated(keep=False)
    dup_indices = np.where(dup_mask)[0]

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
# LOOCV: sklearn RF
# ============================================================================

def run_rf_loocv(Z, y, seed=42):
    """LOOCV with Random Forest. Returns (y_pred, y_prob)."""
    n = len(y)
    n_cls = len(np.unique(y))
    y_pred = np.zeros(n, dtype=np.int64)
    y_prob = np.zeros((n, n_cls), dtype=np.float64)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scaler = StandardScaler()
        Ztr = scaler.fit_transform(Z[mask])
        Zte = scaler.transform(Z[i:i + 1])
        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
        )
        clf.fit(Ztr, y[mask])
        y_pred[i] = clf.predict(Zte)[0]
        proba = clf.predict_proba(Zte)[0]
        full = np.zeros(n_cls, dtype=np.float64)
        for ci, c in enumerate(clf.classes_):
            full[int(c)] = proba[ci]
        y_prob[i] = full

    return y_pred, y_prob


# ============================================================================
# LOOCV: neural MLP
# ============================================================================

def run_mlp_loocv(Z, y, embed_dim, device="cpu", seed=42):
    """LOOCV with MLP[64]-d0.5 head. Returns (y_pred, y_prob)."""
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
        Zte_np = scaler.transform(Z[i:i + 1])

        Ztr = torch.from_numpy(Ztr_np).float().to(dev)
        ytr = torch.from_numpy(y[mask]).long().to(dev)
        Zte = torch.from_numpy(Zte_np).float().to(dev)

        torch.manual_seed(seed)
        head = MLPHead(
            embed_dim, n_cls, hidden_dims=[64], dropout=0.5,
            use_batchnorm=False,
        )
        head = head.to(dev)
        head.train()

        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_loss, patience = float("inf"), 0
        for epoch in range(200):
            logits = head(Ztr)
            loss = loss_fn(logits, ytr)
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
# Figure 1: N=109 vs N=105 Comparison
# Scenario: Shows the impact of removing 4 timestamp-collision events.
#   Collision events share identical sensor windows but carry different
#   labels, making them theoretically unresolvable.
# ============================================================================

def fig1_n109_vs_n105(emb_109, y_109, emb_105, y_105, output_dir: Path):
    """Side-by-side t-SNE: N=109 (with collisions) vs N=105 (cleaned)."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, (emb, y, title) in enumerate([
        (emb_109, y_109, "N=109 (with collisions)"),
        (emb_105, y_105, "N=105 (cleaned)"),
    ]):
        ax = axes[col]
        for cls in sorted(CLASS_COLORS):
            m = y == cls
            if m.any():
                ax.scatter(
                    emb[m, 0], emb[m, 1],
                    c=CLASS_COLORS[cls], label=f"{CLASS_NAMES[cls]} ({m.sum()})",
                    s=25, alpha=0.7, edgecolors="white", linewidths=0.3,
                )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(
            loc="lower right", frameon=True, framealpha=0.7,
            edgecolor="#CCCCCC", fontsize=7, markerscale=1.2,
        )

    fig.suptitle(
        "Enter/Leave Embedding Space: Impact of Timestamp Collision Removal\n"
        "(MantisV2 L3, M+C, 2+1+2 bidirectional)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_n109_vs_n105")


# ============================================================================
# Figure 2: Collision Proof
# Scenario: Proves that collision events occupy identical positions in
#   embedding space (connected by dashed lines) — different labels but
#   indistinguishable input, guaranteeing classification errors.
# ============================================================================

def fig2_collision_analysis(emb_109, y_109, pairs, dup_indices,
                             output_dir: Path):
    """Highlight collision samples — identical embeddings, different labels."""
    setup_style()
    if len(dup_indices) == 0:
        logger.warning("No collision events detected; skipping Fig 2")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    # Background: non-collision (faded)
    non_dup = np.ones(len(y_109), dtype=bool)
    non_dup[dup_indices] = False
    for cls in sorted(CLASS_COLORS):
        m = (y_109 == cls) & non_dup
        if m.any():
            ax.scatter(emb_109[m, 0], emb_109[m, 1], c=CLASS_COLORS[cls],
                       label=CLASS_NAMES[cls], s=15, alpha=0.25,
                       edgecolors="none")

    # Foreground: collision (diamond markers)
    for cls in sorted(CLASS_COLORS):
        m_cls = np.zeros(len(y_109), dtype=bool)
        for idx in dup_indices:
            if y_109[idx] == cls:
                m_cls[idx] = True
        if m_cls.any():
            ax.scatter(emb_109[m_cls, 0], emb_109[m_cls, 1],
                       c=CLASS_COLORS[cls], s=55, alpha=1.0,
                       edgecolors="black", linewidths=1.2,
                       zorder=10, marker="D",
                       label=f"{CLASS_NAMES[cls]} (collision)")

    for a, b in pairs:
        ax.plot([emb_109[a, 0], emb_109[b, 0]],
                [emb_109[a, 1], emb_109[b, 1]],
                color="#333333", linewidth=1.0, linestyle="--",
                alpha=0.6, zorder=9)
        la, lb = CLASS_NAMES[y_109[a]], CLASS_NAMES[y_109[b]]
        mid_x = (emb_109[a, 0] + emb_109[b, 0]) / 2
        mid_y = (emb_109[a, 1] + emb_109[b, 1]) / 2
        ax.annotate(f"{la}/{lb}", (mid_x, mid_y), fontsize=7,
                    fontweight="bold", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFDD",
                              ec="#999999", alpha=0.85, linewidth=0.6))

    ax.set_title(
        f"Timestamp Collision Analysis: {len(dup_indices)} events at "
        f"{len(pairs)} shared timestamps\n"
        "Collision pairs share identical sensor windows "
        "\u2192 overlapping embeddings, different labels",
        fontsize=10, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=True, framealpha=0.8,
              edgecolor="#CCCCCC", markerscale=0.8, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.15)
    save_fig(fig, output_dir, "fig2_collision_analysis")


# ============================================================================
# Figure 3: Classification Overlay (RF vs MLP)
# Scenario: Compares RF and MLP[64]-d0.5 on the N=105 cleaned dataset
#   using LOOCV. Both achieve 96.19% — the same 4 errors at the
#   Enter/Leave boundary confirm the practical ceiling.
# ============================================================================

def fig3_classification_overlay(emb, y, y_pred_rf, y_pred_mlp,
                                 output_dir: Path):
    """Side-by-side: RF vs MLP classification results on N=105."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for col, (name, y_pred) in enumerate([
        ("Random Forest (sklearn)", y_pred_rf),
        ("MLP[64]-d0.5 (neural)", y_pred_mlp),
    ]):
        ax = axes[col]
        correct = y == y_pred
        incorrect = ~correct

        for cls in sorted(CLASS_COLORS):
            m = (y == cls) & correct
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=22, alpha=0.6,
                           edgecolors="white", linewidths=0.3)

        if incorrect.any():
            ax.scatter(emb[incorrect, 0], emb[incorrect, 1], c=ACCENT_RED,
                       s=40, alpha=0.95, edgecolors="#333333", linewidths=1.0,
                       marker="X", zorder=10, label="Misclassified")
            for idx in np.where(incorrect)[0]:
                ax.annotate(
                    f"{CLASS_NAMES[y[idx]]}\u2192{CLASS_NAMES[y_pred[idx]]}",
                    (emb[idx, 0], emb[idx, 1]), fontsize=6.5,
                    fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="#444444", alpha=0.85, linewidth=0.6))

        n_err = incorrect.sum()
        acc = 100 * correct.mean()
        ax.set_title(f"{name}\nAccuracy = {acc:.2f}%  "
                     f"({n_err} error{'s' if n_err != 1 else ''})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, loc="lower right", frameon=True,
                  framealpha=0.7, edgecolor="#CCCCCC", markerscale=0.9)

    fig.suptitle(
        "N=105 Classification Results: sklearn RF vs Neural MLP\n"
        "(MantisV2 L3, M+C, 2+1+2 bidirectional, LOOCV)",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_classification_overlay")


# ============================================================================
# Figure 4: Uncertainty Analysis
# Scenario: Compares prediction uncertainty between RF and MLP via
#   entropy heatmaps and confidence scatter. Reveals whether errors
#   correlate with high uncertainty and if the two models disagree.
# ============================================================================

def fig4_uncertainty_analysis(emb, y, y_prob_rf, y_prob_mlp,
                               y_pred_rf, y_pred_mlp, output_dir: Path):
    """4-panel uncertainty comparison: entropy maps + scatter + confidence."""
    setup_style()

    ent_rf = scipy_entropy(y_prob_rf, axis=1)
    ent_mlp = scipy_entropy(y_prob_mlp, axis=1)
    max_ent = np.log(y_prob_rf.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))

    # (a) RF entropy heatmap
    ax = axes[0, 0]
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=ent_rf, cmap="YlOrRd",
                    s=20, alpha=0.85, vmin=0, vmax=max_ent,
                    edgecolors="white", linewidths=0.15)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(a) RF \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # (b) MLP entropy heatmap
    ax = axes[0, 1]
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=ent_mlp, cmap="YlOrRd",
                    s=20, alpha=0.85, vmin=0, vmax=max_ent,
                    edgecolors="white", linewidths=0.15)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(b) MLP \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # (c) RF vs MLP entropy scatter
    ax = axes[1, 0]
    correct_both = (y == y_pred_rf) & (y == y_pred_mlp)
    wrong_both = (y != y_pred_rf) & (y != y_pred_mlp)
    rf_only_wrong = (y != y_pred_rf) & (y == y_pred_mlp)
    mlp_only_wrong = (y == y_pred_rf) & (y != y_pred_mlp)

    if correct_both.any():
        ax.scatter(ent_rf[correct_both], ent_mlp[correct_both],
                   c="#BBBBBB", s=12, alpha=0.35, label="Both correct")
    if wrong_both.any():
        ax.scatter(ent_rf[wrong_both], ent_mlp[wrong_both], c=ACCENT_RED,
                   s=35, alpha=1.0, marker="X", edgecolors="#333333",
                   linewidths=0.7, label="Both wrong", zorder=10)
    if rf_only_wrong.any():
        ax.scatter(ent_rf[rf_only_wrong], ent_mlp[rf_only_wrong],
                   c="#1f77b4", s=28, alpha=0.9, marker="s",
                   edgecolors="#333333", linewidths=0.4,
                   label="RF wrong only", zorder=9)
    if mlp_only_wrong.any():
        ax.scatter(ent_rf[mlp_only_wrong], ent_mlp[mlp_only_wrong],
                   c="#ff7f0e", s=28, alpha=0.9, marker="^",
                   edgecolors="#333333", linewidths=0.4,
                   label="MLP wrong only", zorder=9)
    ax.plot([0, max_ent], [0, max_ent], "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("RF Entropy"); ax.set_ylabel("MLP Entropy")
    ax.set_title("(c) Per-Sample Entropy: RF vs MLP", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7, frameon=True, framealpha=0.7, edgecolor="#CCCCCC",
              loc="lower right", markerscale=0.9)

    # (d) Confidence scatter
    ax = axes[1, 1]
    conf_rf = y_prob_rf.max(axis=1)
    conf_mlp = y_prob_mlp.max(axis=1)
    colors = np.where(y == y_pred_rf, ACCENT_GREEN, ACCENT_RED)
    ax.scatter(conf_rf, conf_mlp, c=colors, s=15, alpha=0.55,
               edgecolors="white", linewidths=0.15)
    ax.plot([0.3, 1], [0.3, 1], "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("RF Max Probability"); ax.set_ylabel("MLP Max Probability")
    ax.set_title("(d) Prediction Confidence", fontsize=10, fontweight="bold")
    ax.set_xlim(0.25, 1.03); ax.set_ylim(0.25, 1.03)
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT_GREEN,
               markersize=5.5, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT_RED,
               markersize=5.5, label="Incorrect"),
    ]
    ax.legend(handles=legend_elems, fontsize=7, frameon=True, framealpha=0.7,
              edgecolor="#CCCCCC", loc="lower right")

    fig.suptitle("Uncertainty & Confidence Analysis: RF vs MLP "
                 "(N=105, L3, M+C, 2+1+2)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.subplots_adjust(hspace=0.28, wspace=0.28)
    save_fig(fig, output_dir, "fig4_uncertainty_analysis")


# ============================================================================
# Figure 5: Layer Ablation (2x3 grid)
# Scenario: Which transformer layer produces the best class separation?
#   Early layers (L0-L1) capture low-level temporal patterns; deeper
#   layers (L3-L5) form higher-level abstractions. L3 was identified
#   as optimal in the Phase 1 sweep.
# ============================================================================

def fig5_layer_ablation(pretrained, output_token, device,
                        sensor_arr, sensor_ts, event_ts, event_labels,
                        ctx_before, ctx_after, seed, output_dir: Path):
    """2x3 grid: t-SNE from each transformer layer L0-L5."""
    setup_style()
    layers = [0, 1, 2, 3, 4, 5]
    default_layer = 3

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, layer in enumerate(layers):
        ax = axes[i // 3, i % 3]
        logger.info("  Fig5 L%d: loading model + extracting...", layer)
        model = load_mantis(pretrained, layer=layer,
                            output_token=output_token, device=device)
        Z, y = extract_embeddings(model, sensor_arr, sensor_ts, event_ts,
                                   event_labels, ctx_before=ctx_before,
                                   ctx_after=ctx_after)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        emb = tsne_2d(Z, seed=seed)
        for cls in sorted(CLASS_COLORS):
            m = y == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           s=22, alpha=0.65, edgecolors="white",
                           linewidths=0.2)

        tag = " *" if layer == default_layer else ""
        ax.set_title(f"L{layer}{tag}", fontsize=10, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=CLASS_COLORS[c], markersize=6,
                      label=CLASS_NAMES[c])
               for c in sorted(CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CLASS_COLORS), fontsize=8, frameon=True,
               framealpha=0.7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Layer Ablation: Embedding Quality Across L0\u2013L5\n"
        f"(MantisV2, M+C, {ctx_before}+1+{ctx_after} bidirectional, "
        "N=105, * = optimal)",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    save_fig(fig, output_dir, "fig5_layer_ablation")


# ============================================================================
# Figure 6: Context Window Ablation (2x3 grid)
# Scenario: How does the size of the temporal context window affect class
#   separation? Enter/Leave events are highly localized — the optimal
#   window is 2+1+2 (5 min). Larger windows add noise from unrelated
#   sensor activity.
# ============================================================================

def fig6_context_ablation(model, sensor_arr, sensor_ts, event_ts,
                           event_labels, seed, output_dir: Path):
    """2x3 grid: t-SNE with varying context windows."""
    setup_style()
    contexts = [
        (1, 1, "1+1+1 (3 min)"),
        (2, 2, "2+1+2 (5 min) *"),
        (4, 4, "4+1+4 (9 min)"),
        (10, 10, "10+1+10 (21 min)"),
        (20, 20, "20+1+20 (41 min)"),
        (4, 0, "4+1+0 (backward)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, (cb, ca, label) in enumerate(contexts):
        ax = axes[i // 3, i % 3]
        logger.info("  Fig6 context %s: extracting...", label)
        Z, y = extract_embeddings(model, sensor_arr, sensor_ts, event_ts,
                                   event_labels, ctx_before=cb, ctx_after=ca)
        emb = tsne_2d(Z, seed=seed)

        for cls in sorted(CLASS_COLORS):
            m = y == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           s=22, alpha=0.65, edgecolors="white",
                           linewidths=0.2)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=CLASS_COLORS[c], markersize=6,
                      label=CLASS_NAMES[c])
               for c in sorted(CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CLASS_COLORS), fontsize=8, frameon=True,
               framealpha=0.7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Context Window Ablation: Temporal Scope vs Class Separation\n"
        "(MantisV2 L3, M+C, N=105, * = optimal)",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    save_fig(fig, output_dir, "fig6_context_ablation")


# ============================================================================
# Figure 7: Channel Ablation (2x3 grid)
# Scenario: Which sensor channels contribute most to class separation?
#   M (motion) and C (contact) are the two primary channels. T1
#   (temperature) adds marginal value. The optimal combo is M+C.
# ============================================================================

def fig7_channel_ablation(raw_cfg, model, ctx_before, ctx_after,
                           seed, output_dir: Path):
    """2x3 grid: t-SNE with different channel combinations."""
    setup_style()
    combos = [
        ("M only", ["d620900d_motionSensor"]),
        ("C only", ["408981c2_contactSensor"]),
        ("T1 only", ["d620900d_temperatureMeasurement"]),
        ("M+C *", ["d620900d_motionSensor", "408981c2_contactSensor"]),
        ("M+C+T1", ["d620900d_motionSensor", "408981c2_contactSensor",
                     "d620900d_temperatureMeasurement"]),
        ("All 6ch", None),  # None = use all from config
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, (name, channels) in enumerate(combos):
        ax = axes[i // 3, i % 3]
        logger.info("  Fig7 %s: loading data + extracting...", name)

        cfg_ch = copy.deepcopy(raw_cfg)
        if channels is not None:
            cfg_ch["data"]["channels"] = channels
        try:
            sensor_arr, sensor_ts, event_ts, labels, ch_names, _ = \
                load_data(cfg_ch, include_none=True)
            Z, y = extract_embeddings(model, sensor_arr, sensor_ts, event_ts,
                                       labels, ctx_before=ctx_before,
                                       ctx_after=ctx_after)
            emb = tsne_2d(Z, seed=seed)
            n_ch = len(ch_names)
            embed_d = Z.shape[1]

            for cls in sorted(CLASS_COLORS):
                m = y == cls
                if m.any():
                    ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                               s=22, alpha=0.65, edgecolors="white",
                               linewidths=0.2)
            ax.set_title(f"{name}\n({n_ch}ch, {embed_d}-d)", fontsize=9,
                         fontweight="bold")
        except Exception as e:
            logger.warning("  Fig7 %s failed: %s", name, e)
            ax.set_title(f"{name}\n(unavailable)", fontsize=9)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#999999")

        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=CLASS_COLORS[c], markersize=6,
                      label=CLASS_NAMES[c])
               for c in sorted(CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CLASS_COLORS), fontsize=8, frameon=True,
               framealpha=0.7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Channel Ablation: Sensor Contribution to Class Separation\n"
        f"(MantisV2 L3, {ctx_before}+1+{ctx_after} bidirectional, "
        "N=105, * = optimal)",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    save_fig(fig, output_dir, "fig7_channel_ablation")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualization analysis for Enter/Leave detection",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (enter-leave-phase1.yaml)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default="results/visualization_analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures", type=int, nargs="*", default=None,
                        help="Figure numbers to generate (default: all 1-7)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    setup_style()

    figs_to_run = set(args.figures) if args.figures else set(range(1, 8))
    output_dir = Path(args.output_dir)
    raw_cfg = load_config(args.config)
    device = args.device
    pretrained = raw_cfg.get("model", {}).get("pretrained_name",
                                               "paris-noah/MantisV2")
    output_token = raw_cfg.get("model", {}).get("output_token", "combined")

    # Data paths
    data_cfg = raw_cfg.get("data", {})
    events_csv_cleaned = data_cfg.get("events_csv", "")
    events_csv_original = events_csv_cleaned.replace("_with_none_cleaned", "")
    if events_csv_original == events_csv_cleaned:
        events_csv_original = events_csv_cleaned.replace("_cleaned", "")

    # M+C config (optimal channels)
    cfg_mc = copy.deepcopy(raw_cfg)
    cfg_mc["data"]["channels"] = [
        "d620900d_motionSensor", "408981c2_contactSensor",
    ]

    t0 = time.time()

    # ==== Load MantisV2 L3 (representative model) ====
    logger.info("=" * 60)
    logger.info("Loading MantisV2 L3 (representative model)...")
    model = load_mantis(pretrained, layer=3, output_token=output_token,
                        device=device)

    # ==== Load N=109 data (with collisions) ====
    has_109 = bool({1, 2} & figs_to_run)
    if has_109:
        logger.info("Loading N=109 data (with collisions)...")
        try:
            (sensor_arr_109, sensor_ts_109, event_ts_109, labels_109,
             ch_names_109, _) = load_data(
                cfg_mc, include_none=True,
                events_csv_override=events_csv_original)
            logger.info("N=109: %d events", len(labels_109))
        except Exception as e:
            logger.warning("Could not load N=109 data (%s). Skipping.", e)
            has_109 = False

    # ==== Load N=105 data (cleaned, M+C) ====
    logger.info("Loading N=105 data (cleaned)...")
    (sensor_arr_105, sensor_ts_105, event_ts_105, labels_105,
     ch_names_105, _) = load_data(cfg_mc, include_none=True)
    logger.info("N=105: %d events, %d channels",
                 len(labels_105), len(ch_names_105))

    # ==== Extract embeddings (L3, M+C, 2+1+2) ====
    logger.info("=" * 60)
    logger.info("Extracting L3 embeddings for N=105 (M+C, 2+1+2)...")
    Z_105, y_105 = extract_embeddings(
        model, sensor_arr_105, sensor_ts_105, event_ts_105, labels_105,
        ctx_before=2, ctx_after=2)
    embed_dim = Z_105.shape[1]
    logger.info("N=105 embeddings: shape=%s (dim=%d)", Z_105.shape, embed_dim)

    Z_109, y_109 = None, None
    if has_109:
        logger.info("Extracting L3 embeddings for N=109...")
        Z_109, y_109 = extract_embeddings(
            model, sensor_arr_109, sensor_ts_109, event_ts_109, labels_109,
            ctx_before=2, ctx_after=2)

    # ==== Pre-compute t-SNE ONCE per dataset (consistency across figures) ====
    logger.info("=" * 60)
    logger.info("Pre-computing t-SNE (once per dataset, reused by all figs)...")
    emb_105 = tsne_2d(Z_105, seed=args.seed)
    emb_109 = None
    if has_109:
        emb_109 = tsne_2d(Z_109, seed=args.seed)

    # ==== Collision detection ====
    pairs, dup_indices = [], np.array([], dtype=int)
    if has_109:
        pairs, dup_indices = detect_collision_pairs(event_ts_109, y_109)

    # ==== Fig 1: N=109 vs N=105 ====
    if 1 in figs_to_run and has_109:
        logger.info("Fig 1: N=109 vs N=105 t-SNE comparison...")
        fig1_n109_vs_n105(emb_109, y_109, emb_105, y_105, output_dir)

    # ==== Fig 2: Collision analysis ====
    if 2 in figs_to_run and has_109:
        logger.info("Fig 2: Collision sample analysis...")
        fig2_collision_analysis(emb_109, y_109, pairs, dup_indices, output_dir)

    # ==== Run LOOCV (RF + MLP) for Fig 3-4 ====
    y_pred_rf = y_prob_rf = y_pred_mlp = y_prob_mlp = None
    if {3, 4} & figs_to_run:
        logger.info("=" * 60)
        logger.info("Running RF LOOCV on N=105...")
        y_pred_rf, y_prob_rf = run_rf_loocv(Z_105, y_105, seed=args.seed)
        acc_rf = 100 * (y_105 == y_pred_rf).mean()
        logger.info("RF: %.2f%% (%d/%d correct)",
                     acc_rf, (y_105 == y_pred_rf).sum(), len(y_105))

        logger.info("Running MLP LOOCV on N=105 (~30-60s)...")
        y_pred_mlp, y_prob_mlp = run_mlp_loocv(
            Z_105, y_105, embed_dim, device=device, seed=args.seed)
        acc_mlp = 100 * (y_105 == y_pred_mlp).mean()
        logger.info("MLP: %.2f%% (%d/%d correct)",
                     acc_mlp, (y_105 == y_pred_mlp).sum(), len(y_105))

    # ==== Fig 3: Classification overlay ====
    if 3 in figs_to_run:
        logger.info("Fig 3: Classification overlay (RF vs MLP)...")
        fig3_classification_overlay(emb_105, y_105, y_pred_rf, y_pred_mlp,
                                     output_dir)

    # ==== Fig 4: Uncertainty analysis ====
    if 4 in figs_to_run:
        logger.info("Fig 4: Uncertainty analysis...")
        fig4_uncertainty_analysis(emb_105, y_105, y_prob_rf, y_prob_mlp,
                                   y_pred_rf, y_pred_mlp, output_dir)

    # ==== Fig 5: Layer ablation ====
    if 5 in figs_to_run:
        logger.info("=" * 60)
        logger.info("Fig 5: Layer ablation (L0-L5)...")
        fig5_layer_ablation(pretrained, output_token, device,
                            sensor_arr_105, sensor_ts_105, event_ts_105,
                            labels_105, 2, 2, args.seed, output_dir)

    # ==== Fig 6: Context window ablation ====
    if 6 in figs_to_run:
        logger.info("=" * 60)
        logger.info("Fig 6: Context window ablation...")
        fig6_context_ablation(model, sensor_arr_105, sensor_ts_105,
                               event_ts_105, labels_105, args.seed,
                               output_dir)

    # ==== Fig 7: Channel ablation ====
    if 7 in figs_to_run:
        logger.info("=" * 60)
        logger.info("Fig 7: Channel ablation...")
        fig7_channel_ablation(raw_cfg, model, 2, 2, args.seed, output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All figures saved to: %s", output_dir.resolve())
    logger.info("Total time: %.1fs", elapsed)
    if y_pred_rf is not None:
        logger.info("Summary -- RF: %.2f%% | MLP: %.2f%%",
                     100 * (y_105 == y_pred_rf).mean(),
                     100 * (y_105 == y_pred_mlp).mean())


if __name__ == "__main__":
    main()
