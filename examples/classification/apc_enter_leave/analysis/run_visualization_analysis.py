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

Usage:
    cd examples/classification/apc_enter_leave
    python analysis/run_visualization_analysis.py \
        --config training/configs/enter-leave-phase1.yaml \
        --device cuda --output-dir results/visualization_analysis
"""

from __future__ import annotations

import argparse
import copy
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


def tsne_2d_joint(*arrays, seed=42, pca_pre=50):
    """Reduce multiple arrays in shared coordinate space."""
    sizes = [len(a) for a in arrays]
    combined = np.concatenate(arrays, axis=0)
    reduced = tsne_2d(combined, seed=seed, pca_pre=pca_pre)
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
    """Find collision pairs (same-timestamp, different labels).

    Returns (pairs, dup_indices) where pairs is list of (idx_a, idx_b).
    """
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
# ============================================================================

def fig1_n109_vs_n105(Z_109, y_109, Z_105, y_105, output_dir: Path):
    """Side-by-side t-SNE: N=109 (with collisions) vs N=105 (cleaned)."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, (Z, y, title, n_label) in enumerate([
        (Z_109, y_109, "N=109 (with collisions)", "109"),
        (Z_105, y_105, "N=105 (cleaned)", "105"),
    ]):
        emb = tsne_2d(Z)
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
# ============================================================================

def fig2_collision_analysis(Z_109, y_109, event_ts_109, output_dir: Path):
    """Highlight collision samples — identical embeddings, different labels."""
    setup_style()
    pairs, dup_indices = detect_collision_pairs(event_ts_109, y_109)

    if len(dup_indices) == 0:
        logger.warning("No collision events detected; skipping Fig 2")
        return

    emb = tsne_2d(Z_109)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Background: all non-collision samples (faded)
    non_dup = np.ones(len(y_109), dtype=bool)
    non_dup[dup_indices] = False
    for cls in sorted(CLASS_COLORS):
        m = (y_109 == cls) & non_dup
        if m.any():
            ax.scatter(
                emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls], s=15, alpha=0.25, edgecolors="none",
            )

    # Foreground: collision samples (refined diamond markers)
    for cls in sorted(CLASS_COLORS):
        m_cls = np.zeros(len(y_109), dtype=bool)
        for idx in dup_indices:
            if y_109[idx] == cls:
                m_cls[idx] = True
        if m_cls.any():
            ax.scatter(
                emb[m_cls, 0], emb[m_cls, 1], c=CLASS_COLORS[cls],
                s=55, alpha=1.0, edgecolors="black", linewidths=1.2,
                zorder=10, marker="D",
                label=f"{CLASS_NAMES[cls]} (collision)",
            )

    # Connect collision pairs with dashed lines + annotations
    for a, b in pairs:
        ax.plot(
            [emb[a, 0], emb[b, 0]], [emb[a, 1], emb[b, 1]],
            color="#333333", linewidth=1.0, linestyle="--", alpha=0.6, zorder=9,
        )
        la = CLASS_NAMES[y_109[a]]
        lb = CLASS_NAMES[y_109[b]]
        mid_x = (emb[a, 0] + emb[b, 0]) / 2
        mid_y = (emb[a, 1] + emb[b, 1]) / 2
        ax.annotate(
            f"{la}/{lb}", (mid_x, mid_y),
            fontsize=7, fontweight="bold", ha="center", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="#FFFFDD",
                ec="#999999", alpha=0.85, linewidth=0.6,
            ),
        )

    ax.set_title(
        f"Timestamp Collision Analysis: {len(dup_indices)} events at "
        f"{len(pairs)} shared timestamps\n"
        "Collision pairs share identical sensor windows "
        "\u2192 overlapping embeddings, different labels",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Legend below the plot to avoid occlusion
    ax.legend(
        fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, frameon=True, framealpha=0.8,
        edgecolor="#CCCCCC", markerscale=0.8, columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.15)
    save_fig(fig, output_dir, "fig2_collision_analysis")


# ============================================================================
# Figure 3: Classification Overlay (RF vs MLP)
# ============================================================================

def fig3_classification_overlay(Z, y, y_pred_rf, y_pred_mlp, output_dir: Path):
    """Side-by-side: RF vs MLP classification results on N=105."""
    setup_style()

    # Use single shared t-SNE embedding for fair comparison
    emb = tsne_2d(Z)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for col, (name, y_pred) in enumerate([
        ("Random Forest (sklearn)", y_pred_rf),
        ("MLP[64]-d0.5 (neural)", y_pred_mlp),
    ]):
        ax = axes[col]
        correct = y == y_pred
        incorrect = ~correct

        # Correct samples: colored by true class
        for cls in sorted(CLASS_COLORS):
            m = (y == cls) & correct
            if m.any():
                ax.scatter(
                    emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                    label=CLASS_NAMES[cls],
                    s=22, alpha=0.6, edgecolors="white", linewidths=0.3,
                )

        # Misclassified samples: distinct red X markers
        if incorrect.any():
            ax.scatter(
                emb[incorrect, 0], emb[incorrect, 1],
                c=ACCENT_RED,
                s=40, alpha=0.95, edgecolors="#333333", linewidths=1.0,
                marker="X", zorder=10, label="Misclassified",
            )
            # Annotate each error
            for idx in np.where(incorrect)[0]:
                true_cls = CLASS_NAMES[y[idx]]
                pred_cls = CLASS_NAMES[y_pred[idx]]
                ax.annotate(
                    f"{true_cls}\u2192{pred_cls}",
                    (emb[idx, 0], emb[idx, 1]),
                    fontsize=6.5, fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.12", fc="white",
                        ec="#444444", alpha=0.85, linewidth=0.6,
                    ),
                )

        n_err = incorrect.sum()
        acc = 100 * correct.mean()
        ax.set_title(
            f"{name}\n"
            f"Accuracy = {acc:.2f}%  "
            f"({n_err} error{'s' if n_err != 1 else ''})",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(
            fontsize=7, loc="lower right", frameon=True, framealpha=0.7,
            edgecolor="#CCCCCC", markerscale=0.9,
        )

    fig.suptitle(
        "N=105 Classification Results: sklearn RF vs Neural MLP\n"
        "(MantisV2 L3, M+C, 2+1+2 bidirectional, LOOCV)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_classification_overlay")


# ============================================================================
# Figure 4: Uncertainty Analysis
# ============================================================================

def fig4_uncertainty_analysis(
    Z, y, y_prob_rf, y_prob_mlp, y_pred_rf, y_pred_mlp, output_dir: Path,
):
    """4-panel uncertainty comparison: entropy maps + scatter + confidence."""
    setup_style()

    ent_rf = scipy_entropy(y_prob_rf, axis=1)
    ent_mlp = scipy_entropy(y_prob_mlp, axis=1)
    max_ent = np.log(y_prob_rf.shape[1])  # log(n_classes)

    emb = tsne_2d(Z)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))

    # ---- (a) RF entropy heatmap on t-SNE ----
    ax = axes[0, 0]
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=ent_rf, cmap="YlOrRd",
        s=20, alpha=0.85, vmin=0, vmax=max_ent, edgecolors="white",
        linewidths=0.15,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(a) RF \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # ---- (b) MLP entropy heatmap on t-SNE ----
    ax = axes[0, 1]
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=ent_mlp, cmap="YlOrRd",
        s=20, alpha=0.85, vmin=0, vmax=max_ent, edgecolors="white",
        linewidths=0.15,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(b) MLP \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # ---- (c) RF vs MLP entropy scatter ----
    ax = axes[1, 0]
    correct_both = (y == y_pred_rf) & (y == y_pred_mlp)
    wrong_both = (y != y_pred_rf) & (y != y_pred_mlp)
    rf_only_wrong = (y != y_pred_rf) & (y == y_pred_mlp)
    mlp_only_wrong = (y == y_pred_rf) & (y != y_pred_mlp)

    if correct_both.any():
        ax.scatter(
            ent_rf[correct_both], ent_mlp[correct_both],
            c="#BBBBBB", s=12, alpha=0.35, label="Both correct",
        )
    if wrong_both.any():
        ax.scatter(
            ent_rf[wrong_both], ent_mlp[wrong_both],
            c=ACCENT_RED, s=35, alpha=1.0, marker="X",
            edgecolors="#333333", linewidths=0.7,
            label="Both wrong", zorder=10,
        )
    if rf_only_wrong.any():
        ax.scatter(
            ent_rf[rf_only_wrong], ent_mlp[rf_only_wrong],
            c="#1f77b4", s=28, alpha=0.9, marker="s",
            edgecolors="#333333", linewidths=0.4,
            label="RF wrong only", zorder=9,
        )
    if mlp_only_wrong.any():
        ax.scatter(
            ent_rf[mlp_only_wrong], ent_mlp[mlp_only_wrong],
            c="#ff7f0e", s=28, alpha=0.9, marker="^",
            edgecolors="#333333", linewidths=0.4,
            label="MLP wrong only", zorder=9,
        )
    ax.plot([0, max_ent], [0, max_ent], "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("RF Entropy")
    ax.set_ylabel("MLP Entropy")
    ax.set_title(
        "(c) Per-Sample Entropy: RF vs MLP", fontsize=10, fontweight="bold",
    )
    ax.legend(
        fontsize=7, frameon=True, framealpha=0.7, edgecolor="#CCCCCC",
        loc="lower right", markerscale=0.9,
    )

    # ---- (d) Confidence (max prob) comparison ----
    ax = axes[1, 1]
    conf_rf = y_prob_rf.max(axis=1)
    conf_mlp = y_prob_mlp.max(axis=1)
    colors = np.where(y == y_pred_rf, ACCENT_GREEN, ACCENT_RED)
    ax.scatter(conf_rf, conf_mlp, c=colors, s=15, alpha=0.55,
               edgecolors="white", linewidths=0.15)
    ax.plot([0.3, 1], [0.3, 1], "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("RF Max Probability")
    ax.set_ylabel("MLP Max Probability")
    ax.set_title("(d) Prediction Confidence", fontsize=10, fontweight="bold")
    ax.set_xlim(0.25, 1.03)
    ax.set_ylim(0.25, 1.03)
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT_GREEN,
               markersize=5.5, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT_RED,
               markersize=5.5, label="Incorrect"),
    ]
    ax.legend(
        handles=legend_elems, fontsize=7,
        frameon=True, framealpha=0.7, edgecolor="#CCCCCC",
        loc="lower right",
    )

    fig.suptitle(
        "Uncertainty & Confidence Analysis: RF vs MLP "
        "(N=105, L3, M+C, 2+1+2)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(hspace=0.28, wspace=0.28)
    save_fig(fig, output_dir, "fig4_uncertainty_analysis")


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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    setup_style()

    output_dir = Path(args.output_dir)
    raw_cfg = load_config(args.config)
    device = args.device
    pretrained = raw_cfg.get("model", {}).get("pretrained_name",
                                               "paris-noah/MantisV2")
    output_token = raw_cfg.get("model", {}).get("output_token", "combined")

    # Data paths
    data_cfg = raw_cfg.get("data", {})
    events_csv_cleaned = data_cfg.get("events_csv", "")  # N=105
    events_csv_original = events_csv_cleaned.replace(
        "_with_none_cleaned", "",
    )
    if events_csv_original == events_csv_cleaned:
        events_csv_original = events_csv_cleaned.replace("_cleaned", "")

    # Restrict to M+C (optimal 2 channels)
    cfg_mc = copy.deepcopy(raw_cfg)
    cfg_mc["data"]["channels"] = [
        "d620900d_motionSensor",
        "408981c2_contactSensor",
    ]

    t0 = time.time()

    # ==== Load MantisV2 L3 (representative model) ====
    logger.info("=" * 60)
    logger.info("Loading MantisV2 L3 (representative model)...")
    model = load_mantis(pretrained, layer=3, output_token=output_token,
                        device=device)

    # ==== Load N=109 data (with collisions) ====
    logger.info("Loading N=109 data (with collisions)...")
    has_109 = True
    try:
        (sensor_arr, sensor_ts, event_ts_109, labels_109,
         ch_names, cls_names) = load_data(
            cfg_mc, include_none=True,
            events_csv_override=events_csv_original,
        )
        logger.info("N=109: %d events, %d channels",
                     len(labels_109), len(ch_names))
    except Exception as e:
        logger.warning("Could not load N=109 data (%s). Skipping Fig 1-2.", e)
        has_109 = False

    # ==== Load N=105 data (cleaned) ====
    logger.info("Loading N=105 data (cleaned)...")
    (sensor_arr_105, sensor_ts_105, event_ts_105, labels_105,
     ch_names_105, cls_names_105) = load_data(cfg_mc, include_none=True)
    logger.info("N=105: %d events, %d channels",
                 len(labels_105), len(ch_names_105))

    # ==== Extract embeddings (L3, M+C, 2+1+2) ====
    logger.info("=" * 60)
    logger.info("Extracting L3 embeddings for N=105 (M+C, 2+1+2)...")
    Z_105, y_105 = extract_embeddings(
        model, sensor_arr_105, sensor_ts_105, event_ts_105, labels_105,
        ctx_before=2, ctx_after=2,
    )
    embed_dim = Z_105.shape[1]
    logger.info("N=105 embeddings: shape=%s (dim=%d)", Z_105.shape, embed_dim)

    Z_109, y_109 = None, None
    if has_109:
        logger.info("Extracting L3 embeddings for N=109...")
        Z_109, y_109 = extract_embeddings(
            model, sensor_arr, sensor_ts, event_ts_109, labels_109,
            ctx_before=2, ctx_after=2,
        )
        logger.info("N=109 embeddings: shape=%s", Z_109.shape)

    # ==== Fig 1: N=109 vs N=105 ====
    if has_109:
        logger.info("=" * 60)
        logger.info("Fig 1: N=109 vs N=105 t-SNE comparison...")
        fig1_n109_vs_n105(Z_109, y_109, Z_105, y_105, output_dir)

    # ==== Fig 2: Collision analysis ====
    if has_109:
        logger.info("Fig 2: Collision sample analysis...")
        fig2_collision_analysis(Z_109, y_109, event_ts_109, output_dir)

    # ==== Run LOOCV (RF + MLP) ====
    logger.info("=" * 60)
    logger.info("Running RF LOOCV on N=105...")
    y_pred_rf, y_prob_rf = run_rf_loocv(Z_105, y_105, seed=args.seed)
    acc_rf = 100 * (y_105 == y_pred_rf).mean()
    logger.info("RF: %.2f%% (%d/%d correct)",
                 acc_rf, (y_105 == y_pred_rf).sum(), len(y_105))

    logger.info("Running MLP LOOCV on N=105 (this takes ~30-60s)...")
    y_pred_mlp, y_prob_mlp = run_mlp_loocv(
        Z_105, y_105, embed_dim, device=device, seed=args.seed,
    )
    acc_mlp = 100 * (y_105 == y_pred_mlp).mean()
    logger.info("MLP: %.2f%% (%d/%d correct)",
                 acc_mlp, (y_105 == y_pred_mlp).sum(), len(y_105))

    # ==== Fig 3: Classification overlay ====
    logger.info("Fig 3: Classification overlay (RF vs MLP)...")
    fig3_classification_overlay(Z_105, y_105, y_pred_rf, y_pred_mlp,
                                 output_dir)

    # ==== Fig 4: Uncertainty analysis ====
    logger.info("Fig 4: Uncertainty analysis...")
    fig4_uncertainty_analysis(Z_105, y_105, y_prob_rf, y_prob_mlp,
                               y_pred_rf, y_pred_mlp, output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All figures saved to: %s", output_dir.resolve())
    logger.info("Total time: %.1fs", elapsed)
    logger.info(
        "Summary -- RF: %.2f%% (%d errors) | MLP: %.2f%% (%d errors)",
        acc_rf, (y_105 != y_pred_rf).sum(),
        acc_mlp, (y_105 != y_pred_mlp).sum(),
    )


if __name__ == "__main__":
    main()
