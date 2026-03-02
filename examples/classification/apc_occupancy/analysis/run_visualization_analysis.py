#!/usr/bin/env python3
"""Visualization Analysis for Occupancy Detection.

Generates publication-quality t-SNE figures analyzing MantisV2 embeddings
for the binary occupancy (Empty/Occupied) classification task.
Uses train/test split (date-based at 2026-02-15).

Representative model:
    MantisV2 L2, M+C+T1 (3 channels, 1536-d), 125+1+125 bidirectional.

Figures (saved as PNG + PDF):
  Fig 1 — Train/Test embedding space with binary class coloring
  Fig 2 — Classification overlay: SVM_rbf vs MLP on test set
  Fig 3 — Decision boundary: SVM vs MLP (PCA 2D projection)
  Fig 4 — Uncertainty analysis: Per-sample entropy & confidence comparison

Usage:
    cd examples/classification/apc_occupancy
    python analysis/run_visualization_analysis.py \
        --config training/configs/occupancy-phase1.yaml \
        --device cuda --output-dir results/visualization_analysis
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
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent  # apc_occupancy/
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import PreprocessConfig, load_occupancy_data
from data.dataset import DatasetConfig, OccupancyDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (matplotlib tab10 — universally recognized, high contrast)
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    0: "#1f77b4",  # Empty    — standard blue (cool/absence)
    1: "#ff7f0e",  # Occupied — standard orange (warm/presence)
}
CLASS_NAMES = {0: "Empty", 1: "Occupied"}
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


def load_data(raw_cfg: dict, split_date: str = "2026-02-15",
              channels: list[str] | None = None):
    """Load sensor + labels with date-based train/test split.

    Returns (sensor_array, train_labels, test_labels, channel_names, timestamps).
    """
    data_cfg = raw_cfg.get("data", {})
    cfg = PreprocessConfig(
        sensor_csv=data_cfg.get("sensor_csv", ""),
        label_csv=data_cfg.get("label_csv", ""),
        label_format=data_cfg.get("label_format", "events"),
        initial_occupancy=data_cfg.get("initial_occupancy", 0),
        nan_threshold=data_cfg.get("nan_threshold", 0.5),
        channels=channels or data_cfg.get("channels"),
        exclude_channels=data_cfg.get("exclude_channels", []),
        binarize=data_cfg.get("binarize", True),
        add_time_features=data_cfg.get("add_time_features", False),
    )
    return load_occupancy_data(cfg, split_date=split_date)


def load_mantis(pretrained: str, layer: int, output_token: str, device: str):
    """Load MantisV2 + MantisTrainer."""
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
    model, sensor_array, labels, timestamps, channel_names,
    ctx_before=125, ctx_after=125, stride=1,
):
    """Build sliding-window dataset and extract embeddings."""
    ds_cfg = DatasetConfig(
        context_mode="bidirectional",
        context_before=ctx_before,
        context_after=ctx_after,
        stride=stride,
    )
    dataset = OccupancyDataset(sensor_array, labels, timestamps, ds_cfg)
    X, y = dataset.get_numpy_arrays()
    if len(X) == 0:
        return np.array([]), np.array([])
    Z = model.transform(X)
    if int(np.isnan(Z).sum()) > 0:
        Z = np.nan_to_num(Z, nan=0.0)
    return Z, y


def subsample_balanced(Z, y, max_per_class=500, seed=42):
    """Balanced subsampling for visualization."""
    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > max_per_class:
            chosen = rng.choice(cls_idx, max_per_class, replace=False)
            indices.extend(chosen.tolist())
        else:
            indices.extend(cls_idx.tolist())
    indices = sorted(indices)
    return Z[indices], y[indices], np.array(indices)


# ============================================================================
# Classifiers
# ============================================================================

def run_svm_classifier(Z_train, y_train, Z_test, y_test, seed=42):
    """Train SVM_rbf on train, predict on test."""
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(Z_train)
    Zte = scaler.transform(Z_test)
    clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    clf.fit(Ztr, y_train)
    y_pred = clf.predict(Zte)
    y_prob = clf.predict_proba(Zte)
    return y_pred, y_prob, clf, scaler


def run_mlp_classifier(Z_train, y_train, Z_test, embed_dim,
                        hidden_dims=None, device="cpu", seed=42):
    """Train MLP head on train, predict on test."""
    from training.heads import MLPHead

    if hidden_dims is None:
        hidden_dims = [128]

    scaler = StandardScaler()
    Ztr_np = scaler.fit_transform(Z_train)
    Zte_np = scaler.transform(Z_test)

    n_cls = 2
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    head = MLPHead(embed_dim, n_cls, hidden_dims=hidden_dims, dropout=0.5,
                   use_batchnorm=True)
    head = head.to(dev)

    Ztr = torch.from_numpy(Ztr_np).float().to(dev)
    ytr = torch.from_numpy(y_train).long().to(dev)
    Zte = torch.from_numpy(Zte_np).float().to(dev)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    loss_fn = torch.nn.CrossEntropyLoss()

    head.train()
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
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    return y_pred, probs


# ============================================================================
# Figure 1: Train/Test Embedding Space
# ============================================================================

def fig1_train_test_embeddings(Z_train, y_train, Z_test, y_test,
                                 output_dir: Path):
    """Side-by-side t-SNE: Train vs Test with binary class coloring."""
    setup_style()

    # Subsample for visualization clarity
    max_viz = 600
    Ztr_viz, ytr_viz, _ = subsample_balanced(Z_train, y_train,
                                              max_per_class=max_viz)
    Zte_viz, yte_viz, _ = subsample_balanced(Z_test, y_test,
                                              max_per_class=max_viz)

    # Joint reduction for shared coordinate space
    [emb_tr, emb_te] = tsne_2d_joint(Ztr_viz, Zte_viz)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for col, (emb, y, title_prefix) in enumerate([
        (emb_tr, ytr_viz, "Train Set"),
        (emb_te, yte_viz, "Test Set"),
    ]):
        ax = axes[col]
        for cls in sorted(CLASS_COLORS):
            m = y == cls
            if m.any():
                ax.scatter(
                    emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                    label=f"{CLASS_NAMES[cls]} ({m.sum()})",
                    s=15, alpha=0.55, edgecolors="white", linewidths=0.15,
                )
        n_total = len(y)
        ax.set_title(
            f"{title_prefix} (N={n_total})", fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(
            loc="lower right", frameon=True, framealpha=0.7,
            edgecolor="#CCCCCC", fontsize=7, markerscale=1.5,
        )

    fig.suptitle(
        "Occupancy Embedding Space: Train vs Test\n"
        "(MantisV2 L2, M+C+T1, 125+1+125 bidirectional, "
        "split at 2026-02-15)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_train_test_embeddings")


# ============================================================================
# Figure 2: Classification Overlay (SVM vs MLP)
# ============================================================================

def fig2_classification_overlay(Z_test, y_test, y_pred_svm, y_pred_mlp,
                                  output_dir: Path):
    """Side-by-side: SVM vs MLP classification results on test set."""
    setup_style()

    # Subsample for clear visualization
    max_viz = 600
    Z_viz, y_viz, viz_idx = subsample_balanced(Z_test, y_test,
                                                max_per_class=max_viz)
    pred_svm_viz = y_pred_svm[viz_idx]
    pred_mlp_viz = y_pred_mlp[viz_idx]

    # Shared t-SNE embedding
    emb = tsne_2d(Z_viz)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for col, (name, y_pred) in enumerate([
        ("SVM_rbf (sklearn)", pred_svm_viz),
        ("MLP[128]-d0.5 (neural)", pred_mlp_viz),
    ]):
        ax = axes[col]
        correct = y_viz == y_pred
        incorrect = ~correct

        # Correct samples
        for cls in sorted(CLASS_COLORS):
            m = (y_viz == cls) & correct
            if m.any():
                ax.scatter(
                    emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                    label=CLASS_NAMES[cls],
                    s=14, alpha=0.5, edgecolors="white", linewidths=0.15,
                )

        # Misclassified samples: distinct red X markers
        if incorrect.any():
            ax.scatter(
                emb[incorrect, 0], emb[incorrect, 1],
                c=ACCENT_RED,
                s=30, alpha=0.9, edgecolors="#333333", linewidths=0.8,
                marker="X", zorder=10, label="Misclassified",
            )

        n_err = incorrect.sum()
        n_total = len(y_viz)
        acc = 100 * correct.mean()
        ax.set_title(
            f"{name}\n"
            f"Accuracy = {acc:.2f}%  ({n_err}/{n_total} errors)",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(
            fontsize=7, loc="lower right", frameon=True, framealpha=0.7,
            edgecolor="#CCCCCC", markerscale=0.9,
        )

    fig.suptitle(
        "Test Set Classification: SVM_rbf vs MLP[128]-d0.5\n"
        "(MantisV2 L2, M+C+T1, 125+1+125 bidirectional)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig2_classification_overlay")


# ============================================================================
# Figure 3: Decision Boundary (PCA 2D)
# ============================================================================

def fig3_decision_boundary(Z_train, y_train, Z_test, y_test,
                             y_pred_svm, y_pred_mlp, output_dir: Path):
    """Side-by-side PCA 2D: SVM vs MLP decision boundary."""
    setup_style()

    # Subsample train for faster SVM fitting on 2D
    Z_tr_sub, y_tr_sub, _ = subsample_balanced(Z_train, y_train,
                                                 max_per_class=1000)
    Z_te_sub, y_te_sub, te_idx = subsample_balanced(Z_test, y_test,
                                                      max_per_class=600)
    pred_svm_sub = y_pred_svm[te_idx]
    pred_mlp_sub = y_pred_mlp[te_idx]

    # PCA for decision boundary (t-SNE doesn't preserve for boundaries)
    scaler = StandardScaler()
    Z_all = np.concatenate([Z_tr_sub, Z_te_sub])
    Z_scaled = scaler.fit_transform(Z_all)
    pca = PCA(n_components=2, random_state=42)
    Z_2d = pca.fit_transform(Z_scaled)

    n_tr = len(Z_tr_sub)
    Z_2d_train = Z_2d[:n_tr]
    Z_2d_test = Z_2d[n_tr:]
    ev1 = pca.explained_variance_ratio_[0]
    ev2 = pca.explained_variance_ratio_[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for col, (name, y_pred) in enumerate([
        ("SVM_rbf (sklearn)", pred_svm_sub),
        ("MLP[128]-d0.5 (neural)", pred_mlp_sub),
    ]):
        ax = axes[col]

        # Fit an SVM on PCA-2D for boundary visualization
        clf_2d = SVC(kernel="rbf", C=1.0, gamma="scale")
        clf_2d.fit(Z_2d_train, y_tr_sub)

        # Decision region mesh
        margin = 1.5
        x_min = min(Z_2d_train[:, 0].min(), Z_2d_test[:, 0].min()) - margin
        x_max = max(Z_2d_train[:, 0].max(), Z_2d_test[:, 0].max()) + margin
        y_min = min(Z_2d_train[:, 1].min(), Z_2d_test[:, 1].min()) - margin
        y_max = max(Z_2d_train[:, 1].max(), Z_2d_test[:, 1].max()) + margin
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 250),
            np.linspace(y_min, y_max, 250),
        )
        zz = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(
            xx, yy, zz, alpha=0.10, levels=[-0.5, 0.5, 1.5],
            colors=[CLASS_COLORS[0], CLASS_COLORS[1]],
        )
        ax.contour(
            xx, yy, zz, levels=[0.5], colors=["#444444"],
            linewidths=1.2, alpha=0.5, linestyles="--",
        )

        # Test points
        correct = y_te_sub == y_pred
        incorrect = ~correct
        for cls in sorted(CLASS_COLORS):
            m = (y_te_sub == cls) & correct
            if m.any():
                ax.scatter(
                    Z_2d_test[m, 0], Z_2d_test[m, 1], c=CLASS_COLORS[cls],
                    s=12, alpha=0.5, edgecolors="white", linewidths=0.15,
                    label=CLASS_NAMES[cls],
                )
        if incorrect.any():
            ax.scatter(
                Z_2d_test[incorrect, 0], Z_2d_test[incorrect, 1],
                c=ACCENT_RED, s=25, marker="X", alpha=0.9, zorder=10,
                edgecolors="#333333", linewidths=0.6,
                label="Error",
            )

        acc = 100 * correct.mean()
        ax.set_title(f"{name}\nAcc = {acc:.2f}%", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel(f"PC1 ({ev1:.1%})")
        ax.set_ylabel(f"PC2 ({ev2:.1%})")
        ax.legend(
            fontsize=7, loc="lower right", frameon=True, framealpha=0.7,
            edgecolor="#CCCCCC", markerscale=1.0,
        )

    fig.suptitle(
        "Decision Boundary (PCA 2D): SVM_rbf vs MLP\n"
        "(L2, M+C+T1, 125+1+125 bidirectional)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_decision_boundary")


# ============================================================================
# Figure 4: Uncertainty Analysis
# ============================================================================

def fig4_uncertainty_analysis(Z_test, y_test, y_prob_svm, y_prob_mlp,
                                y_pred_svm, y_pred_mlp, output_dir: Path):
    """4-panel uncertainty comparison: entropy maps + scatter + confidence."""
    setup_style()

    # Subsample
    max_viz = 600
    Z_viz, y_viz, viz_idx = subsample_balanced(Z_test, y_test,
                                                max_per_class=max_viz)
    prob_svm = y_prob_svm[viz_idx]
    prob_mlp = y_prob_mlp[viz_idx]
    pred_svm = y_pred_svm[viz_idx]
    pred_mlp = y_pred_mlp[viz_idx]

    ent_svm = scipy_entropy(prob_svm, axis=1)
    ent_mlp = scipy_entropy(prob_mlp, axis=1)
    max_ent = np.log(2)  # binary

    emb = tsne_2d(Z_viz)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))

    # ---- (a) SVM entropy heatmap ----
    ax = axes[0, 0]
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=ent_svm, cmap="YlOrRd",
        s=14, alpha=0.8, vmin=0, vmax=max_ent, edgecolors="white",
        linewidths=0.15,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(a) SVM \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # ---- (b) MLP entropy heatmap ----
    ax = axes[0, 1]
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=ent_mlp, cmap="YlOrRd",
        s=14, alpha=0.8, vmin=0, vmax=max_ent, edgecolors="white",
        linewidths=0.15,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(b) MLP \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # ---- (c) SVM vs MLP entropy scatter ----
    ax = axes[1, 0]
    correct_both = (y_viz == pred_svm) & (y_viz == pred_mlp)
    wrong_both = (y_viz != pred_svm) & (y_viz != pred_mlp)
    svm_wrong = (y_viz != pred_svm) & (y_viz == pred_mlp)
    mlp_wrong = (y_viz == pred_svm) & (y_viz != pred_mlp)

    if correct_both.any():
        ax.scatter(
            ent_svm[correct_both], ent_mlp[correct_both],
            c="#BBBBBB", s=10, alpha=0.3, label="Both correct",
        )
    if wrong_both.any():
        ax.scatter(
            ent_svm[wrong_both], ent_mlp[wrong_both],
            c=ACCENT_RED, s=30, alpha=1.0, marker="X",
            edgecolors="#333333", linewidths=0.7,
            label="Both wrong", zorder=10,
        )
    if svm_wrong.any():
        ax.scatter(
            ent_svm[svm_wrong], ent_mlp[svm_wrong],
            c="#1f77b4", s=22, alpha=0.9, marker="s",
            edgecolors="#333333", linewidths=0.4,
            label="SVM wrong only", zorder=9,
        )
    if mlp_wrong.any():
        ax.scatter(
            ent_svm[mlp_wrong], ent_mlp[mlp_wrong],
            c="#ff7f0e", s=22, alpha=0.9, marker="^",
            edgecolors="#333333", linewidths=0.4,
            label="MLP wrong only", zorder=9,
        )
    ax.plot([0, max_ent], [0, max_ent], "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("SVM Entropy")
    ax.set_ylabel("MLP Entropy")
    ax.set_title(
        "(c) Per-Sample Entropy: SVM vs MLP", fontsize=10, fontweight="bold",
    )
    ax.legend(
        fontsize=7, frameon=True, framealpha=0.7, edgecolor="#CCCCCC",
        loc="lower right", markerscale=0.9,
    )

    # ---- (d) Confidence (max prob) comparison ----
    ax = axes[1, 1]
    conf_svm = prob_svm.max(axis=1)
    conf_mlp = prob_mlp.max(axis=1)
    colors = np.where(y_viz == pred_svm, ACCENT_GREEN, ACCENT_RED)
    ax.scatter(conf_svm, conf_mlp, c=colors, s=12, alpha=0.5,
               edgecolors="white", linewidths=0.15)
    ax.plot([0.5, 1], [0.5, 1], "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("SVM Max Probability")
    ax.set_ylabel("MLP Max Probability")
    ax.set_title("(d) Prediction Confidence", fontsize=10, fontweight="bold")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT_GREEN,
               markersize=5.5, label="Correct (SVM)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT_RED,
               markersize=5.5, label="Incorrect (SVM)"),
    ]
    ax.legend(
        handles=legend_elems, fontsize=7,
        frameon=True, framealpha=0.7, edgecolor="#CCCCCC",
        loc="lower right",
    )

    fig.suptitle(
        "Uncertainty & Confidence Analysis: SVM_rbf vs MLP[128]-d0.5\n"
        "(Test set, L2, M+C+T1, 125+1+125)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(hspace=0.28, wspace=0.28)
    save_fig(fig, output_dir, "fig4_uncertainty_analysis")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualization analysis for Occupancy detection",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (occupancy-phase1.yaml)")
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
    output_token = "combined"
    split_date = raw_cfg.get("split_date", "2026-02-15")
    ctx_before = raw_cfg.get("default_context_before", 125)
    ctx_after = raw_cfg.get("default_context_after", 125)
    default_layer = raw_cfg.get("default_layer", 2)

    # Optimal channels: M+C+T1
    CHANNELS_MCT1 = [
        "d620900d_motionSensor",
        "408981c2_contactSensor",
        "d620900d_temperatureMeasurement",
    ]

    t0 = time.time()

    # ==== Load MantisV2 L2 (representative model) ====
    logger.info("=" * 60)
    logger.info("Loading MantisV2 L%d (representative model)...", default_layer)
    model = load_mantis(pretrained, layer=default_layer,
                        output_token=output_token, device=device)

    # ==== Load data ====
    logger.info("Loading data with M+C+T1 channels...")
    result = load_data(raw_cfg, split_date=split_date, channels=CHANNELS_MCT1)
    sensor_arr, train_labels, test_labels, ch_names, timestamps = result
    logger.info("Channels: %s", ch_names)

    # ==== Extract embeddings ====
    logger.info("=" * 60)
    logger.info("Extracting embeddings (L%d, M+C+T1, %d+1+%d)...",
                 default_layer, ctx_before, ctx_after)

    Z_train, y_train = extract_embeddings(
        model, sensor_arr, train_labels, timestamps, ch_names,
        ctx_before=ctx_before, ctx_after=ctx_after,
    )
    Z_test, y_test = extract_embeddings(
        model, sensor_arr, test_labels, timestamps, ch_names,
        ctx_before=ctx_before, ctx_after=ctx_after,
    )
    embed_dim = Z_train.shape[1]
    logger.info("Train: %s, Test: %s (dim=%d)",
                 Z_train.shape, Z_test.shape, embed_dim)

    # ==== Classifiers ====
    logger.info("=" * 60)
    logger.info("Training SVM_rbf...")
    y_pred_svm, y_prob_svm, _, _ = run_svm_classifier(
        Z_train, y_train, Z_test, y_test, seed=args.seed,
    )
    acc_svm = 100 * (y_test == y_pred_svm).mean()
    logger.info("SVM Accuracy: %.2f%%", acc_svm)

    logger.info("Training MLP[128]-d0.5...")
    y_pred_mlp, y_prob_mlp = run_mlp_classifier(
        Z_train, y_train, Z_test, embed_dim,
        hidden_dims=[128], device=device, seed=args.seed,
    )
    acc_mlp = 100 * (y_test == y_pred_mlp).mean()
    logger.info("MLP Accuracy: %.2f%%", acc_mlp)

    # ==== Fig 1: Train/Test embedding space ====
    logger.info("=" * 60)
    logger.info("Fig 1: Train/Test embedding space...")
    fig1_train_test_embeddings(Z_train, y_train, Z_test, y_test, output_dir)

    # ==== Fig 2: Classification overlay ====
    logger.info("Fig 2: Classification overlay (SVM vs MLP)...")
    fig2_classification_overlay(Z_test, y_test, y_pred_svm, y_pred_mlp,
                                  output_dir)

    # ==== Fig 3: Decision boundary ====
    logger.info("Fig 3: Decision boundary (PCA 2D)...")
    fig3_decision_boundary(Z_train, y_train, Z_test, y_test,
                             y_pred_svm, y_pred_mlp, output_dir)

    # ==== Fig 4: Uncertainty analysis ====
    logger.info("Fig 4: Uncertainty analysis...")
    fig4_uncertainty_analysis(Z_test, y_test, y_prob_svm, y_prob_mlp,
                                y_pred_svm, y_pred_mlp, output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All figures saved to: %s", output_dir.resolve())
    logger.info("Total time: %.1fs", elapsed)
    logger.info(
        "Summary -- SVM: %.2f%% (%d errors) | MLP: %.2f%% (%d errors)",
        acc_svm, (y_test != y_pred_svm).sum(),
        acc_mlp, (y_test != y_pred_mlp).sum(),
    )


if __name__ == "__main__":
    main()
