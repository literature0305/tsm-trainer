#!/usr/bin/env python3
"""Visualization Analysis for Occupancy Detection.

Generates publication-quality t-SNE figures analyzing MantisV2 embeddings
for the binary occupancy (Empty/Occupied) classification task.
Uses train/test split (date-based at 2026-02-15).

Representative model:
    MantisV2 L2, M+C+T1 (3 channels, 1536-d), 120+1+120 bidirectional.

Figures (saved as PNG + PDF):
  Fig 1 — Train/Test embedding space: Joint t-SNE showing class distribution
  Fig 2 — Classification overlay: SVM_rbf vs MLP on test set (correct/errors)
  Fig 3 — Decision boundary: SVM vs MLP (PCA 2D projection)
  Fig 4 — Uncertainty analysis: Per-sample entropy & confidence comparison
  Fig 5 — Layer ablation: L0-L5 embedding quality comparison (test set)
  Fig 6 — Context window ablation: Impact of temporal context size (test set)
  Fig 7 — Channel ablation: Contribution of each sensor channel (test set)

Usage:
    cd examples/classification/apc_occupancy
    python analysis/run_visualization_analysis.py \
        --config training/configs/occupancy-phase1.yaml \
        --device cuda --output-dir results/visualization_analysis

    # Selective figures:
    python analysis/run_visualization_analysis.py \
        --config training/configs/occupancy-phase1.yaml \
        --device cuda --figures 5 6 7
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
    ctx_before=120, ctx_after=120, stride=1,
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
# Scenario: Shows how train and test sets distribute in MantisV2 embedding
#   space. Joint t-SNE preserves relative positions between splits.
#   Good separation indicates the model captures occupancy patterns;
#   overlap regions hint at classification boundary difficulty.
# ============================================================================

def fig1_train_test_embeddings(emb_tr, y_tr, emb_te, y_te, output_dir: Path):
    """Side-by-side t-SNE: Train vs Test with binary class coloring.

    Uses pre-computed joint t-SNE coordinates for consistency.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for col, (emb, y, title_prefix) in enumerate([
        (emb_tr, y_tr, "Train Set"),
        (emb_te, y_te, "Test Set"),
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
        "(MantisV2 L2, M+C+T1, 120+1+120 bidirectional, "
        "split at 2026-02-15)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_train_test_embeddings")


# ============================================================================
# Figure 2: Classification Overlay (SVM vs MLP)
# Scenario: Compares SVM_rbf and MLP[128]-d0.5 on the held-out test set.
#   Red X markers show misclassified samples — typically located at the
#   boundary between Empty and Occupied clusters. Highlights where each
#   classifier struggles and whether they make similar mistakes.
# ============================================================================

def fig2_classification_overlay(emb_te, y_te, pred_svm, pred_mlp,
                                  output_dir: Path):
    """Side-by-side: SVM vs MLP classification results on test set.

    Uses pre-computed t-SNE coordinates for consistency with Fig 1/4.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for col, (name, y_pred) in enumerate([
        ("SVM_rbf (sklearn)", pred_svm),
        ("MLP[128]-d0.5 (neural)", pred_mlp),
    ]):
        ax = axes[col]
        correct = y_te == y_pred
        incorrect = ~correct

        for cls in sorted(CLASS_COLORS):
            m = (y_te == cls) & correct
            if m.any():
                ax.scatter(
                    emb_te[m, 0], emb_te[m, 1], c=CLASS_COLORS[cls],
                    label=CLASS_NAMES[cls],
                    s=14, alpha=0.5, edgecolors="white", linewidths=0.15,
                )

        if incorrect.any():
            ax.scatter(
                emb_te[incorrect, 0], emb_te[incorrect, 1],
                c=ACCENT_RED,
                s=30, alpha=0.9, edgecolors="#333333", linewidths=0.8,
                marker="X", zorder=10, label="Misclassified",
            )

        n_err = incorrect.sum()
        n_total = len(y_te)
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
        "(MantisV2 L2, M+C+T1, 120+1+120 bidirectional)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig2_classification_overlay")


# ============================================================================
# Figure 3: Decision Boundary (PCA 2D)
# Scenario: Projects embeddings to 2D via PCA (preserves linear structure)
#   and visualizes SVM decision boundaries. Unlike t-SNE, PCA supports
#   decision surface rendering. The contour shows where the classifier
#   transitions between Empty and Occupied predictions.
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
        "(L2, M+C+T1, 120+1+120 bidirectional)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_decision_boundary")


# ============================================================================
# Figure 4: Uncertainty Analysis
# Scenario: Examines prediction uncertainty via entropy heatmaps and
#   confidence scatter. High-entropy regions in the t-SNE map indicate
#   where the model is uncertain. Panel (c) shows whether SVM and MLP
#   disagree on the same samples. Panel (d) reveals calibration.
# ============================================================================

def fig4_uncertainty_analysis(emb_te, y_te, prob_svm, prob_mlp,
                                pred_svm, pred_mlp, output_dir: Path):
    """4-panel uncertainty comparison: entropy maps + scatter + confidence.

    Uses pre-computed t-SNE coordinates for consistency with Fig 1/2.
    """
    setup_style()

    ent_svm = scipy_entropy(prob_svm, axis=1)
    ent_mlp = scipy_entropy(prob_mlp, axis=1)
    max_ent = np.log(2)  # binary

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))

    # ---- (a) SVM entropy heatmap ----
    ax = axes[0, 0]
    sc = ax.scatter(
        emb_te[:, 0], emb_te[:, 1], c=ent_svm, cmap="YlOrRd",
        s=14, alpha=0.8, vmin=0, vmax=max_ent, edgecolors="white",
        linewidths=0.15,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(a) SVM \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # ---- (b) MLP entropy heatmap ----
    ax = axes[0, 1]
    sc = ax.scatter(
        emb_te[:, 0], emb_te[:, 1], c=ent_mlp, cmap="YlOrRd",
        s=14, alpha=0.8, vmin=0, vmax=max_ent, edgecolors="white",
        linewidths=0.15,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Entropy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("(b) MLP \u2014 Prediction Entropy", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # ---- (c) SVM vs MLP entropy scatter ----
    ax = axes[1, 0]
    correct_both = (y_te == pred_svm) & (y_te == pred_mlp)
    wrong_both = (y_te != pred_svm) & (y_te != pred_mlp)
    svm_wrong = (y_te != pred_svm) & (y_te == pred_mlp)
    mlp_wrong = (y_te == pred_svm) & (y_te != pred_mlp)

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
    colors = np.where(y_te == pred_svm, ACCENT_GREEN, ACCENT_RED)
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
        "(Test set, L2, M+C+T1, 120+1+120)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(hspace=0.28, wspace=0.28)
    save_fig(fig, output_dir, "fig4_uncertainty_analysis")


# ============================================================================
# Figure 5: Layer Ablation (2x3 grid)
# Scenario: Which transformer layer produces the best class separation?
#   Early layers (L0-L1) capture low-level temporal patterns; deeper
#   layers (L3-L5) form higher-level abstractions. Phase 1 sweep found
#   layers are nearly interchangeable for occupancy (±0.01 AUC), but
#   L0 and L2 performed marginally best.
# ============================================================================

def fig5_layer_ablation(pretrained, output_token, device,
                        sensor_arr, test_labels, timestamps, ch_names,
                        ctx_before, ctx_after, seed, output_dir: Path):
    """2x3 grid: t-SNE from each transformer layer L0-L5 (test set)."""
    setup_style()
    layers = [0, 1, 2, 3, 4, 5]
    default_layer = 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, layer in enumerate(layers):
        ax = axes[i // 3, i % 3]
        logger.info("  Fig5 L%d: loading model + extracting...", layer)
        model = load_mantis(pretrained, layer=layer,
                            output_token=output_token, device=device)
        Z_te, y_te = extract_embeddings(
            model, sensor_arr, test_labels, timestamps, ch_names,
            ctx_before=ctx_before, ctx_after=ctx_after)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(Z_te) == 0:
            ax.set_title(f"L{layer}\n(no data)", fontsize=9)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#999999")
            continue

        Z_sub, y_sub, _ = subsample_balanced(Z_te, y_te, max_per_class=400,
                                              seed=seed)
        emb = tsne_2d(Z_sub, seed=seed)
        for cls in sorted(CLASS_COLORS):
            m = y_sub == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           s=14, alpha=0.55, edgecolors="white",
                           linewidths=0.15)

        tag = " *" if layer == default_layer else ""
        ax.set_title(f"L{layer}{tag}  (N={len(y_sub)})", fontsize=10,
                     fontweight="bold")
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
        f"(MantisV2, M+C+T1, {ctx_before}+1+{ctx_after} bidirectional, "
        "test set, * = default)",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    save_fig(fig, output_dir, "fig5_layer_ablation")


# ============================================================================
# Figure 6: Context Window Ablation (2x3 grid)
# Scenario: How does the size of the temporal context window affect class
#   separation? Occupancy states persist over long periods, so larger
#   context windows (120+1+120 = 241 min) were found to be optimal.
#   This is the opposite of Enter/Leave where short windows dominate.
# ============================================================================

def fig6_context_ablation(model, sensor_arr, test_labels, timestamps,
                           ch_names, seed, output_dir: Path):
    """2x3 grid: t-SNE with varying context windows (test set)."""
    setup_style()
    contexts = [
        (5, 5, "5+1+5 (11 min)"),
        (15, 15, "15+1+15 (31 min)"),
        (30, 30, "30+1+30 (61 min)"),
        (60, 60, "60+1+60 (121 min)"),
        (120, 120, "120+1+120 (241 min) *"),
        (180, 180, "180+1+180 (361 min)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, (cb, ca, label) in enumerate(contexts):
        ax = axes[i // 3, i % 3]
        logger.info("  Fig6 context %s: extracting...", label)
        Z_te, y_te = extract_embeddings(
            model, sensor_arr, test_labels, timestamps, ch_names,
            ctx_before=cb, ctx_after=ca)

        if len(Z_te) == 0:
            ax.set_title(f"{label}\n(no data)", fontsize=9)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#999999")
            continue

        Z_sub, y_sub, _ = subsample_balanced(Z_te, y_te, max_per_class=400,
                                              seed=seed)
        emb = tsne_2d(Z_sub, seed=seed)

        for cls in sorted(CLASS_COLORS):
            m = y_sub == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           s=14, alpha=0.55, edgecolors="white",
                           linewidths=0.15)

        ax.set_title(f"{label}  (N={len(y_sub)})", fontsize=9,
                     fontweight="bold")
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
        "(MantisV2 L2, M+C+T1, test set, * = default)",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    save_fig(fig, output_dir, "fig6_context_ablation")


# ============================================================================
# Figure 7: Channel Ablation (2x3 grid)
# Scenario: Which sensor channels contribute most to class separation?
#   M (motion) is the strongest single channel for occupancy. C (contact)
#   adds entry/exit information. T1 (temperature) adds thermal signature.
#   Phase 1 found M+C+T1 optimal (AUC=0.8937 with L0).
# ============================================================================

def fig7_channel_ablation(raw_cfg, pretrained, output_token, device,
                           split_date, ctx_before, ctx_after, default_layer,
                           seed, output_dir: Path):
    """2x3 grid: t-SNE with different channel combinations (test set)."""
    setup_style()
    combos = [
        ("M only", ["d620900d_motionSensor"]),
        ("C only", ["408981c2_contactSensor"]),
        ("T1 only", ["d620900d_temperatureMeasurement"]),
        ("M+C", ["d620900d_motionSensor", "408981c2_contactSensor"]),
        ("M+T1", ["d620900d_motionSensor",
                   "d620900d_temperatureMeasurement"]),
        ("M+C+T1 *", ["d620900d_motionSensor", "408981c2_contactSensor",
                       "d620900d_temperatureMeasurement"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, (name, channels) in enumerate(combos):
        ax = axes[i // 3, i % 3]
        logger.info("  Fig7 %s: loading data + extracting...", name)

        try:
            result = load_data(raw_cfg, split_date=split_date,
                               channels=channels)
            sensor_arr, _, test_labels, ch_names, timestamps = result

            model = load_mantis(pretrained, layer=default_layer,
                                output_token=output_token, device=device)
            Z_te, y_te = extract_embeddings(
                model, sensor_arr, test_labels, timestamps, ch_names,
                ctx_before=ctx_before, ctx_after=ctx_after)
            del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(Z_te) == 0:
                raise ValueError("Empty embeddings")

            Z_sub, y_sub, _ = subsample_balanced(Z_te, y_te,
                                                  max_per_class=400, seed=seed)
            emb = tsne_2d(Z_sub, seed=seed)
            n_ch = len(ch_names)
            embed_d = Z_te.shape[1]

            for cls in sorted(CLASS_COLORS):
                m = y_sub == cls
                if m.any():
                    ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                               s=14, alpha=0.55, edgecolors="white",
                               linewidths=0.15)
            ax.set_title(f"{name}\n({n_ch}ch, {embed_d}-d, N={len(y_sub)})",
                         fontsize=9, fontweight="bold")
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
        f"(MantisV2 L{default_layer}, {ctx_before}+1+{ctx_after} "
        "bidirectional, test set, * = default)",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    save_fig(fig, output_dir, "fig7_channel_ablation")


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
    output_token = "combined"
    split_date = raw_cfg.get("split_date", "2026-02-15")
    ctx_before = raw_cfg.get("default_context_before", 120)
    ctx_after = raw_cfg.get("default_context_after", 120)
    default_layer = raw_cfg.get("default_layer", 2)

    # Optimal channels: M+C+T1
    CHANNELS_MCT1 = [
        "d620900d_motionSensor",
        "408981c2_contactSensor",
        "d620900d_temperatureMeasurement",
    ]

    t0 = time.time()

    # ==== Load MantisV2 (representative model) ====
    logger.info("=" * 60)
    logger.info("Loading MantisV2 L%d (representative model)...",
                 default_layer)
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

    # ==== Classifiers (for Fig 2-4) ====
    y_pred_svm = y_prob_svm = y_pred_mlp = y_prob_mlp = None
    if {2, 3, 4} & figs_to_run:
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

    # ==== Pre-compute t-SNE ONCE (consistency across Fig 1/2/4) ====
    max_viz = 600
    emb_tr_sub = emb_te_sub = None
    y_tr_sub = y_te_sub = None
    te_idx = None

    if {1, 2, 4} & figs_to_run:
        logger.info("=" * 60)
        logger.info("Pre-computing joint t-SNE (train+test, subsampled, "
                     "reused by Fig 1/2/4)...")
        Z_tr_sub, y_tr_sub, _ = subsample_balanced(
            Z_train, y_train, max_per_class=max_viz, seed=args.seed)
        Z_te_sub, y_te_sub, te_idx = subsample_balanced(
            Z_test, y_test, max_per_class=max_viz, seed=args.seed)
        [emb_tr_sub, emb_te_sub] = tsne_2d_joint(
            Z_tr_sub, Z_te_sub, seed=args.seed)
        logger.info("Joint t-SNE: train=%s, test=%s",
                     emb_tr_sub.shape, emb_te_sub.shape)

    # ==== Fig 1: Train/Test embedding space ====
    if 1 in figs_to_run:
        logger.info("=" * 60)
        logger.info("Fig 1: Train/Test embedding space...")
        fig1_train_test_embeddings(emb_tr_sub, y_tr_sub,
                                     emb_te_sub, y_te_sub, output_dir)

    # ==== Fig 2: Classification overlay ====
    if 2 in figs_to_run:
        logger.info("Fig 2: Classification overlay (SVM vs MLP)...")
        fig2_classification_overlay(
            emb_te_sub, y_te_sub,
            y_pred_svm[te_idx], y_pred_mlp[te_idx], output_dir)

    # ==== Fig 3: Decision boundary ====
    if 3 in figs_to_run:
        logger.info("Fig 3: Decision boundary (PCA 2D)...")
        fig3_decision_boundary(Z_train, y_train, Z_test, y_test,
                                 y_pred_svm, y_pred_mlp, output_dir)

    # ==== Fig 4: Uncertainty analysis ====
    if 4 in figs_to_run:
        logger.info("Fig 4: Uncertainty analysis...")
        fig4_uncertainty_analysis(
            emb_te_sub, y_te_sub,
            y_prob_svm[te_idx], y_prob_mlp[te_idx],
            y_pred_svm[te_idx], y_pred_mlp[te_idx], output_dir)

    # ==== Fig 5: Layer ablation ====
    if 5 in figs_to_run:
        logger.info("=" * 60)
        logger.info("Fig 5: Layer ablation (L0-L5)...")
        fig5_layer_ablation(pretrained, output_token, device,
                            sensor_arr, test_labels, timestamps, ch_names,
                            ctx_before, ctx_after, args.seed, output_dir)

    # ==== Fig 6: Context window ablation ====
    if 6 in figs_to_run:
        logger.info("=" * 60)
        logger.info("Fig 6: Context window ablation...")
        fig6_context_ablation(model, sensor_arr, test_labels, timestamps,
                               ch_names, args.seed, output_dir)

    # ==== Fig 7: Channel ablation ====
    if 7 in figs_to_run:
        logger.info("=" * 60)
        logger.info("Fig 7: Channel ablation...")
        fig7_channel_ablation(raw_cfg, pretrained, output_token, device,
                               split_date, ctx_before, ctx_after,
                               default_layer, args.seed, output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All figures saved to: %s", output_dir.resolve())
    logger.info("Total time: %.1fs", elapsed)
    if y_pred_svm is not None:
        logger.info(
            "Summary -- SVM: %.2f%% (%d errors) | MLP: %.2f%% (%d errors)",
            100 * (y_test == y_pred_svm).mean(),
            (y_test != y_pred_svm).sum(),
            100 * (y_test == y_pred_mlp).mean(),
            (y_test != y_pred_mlp).sum(),
        )


if __name__ == "__main__":
    main()
