#!/usr/bin/env python3
"""Comprehensive Visualization Analysis for Occupancy Detection.

Generates publication-quality figures analyzing MantisV2 embeddings for the
binary occupancy (Empty/Occupied) classification task. Uses train/test split
(date-based at 2026-02-15) to evaluate sklearn (SVM_rbf) vs neural (MLP) heads.

Figures produced (saved as PNG + PDF):
  Fig 1 — Train/Test embedding space: t-SNE + UMAP with binary class coloring
  Fig 2 — Classification result overlay: SVM vs MLP correct/incorrect markers
  Fig 3 — Decision boundary approximation: SVM vs MLP on 2D projections
  Fig 4 — Uncertainty analysis: per-sample entropy & confidence comparison
  Fig 5 — Layer progression: L0-L5 embedding quality (t-SNE + accuracy)
  Fig 6 — Context window effect: class separation across context sizes
  Fig 7 — Channel ablation: M+C vs M+C+T1 embedding comparison

Usage:
    cd examples/classification/apc_occupancy
    python analysis/run_visualization_analysis.py \\
        --config training/configs/occupancy-phase1.yaml \\
        --device cuda --output-dir results/visualization_analysis

    # Quick mode (skip slow layer sweep & context sweep)
    python analysis/run_visualization_analysis.py \\
        --config training/configs/occupancy-phase1.yaml \\
        --device cuda --quick

Dependencies:
    pip install umap-learn  # Optional, falls back to t-SNE if missing
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
CLASS_COLORS = {0: "#009E73", 1: "#CC3311"}
CLASS_NAMES = {0: "Empty", 1: "Occupied"}
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
    """Reduce high-dim embeddings to 2D. Pipeline: Scale -> PCA -> method."""
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


def load_data(
    raw_cfg: dict,
    split_date: str = "2026-02-15",
    channels: list[str] | None = None,
):
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
    result = load_occupancy_data(cfg, split_date=split_date)
    # Returns: sensor_array, train_labels, test_labels, channel_names, timestamps
    return result


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
    model,
    sensor_array: np.ndarray,
    labels: np.ndarray,
    timestamps,
    channel_names: list[str],
    ctx_before: int = 125,
    ctx_after: int = 125,
    stride: int = 1,
):
    """Build dataset and extract train/test embeddings.

    Returns (Z_train, y_train, Z_test, y_test).
    """
    import pandas as pd

    ds_cfg = DatasetConfig(
        context_mode="bidirectional",
        context_before=ctx_before,
        context_after=ctx_after,
        stride=stride,
    )

    # Create dataset from the full array using train_labels (labeled portion = train)
    # We need to create separate train/test datasets
    dataset = OccupancyDataset(sensor_array, labels, timestamps, ds_cfg)

    X, y = dataset.get_numpy_arrays()
    if len(X) == 0:
        return np.array([]), np.array([]), None, None

    Z = model.transform(X)
    nan_count = int(np.isnan(Z).sum())
    if nan_count > 0:
        Z = np.nan_to_num(Z, nan=0.0)

    return Z, y


def subsample_balanced(Z, y, max_per_class=500, seed=42):
    """Subsample for visualization if dataset is too large."""
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
# Classification helpers (SVM + MLP)
# ============================================================================

def run_svm_classifier(Z_train, y_train, Z_test, y_test, seed=42):
    """Train SVM_rbf on train, predict on test. Returns (y_pred, y_prob)."""
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(Z_train)
    Zte = scaler.transform(Z_test)

    clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    clf.fit(Ztr, y_train)

    y_pred = clf.predict(Zte)
    y_prob = clf.predict_proba(Zte)  # (n_test, 2)

    return y_pred, y_prob, clf, scaler


def run_mlp_classifier(Z_train, y_train, Z_test, y_test, embed_dim,
                        hidden_dims=None, device="cpu", seed=42):
    """Train MLP head on train, predict on test. Returns (y_pred, y_prob)."""
    from training.heads import MLPHead

    if hidden_dims is None:
        hidden_dims = [128]

    scaler = StandardScaler()
    Ztr_np = scaler.fit_transform(Z_train)
    Zte_np = scaler.transform(Z_test)

    n_cls = len(np.unique(np.concatenate([y_train, y_test])))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    head = MLPHead(embed_dim, n_cls, hidden_dims=hidden_dims, dropout=0.5, use_batchnorm=True)
    head = head.to(dev)

    Ztr = torch.from_numpy(Ztr_np).float().to(dev)
    ytr = torch.from_numpy(y_train).long().to(dev)
    Zte = torch.from_numpy(Zte_np).float().to(dev)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    loss_fn = torch.nn.CrossEntropyLoss()

    head.train()
    best_loss, patience_count = float("inf"), 0
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
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= 30 or lv < 0.01:
            break

    head.eval()
    with torch.no_grad():
        logits = head(Zte)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)

    return y_pred, probs, head, scaler


# ============================================================================
# Figure generators
# ============================================================================

def fig1_train_test_embeddings(Z_train, y_train, Z_test, y_test, output_dir: Path):
    """Fig 1: Train/Test embedding space via t-SNE and UMAP with class coloring."""
    setup_style()
    methods = ["tsne", "umap"] if _UMAP_AVAILABLE else ["tsne", "pca"]

    # Subsample for visualization (occupancy has thousands of samples)
    max_viz = 800
    Ztr_viz, ytr_viz, _ = subsample_balanced(Z_train, y_train, max_per_class=max_viz)
    Zte_viz, yte_viz, _ = subsample_balanced(Z_test, y_test, max_per_class=max_viz)

    fig, axes = plt.subplots(2, len(methods), figsize=(6 * len(methods), 10))

    for col, method in enumerate(methods):
        method_label = "t-SNE" if method == "tsne" else method.upper()

        # Joint reduction for shared coordinate space
        [emb_tr, emb_te] = reduce_2d_joint(
            Ztr_viz, Zte_viz, method=method,
        )

        # Row 0: Train set
        ax = axes[0, col]
        for cls in sorted(CLASS_COLORS):
            m = ytr_viz == cls
            if m.any():
                ax.scatter(emb_tr[m, 0], emb_tr[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=15, alpha=0.5, edgecolors="none")
        n_emp = (ytr_viz == 0).sum()
        n_occ = (ytr_viz == 1).sum()
        ax.set_title(f"Train Set — {method_label}\n(Empty={n_emp}, Occupied={n_occ})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=2)

        # Row 1: Test set
        ax = axes[1, col]
        for cls in sorted(CLASS_COLORS):
            m = yte_viz == cls
            if m.any():
                ax.scatter(emb_te[m, 0], emb_te[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=15, alpha=0.5, edgecolors="none")
        n_emp = (yte_viz == 0).sum()
        n_occ = (yte_viz == 1).sum()
        ax.set_title(f"Test Set — {method_label}\n(Empty={n_emp}, Occupied={n_occ})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=2)

    fig.suptitle(
        "Occupancy Embedding Space: Train vs Test\n"
        "(MantisV2 L2, M+C+T1, 125+1+125, split at 2026-02-15)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    save_fig(fig, output_dir, "fig1_train_test_embeddings")


def fig2_classification_overlay(
    Z_test, y_test, y_pred_svm, y_pred_mlp, output_dir: Path,
):
    """Fig 2: SVM vs MLP classification results on test set (correct/incorrect overlay)."""
    setup_style()
    methods = ["tsne", "umap"] if _UMAP_AVAILABLE else ["tsne", "pca"]

    # Subsample for clear visualization
    Z_viz, y_viz, viz_idx = subsample_balanced(Z_test, y_test, max_per_class=600)
    pred_svm_viz = y_pred_svm[viz_idx]
    pred_mlp_viz = y_pred_mlp[viz_idx]

    fig, axes = plt.subplots(2, len(methods), figsize=(6 * len(methods), 10))

    for col, method in enumerate(methods):
        method_label = "t-SNE" if method == "tsne" else method.upper()
        emb = reduce_2d(Z_viz, method=method)

        for row, (name, y_pred) in enumerate([
            ("SVM_rbf (sklearn)", pred_svm_viz),
            ("MLP[128]-d0.5 (neural)", pred_mlp_viz),
        ]):
            ax = axes[row, col]
            correct = y_viz == y_pred
            incorrect = ~correct

            # Correct samples
            for cls in sorted(CLASS_COLORS):
                m = (y_viz == cls) & correct
                if m.any():
                    ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                               label=f"{CLASS_NAMES[cls]} (correct)",
                               s=12, alpha=0.4, edgecolors="none")

            # Incorrect samples with prominent markers
            if incorrect.any():
                ax.scatter(emb[incorrect, 0], emb[incorrect, 1],
                           c=[CLASS_COLORS[int(c)] for c in y_viz[incorrect]],
                           s=60, alpha=1.0, edgecolors="red", linewidths=1.5,
                           marker="X", zorder=10, label="Misclassified")

            n_err = incorrect.sum()
            n_total = len(y_viz)
            acc = 100 * correct.mean()
            ax.set_title(f"{name} — {method_label}\nAcc={acc:.2f}% ({n_err}/{n_total} errors)",
                         fontsize=10)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(fontsize=7, loc="best", markerscale=0.8)

    fig.suptitle(
        "Test Set Classification: SVM_rbf (sklearn) vs MLP[128]-d0.5 (neural)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    save_fig(fig, output_dir, "fig2_classification_overlay")


def fig3_decision_boundary(
    Z_train, y_train, Z_test, y_test,
    y_pred_svm, y_pred_mlp,
    output_dir: Path,
):
    """Fig 3: Decision boundary approximation on 2D projections."""
    setup_style()

    # Use PCA for stable decision boundary visualization
    # (t-SNE doesn't preserve global structure for boundary estimation)
    scaler = StandardScaler()
    Z_all = np.concatenate([Z_train, Z_test])
    Z_scaled = scaler.fit_transform(Z_all)
    pca = PCA(n_components=2, random_state=42)
    Z_2d = pca.fit_transform(Z_scaled)

    n_train = len(Z_train)
    Z_2d_train = Z_2d[:n_train]
    Z_2d_test = Z_2d[n_train:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): Training data distribution
    ax = axes[0]
    for cls in sorted(CLASS_COLORS):
        m = y_train == cls
        if m.any():
            ax.scatter(Z_2d_train[m, 0], Z_2d_train[m, 1], c=CLASS_COLORS[cls],
                       label=CLASS_NAMES[cls], s=8, alpha=0.3, edgecolors="none")
    ax.set_title("Train Set (PCA 2D)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(markerscale=2)

    # Panel (b): SVM decision region
    ax = axes[1]
    _plot_decision_region(ax, Z_2d_train, y_train, Z_2d_test, y_test, y_pred_svm,
                          "SVM_rbf", pca)

    # Panel (c): MLP decision region (approximated with SVM on 2D)
    ax = axes[2]
    _plot_decision_region(ax, Z_2d_train, y_train, Z_2d_test, y_test, y_pred_mlp,
                          "MLP[128]-d0.5", pca)

    fig.suptitle(
        "Decision Boundary Analysis (PCA 2D Projection, L2, M+C+T1, 125+1+125)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    save_fig(fig, output_dir, "fig3_decision_boundary")


def _plot_decision_region(ax, Z_2d_train, y_train, Z_2d_test, y_test, y_pred, name, pca):
    """Plot decision region contour + test scatter."""
    # Fit an SVM on 2D for visualization of predicted boundary
    from sklearn.svm import SVC as SVC2D
    clf_2d = SVC2D(kernel="rbf", C=1.0, gamma="scale")
    clf_2d.fit(Z_2d_train, y_train)

    # Create mesh
    x_min, x_max = Z_2d_test[:, 0].min() - 1, Z_2d_test[:, 0].max() + 1
    y_min, y_max = Z_2d_test[:, 1].min() - 1, Z_2d_test[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf_2d.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.15, levels=[-0.5, 0.5, 1.5],
                colors=[CLASS_COLORS[0], CLASS_COLORS[1]])
    ax.contour(xx, yy, zz, levels=[0.5], colors=["black"], linewidths=1.5, alpha=0.5)

    # Test points colored by correctness
    correct = y_test == y_pred
    incorrect = ~correct
    for cls in sorted(CLASS_COLORS):
        m = (y_test == cls) & correct
        if m.any():
            ax.scatter(Z_2d_test[m, 0], Z_2d_test[m, 1], c=CLASS_COLORS[cls],
                       s=10, alpha=0.4, edgecolors="none", label=f"{CLASS_NAMES[cls]}")
    if incorrect.any():
        ax.scatter(Z_2d_test[incorrect, 0], Z_2d_test[incorrect, 1],
                   c="red", s=30, marker="X", alpha=0.8, zorder=10, label="Error")

    acc = 100 * correct.mean()
    ax.set_title(f"{name}\nAcc={acc:.2f}%", fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=7, markerscale=1.2)


def fig4_uncertainty_analysis(
    Z_test, y_test, y_prob_svm, y_prob_mlp, y_pred_svm, y_pred_mlp,
    output_dir: Path,
):
    """Fig 4: Per-sample uncertainty (entropy) comparison between SVM and MLP."""
    setup_style()

    # Subsample for clarity
    Z_viz, y_viz, viz_idx = subsample_balanced(Z_test, y_test, max_per_class=600)
    prob_svm_viz = y_prob_svm[viz_idx]
    prob_mlp_viz = y_prob_mlp[viz_idx]
    pred_svm_viz = y_pred_svm[viz_idx]
    pred_mlp_viz = y_pred_mlp[viz_idx]

    # Compute entropy per sample
    ent_svm = scipy_entropy(prob_svm_viz, axis=1)
    ent_mlp = scipy_entropy(prob_mlp_viz, axis=1)
    max_ent = np.log(2)  # binary classification max entropy

    emb = reduce_2d(Z_viz, method="tsne")

    fig = plt.figure(figsize=(18, 10))

    # Panel (a): t-SNE colored by SVM entropy
    ax1 = fig.add_subplot(2, 3, 1)
    sc1 = ax1.scatter(emb[:, 0], emb[:, 1], c=ent_svm, cmap="YlOrRd",
                       s=15, alpha=0.7, vmin=0, vmax=max_ent)
    plt.colorbar(sc1, ax=ax1, label="Entropy", shrink=0.8)
    ax1.set_title("SVM — Entropy Map (t-SNE)")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    # Panel (b): t-SNE colored by MLP entropy
    ax2 = fig.add_subplot(2, 3, 2)
    sc2 = ax2.scatter(emb[:, 0], emb[:, 1], c=ent_mlp, cmap="YlOrRd",
                       s=15, alpha=0.7, vmin=0, vmax=max_ent)
    plt.colorbar(sc2, ax=ax2, label="Entropy", shrink=0.8)
    ax2.set_title("MLP — Entropy Map (t-SNE)")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")

    # Panel (c): SVM entropy vs MLP entropy scatter
    ax3 = fig.add_subplot(2, 3, 3)
    correct_both = (y_viz == pred_svm_viz) & (y_viz == pred_mlp_viz)
    wrong_both = (y_viz != pred_svm_viz) & (y_viz != pred_mlp_viz)
    svm_only_wrong = (y_viz != pred_svm_viz) & (y_viz == pred_mlp_viz)
    mlp_only_wrong = (y_viz == pred_svm_viz) & (y_viz != pred_mlp_viz)

    if correct_both.any():
        ax3.scatter(ent_svm[correct_both], ent_mlp[correct_both],
                    c="#009E73", s=10, alpha=0.3, label="Both correct")
    if wrong_both.any():
        ax3.scatter(ent_svm[wrong_both], ent_mlp[wrong_both],
                    c="#CC3311", s=50, alpha=1.0, marker="X", label="Both wrong")
    if svm_only_wrong.any():
        ax3.scatter(ent_svm[svm_only_wrong], ent_mlp[svm_only_wrong],
                    c=ACCENT_BLUE, s=40, alpha=0.9, marker="s", label="SVM wrong only")
    if mlp_only_wrong.any():
        ax3.scatter(ent_svm[mlp_only_wrong], ent_mlp[mlp_only_wrong],
                    c=ACCENT_ORANGE, s=40, alpha=0.9, marker="^", label="MLP wrong only")
    ax3.plot([0, max_ent], [0, max_ent], "k--", alpha=0.3, label="y=x")
    ax3.set_xlabel("SVM Entropy")
    ax3.set_ylabel("MLP Entropy")
    ax3.set_title("SVM vs MLP Uncertainty")
    ax3.legend(fontsize=7)

    # Panel (d): Entropy histogram comparison
    ax4 = fig.add_subplot(2, 3, 4)
    bins = np.linspace(0, max_ent, 30)
    ax4.hist(ent_svm, bins=bins, alpha=0.5, color=ACCENT_BLUE, label="SVM", density=True)
    ax4.hist(ent_mlp, bins=bins, alpha=0.5, color=ACCENT_ORANGE, label="MLP", density=True)
    ax4.axvline(np.median(ent_svm), color=ACCENT_BLUE, linestyle="--", linewidth=1.5)
    ax4.axvline(np.median(ent_mlp), color=ACCENT_ORANGE, linestyle="--", linewidth=1.5)
    ax4.set_xlabel("Entropy")
    ax4.set_ylabel("Density")
    ax4.set_title("Entropy Distribution")
    ax4.legend()

    # Panel (e): Per-class entropy boxplot
    ax5 = fig.add_subplot(2, 3, 5)
    data_svm = [ent_svm[y_viz == c] for c in sorted(CLASS_COLORS) if (y_viz == c).any()]
    data_mlp = [ent_mlp[y_viz == c] for c in sorted(CLASS_COLORS) if (y_viz == c).any()]
    labels_box = [CLASS_NAMES[c] for c in sorted(CLASS_COLORS) if (y_viz == c).any()]
    n_cls = len(labels_box)
    pos_svm = np.arange(n_cls) - 0.18
    pos_mlp = np.arange(n_cls) + 0.18
    bp1 = ax5.boxplot(data_svm, positions=pos_svm, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor=ACCENT_BLUE, alpha=0.4))
    bp2 = ax5.boxplot(data_mlp, positions=pos_mlp, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor=ACCENT_ORANGE, alpha=0.4))
    ax5.set_xticks(np.arange(n_cls))
    ax5.set_xticklabels(labels_box)
    ax5.set_ylabel("Entropy")
    ax5.set_title("Per-Class Uncertainty")
    ax5.legend([bp1["boxes"][0], bp2["boxes"][0]], ["SVM", "MLP"], fontsize=7)

    # Panel (f): Confidence (max prob) comparison
    ax6 = fig.add_subplot(2, 3, 6)
    conf_svm = prob_svm_viz.max(axis=1)
    conf_mlp = prob_mlp_viz.max(axis=1)
    colors = ["#009E73" if y_viz[i] == pred_svm_viz[i] else "#CC3311"
              for i in range(len(y_viz))]
    ax6.scatter(conf_svm, conf_mlp, c=colors, s=10, alpha=0.4)
    ax6.plot([0.5, 1], [0.5, 1], "k--", alpha=0.3)
    ax6.set_xlabel("SVM Max Probability")
    ax6.set_ylabel("MLP Max Probability")
    ax6.set_title("Prediction Confidence")
    ax6.set_xlim(0.45, 1.02)
    ax6.set_ylim(0.45, 1.02)
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#009E73",
               markersize=6, label="Correct (SVM)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CC3311",
               markersize=6, label="Incorrect (SVM)"),
    ]
    ax6.legend(handles=legend_elems, fontsize=7)

    fig.suptitle(
        "Uncertainty & Confidence Analysis: SVM_rbf vs MLP[128]-d0.5\n"
        "(Test set, L2, M+C+T1, 125+1+125)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    save_fig(fig, output_dir, "fig4_uncertainty_analysis")


def fig5_layer_progression(
    sensor_array, train_labels, test_labels, timestamps, channel_names,
    model_loader, ctx_before, ctx_after, device, output_dir: Path,
):
    """Fig 5: Embedding evolution across L0-L5 with SVM accuracy."""
    setup_style()
    layers = [0, 1, 2, 3, 4, 5]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    accuracies = {}

    for idx, layer in enumerate(layers):
        logger.info("  Layer L%d: extracting embeddings...", layer)
        model = model_loader(layer)

        Z_tr, y_tr = extract_embeddings(
            model, sensor_array, train_labels, timestamps, channel_names,
            ctx_before=ctx_before, ctx_after=ctx_after,
        )
        Z_te, y_te = extract_embeddings(
            model, sensor_array, test_labels, timestamps, channel_names,
            ctx_before=ctx_before, ctx_after=ctx_after,
        )

        if len(Z_tr) == 0 or len(Z_te) == 0:
            logger.warning("Empty embeddings for L%d, skipping", layer)
            continue

        # SVM classification for accuracy
        scaler = StandardScaler()
        Ztr_s = scaler.fit_transform(Z_tr)
        Zte_s = scaler.transform(Z_te)
        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        clf.fit(Ztr_s, y_tr)
        y_pred = clf.predict(Zte_s)
        acc = 100 * (y_te == y_pred).mean()
        accuracies[layer] = acc

        # Subsample test set for visualization
        Z_viz, y_viz, _ = subsample_balanced(Z_te, y_te, max_per_class=500)
        emb = reduce_2d(Z_viz, method="tsne")

        ax = axes_flat[idx]
        for cls in sorted(CLASS_COLORS):
            m = y_viz == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=12, alpha=0.5, edgecolors="none")
        ax.set_title(f"Layer {layer} — SVM Acc={acc:.1f}%", fontsize=10)
        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")
        if idx == 0:
            ax.legend(fontsize=7, markerscale=1.5)

        # Free memory
        del Z_tr, Z_te, y_tr, y_te
        gc.collect()

    fig.suptitle(
        "MantisV2 Layer Progression (t-SNE, M+C+T1, 125+1+125, SVM_rbf)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    save_fig(fig, output_dir, "fig5_layer_progression")

    # Save accuracy summary
    summary = output_dir / "fig5_layer_accuracies.txt"
    with open(summary, "w") as f:
        f.write("Layer vs SVM Accuracy (M+C+T1, 125+1+125, test set)\n")
        f.write("=" * 50 + "\n")
        for l, acc in sorted(accuracies.items()):
            f.write(f"  L{l}: {acc:.2f}%\n")
    logger.info("Layer summary saved: %s", summary)


def fig6_context_window_effect(
    sensor_array, train_labels, test_labels, timestamps, channel_names,
    model_loader, device, output_dir: Path,
):
    """Fig 6: Class separation across different context window sizes."""
    setup_style()
    contexts = [
        (10, 10, "10+1+10 (21min)"),
        (30, 30, "30+1+30 (61min)"),
        (60, 60, "60+1+60 (121min)"),
        (125, 125, "125+1+125 (251min)"),
        (180, 180, "180+1+180 (361min)"),
    ]
    layer = 2  # Optimal layer for sklearn

    fig, axes = plt.subplots(1, len(contexts), figsize=(4.5 * len(contexts), 4.5))
    accuracies = []

    for col, (cb, ca, label) in enumerate(contexts):
        logger.info("  Context %s: extracting embeddings...", label)
        model = model_loader(layer)

        Z_tr, y_tr = extract_embeddings(
            model, sensor_array, train_labels, timestamps, channel_names,
            ctx_before=cb, ctx_after=ca,
        )
        Z_te, y_te = extract_embeddings(
            model, sensor_array, test_labels, timestamps, channel_names,
            ctx_before=cb, ctx_after=ca,
        )

        if len(Z_tr) == 0 or len(Z_te) == 0:
            logger.warning("Empty embeddings for context %s, skipping", label)
            accuracies.append(0.0)
            continue

        # SVM classification for accuracy
        scaler = StandardScaler()
        Ztr_s = scaler.fit_transform(Z_tr)
        Zte_s = scaler.transform(Z_te)
        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        clf.fit(Ztr_s, y_tr)
        y_pred = clf.predict(Zte_s)
        acc = 100 * (y_te == y_pred).mean()
        accuracies.append(acc)

        # Subsample for visualization
        Z_viz, y_viz, _ = subsample_balanced(Z_te, y_te, max_per_class=400)
        emb = reduce_2d(Z_viz, method="tsne")

        ax = axes[col]
        for cls in sorted(CLASS_COLORS):
            m = y_viz == cls
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                           label=CLASS_NAMES[cls], s=12, alpha=0.5, edgecolors="none")
        ax.set_title(f"{label}\nAcc={acc:.1f}%", fontsize=10)
        ax.set_xlabel("Comp 1")
        if col == 0:
            ax.set_ylabel("Comp 2")
        ax.legend(fontsize=6, markerscale=1.5)

        del Z_tr, Z_te
        gc.collect()

    fig.suptitle(
        "Context Window Effect on Occupancy Detection (L2, M+C+T1, SVM_rbf)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    save_fig(fig, output_dir, "fig6_context_window_effect")

    # Save accuracy summary
    summary = output_dir / "fig6_context_accuracies.txt"
    with open(summary, "w") as f:
        f.write("Context Window vs SVM Accuracy (L2, M+C+T1, test set)\n")
        f.write("=" * 50 + "\n")
        for (cb, ca, label), acc in zip(contexts, accuracies):
            f.write(f"  {label:25s}  Acc = {acc:.2f}%\n")
    logger.info("Context summary saved: %s", summary)


def fig7_channel_ablation(
    raw_cfg, sensor_array_mc, train_labels_mc, test_labels_mc,
    timestamps_mc, ch_names_mc,
    sensor_array_mct, train_labels_mct, test_labels_mct,
    timestamps_mct, ch_names_mct,
    model_loader, ctx_before, ctx_after, output_dir: Path,
):
    """Fig 7: M+C vs M+C+T1 embedding comparison."""
    setup_style()
    methods = ["tsne", "umap"] if _UMAP_AVAILABLE else ["tsne", "pca"]
    layer = 2

    fig, axes = plt.subplots(2, len(methods), figsize=(6 * len(methods), 10))

    model = model_loader(layer)

    configs = [
        ("M+C (2ch, 1024-d)", sensor_array_mc, test_labels_mc,
         timestamps_mc, ch_names_mc),
        ("M+C+T1 (3ch, 1536-d)", sensor_array_mct, test_labels_mct,
         timestamps_mct, ch_names_mct),
    ]

    for row, (ch_label, s_arr, t_labels, ts, ch_names) in enumerate(configs):
        Z_te, y_te = extract_embeddings(
            model, s_arr, t_labels, ts, ch_names,
            ctx_before=ctx_before, ctx_after=ctx_after,
        )

        if len(Z_te) == 0:
            logger.warning("Empty embeddings for %s", ch_label)
            continue

        # SVM accuracy
        Z_tr, y_tr = extract_embeddings(
            model, s_arr,
            # Find the corresponding train_labels array
            train_labels_mc if "2ch" in ch_label else train_labels_mct,
            ts, ch_names,
            ctx_before=ctx_before, ctx_after=ctx_after,
        )
        scaler = StandardScaler()
        Ztr_s = scaler.fit_transform(Z_tr)
        Zte_s = scaler.transform(Z_te)
        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        clf.fit(Ztr_s, y_tr)
        y_pred = clf.predict(Zte_s)
        acc = 100 * (y_te == y_pred).mean()

        Z_viz, y_viz, _ = subsample_balanced(Z_te, y_te, max_per_class=500)

        for col, method in enumerate(methods):
            method_label = "t-SNE" if method == "tsne" else method.upper()
            emb = reduce_2d(Z_viz, method=method)

            ax = axes[row, col]
            for cls in sorted(CLASS_COLORS):
                m = y_viz == cls
                if m.any():
                    ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[cls],
                               label=CLASS_NAMES[cls], s=12, alpha=0.5, edgecolors="none")
            ax.set_title(f"{ch_label} — {method_label}\nSVM Acc={acc:.1f}%", fontsize=10)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(fontsize=7, markerscale=1.5)

    fig.suptitle(
        "Channel Ablation: M+C vs M+C+T1 Embedding Space (L2, 125+1+125)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    save_fig(fig, output_dir, "fig7_channel_ablation")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive visualization analysis for Occupancy detection",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g., occupancy-phase1.yaml)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/visualization_analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow experiments (layer sweep, context sweep)")
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
    output_token = "combined"
    split_date = raw_cfg.get("split_date", "2026-02-15")

    # Default context from config
    ctx_before = raw_cfg.get("default_context_before", 125)
    ctx_after = raw_cfg.get("default_context_after", 125)
    default_layer = raw_cfg.get("default_layer", 2)

    # Channel configurations
    CHANNELS_MC = [
        "d620900d_motionSensor",
        "408981c2_contactSensor",
    ]
    CHANNELS_MCT1 = [
        "d620900d_motionSensor",
        "408981c2_contactSensor",
        "d620900d_temperatureMeasurement",
    ]

    # Model loader factory (caches models per layer)
    _model_cache = {}
    def model_loader(layer):
        if layer not in _model_cache:
            _model_cache[layer] = load_mantis(pretrained, layer, output_token, device)
        return _model_cache[layer]

    t0 = time.time()

    # ==== Load data with M+C+T1 (optimal channels) ====
    logger.info("=" * 60)
    logger.info("Loading data with M+C+T1 channels...")
    result_mct = load_data(raw_cfg, split_date=split_date, channels=CHANNELS_MCT1)
    sensor_arr_mct, train_labels_mct, test_labels_mct, ch_names_mct, timestamps_mct = result_mct
    n_train = (train_labels_mct >= 0).sum()
    n_test = (test_labels_mct >= 0).sum()
    logger.info("M+C+T1: train=%d, test=%d, channels=%s", n_train, n_test, ch_names_mct)

    # ==== Load data with M+C (for channel comparison) ====
    logger.info("Loading data with M+C channels...")
    result_mc = load_data(raw_cfg, split_date=split_date, channels=CHANNELS_MC)
    sensor_arr_mc, train_labels_mc, test_labels_mc, ch_names_mc, timestamps_mc = result_mc
    logger.info("M+C: channels=%s", ch_names_mc)

    # ==== Extract embeddings at optimal layer (L2) with M+C+T1 ====
    logger.info("=" * 60)
    logger.info("Extracting L%d embeddings with M+C+T1 (125+1+125)...", default_layer)
    model = model_loader(default_layer)

    Z_train, y_train = extract_embeddings(
        model, sensor_arr_mct, train_labels_mct, timestamps_mct, ch_names_mct,
        ctx_before=ctx_before, ctx_after=ctx_after,
    )
    Z_test, y_test = extract_embeddings(
        model, sensor_arr_mct, test_labels_mct, timestamps_mct, ch_names_mct,
        ctx_before=ctx_before, ctx_after=ctx_after,
    )
    embed_dim = Z_train.shape[1]
    logger.info("Train embeddings: shape=%s", Z_train.shape)
    logger.info("Test embeddings: shape=%s", Z_test.shape)

    # ==== Run SVM classifier ====
    logger.info("=" * 60)
    logger.info("Training SVM_rbf on train set, predicting test set...")
    y_pred_svm, y_prob_svm, _, _ = run_svm_classifier(
        Z_train, y_train, Z_test, y_test, seed=args.seed,
    )
    acc_svm = 100 * (y_test == y_pred_svm).mean()
    logger.info("SVM Accuracy: %.2f%% (%d/%d)", acc_svm,
                (y_test == y_pred_svm).sum(), len(y_test))

    # ==== Run MLP classifier ====
    logger.info("Training MLP[128]-d0.5 on train set, predicting test set...")
    y_pred_mlp, y_prob_mlp, _, _ = run_mlp_classifier(
        Z_train, y_train, Z_test, y_test, embed_dim,
        hidden_dims=[128], device=device, seed=args.seed,
    )
    acc_mlp = 100 * (y_test == y_pred_mlp).mean()
    logger.info("MLP Accuracy: %.2f%% (%d/%d)", acc_mlp,
                (y_test == y_pred_mlp).sum(), len(y_test))

    # ==== Fig 1: Train/Test embedding space ====
    logger.info("=" * 60)
    logger.info("Generating Fig 1: Train/Test embedding space...")
    fig1_train_test_embeddings(Z_train, y_train, Z_test, y_test, output_dir)

    # ==== Fig 2: Classification overlay ====
    logger.info("Generating Fig 2: Classification overlay (SVM vs MLP)...")
    fig2_classification_overlay(Z_test, y_test, y_pred_svm, y_pred_mlp, output_dir)

    # ==== Fig 3: Decision boundary ====
    logger.info("Generating Fig 3: Decision boundary analysis...")
    fig3_decision_boundary(Z_train, y_train, Z_test, y_test,
                            y_pred_svm, y_pred_mlp, output_dir)

    # ==== Fig 4: Uncertainty analysis ====
    logger.info("Generating Fig 4: Uncertainty analysis...")
    fig4_uncertainty_analysis(Z_test, y_test, y_prob_svm, y_prob_mlp,
                              y_pred_svm, y_pred_mlp, output_dir)

    # ==== Fig 5: Layer progression (slow) ====
    if not args.quick:
        logger.info("=" * 60)
        logger.info("Generating Fig 5: Layer progression (slow — 6 layer extractions)...")
        fig5_layer_progression(
            sensor_arr_mct, train_labels_mct, test_labels_mct,
            timestamps_mct, ch_names_mct,
            model_loader, ctx_before, ctx_after, device, output_dir,
        )
    else:
        logger.info("Skipping Fig 5 (--quick mode)")

    # ==== Fig 6: Context window effect (slow) ====
    if not args.quick:
        logger.info("Generating Fig 6: Context window effect (slow — 5 context sizes)...")
        fig6_context_window_effect(
            sensor_arr_mct, train_labels_mct, test_labels_mct,
            timestamps_mct, ch_names_mct,
            model_loader, device, output_dir,
        )
    else:
        logger.info("Skipping Fig 6 (--quick mode)")

    # ==== Fig 7: Channel ablation ====
    if not args.quick:
        logger.info("Generating Fig 7: Channel ablation (M+C vs M+C+T1)...")
        fig7_channel_ablation(
            raw_cfg,
            sensor_arr_mc, train_labels_mc, test_labels_mc,
            timestamps_mc, ch_names_mc,
            sensor_arr_mct, train_labels_mct, test_labels_mct,
            timestamps_mct, ch_names_mct,
            model_loader, ctx_before, ctx_after, output_dir,
        )
    else:
        logger.info("Skipping Fig 7 (--quick mode)")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All figures saved to: %s", output_dir.resolve())
    logger.info("Total time: %.1fs", elapsed)
    logger.info(
        "Summary — SVM: %.2f%% (%d errors) | MLP: %.2f%% (%d errors)",
        acc_svm, (y_test != y_pred_svm).sum(),
        acc_mlp, (y_test != y_pred_mlp).sum(),
    )


if __name__ == "__main__":
    main()
