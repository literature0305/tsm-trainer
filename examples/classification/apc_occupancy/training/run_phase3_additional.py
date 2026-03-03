#!/usr/bin/env python3
"""Phase 3 Additional Experiments: MLP Training Fixes + Cross-Combination Grid.

Addresses critical findings from Phase 2 analysis (818 experiments):

  CRITICAL BUG 1 — Early stopping based on training loss (no validation set)
    → Selects the most overfit model. SVM has structural regularization (margin),
      MLP relies on this flawed early stopping.
    FIX: train_head_v2() holds out 15% for validation-based early stopping.

  CRITICAL BUG 2 — Z_train_std computed AFTER StandardScaler (all ≈ 1.0)
    → FroFA/AdaptiveNoise degenerate to uniform Gaussian noise.
      Per-dimension variance scaling is completely neutralized.
    FIX: run_neural_train_test_v2() computes dimension importance from
      raw embeddings, then passes normalized weights to augmentation.

  SIGNIFICANT — Dropout 0.5 too aggressive for small hidden dims ([64], [128])
    → Effective capacity < 32/64 neurons. SVM_rbf has infinite-dim kernel.
    FIX: Test dropout 0.1-0.3 systematically.

  SIGNIFICANT — CosineAnnealing T_max=200 but early stopping at ~130
    → LR never reaches its minimum for fine-grained optimization.
    FIX: Use CosineAnnealing with T_max adapted to actual training.

  KEY GAP — M+C+T1 × Layer × MLP at 251min NEVER TESTED
    → Best single factors (T1 channel, L3 layer, 251min context) were
      never combined. MLP at 251min + M+C+T1 could match SVM.

Global best (Phase 2): M+C+T1 | 251min | SVM_rbf | L2 → AUC=0.9859

Six experiment groups across 3 GPU servers:

  GPU 1: Group J + K  — Training fix verification + Layer×Classifier grid
    J: 24 experiments (fix ablation)
    K: 48 experiments (M+C+T1 × 6 layers × 8 classifiers at 251min)

  GPU 2: Group L + N  — Context tuning + SVM hyperparameter grid
    L: 72 experiments (12 contexts × 2 layers × 3 classifiers)
    N: 60 experiments (SVM C×gamma grid at 251min)

  GPU 3: Group M + O  — Improved MLP recipe + Final combinations
    M: 72 experiments (LR×dropout×hidden grid with all fixes)
    O: 36 experiments (multi-seed, multi-layer fusion, TTA)

  Total: ~312 experiments.

Usage:
  cd examples/classification/apc_occupancy

  # 3-GPU parallel execution (recommended)
  python training/run_phase3_additional.py \\
      --config training/configs/occupancy-phase2.yaml --group J K --device cuda

  python training/run_phase3_additional.py \\
      --config training/configs/occupancy-phase2.yaml --group L N --device cuda

  python training/run_phase3_additional.py \\
      --config training/configs/occupancy-phase2.yaml --group M O --device cuda

  # Single group
  python training/run_phase3_additional.py \\
      --config training/configs/occupancy-phase2.yaml --group J --device cuda

  # All groups sequential (single server, ~8-16h)
  python training/run_phase3_additional.py \\
      --config training/configs/occupancy-phase2.yaml --device cuda
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Headless matplotlib
import matplotlib as mpl
if not os.environ.get("DISPLAY"):
    mpl.use("Agg")
import matplotlib.pyplot as plt

# Local imports -- run from apc_occupancy/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_unified_split
from data.dataset import DatasetConfig, OccupancyDataset
from evaluation.metrics import compute_metrics, compute_wilson_ci, ClassificationMetrics
from training.heads import build_head
from training.augmentation import (
    apply_augmentation,
    apply_pretrain_augmentation,
)
from visualization.style import (
    setup_style,
    save_figure,
    configure_output,
)

# Reuse utilities from Phase 2 sweep
from training.run_phase2_sweep import (
    TrainConfig,
    load_mantis_model,
    extract_embeddings,
    run_sklearn_train_test,
    run_neural_train_test,
    run_neural_train_test_multi_layer,
    run_ensemble_train_test,
    run_tta_train_test,
    build_sklearn_classifier,
    _build_optimizer,
    _build_scheduler,
    _make_result_row,
    _format_metrics_log,
    _save_group_results,
    generate_summary,
    plot_group_bar,
    CHANNEL_MAP,
    ALL_LAYERS,
)

logger = logging.getLogger(__name__)


def _resolve_channels(keys: list[str]) -> list[str]:
    """Convert short keys (M, C, T1, ...) to full channel names."""
    return [CHANNEL_MAP[k] for k in keys]


# ============================================================================
# Fixed training functions (v2)
# ============================================================================

def train_head_v2(
    head: nn.Module,
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainConfig,
    n_classes: int,
    Z_train_std: torch.Tensor | None = None,
    val_split: float = 0.15,
) -> nn.Module:
    """Train a classification head with VALIDATION-BASED early stopping.

    Key fixes over train_head():
      1. Holds out val_split fraction for validation loss monitoring.
         Best model selected by lowest validation loss (not training loss).
      2. When val_split=0, falls back to training-loss-based ES (legacy mode).

    Parameters
    ----------
    head : nn.Module
        Classification head to train.
    Z_train : Tensor, shape (N, D)
        Training embeddings (already scaled).
    y_train : Tensor, shape (N,) or (N, C) for soft labels
        Training labels.
    config : TrainConfig
        Training hyperparameters.
    n_classes : int
        Number of classes.
    Z_train_std : Tensor, optional
        Per-dimension importance weights for augmentation.
    val_split : float
        Fraction of training data to hold out for validation (0.0-0.3).
    """
    device = torch.device(config.device)
    head = head.to(device)
    head.train()

    Z_train = Z_train.to(device)
    y_train = y_train.to(device)
    if Z_train_std is not None:
        Z_train_std = Z_train_std.to(device)

    n_total = len(Z_train)

    # Split train/val
    use_val = val_split > 0 and n_total > 20  # need enough samples
    if use_val:
        n_val = max(int(n_total * val_split), 2)
        perm = torch.randperm(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        Z_tr, y_tr = Z_train[train_idx], y_train[train_idx]
        Z_val, y_val = Z_train[val_idx], y_train[val_idx]
    else:
        Z_tr, y_tr = Z_train, y_train
        Z_val, y_val = None, None

    n_train = len(Z_tr)
    use_minibatch = config.batch_size > 0 and n_train > config.batch_size
    batch_size = config.batch_size if use_minibatch else n_train
    steps_per_epoch = math.ceil(n_train / batch_size) if use_minibatch else 1

    # Optimizer + scheduler
    optimizer = _build_optimizer(head.parameters(), config)
    scheduler = _build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
    is_per_step = config.scheduler == "onecycle"

    # Training loss function
    aug_cfg = config.augmentation or {}
    strategy = aug_cfg.get("strategy", "")
    use_soft = (
        strategy not in ("frofa", "adaptive_noise", "within_class_mixup", "")
        or aug_cfg.get("mixup_alpha", 0) > 0
    )

    if use_soft:
        def loss_fn(logits, targets):
            lp = torch.nn.functional.log_softmax(logits, dim=1)
            return -(targets * lp).sum(dim=1).mean()
    else:
        weight = None
        if config.class_weight is not None:
            weight = torch.tensor(config.class_weight, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=config.label_smoothing,
        )

    # Validation loss (always standard CE, no augmentation, no label smoothing)
    val_loss_fn = nn.CrossEntropyLoss()

    best_monitor_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        head.train()

        # Apply epoch-level augmentation on training portion only
        if config.augmentation is not None:
            Z_epoch, y_epoch = apply_augmentation(
                Z_tr, y_tr, config.augmentation, Z_train_std=Z_train_std,
            )
        else:
            Z_epoch, y_epoch = Z_tr, y_tr

        if use_minibatch:
            perm_train = torch.randperm(len(Z_epoch), device=device)
            for start in range(0, len(Z_epoch), batch_size):
                idx = perm_train[start : start + batch_size]
                logits = head(Z_epoch[idx])
                loss = loss_fn(logits, y_epoch[idx])
                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), config.grad_clip)
                optimizer.step()
                if is_per_step and scheduler is not None:
                    scheduler.step()
        else:
            logits = head(Z_epoch)
            loss = loss_fn(logits, y_epoch)
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), config.grad_clip)
            optimizer.step()

        # Per-epoch scheduler
        if not is_per_step and scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(loss.item())
            else:
                scheduler.step()

        # Monitor: validation loss (if available) or training loss (fallback)
        if use_val:
            head.eval()
            with torch.no_grad():
                val_logits = head(Z_val)
                monitor_loss = val_loss_fn(val_logits, y_val).item()
        else:
            monitor_loss = loss.item()

        if monitor_loss < best_monitor_loss - 1e-5:
            best_monitor_loss = monitor_loss
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    return head


def run_neural_train_test_v2(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    pretrain_aug_config: dict | None = None,
    epoch_aug_config: dict | None = None,
    seed: int = 42,
    val_split: float = 0.15,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Neural train/test with training fixes.

    Key fixes over run_neural_train_test():
      1. Z_train_std (dimension importance) computed from RAW embeddings
         BEFORE StandardScaler. After StandardScaler, all dims have std ≈ 1.0
         which neutralizes FroFA/AdaptiveNoise per-dimension scaling.
         We compute relative importance = raw_std / mean(raw_std) to preserve
         the natural variance structure while operating in scaled space.
      2. Uses train_head_v2() with validation-based early stopping.
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)

    # Pre-training augmentation (DC/SMOTE) on numpy
    if pretrain_aug_config is not None:
        Z_train_aug, y_train_aug = apply_pretrain_augmentation(
            Z_train, y_train, pretrain_aug_config, rng,
        )
    else:
        Z_train_aug, y_train_aug = Z_train, y_train

    # FIX: Compute Z_train_std from RAW embeddings BEFORE StandardScaler.
    # After StandardScaler, each dim has std ≈ 1.0, making FroFA/AdaptiveNoise
    # degenerate to uniform Gaussian noise. By preserving the relative variance
    # structure (importance = raw_std / mean_raw_std), augmentation correctly
    # applies larger perturbations to high-variance dims.
    Z_raw = torch.from_numpy(Z_train_aug).float()
    Z_raw_std = Z_raw.std(dim=0).clamp(min=1e-8)
    Z_importance = Z_raw_std / Z_raw_std.mean()

    # Standard scaling
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train_aug)
    Z_test_s = scaler.transform(Z_test)

    # Convert to tensors
    Z_train_t = torch.from_numpy(Z_train_s).float()
    y_train_t = torch.from_numpy(y_train_aug).long()
    Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

    # Build train config with epoch augmentation
    tc = TrainConfig(
        epochs=train_config.epochs,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        label_smoothing=train_config.label_smoothing,
        early_stopping_patience=train_config.early_stopping_patience,
        augmentation=epoch_aug_config,
        device=train_config.device,
        optimizer=train_config.optimizer,
        scheduler=train_config.scheduler,
        grad_clip=train_config.grad_clip,
        class_weight=train_config.class_weight,
        batch_size=train_config.batch_size,
    )

    # Train with v2 (validation-based early stopping + proper std)
    torch.manual_seed(seed)
    head = head_factory()
    head = train_head_v2(
        head, Z_train_t, y_train_t, tc, n_classes,
        Z_train_std=Z_importance, val_split=val_split,
    )

    # Predict
    with torch.no_grad():
        logits = head(Z_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = probs.argmax(axis=1).astype(np.int64)
    y_prob = probs[:, 1] if n_classes == 2 else probs

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


# ============================================================================
# Shared run-and-log helper
# ============================================================================

def _run_and_log(
    name: str,
    run_fn,
    extra: dict | None = None,
) -> dict | None:
    """Execute experiment, log results, return result row or None on error."""
    t0 = time.time()
    try:
        metrics, _, _ = run_fn()
        elapsed = time.time() - t0
        logger.info(_format_metrics_log(name, metrics, elapsed))
        return _make_result_row(name, metrics, elapsed, extra=extra)
    except Exception as e:
        logger.error("  [%s] FAILED: %s", name, e, exc_info=True)
        row = {"name": name, "error": str(e)}
        if extra:
            row.update(extra)
        return row


# ============================================================================
# Group J: Training Fix Verification (24 experiments)
# ============================================================================

def run_group_j(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Ablation: compare old train_head() vs new train_head_v2().

    6 fix variants × 4 head configs = 24 experiments.
    Quantifies the impact of each training fix independently.
    """
    logger.info("=" * 70)
    logger.info("GROUP J: Training Fix Verification (24 experiments)")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    exp_idx = 0

    head_configs = [
        {"name": "MLP[64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}},
        {"name": "MLP[64]-d0.2", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.2}},
        {"name": "MLP[128,64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}},
        {"name": "MLP[128,64]-d0.2", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.2}},
    ]

    # Fix variants: (use_v2_training, val_split, description)
    fix_variants = [
        # Baseline: old training (train loss ES, post-scaler std)
        {"name": "old_baseline", "use_v2": False, "val_split": 0.0,
         "desc": "Original: train-loss ES, post-scaler std"},
        # Fix 1: Validation-based ES only
        {"name": "fix_valES", "use_v2": True, "val_split": 0.15,
         "desc": "Fix: validation-based early stopping (15%)"},
        # Fix 2: Proper Z_train_std only (still train-loss ES)
        {"name": "fix_std", "use_v2": True, "val_split": 0.0,
         "desc": "Fix: proper Z_train_std (raw importance)"},
        # Fix 3: Both fixes combined
        {"name": "fix_both", "use_v2": True, "val_split": 0.15,
         "desc": "Fix: val ES + proper std"},
        # Fix 4: Both fixes + FroFA augmentation
        {"name": "fix_both+FroFA", "use_v2": True, "val_split": 0.15,
         "desc": "Fix: val ES + proper std + FroFA(s=0.1)"},
        # Fix 5: Both fixes + AdaptNoise augmentation
        {"name": "fix_both+AdaptN", "use_v2": True, "val_split": 0.15,
         "desc": "Fix: val ES + proper std + AdaptNoise(s=0.1)"},
    ]

    for fix in fix_variants:
        for hcfg in head_configs:
            exp_idx += 1
            exp_name = f"{fix['name']}|{hcfg['name']}"
            logger.info("[%d/24] %s — %s", exp_idx, exp_name, fix["desc"])

            def head_factory(hc=hcfg):
                return build_head(hc["type"], embed_dim, n_classes, **hc["kwargs"])

            # Determine augmentation
            epoch_aug = None
            if "FroFA" in fix["name"]:
                epoch_aug = {"strategy": "frofa", "strength": 0.1}
            elif "AdaptN" in fix["name"]:
                epoch_aug = {"strategy": "adaptive_noise", "scale": 0.1}

            if fix["use_v2"]:
                row = _run_and_log(
                    exp_name,
                    lambda hf=head_factory, ea=epoch_aug, vs=fix["val_split"]: (
                        run_neural_train_test_v2(
                            Z_train, y_train, Z_test, y_test,
                            hf, train_config,
                            epoch_aug_config=ea, seed=seed,
                            val_split=vs,
                        )
                    ),
                    extra={
                        "group": "J", "fix": fix["name"],
                        "head": hcfg["name"], "version": "v2",
                    },
                )
            else:
                row = _run_and_log(
                    exp_name,
                    lambda hf=head_factory, ea=epoch_aug: run_neural_train_test(
                        Z_train, y_train, Z_test, y_test,
                        hf, train_config,
                        epoch_aug_config=ea, seed=seed,
                    ),
                    extra={
                        "group": "J", "fix": fix["name"],
                        "head": hcfg["name"], "version": "v1",
                    },
                )
            results.append(row)

    _save_group_results(results, "J", "group_j_training_fix", output_dir)
    return results


# ============================================================================
# Group K: M+C+T1 × Layer × Classifier at 251min (48 experiments)
# ============================================================================

GROUP_K_CLASSIFIERS = [
    {"name": "SVM_rbf", "type": "sklearn",
     "clf_config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "LogReg", "type": "sklearn",
     "clf_config": {"type": "logistic_regression"}},
    {"name": "RF", "type": "sklearn",
     "clf_config": {"type": "random_forest"}},
    {"name": "ExtraTrees", "type": "sklearn",
     "clf_config": {"type": "extra_trees"}},
    # Neural heads with v2 training (validation ES + proper std)
    {"name": "MLP[64]-d0.3_v2", "type": "neural_v2",
     "head_type": "mlp", "head_kwargs": {"hidden_dims": [64], "dropout": 0.3}},
    {"name": "MLP[128,64]-d0.3_v2", "type": "neural_v2",
     "head_type": "mlp", "head_kwargs": {"hidden_dims": [128, 64], "dropout": 0.3}},
    # Neural heads with old training (for comparison)
    {"name": "MLP[64]-d0.5_old", "type": "neural",
     "head_type": "mlp", "head_kwargs": {"hidden_dims": [64], "dropout": 0.5}},
    {"name": "Linear_v2", "type": "neural_v2",
     "head_type": "linear", "head_kwargs": {}},
]


def run_group_k(
    train_csv: str,
    test_csv: str,
    label_column: str,
    split_date: str | None,
    stride: int,
    pretrained: str,
    output_token: str,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """M+C+T1 × 6 layers × 8 classifiers at 251min context.

    6 × 8 = 48 experiments. Requires re-extraction per layer.
    Fills the BIGGEST gap: layer sweep with M+C+T1 at optimal context.
    """
    logger.info("=" * 70)
    logger.info("GROUP K: M+C+T1 × Layer × Classifier @ 251min (48 experiments)")
    logger.info("=" * 70)

    device = train_config.device
    channels = _resolve_channels(["M", "C", "T1"])

    # Load unified data
    logger.info("Loading unified data for M+C+T1...")
    sensor, train_labels, test_labels, ch_names, timestamps = load_unified_split(
        train_csv, test_csv,
        label_column=label_column, channels=channels,
        split_date=split_date,
    )

    # Build 251min context datasets (before=150, after=100 → 150+1+100=251)
    ds_cfg = DatasetConfig(
        context_mode="bidirectional",
        context_before=150, context_after=100,
        stride=stride,
    )
    train_dataset = OccupancyDataset(sensor, train_labels, timestamps, ds_cfg)
    test_dataset = OccupancyDataset(sensor, test_labels, timestamps, ds_cfg)

    n_train = len(train_dataset)
    n_test = len(test_dataset)
    logger.info("251min datasets: train=%d, test=%d", n_train, n_test)

    y_train = train_dataset.labels
    y_test = test_dataset.labels
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))

    results = []
    exp_idx = 0
    total = len(ALL_LAYERS) * len(GROUP_K_CLASSIFIERS)

    for layer in ALL_LAYERS:
        logger.info("Extracting L%d embeddings (M+C+T1, 251min)...", layer)
        model = load_mantis_model(pretrained, layer, output_token, device)
        Z_train = extract_embeddings(model, train_dataset, device)
        Z_test = extract_embeddings(model, test_dataset, device)
        embed_dim = Z_train.shape[1]
        logger.info("  L%d: train=%s, test=%s", layer, Z_train.shape, Z_test.shape)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        for clf_cfg in GROUP_K_CLASSIFIERS:
            exp_idx += 1
            exp_name = f"L{layer}|{clf_cfg['name']}"
            logger.info("[%d/%d] %s", exp_idx, total, exp_name)

            if clf_cfg["type"] == "sklearn":
                row = _run_and_log(
                    exp_name,
                    lambda cc=clf_cfg["clf_config"], zt=Z_train, ze=Z_test: (
                        run_sklearn_train_test(zt, y_train, ze, y_test, cc)
                    ),
                    extra={
                        "group": "K", "layer": layer, "classifier": clf_cfg["name"],
                        "channels": "M+C+T1", "context_min": 251,
                    },
                )
            elif clf_cfg["type"] == "neural_v2":
                def head_factory(hc=clf_cfg, ed=embed_dim):
                    return build_head(hc["head_type"], ed, n_classes, **hc["head_kwargs"])

                row = _run_and_log(
                    exp_name,
                    lambda hf=head_factory, zt=Z_train, ze=Z_test: (
                        run_neural_train_test_v2(
                            zt, y_train, ze, y_test,
                            hf, train_config, seed=seed, val_split=0.15,
                        )
                    ),
                    extra={
                        "group": "K", "layer": layer, "classifier": clf_cfg["name"],
                        "channels": "M+C+T1", "context_min": 251,
                    },
                )
            else:  # neural (old)
                def head_factory(hc=clf_cfg, ed=embed_dim):
                    return build_head(hc["head_type"], ed, n_classes, **hc["head_kwargs"])

                row = _run_and_log(
                    exp_name,
                    lambda hf=head_factory, zt=Z_train, ze=Z_test: (
                        run_neural_train_test(
                            zt, y_train, ze, y_test,
                            hf, train_config, seed=seed,
                        )
                    ),
                    extra={
                        "group": "K", "layer": layer, "classifier": clf_cfg["name"],
                        "channels": "M+C+T1", "context_min": 251,
                    },
                )
            results.append(row)

    _save_group_results(results, "K", "group_k_layer_classifier", output_dir)
    return results


# ============================================================================
# Group L: Context Fine-Tuning for M+C+T1 (72 experiments)
# ============================================================================

PHASE3_CONTEXTS = [
    # Fine-grained search around the 200-350min sweet spot.
    # All asymmetric with after=100 (deployment limit).
    {"name": "181min", "before": 90, "after": 90},
    {"name": "201min", "before": 100, "after": 100},
    {"name": "221min", "before": 120, "after": 100},
    {"name": "231min", "before": 130, "after": 100},
    {"name": "241min", "before": 140, "after": 100},
    {"name": "251min", "before": 150, "after": 100},
    {"name": "261min", "before": 160, "after": 100},
    {"name": "271min", "before": 170, "after": 100},
    {"name": "301min", "before": 200, "after": 100},
    {"name": "321min", "before": 220, "after": 100},
    {"name": "351min", "before": 250, "after": 100},
    {"name": "401min", "before": 300, "after": 100},
]

GROUP_L_CLASSIFIERS = [
    {"name": "SVM_rbf", "type": "sklearn",
     "clf_config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "MLP[64]-d0.3_v2", "type": "neural_v2",
     "head_type": "mlp", "head_kwargs": {"hidden_dims": [64], "dropout": 0.3}},
    {"name": "LogReg", "type": "sklearn",
     "clf_config": {"type": "logistic_regression"}},
]


def run_group_l(
    train_csv: str,
    test_csv: str,
    label_column: str,
    split_date: str | None,
    stride: int,
    pretrained: str,
    output_token: str,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Context fine-tuning: 12 contexts × 2 layers × 3 classifiers = 72 experiments.

    Fine-grained search in the 181-401min range to pinpoint optimal context.
    Tests both L2 (best for SVM) and L3 (best for MLP).
    """
    logger.info("=" * 70)
    logger.info("GROUP L: Context Fine-Tuning (72 experiments)")
    logger.info("=" * 70)

    device = train_config.device
    channels = _resolve_channels(["M", "C", "T1"])
    test_layers = [2, 3]
    total = len(PHASE3_CONTEXTS) * len(test_layers) * len(GROUP_L_CLASSIFIERS)

    # Load unified sensor data
    logger.info("Loading unified data for M+C+T1...")
    sensor, train_labels, test_labels, ch_names, timestamps = load_unified_split(
        train_csv, test_csv,
        label_column=label_column, channels=channels,
        split_date=split_date,
    )

    results = []
    exp_idx = 0

    for layer in test_layers:
        logger.info("Loading MantisV2 model L%d...", layer)
        model = load_mantis_model(pretrained, layer, output_token, device)

        for ctx_cfg in PHASE3_CONTEXTS:
            ctx_min = ctx_cfg["before"] + 1 + ctx_cfg["after"]
            ds_cfg = DatasetConfig(
                context_mode="bidirectional",
                context_before=ctx_cfg["before"],
                context_after=ctx_cfg["after"],
                stride=stride,
            )
            train_dataset = OccupancyDataset(sensor, train_labels, timestamps, ds_cfg)
            test_dataset = OccupancyDataset(sensor, test_labels, timestamps, ds_cfg)

            n_train = len(train_dataset)
            n_test = len(test_dataset)
            if n_train < 10 or n_test < 10:
                logger.warning("Skipping L%d/%s: train=%d test=%d",
                               layer, ctx_cfg["name"], n_train, n_test)
                continue

            Z_train = extract_embeddings(model, train_dataset, device)
            Z_test = extract_embeddings(model, test_dataset, device)
            y_train = train_dataset.labels
            y_test = test_dataset.labels
            embed_dim = Z_train.shape[1]
            n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))

            for clf_cfg in GROUP_L_CLASSIFIERS:
                exp_idx += 1
                exp_name = f"L{layer}|{ctx_cfg['name']}|{clf_cfg['name']}"
                logger.info("[%d/%d] %s (train=%d test=%d)",
                            exp_idx, total, exp_name, n_train, n_test)

                if clf_cfg["type"] == "sklearn":
                    row = _run_and_log(
                        exp_name,
                        lambda cc=clf_cfg["clf_config"], zt=Z_train, ze=Z_test,
                               yt=y_train, ye=y_test: (
                            run_sklearn_train_test(zt, yt, ze, ye, cc)
                        ),
                        extra={
                            "group": "L", "layer": layer, "context": ctx_cfg["name"],
                            "context_min": ctx_min, "classifier": clf_cfg["name"],
                            "channels": "M+C+T1",
                        },
                    )
                else:  # neural_v2
                    def head_factory(hc=clf_cfg, ed=embed_dim, nc=n_classes):
                        return build_head(hc["head_type"], ed, nc, **hc["head_kwargs"])

                    row = _run_and_log(
                        exp_name,
                        lambda hf=head_factory, zt=Z_train, ze=Z_test,
                               yt=y_train, ye=y_test: (
                            run_neural_train_test_v2(
                                zt, yt, ze, ye,
                                hf, train_config, seed=seed, val_split=0.15,
                            )
                        ),
                        extra={
                            "group": "L", "layer": layer, "context": ctx_cfg["name"],
                            "context_min": ctx_min, "classifier": clf_cfg["name"],
                            "channels": "M+C+T1",
                        },
                    )
                results.append(row)

            gc.collect()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save_group_results(results, "L", "group_l_context_tuning", output_dir)
    return results


# ============================================================================
# Group M: Improved MLP Recipe (72 experiments)
# ============================================================================

def run_group_m(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    base_train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Systematic MLP recipe search with all training fixes applied.

    Part M1: Architecture grid (27) — LR × dropout × hidden
    Part M2: Training HP grid (27) — LS × CW × scheduler
    Part M3: Advanced configs (18) — BS, optimizer, augmentation

    Total: 72 experiments. All use train_head_v2 + proper Z_train_std.
    """
    logger.info("=" * 70)
    logger.info("GROUP M: Improved MLP Recipe (72 experiments)")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    exp_idx = 0

    # ---- Part M1: Architecture grid = 3 LR × 3 dropout × 3 hidden = 27 ----
    logger.info("--- Part M1: Architecture Grid (27 experiments) ---")

    lr_values = [1e-4, 3e-4, 5e-4]
    dropout_values = [0.1, 0.2, 0.3]
    hidden_configs = [
        {"name": "[64]", "dims": [64]},
        {"name": "[128]", "dims": [128]},
        {"name": "[128,64]", "dims": [128, 64]},
    ]

    for lr in lr_values:
        for dp in dropout_values:
            for hcfg in hidden_configs:
                exp_idx += 1
                exp_name = f"LR={lr:.0e}_d{dp}_{hcfg['name']}"
                logger.info("[%d] %s", exp_idx, exp_name)

                def head_factory(dims=hcfg["dims"], d=dp):
                    return build_head("mlp", embed_dim, n_classes,
                                      hidden_dims=dims, dropout=d)

                tc = TrainConfig(
                    epochs=base_train_config.epochs,
                    lr=lr, weight_decay=0.01,
                    early_stopping_patience=30,
                    device=base_train_config.device,
                )
                row = _run_and_log(
                    exp_name,
                    lambda hf=head_factory, tc_=tc: run_neural_train_test_v2(
                        Z_train, y_train, Z_test, y_test,
                        hf, tc_, seed=seed, val_split=0.15,
                    ),
                    extra={
                        "group": "M", "subgroup": "arch_grid",
                        "lr": lr, "dropout": dp, "hidden": hcfg["name"],
                    },
                )
                results.append(row)

    # ---- Part M2: Training HP grid = 3 LS × 3 CW × 3 scheduler = 27 ----
    logger.info("--- Part M2: Training HP Grid (27 experiments) ---")

    ls_values = [0.0, 0.1, 0.15]
    cw_configs = [
        {"name": "none", "weight": None},
        {"name": "1:1.5", "weight": [1.0, 1.5]},
        {"name": "1:2", "weight": [1.0, 2.0]},
    ]
    scheduler_values = ["cosine", "onecycle", "plateau"]

    for ls in ls_values:
        for cw in cw_configs:
            for sched in scheduler_values:
                exp_idx += 1
                exp_name = f"LS={ls}_CW={cw['name']}_sched={sched}"
                logger.info("[%d] %s", exp_idx, exp_name)

                # Use MLP[64]-d0.2 as the architecture (likely best from M1)
                def head_factory():
                    return build_head("mlp", embed_dim, n_classes,
                                      hidden_dims=[64], dropout=0.2)

                tc = TrainConfig(
                    epochs=base_train_config.epochs,
                    lr=3e-4,
                    weight_decay=0.01,
                    label_smoothing=ls,
                    early_stopping_patience=30,
                    device=base_train_config.device,
                    scheduler=sched,
                    class_weight=cw["weight"],
                )
                row = _run_and_log(
                    exp_name,
                    lambda hf=head_factory, tc_=tc: run_neural_train_test_v2(
                        Z_train, y_train, Z_test, y_test,
                        hf, tc_, seed=seed, val_split=0.15,
                    ),
                    extra={
                        "group": "M", "subgroup": "hp_grid",
                        "label_smoothing": ls, "class_weight": cw["name"],
                        "scheduler": sched,
                    },
                )
                results.append(row)

    # ---- Part M3: Advanced configs (18) ----
    logger.info("--- Part M3: Advanced Configs (18 experiments) ---")

    # M3a: Batch size sweep (6)
    for bs in [32, 64, 128, 256, 512, 0]:
        exp_idx += 1
        bs_label = "full" if bs == 0 else str(bs)
        exp_name = f"BS={bs_label}"
        logger.info("[%d] %s", exp_idx, exp_name)

        def head_factory():
            return build_head("mlp", embed_dim, n_classes,
                              hidden_dims=[64], dropout=0.2)

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=3e-4, weight_decay=0.01,
            early_stopping_patience=30,
            device=base_train_config.device,
            batch_size=bs,
        )
        row = _run_and_log(
            exp_name,
            lambda hf=head_factory, tc_=tc: run_neural_train_test_v2(
                Z_train, y_train, Z_test, y_test,
                hf, tc_, seed=seed, val_split=0.15,
            ),
            extra={"group": "M", "subgroup": "batch_size", "batch_size": bs_label},
        )
        results.append(row)

    # M3b: Optimizer comparison (4)
    for opt in ["adamw", "adam", "sgd", "rmsprop"]:
        exp_idx += 1
        exp_name = f"opt={opt}"
        logger.info("[%d] %s", exp_idx, exp_name)

        def head_factory():
            return build_head("mlp", embed_dim, n_classes,
                              hidden_dims=[64], dropout=0.2)

        lr_opt = 5e-3 if opt == "sgd" else 3e-4
        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=lr_opt, weight_decay=0.01,
            early_stopping_patience=30,
            device=base_train_config.device,
            optimizer=opt,
        )
        row = _run_and_log(
            exp_name,
            lambda hf=head_factory, tc_=tc: run_neural_train_test_v2(
                Z_train, y_train, Z_test, y_test,
                hf, tc_, seed=seed, val_split=0.15,
            ),
            extra={"group": "M", "subgroup": "optimizer", "optimizer": opt},
        )
        results.append(row)

    # M3c: Augmentation with fixed training (5)
    aug_configs = [
        {"name": "no_aug", "pretrain": None, "epoch": None},
        {"name": "FroFA_s01", "pretrain": None,
         "epoch": {"strategy": "frofa", "strength": 0.1}},
        {"name": "AdaptN_s01", "pretrain": None,
         "epoch": {"strategy": "adaptive_noise", "scale": 0.1}},
        {"name": "DC_a05_n50", "pretrain": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50},
         "epoch": None},
        {"name": "DC+FroFA", "pretrain": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50},
         "epoch": {"strategy": "frofa", "strength": 0.1}},
    ]
    for aug in aug_configs:
        exp_idx += 1
        exp_name = f"aug={aug['name']}"
        logger.info("[%d] %s", exp_idx, exp_name)

        def head_factory():
            return build_head("mlp", embed_dim, n_classes,
                              hidden_dims=[64], dropout=0.2)

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=3e-4, weight_decay=0.01,
            early_stopping_patience=30,
            device=base_train_config.device,
        )
        row = _run_and_log(
            exp_name,
            lambda hf=head_factory, tc_=tc, pa=aug["pretrain"], ea=aug["epoch"]: (
                run_neural_train_test_v2(
                    Z_train, y_train, Z_test, y_test,
                    hf, tc_, pretrain_aug_config=pa,
                    epoch_aug_config=ea, seed=seed, val_split=0.15,
                )
            ),
            extra={"group": "M", "subgroup": "augmentation", "augmentation": aug["name"]},
        )
        results.append(row)

    # M3d: Larger architectures with best HP (3)
    large_archs = [
        {"name": "MLP[256]-d0.2", "dims": [256], "dp": 0.2},
        {"name": "MLP[256,128]-d0.2", "dims": [256, 128], "dp": 0.2},
        {"name": "MLP[512,256]-d0.2", "dims": [512, 256], "dp": 0.2},
    ]
    for arch in large_archs:
        exp_idx += 1
        exp_name = arch["name"]
        logger.info("[%d] %s", exp_idx, exp_name)

        def head_factory(a=arch):
            return build_head("mlp", embed_dim, n_classes,
                              hidden_dims=a["dims"], dropout=a["dp"])

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=3e-4, weight_decay=0.01,
            label_smoothing=0.1,
            early_stopping_patience=30,
            device=base_train_config.device,
            class_weight=[1.0, 1.5],
        )
        row = _run_and_log(
            exp_name,
            lambda hf=head_factory, tc_=tc: run_neural_train_test_v2(
                Z_train, y_train, Z_test, y_test,
                hf, tc_, seed=seed, val_split=0.15,
            ),
            extra={"group": "M", "subgroup": "large_arch", "head": arch["name"]},
        )
        results.append(row)

    _save_group_results(results, "M", "group_m_mlp_recipe", output_dir)
    return results


# ============================================================================
# Group N: SVM/LogReg Hyperparameter Grid (60 experiments)
# ============================================================================

def run_group_n(
    all_Z_train: dict[int, np.ndarray],
    all_Z_test: dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> list[dict]:
    """SVM C×gamma grid + LogReg C grid at M+C+T1, 251min.

    SVM: 6 C × 5 gamma × 2 layers = 60 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP N: SVM/LogReg Hyperparameter Grid (60 experiments)")
    logger.info("=" * 70)

    results = []
    exp_idx = 0

    C_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    gamma_values = ["scale", "auto", 0.001, 0.005, 0.01]

    for layer in [2, 3]:
        Z_train = all_Z_train[layer]
        Z_test_l = all_Z_test[layer]

        # SVM grid: 6 C × 5 gamma = 30 per layer
        for C in C_values:
            for gamma in gamma_values:
                exp_idx += 1
                gamma_str = gamma if isinstance(gamma, str) else f"{gamma:.4f}"
                exp_name = f"L{layer}|SVM_C={C}_g={gamma_str}"
                logger.info("[%d/60] %s", exp_idx, exp_name)

                clf_config = {
                    "type": "svm", "kernel": "rbf",
                    "C": C, "gamma": gamma,
                }
                row = _run_and_log(
                    exp_name,
                    lambda cc=clf_config, zt=Z_train, ze=Z_test_l: (
                        run_sklearn_train_test(zt, y_train, ze, y_test, cc)
                    ),
                    extra={
                        "group": "N", "layer": layer, "classifier": "SVM_rbf",
                        "C": C, "gamma": gamma_str,
                        "channels": "M+C+T1", "context_min": 251,
                    },
                )
                results.append(row)

    _save_group_results(results, "N", "group_n_svm_grid", output_dir)
    return results


# ============================================================================
# Group O: Final Combinations + Multi-Layer + Stability (36 experiments)
# ============================================================================

def run_group_o(
    all_Z_train: dict[int, np.ndarray],
    all_Z_test: dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Final combination experiments: multi-seed, multi-layer, TTA.

    Part O1: Multi-seed stability for top configs (15)
    Part O2: Multi-layer fusion at 251min with M+C+T1 (12)
    Part O3: TTA on best neural configs (6)
    Part O4: Best config × 3 val_split values (3)

    Total: 36 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP O: Final Combinations (36 experiments)")
    logger.info("=" * 70)

    embed_dim = next(iter(all_Z_train.values())).shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    exp_idx = 0

    # Primary embeddings (L2 and L3)
    Z_train_L2 = all_Z_train[2]
    Z_test_L2 = all_Z_test[2]
    Z_train_L3 = all_Z_train[3]
    Z_test_L3 = all_Z_test[3]

    # ---- Part O1: Multi-seed stability (15 experiments) ----
    logger.info("--- Part O1: Multi-seed Stability (15 experiments) ---")

    stability_configs = [
        # Best MLP v2 config
        {"name": "MLP[64]-d0.2_v2_L2", "layer": 2,
         "head_type": "mlp", "head_kwargs": {"hidden_dims": [64], "dropout": 0.2}},
        {"name": "MLP[64]-d0.2_v2_L3", "layer": 3,
         "head_type": "mlp", "head_kwargs": {"hidden_dims": [64], "dropout": 0.2}},
        {"name": "MLP[128,64]-d0.2_v2_L2", "layer": 2,
         "head_type": "mlp", "head_kwargs": {"hidden_dims": [128, 64], "dropout": 0.2}},
    ]

    for cfg in stability_configs:
        Z_tr = all_Z_train[cfg["layer"]]
        Z_te = all_Z_test[cfg["layer"]]

        for s in [42, 142, 242, 342, 442]:
            exp_idx += 1
            exp_name = f"{cfg['name']}_seed{s}"
            logger.info("[%d] %s", exp_idx, exp_name)

            def head_factory(c=cfg, ed=Z_tr.shape[1]):
                return build_head(c["head_type"], ed, n_classes, **c["head_kwargs"])

            tc = TrainConfig(
                epochs=train_config.epochs,
                lr=3e-4, weight_decay=0.01,
                early_stopping_patience=30,
                device=train_config.device,
            )
            row = _run_and_log(
                exp_name,
                lambda hf=head_factory, tc_=tc, zt=Z_tr, ze=Z_te, sd=s: (
                    run_neural_train_test_v2(
                        zt, y_train, ze, y_test,
                        hf, tc_, seed=sd, val_split=0.15,
                    )
                ),
                extra={
                    "group": "O", "subgroup": "stability",
                    "config": cfg["name"], "seed": s,
                    "channels": "M+C+T1", "context_min": 251,
                },
            )
            results.append(row)

    # ---- Part O2: Multi-layer fusion at 251min (12 experiments) ----
    logger.info("--- Part O2: Multi-Layer Fusion (12 experiments) ---")

    fusion_combos = [
        {"name": "Concat_L2+L3", "layers": [2, 3], "method": "concat"},
        {"name": "Concat_L0+L3", "layers": [0, 3], "method": "concat"},
        {"name": "Concat_L2+L3+L5", "layers": [2, 3, 5], "method": "concat"},
        {"name": "Concat_L0+L2+L3", "layers": [0, 2, 3], "method": "concat"},
        {"name": "Fusion_L2+L3", "layers": [2, 3], "method": "fusion"},
        {"name": "Fusion_L0+L3+L5", "layers": [0, 3, 5], "method": "fusion"},
        {"name": "Fusion_L2+L3+L4", "layers": [2, 3, 4], "method": "fusion"},
        {"name": "Fusion_All", "layers": [0, 1, 2, 3, 4, 5], "method": "fusion"},
        {"name": "Attn_L2+L3", "layers": [2, 3], "method": "attention"},
        {"name": "Attn_L0+L3+L5", "layers": [0, 3, 5], "method": "attention"},
        {"name": "Attn_L2+L3+L4", "layers": [2, 3, 4], "method": "attention"},
        {"name": "Attn_All", "layers": [0, 1, 2, 3, 4, 5], "method": "attention"},
    ]

    for combo in fusion_combos:
        exp_idx += 1
        exp_name = combo["name"]
        logger.info("[%d] %s", exp_idx, exp_name)

        nl = len(combo["layers"])

        if combo["method"] == "concat":
            Z_tr_cat = np.concatenate([all_Z_train[l] for l in combo["layers"]], axis=1)
            Z_te_cat = np.concatenate([all_Z_test[l] for l in combo["layers"]], axis=1)
            cat_dim = Z_tr_cat.shape[1]

            def head_factory(cd=cat_dim):
                return build_head("mlp", cd, n_classes,
                                  hidden_dims=[min(256, cd // 2), 64], dropout=0.3)

            row = _run_and_log(
                exp_name,
                lambda hf=head_factory, zt=Z_tr_cat, ze=Z_te_cat: (
                    run_neural_train_test_v2(
                        zt, y_train, ze, y_test,
                        hf, train_config, seed=seed, val_split=0.15,
                    )
                ),
                extra={
                    "group": "O", "subgroup": "fusion",
                    "method": "concat", "layers": combo["layers"],
                    "channels": "M+C+T1", "context_min": 251,
                },
            )
        elif combo["method"] == "fusion":
            def head_factory(n_layers=nl):
                return build_head(
                    "multi_layer_fusion", embed_dim, n_classes,
                    n_layers=n_layers, hidden_dims=[64], dropout=0.3,
                )

            row = _run_and_log(
                exp_name,
                lambda li=combo["layers"], hf=head_factory: (
                    run_neural_train_test_multi_layer(
                        all_Z_train, all_Z_test, li, y_train, y_test,
                        hf, train_config, seed,
                    )
                ),
                extra={
                    "group": "O", "subgroup": "fusion",
                    "method": "fusion", "layers": combo["layers"],
                    "channels": "M+C+T1", "context_min": 251,
                },
            )
        else:  # attention
            def head_factory(n_layers=nl):
                return build_head(
                    "attention_pool", embed_dim, n_classes,
                    n_layers=n_layers, hidden_dims=[64], dropout=0.3,
                )

            row = _run_and_log(
                exp_name,
                lambda li=combo["layers"], hf=head_factory: (
                    run_neural_train_test_multi_layer(
                        all_Z_train, all_Z_test, li, y_train, y_test,
                        hf, train_config, seed,
                    )
                ),
                extra={
                    "group": "O", "subgroup": "fusion",
                    "method": "attention", "layers": combo["layers"],
                    "channels": "M+C+T1", "context_min": 251,
                },
            )
        results.append(row)

    # ---- Part O3: TTA on best neural configs (6 experiments) ----
    logger.info("--- Part O3: TTA (6 experiments) ---")

    tta_configs = [
        {"name": "TTA-5_FroFA_L2", "layer": 2, "tta_k": 5,
         "strategy": "frofa", "strength": 0.1},
        {"name": "TTA-10_FroFA_L2", "layer": 2, "tta_k": 10,
         "strategy": "frofa", "strength": 0.1},
        {"name": "TTA-5_AdaptN_L2", "layer": 2, "tta_k": 5,
         "strategy": "adaptive_noise", "strength": 0.1},
        {"name": "TTA-5_FroFA_L3", "layer": 3, "tta_k": 5,
         "strategy": "frofa", "strength": 0.1},
        {"name": "TTA-10_FroFA_L3", "layer": 3, "tta_k": 10,
         "strategy": "frofa", "strength": 0.1},
        {"name": "TTA-5_AdaptN_L3", "layer": 3, "tta_k": 5,
         "strategy": "adaptive_noise", "strength": 0.1},
    ]

    for tta_cfg in tta_configs:
        exp_idx += 1
        exp_name = tta_cfg["name"]
        logger.info("[%d] %s", exp_idx, exp_name)

        Z_tr = all_Z_train[tta_cfg["layer"]]
        Z_te = all_Z_test[tta_cfg["layer"]]

        def head_factory(ed=Z_tr.shape[1]):
            return build_head("mlp", ed, n_classes,
                              hidden_dims=[64], dropout=0.2)

        row = _run_and_log(
            exp_name,
            lambda hf=head_factory, zt=Z_tr, ze=Z_te, tc=tta_cfg: (
                run_tta_train_test(
                    zt, y_train, ze, y_test,
                    hf, train_config,
                    tta_k=tc["tta_k"], tta_strategy=tc["strategy"],
                    tta_strength=tc["strength"], seed=seed,
                )
            ),
            extra={
                "group": "O", "subgroup": "tta",
                "tta_k": tta_cfg["tta_k"], "tta_strategy": tta_cfg["strategy"],
                "layer": tta_cfg["layer"],
                "channels": "M+C+T1", "context_min": 251,
            },
        )
        results.append(row)

    # ---- Part O4: val_split ablation (3 experiments) ----
    logger.info("--- Part O4: val_split Ablation (3 experiments) ---")

    for vs in [0.05, 0.10, 0.20]:
        exp_idx += 1
        exp_name = f"val_split={vs}"
        logger.info("[%d] %s", exp_idx, exp_name)

        def head_factory():
            return build_head("mlp", Z_train_L2.shape[1], n_classes,
                              hidden_dims=[64], dropout=0.2)

        tc = TrainConfig(
            epochs=train_config.epochs,
            lr=3e-4, weight_decay=0.01,
            early_stopping_patience=30,
            device=train_config.device,
        )
        row = _run_and_log(
            exp_name,
            lambda hf=head_factory, tc_=tc, v=vs: run_neural_train_test_v2(
                Z_train_L2, y_train, Z_test_L2, y_test,
                hf, tc_, seed=seed, val_split=v,
            ),
            extra={
                "group": "O", "subgroup": "val_split",
                "val_split": vs, "layer": 2,
                "channels": "M+C+T1", "context_min": 251,
            },
        )
        results.append(row)

    _save_group_results(results, "O", "group_o_final", output_dir)
    return results


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Additional Experiments: MLP Training Fixes + Grid Search"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--group", nargs="+",
                        choices=["J", "K", "L", "M", "N", "O"], default=None,
                        help="Run specific group(s) (e.g. --group J K for GPU-1)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config or 42)")
    parser.add_argument("--split-date", default=None,
                        help="Override split date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    output_dir = Path(cfg.get("output_dir", "results/phase2_sweep")).parent / "phase3_additional"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_output(formats=["png"], dpi=200)

    # Data config
    data_cfg = cfg["data"]
    label_column = data_cfg.get("label_column", "occupancy_label")
    split_date = args.split_date if args.split_date else cfg.get("split_date")
    stride = cfg.get("stride", 1)
    pretrained = cfg["model"]["pretrained_name"]
    output_token = cfg["model"].get("output_token", "combined")

    # Training config
    train_raw = cfg.get("training", {})
    base_train_config = TrainConfig(
        epochs=train_raw.get("epochs", 200),
        lr=train_raw.get("lr", 1e-3),
        weight_decay=train_raw.get("weight_decay", 0.01),
        label_smoothing=train_raw.get("label_smoothing", 0.0),
        early_stopping_patience=train_raw.get("early_stopping_patience", 30),
        device=device,
        batch_size=train_raw.get("batch_size", 0),
    )

    groups = args.group if args.group else ["J", "K", "L", "M", "N", "O"]

    logger.info("=" * 70)
    logger.info("Phase 3 Additional Experiments")
    logger.info("  Groups: %s", groups)
    logger.info("  Device: %s", device)
    logger.info("  Seed: %d", seed)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Determine data needs
    # -----------------------------------------------------------------------
    needs_default_251min = any(g in groups for g in ["J", "M", "N", "O"])
    needs_reextract_k = "K" in groups
    needs_reextract_l = "L" in groups

    all_Z_train_251: dict[int, np.ndarray] = {}
    all_Z_test_251: dict[int, np.ndarray] = {}
    y_train_251 = None
    y_test_251 = None

    # Load 251min context embeddings for J/M/N/O
    if needs_default_251min:
        channels = _resolve_channels(["M", "C", "T1"])
        logger.info("Loading unified data for M+C+T1 at 251min context...")

        sensor, train_labels, test_labels, ch_names, timestamps = load_unified_split(
            data_cfg["train_csv"], data_cfg["test_csv"],
            label_column=label_column, channels=channels,
            split_date=split_date,
        )

        ds_cfg = DatasetConfig(
            context_mode="bidirectional",
            context_before=150, context_after=100,
            stride=stride,
        )
        train_dataset = OccupancyDataset(sensor, train_labels, timestamps, ds_cfg)
        test_dataset = OccupancyDataset(sensor, test_labels, timestamps, ds_cfg)

        y_train_251 = train_dataset.labels
        y_test_251 = test_dataset.labels
        logger.info("251min datasets: train=%d, test=%d",
                     len(train_dataset), len(test_dataset))

        # Determine which layers to extract
        layers_needed = {2}  # L2 always needed for M
        if any(g in groups for g in ["N", "O"]):
            layers_needed.add(3)
        if "O" in groups:
            layers_needed.update(ALL_LAYERS)  # multi-layer fusion needs all

        for l in sorted(layers_needed):
            logger.info("Extracting L%d embeddings (M+C+T1, 251min)...", l)
            model = load_mantis_model(pretrained, l, output_token, device)
            all_Z_train_251[l] = extract_embeddings(model, train_dataset, device)
            all_Z_test_251[l] = extract_embeddings(model, test_dataset, device)
            logger.info("  L%d: train=%s, test=%s",
                        l, all_Z_train_251[l].shape, all_Z_test_251[l].shape)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # -----------------------------------------------------------------------
    # Run groups
    # -----------------------------------------------------------------------
    t_start = time.time()
    all_results = []

    for group in groups:
        logger.info("")
        t_group = time.time()

        if group == "J":
            # J uses 251min L2 embeddings (same as default for M)
            results = run_group_j(
                all_Z_train_251[2], y_train_251,
                all_Z_test_251[2], y_test_251,
                base_train_config, seed, output_dir,
            )
        elif group == "K":
            # K requires its own extraction (all layers at 251min)
            results = run_group_k(
                train_csv=data_cfg["train_csv"],
                test_csv=data_cfg["test_csv"],
                label_column=label_column,
                split_date=split_date,
                stride=stride,
                pretrained=pretrained,
                output_token=output_token,
                train_config=base_train_config,
                seed=seed,
                output_dir=output_dir,
            )
        elif group == "L":
            # L requires its own extraction (multiple contexts)
            results = run_group_l(
                train_csv=data_cfg["train_csv"],
                test_csv=data_cfg["test_csv"],
                label_column=label_column,
                split_date=split_date,
                stride=stride,
                pretrained=pretrained,
                output_token=output_token,
                train_config=base_train_config,
                seed=seed,
                output_dir=output_dir,
            )
        elif group == "M":
            results = run_group_m(
                all_Z_train_251[2], y_train_251,
                all_Z_test_251[2], y_test_251,
                base_train_config, seed, output_dir,
            )
        elif group == "N":
            results = run_group_n(
                all_Z_train_251, all_Z_test_251,
                y_train_251, y_test_251,
                output_dir,
            )
        elif group == "O":
            results = run_group_o(
                all_Z_train_251, all_Z_test_251,
                y_train_251, y_test_251,
                base_train_config, seed, output_dir,
            )
        else:
            continue

        all_results.extend(results)
        group_time = time.time() - t_group
        logger.info("Group %s completed: %d experiments in %.1fs (%.1f min)",
                     group, len(results), group_time, group_time / 60)

        try:
            plot_group_bar(
                results,
                f"Phase 3 Group {group}: AUC Ranking",
                output_dir / "plots" / f"group_{group.lower()}_bar",
            )
        except Exception:
            logger.warning("Failed to plot Group %s bar", group, exc_info=True)

    generate_summary(all_results, output_dir)

    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Done! Total: %d experiments in %.1fs (%.1f min)",
                len(all_results), total_time, total_time / 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
