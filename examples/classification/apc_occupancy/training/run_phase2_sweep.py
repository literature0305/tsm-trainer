#!/usr/bin/env python3
"""Phase 2 Comprehensive Sweep: Neural Classification Heads on MantisV2 Frozen Embeddings.

v4: Unified sensor array with date-based label split.
Loads both train/test CSVs from prepare_split.py, concatenates into one
continuous sensor timeline, then splits labels by configurable --split-date.
Context windows can cross the train/test boundary (sensor overlap allowed;
label leakage prevented via masked label arrays).

Phase 1 findings (to beat):
  - Best channels: M+C+T1 (3ch) — dominant over M+C (2ch)
  - Best context: ~201min bidirectional (100+1+100) — max symmetric
  - Best layer: L2 (single), L2+L5 (fusion)
  - Best sklearn: SVM_rbf → AUC≈0.986, EER≈0.047, Acc≈0.951
  - Only output_token="combined" works in MantisV2
  - NOTE: Future context capped at MAX_CONTEXT_AFTER=100 (deployment limit)

Six sweep groups reorganized for 3-way parallel execution:

  GPU 1: Group D + E — Augmentation x Head + Context Window (~216 experiments)

  Group D — Augmentation x Head Architecture (120 experiments)
    - 10 augmentation strategies x 12 head architectures
    - Fixed: M+C+T1, 251min, L2
    - Goal: Find best augmentation + head combination

  Group E — Context Window Exploration (96 experiments)
    - 12 context sizes x 2 channel configs x 4 classifiers
    - Requires re-extraction per context window
    - Goal: Confirm optimal context window for neural heads

  GPU 2: Group F + G — Layer Fusion + Training Hyperparams (~170 experiments)

  Group F — Layer x Fusion Strategy (80 experiments)
    - 6 single layers x 4 heads + concat + fusion + attention
    - Fixed: M+C+T1, 251min
    - Goal: Optimal layer utilization strategy

  Group G — Training Hyperparameters (104 experiments)
    - LR x WD grid + optimizer/scheduler + label smoothing + epoch/patience
    - Batch size sweep + BS×LR interaction
    - Fixed: best defaults
    - Goal: Fine-tune training recipe

  GPU 3: Group H + I — Augmentation Deep Dive + TTA/Ensemble (~138 experiments)

  Group H — Augmentation Deep Dive (70 experiments)
    - DC/SMOTE/FroFA/AdaptNoise/Mixup param grids + combined strategies
    - Goal: Systematic augmentation optimization

  Group I — TTA + Ensemble + Final (68 experiments)
    - TTA sweep + multi-seed ensemble + hybrid + sklearn baseline
    - Goal: Inference-time boosting + final comparison vs SVM

Total: ~538 experiments across 3 GPU servers (6 groups).

Usage:
  cd examples/classification/apc_occupancy

  # Run specific group
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --group D --device cuda

  # Run GPU-1 groups (D+E)
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --group D E --device cuda

  # Run GPU-2 groups (F+G)
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --group F G --device cuda

  # Run GPU-3 groups (H+I)
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --group H I --device cuda

  # Run all groups sequentially (single server, ~12-24h)
  python training/run_phase2_sweep.py \\
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
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

ALL_LAYERS = [0, 1, 2, 3, 4, 5]

# ============================================================================
# Channel Map (verified available channels from Phase 1.5)
# ============================================================================

CHANNEL_MAP = {
    "M": "d620900d_motionSensor",
    "C": "408981c2_contactSensor",
    "T1": "d620900d_temperatureMeasurement",
    "T2": "ccea734e_temperatureMeasurement",
    "P": "f2e891c6_powerMeter",
}


def _resolve_channels(keys: list[str]) -> list[str]:
    """Convert short keys (M, C, T1, ...) to full channel names."""
    return [CHANNEL_MAP[k] for k in keys]


def _ch_label(keys: list[str]) -> str:
    """Generate human-readable channel label from keys."""
    return "+".join(keys)


# ============================================================================
# Training config (extended for Phase 2)
# ============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters for a single neural head."""
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    early_stopping_patience: int = 30
    augmentation: dict | None = None
    device: str = "cpu"
    # Phase 2 extensions
    optimizer: str = "adamw"       # adamw, adam, sgd, rmsprop
    scheduler: str = "cosine"      # cosine, step, onecycle, plateau, none
    grad_clip: float = 0.0         # 0 = disabled
    class_weight: list[float] | None = None
    batch_size: int = 0            # 0 = full-batch, >0 = mini-batch with DataLoader


# ============================================================================
# Model + Embedding utilities
# ============================================================================

def load_mantis_model(pretrained_name: str, layer: int, output_token: str, device: str):
    """Load MantisV2 model + trainer wrapper."""
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    network = MantisV2(
        device=device,
        return_transf_layer=layer,
        output_token=output_token,
    )
    network = network.from_pretrained(pretrained_name)
    model = MantisTrainer(device=device, network=network)
    return model


def extract_embeddings(model, dataset: OccupancyDataset, device: str) -> np.ndarray:
    """Extract frozen embeddings for all windows in the dataset.

    Returns shape (n_windows, n_channels * embed_dim).
    """
    X, _ = dataset.get_numpy_arrays()  # (N, C, L)
    n_samples, n_channels, seq_len = X.shape

    # MantisV2 Channel Independence: extract per-channel, then concatenate
    all_embeddings = []
    for ch in range(n_channels):
        X_ch = X[:, [ch], :]  # (N, 1, L)
        Z_ch = model.transform(X_ch)  # (N, D)
        all_embeddings.append(Z_ch)

    Z = np.concatenate(all_embeddings, axis=-1)  # (N, C*D)

    # NaN safety
    if np.isnan(Z).any():
        n_nan = np.isnan(Z).sum()
        logger.warning("Found %d NaN values in embeddings, replacing with 0", n_nan)
        Z = np.nan_to_num(Z, nan=0.0)

    return Z


# ============================================================================
# Neural training loop (extended)
# ============================================================================

def _build_optimizer(params, config: TrainConfig):
    """Build optimizer from config."""
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    elif config.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=config.lr, weight_decay=config.weight_decay)
    else:  # adamw (default)
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def _build_scheduler(optimizer, config: TrainConfig, steps_per_epoch: int = 1):
    """Build scheduler from config.

    For OneCycleLR, total_steps = epochs * steps_per_epoch.
    """
    total_steps = config.epochs * steps_per_epoch
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif config.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, total_steps=total_steps,
        )
    elif config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5,
        )
    elif config.scheduler == "none":
        return None
    else:  # cosine (default)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)


def train_head(
    head: nn.Module,
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainConfig,
    n_classes: int,
    Z_train_std: torch.Tensor | None = None,
) -> nn.Module:
    """Train a classification head on embedding data.

    Supports both full-batch and mini-batch training.

    Key improvements over v1:
      - Mini-batch training via DataLoader (batch_size > 0)
      - Best model state dict save/restore
      - No hard-coded loss floor (removed loss < 0.01 early exit)
      - Correct OneCycleLR total_steps for mini-batch
      - Per-step scheduler for OneCycleLR, per-epoch for others
    """
    device = torch.device(config.device)
    head = head.to(device)
    head.train()

    Z_train = Z_train.to(device)
    y_train = y_train.to(device)
    if Z_train_std is not None:
        Z_train_std = Z_train_std.to(device)

    n_samples = len(Z_train)
    use_minibatch = config.batch_size > 0 and n_samples > config.batch_size
    batch_size = config.batch_size if use_minibatch else n_samples
    steps_per_epoch = math.ceil(n_samples / batch_size) if use_minibatch else 1

    # Optimizer + scheduler
    optimizer = _build_optimizer(head.parameters(), config)
    scheduler = _build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
    is_per_step_scheduler = config.scheduler == "onecycle"

    # Loss function
    aug_cfg = config.augmentation or {}
    strategy = aug_cfg.get("strategy", "")
    use_soft_labels = (
        strategy not in ("frofa", "adaptive_noise", "within_class_mixup", "")
        or aug_cfg.get("mixup_alpha", 0) > 0
    )

    if use_soft_labels:
        def loss_fn(logits, targets):
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            return -(targets * log_probs).sum(dim=1).mean()
    else:
        weight = None
        if config.class_weight is not None:
            weight = torch.tensor(config.class_weight, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=config.label_smoothing,
        )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Apply epoch-level augmentation (operates on full training set)
        if config.augmentation is not None:
            Z_epoch, y_epoch = apply_augmentation(
                Z_train, y_train, config.augmentation,
                Z_train_std=Z_train_std,
            )
        else:
            Z_epoch, y_epoch = Z_train, y_train

        if use_minibatch:
            # Mini-batch training with shuffle
            perm = torch.randperm(len(Z_epoch), device=device)
            epoch_loss_sum = 0.0
            n_batches = 0

            for start in range(0, len(Z_epoch), batch_size):
                idx = perm[start : start + batch_size]
                Z_batch = Z_epoch[idx]
                y_batch = y_epoch[idx]

                logits = head(Z_batch)
                loss = loss_fn(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), config.grad_clip)
                optimizer.step()

                if is_per_step_scheduler and scheduler is not None:
                    scheduler.step()

                epoch_loss_sum += loss.item() * len(idx)
                n_batches += 1

            epoch_loss = epoch_loss_sum / len(Z_epoch)
        else:
            # Full-batch training
            logits = head(Z_epoch)
            loss = loss_fn(logits, y_epoch)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss = loss.item()

        # Per-epoch scheduler step (except OneCycleLR which steps per-batch)
        if not is_per_step_scheduler and scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        # Early stopping with best model checkpointing
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break

    # Restore best model
    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    return head


# ============================================================================
# Train/test evaluation functions
# ============================================================================

def run_neural_train_test(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    pretrain_aug_config: dict | None = None,
    epoch_aug_config: dict | None = None,
    seed: int = 42,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Train neural head and evaluate on test set."""
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

    # Standard scaling
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train_aug)
    Z_test_s = scaler.transform(Z_test)

    # Convert to tensors
    Z_train_t = torch.from_numpy(Z_train_s).float()
    y_train_t = torch.from_numpy(y_train_aug).long()
    Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

    Z_train_std = Z_train_t.std(dim=0)

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

    # Train fresh head
    torch.manual_seed(seed)
    head = head_factory()
    head = train_head(head, Z_train_t, y_train_t, tc, n_classes, Z_train_std)

    # Predict
    with torch.no_grad():
        logits = head(Z_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = probs.argmax(axis=1).astype(np.int64)
    y_prob = probs[:, 1] if n_classes == 2 else probs

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


def run_neural_train_test_multi_layer(
    all_Z_train: dict[int, np.ndarray],
    all_Z_test: dict[int, np.ndarray],
    layer_indices: list[int],
    y_train: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    seed: int = 42,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Train/test with multi-layer fusion heads.

    Uses TrainConfig optimizer/scheduler settings (no longer hardcoded).
    Best model state dict is saved and restored after training.
    """
    from sklearn.preprocessing import StandardScaler

    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)

    # Per-layer scaling
    train_tensors = []
    test_tensors = []
    for li in layer_indices:
        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(all_Z_train[li])
        Z_test_s = scaler.transform(all_Z_test[li])
        train_tensors.append(torch.from_numpy(Z_train_s).float())
        test_tensors.append(torch.from_numpy(Z_test_s).float().to(device))

    y_train_t = torch.from_numpy(y_train).long()

    # Train
    torch.manual_seed(seed)
    head = head_factory()
    head = head.to(device)
    head.train()

    optimizer = _build_optimizer(head.parameters(), train_config)
    scheduler = _build_scheduler(optimizer, train_config, steps_per_epoch=1)
    is_per_step_scheduler = train_config.scheduler == "onecycle"

    loss_fn = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)

    train_inputs = [t.to(device) for t in train_tensors]
    y_train_dev = y_train_t.to(device)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(train_config.epochs):
        logits = head(train_inputs)
        loss = loss_fn(logits, y_train_dev)
        optimizer.zero_grad()
        loss.backward()

        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), train_config.grad_clip)

        optimizer.step()

        if scheduler is not None:
            if train_config.scheduler == "plateau":
                scheduler.step(loss.item())
            else:
                scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss - 1e-5:
            best_loss = loss_val
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= train_config.early_stopping_patience:
            break

    # Restore best model
    if best_state is not None:
        head.load_state_dict(best_state)

    # Predict
    head.eval()
    with torch.no_grad():
        logits = head(test_tensors)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = probs.argmax(axis=1).astype(np.int64)
    y_prob = probs[:, 1] if n_classes == 2 else probs

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


def run_tta_train_test(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    tta_k: int = 5,
    tta_strategy: str = "frofa",
    tta_strength: float = 0.1,
    pretrain_aug_config: dict | None = None,
    seed: int = 42,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Test-Time Augmentation: average softmax over K augmented copies."""
    from sklearn.preprocessing import StandardScaler
    from training.augmentation import frofa_augmentation, adaptive_noise

    rng = np.random.default_rng(seed)
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)

    # Pre-training augmentation
    if pretrain_aug_config is not None:
        Z_train_aug, y_train_aug = apply_pretrain_augmentation(
            Z_train, y_train, pretrain_aug_config, rng,
        )
    else:
        Z_train_aug, y_train_aug = Z_train, y_train

    # Standard scaling
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train_aug)
    Z_test_s = scaler.transform(Z_test)

    Z_train_t = torch.from_numpy(Z_train_s).float()
    y_train_t = torch.from_numpy(y_train_aug).long()
    Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

    Z_train_std = Z_train_t.std(dim=0)

    # Train head
    torch.manual_seed(seed)
    head = head_factory()
    head = train_head(head, Z_train_t, y_train_t, train_config, n_classes, Z_train_std)

    # TTA: average predictions over K augmented copies
    n_test = len(y_test)
    prob_sum = np.zeros((n_test, n_classes), dtype=np.float64)

    with torch.no_grad():
        # Original prediction
        logits = head(Z_test_t)
        prob_sum += torch.softmax(logits, dim=1).cpu().numpy()

        # K augmented predictions
        for k in range(tta_k):
            gen = torch.Generator(device=Z_test_t.device)
            gen.manual_seed(seed + k + 1)

            if tta_strategy == "frofa":
                Z_aug = frofa_augmentation(
                    Z_test_t, strength=tta_strength, generator=gen,
                    Z_train_std=Z_train_std.to(device),
                )
            elif tta_strategy == "adaptive_noise":
                Z_aug = adaptive_noise(
                    Z_test_t, Z_train_std.to(device),
                    scale=tta_strength, generator=gen,
                )
            else:
                Z_aug = Z_test_t

            logits_aug = head(Z_aug)
            prob_sum += torch.softmax(logits_aug, dim=1).cpu().numpy()

    # Average over 1 + K predictions
    probs_avg = prob_sum / (1 + tta_k)
    y_pred = probs_avg.argmax(axis=1).astype(np.int64)
    y_prob = probs_avg[:, 1] if n_classes == 2 else probs_avg

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


def run_ensemble_train_test(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    n_seeds: int = 5,
    base_seed: int = 42,
    pretrain_aug_config: dict | None = None,
    epoch_aug_config: dict | None = None,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Multi-seed ensemble: average softmax across seeds."""
    from sklearn.preprocessing import StandardScaler

    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)
    n_test = len(y_test)
    prob_sum = np.zeros((n_test, n_classes), dtype=np.float64)

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        rng = np.random.default_rng(seed)

        if pretrain_aug_config is not None:
            Z_train_aug, y_train_aug = apply_pretrain_augmentation(
                Z_train, y_train, pretrain_aug_config, rng,
            )
        else:
            Z_train_aug, y_train_aug = Z_train, y_train

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train_aug)
        Z_test_s = scaler.transform(Z_test)

        Z_train_t = torch.from_numpy(Z_train_s).float()
        y_train_t = torch.from_numpy(y_train_aug).long()
        Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

        Z_train_std = Z_train_t.std(dim=0)

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
        )

        torch.manual_seed(seed)
        head = head_factory()
        head = train_head(head, Z_train_t, y_train_t, tc, n_classes, Z_train_std)

        with torch.no_grad():
            logits = head(Z_test_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        prob_sum += probs

    probs_avg = prob_sum / n_seeds
    y_pred = probs_avg.argmax(axis=1).astype(np.int64)
    y_prob = probs_avg[:, 1] if n_classes == 2 else probs_avg

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


# ============================================================================
# Sklearn baseline
# ============================================================================

def build_sklearn_classifier(config: dict):
    """Build sklearn classifier from config dict."""
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, AdaBoostClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    clf_type = config["type"]
    if clf_type == "svm":
        return SVC(
            kernel=config.get("kernel", "rbf"),
            C=config.get("C", 1.0),
            gamma=config.get("gamma", "scale"),
            probability=True,
            random_state=42,
        )
    elif clf_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif clf_type == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif clf_type == "extra_trees":
        return ExtraTreesClassifier(n_estimators=100, random_state=42)
    elif clf_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif clf_type == "adaboost":
        return AdaBoostClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


def run_sklearn_train_test(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    clf_config: dict,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Run sklearn classifier with train/test split."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Z_train)
    Z_te = scaler.transform(Z_test)

    clf = build_sklearn_classifier(clf_config)
    clf.fit(Z_tr, y_train)

    y_pred = clf.predict(Z_te)

    # Probability extraction
    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob_full = clf.predict_proba(Z_te)
        if hasattr(clf, "classes_") and len(clf.classes_) == 2:
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            y_prob = y_prob_full[:, pos_idx]
        elif y_prob_full.shape[1] >= 2:
            y_prob = y_prob_full[:, 1]
    elif hasattr(clf, "decision_function"):
        y_prob = clf.decision_function(Z_te)

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


# ============================================================================
# Result helpers
# ============================================================================

def _make_result_row(
    name: str, metrics: ClassificationMetrics, elapsed: float,
    extra: dict | None = None,
) -> dict:
    """Build a standardized result dict (AUC-primary)."""
    auc_val = metrics.roc_auc if not math.isnan(metrics.roc_auc) else None
    eer_val = metrics.eer if not math.isnan(metrics.eer) else None
    row = {
        "name": name,
        "auc": round(auc_val, 4) if auc_val is not None else None,
        "eer": round(eer_val, 4) if eer_val is not None else None,
        "accuracy": round(metrics.accuracy, 4),
        "f1": round(metrics.f1, 4),
        "f1_macro": round(metrics.f1_macro, 4),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "ci_lower": round(metrics.ci_lower, 4) if metrics.ci_lower is not None else None,
        "ci_upper": round(metrics.ci_upper, 4) if metrics.ci_upper is not None else None,
        "n_samples": metrics.n_samples,
        "time_s": round(elapsed, 1),
    }
    if extra:
        row.update(extra)
    return row


def _format_metrics_log(name: str, metrics: ClassificationMetrics, elapsed: float) -> str:
    """Format comprehensive metric log line."""
    parts = [f"  [{name}]"]
    if not math.isnan(metrics.roc_auc):
        parts.append(f"AUC={metrics.roc_auc:.4f}")
    else:
        parts.append("AUC=N/A")
    if not math.isnan(metrics.eer):
        parts.append(f"EER={metrics.eer:.4f}")
    parts.append(f"Acc={metrics.accuracy:.4f}")
    parts.append(f"F1={metrics.f1:.4f}")
    parts.append(f"P={metrics.precision:.4f}")
    parts.append(f"R={metrics.recall:.4f}")
    if metrics.ci_lower is not None:
        parts.append(f"CI=[{metrics.ci_lower:.4f},{metrics.ci_upper:.4f}]")
    if metrics.confusion_matrix is not None and metrics.confusion_matrix.shape == (2, 2):
        tn, fp, fn, tp = metrics.confusion_matrix.ravel()
        parts.append(f"(TP={tp} TN={tn} FP={fp} FN={fn})")
    parts.append(f"({elapsed:.1f}s)")
    return "  ".join(parts)


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
        logger.error("  [%s] FAILED: %s", name, e)
        row = {"name": name, "error": str(e)}
        if extra:
            row.update(extra)
        return row


# ============================================================================
# Group D: Augmentation x Head Architecture (120 experiments)
# ============================================================================

AUGMENTATION_CONFIGS_D = [
    {"name": "no_aug", "pretrain_aug": None, "epoch_aug": None},
    {"name": "DC_a05_n50", "pretrain_aug": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch_aug": None},
    {"name": "DC_a03_n100", "pretrain_aug": {"strategy": "dc", "alpha": 0.3, "n_synthetic": 100}, "epoch_aug": None},
    {"name": "DC_a07_n30", "pretrain_aug": {"strategy": "dc", "alpha": 0.7, "n_synthetic": 30}, "epoch_aug": None},
    {"name": "SMOTE_k5_n30", "pretrain_aug": {"strategy": "smote", "k": 5, "n_synthetic": 30}, "epoch_aug": None},
    {"name": "SMOTE_k3_n50", "pretrain_aug": {"strategy": "smote", "k": 3, "n_synthetic": 50}, "epoch_aug": None},
    {"name": "FroFA_s01", "pretrain_aug": None, "epoch_aug": {"strategy": "frofa", "strength": 0.1}},
    {"name": "FroFA_s005", "pretrain_aug": None, "epoch_aug": {"strategy": "frofa", "strength": 0.05}},
    {"name": "AdaptNoise_s01", "pretrain_aug": None, "epoch_aug": {"strategy": "adaptive_noise", "scale": 0.1}},
    {"name": "WCMixup_a03", "pretrain_aug": None, "epoch_aug": {"strategy": "within_class_mixup", "alpha": 0.3}},
]

HEAD_CONFIGS_D = [
    {"name": "Linear", "type": "linear", "kwargs": {}},
    {"name": "MLP[32]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [32], "dropout": 0.5}},
    {"name": "MLP[64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}},
    {"name": "MLP[128]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.5}},
    {"name": "MLP[256]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [256], "dropout": 0.5}},
    {"name": "MLP[64,32]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64, 32], "dropout": 0.5}},
    {"name": "MLP[128,64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}},
    {"name": "MLP[256,128]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [256, 128], "dropout": 0.5}},
    {"name": "MLP[64]-d0.3", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.3}},
    {"name": "MLP[64]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.7}},
    {"name": "MLP[128]-d0.3", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.3}},
    {"name": "MLP[128]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.7}},
]


def run_group_d(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Sweep augmentation x head architecture.

    10 augmentations x 12 heads = 120 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP D: Augmentation x Head Architecture (120 experiments)")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    total = len(AUGMENTATION_CONFIGS_D) * len(HEAD_CONFIGS_D)

    for i, aug_cfg in enumerate(AUGMENTATION_CONFIGS_D):
        for j, head_cfg in enumerate(HEAD_CONFIGS_D):
            exp_idx = i * len(HEAD_CONFIGS_D) + j + 1
            exp_name = f"{aug_cfg['name']}|{head_cfg['name']}"
            logger.info("[%d/%d] %s", exp_idx, total, exp_name)

            def head_factory(hc=head_cfg):
                return build_head(hc["type"], embed_dim, n_classes, **hc["kwargs"])

            row = _run_and_log(
                exp_name,
                lambda ac=aug_cfg, hf=head_factory: run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    hf, train_config,
                    pretrain_aug_config=ac["pretrain_aug"],
                    epoch_aug_config=ac["epoch_aug"],
                    seed=seed,
                ),
                extra={
                    "group": "D", "augmentation": aug_cfg["name"],
                    "head": head_cfg["name"], "head_type": head_cfg["type"],
                },
            )
            results.append(row)

    _save_group_results(results, "D", "group_d_aug_head", output_dir)
    return results


# ============================================================================
# Group E: Context Window Exploration (96 experiments)
# ============================================================================

# Context configs: symmetric up to 100+1+100, then asymmetric with future=100.
# MAX_CONTEXT_AFTER = 100 enforced in DatasetConfig.__post_init__.
GROUP_E_CONTEXTS = [
    {"name": "3min", "before": 1, "after": 1},
    {"name": "11min", "before": 5, "after": 5},
    {"name": "21min", "before": 10, "after": 10},
    {"name": "41min", "before": 20, "after": 20},
    {"name": "61min", "before": 30, "after": 30},
    {"name": "91min", "before": 45, "after": 45},
    {"name": "121min", "before": 60, "after": 60},
    {"name": "181min", "before": 90, "after": 90},
    {"name": "201min", "before": 100, "after": 100},
    # Asymmetric: extend past while keeping future at max (100)
    {"name": "221min_asym", "before": 120, "after": 100},
    {"name": "341min_asym", "before": 240, "after": 100},
    {"name": "461min_asym", "before": 360, "after": 100},
]

GROUP_E_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
]

GROUP_E_CLASSIFIERS = [
    {"name": "MLP[64]-d0.5", "type": "neural", "head_type": "mlp", "head_kwargs": {"hidden_dims": [64], "dropout": 0.5}},
    {"name": "Linear", "type": "neural", "head_type": "linear", "head_kwargs": {}},
    {"name": "SVM_rbf", "type": "sklearn", "clf_config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "LogReg", "type": "sklearn", "clf_config": {"type": "logistic_regression"}},
]


def run_group_e(
    train_csv: str,
    test_csv: str,
    label_column: str,
    split_date: str | None,
    stride: int,
    pretrained: str,
    output_token: str,
    primary_layer: int,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Sweep context windows with neural + sklearn classifiers.

    12 contexts x 2 channels x 4 classifiers = 96 experiments.
    Uses unified sensor array — context windows can cross split boundary.
    """
    logger.info("=" * 70)
    logger.info("GROUP E: Context Window Exploration (96 experiments)")
    logger.info("=" * 70)

    device = train_config.device
    total = len(GROUP_E_CONTEXTS) * len(GROUP_E_CHANNELS) * len(GROUP_E_CLASSIFIERS)

    # Load MantisV2 model once
    logger.info("Loading MantisV2 model (L%d)...", primary_layer)
    model = load_mantis_model(pretrained, primary_layer, output_token, device)

    results = []
    exp_idx = 0

    for ch_cfg in GROUP_E_CHANNELS:
        # Load unified sensor data for this channel set
        channels = _resolve_channels(ch_cfg["keys"])
        logger.info("Loading unified data for channels: %s", ch_cfg["name"])

        sensor, train_labels, test_labels, ch_names, timestamps = load_unified_split(
            train_csv, test_csv,
            label_column=label_column, channels=channels,
            split_date=split_date,
        )

        for ctx_cfg in GROUP_E_CONTEXTS:
            # Both datasets share the unified sensor array — context windows
            # can cross the split boundary (sensor overlap allowed).
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
                logger.warning("Skipping %s/%s: train=%d test=%d",
                               ch_cfg["name"], ctx_cfg["name"], n_train, n_test)
                continue

            # Extract embeddings separately for train and test
            Z_train = extract_embeddings(model, train_dataset, device)
            Z_test = extract_embeddings(model, test_dataset, device)
            y_train = train_dataset.labels
            y_test = test_dataset.labels
            embed_dim = Z_train.shape[1]
            n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))

            for clf_cfg in GROUP_E_CLASSIFIERS:
                exp_idx += 1
                exp_name = f"{ch_cfg['name']}|{ctx_cfg['name']}|{clf_cfg['name']}"
                logger.info("[%d/%d] %s (train=%d test=%d)", exp_idx, total, exp_name, n_train, n_test)

                if clf_cfg["type"] == "neural":
                    htype = clf_cfg["head_type"]
                    hkw = clf_cfg["head_kwargs"]

                    def head_factory(ht=htype, kw=hkw, ed=embed_dim, nc=n_classes):
                        return build_head(ht, ed, nc, **kw)

                    row = _run_and_log(
                        exp_name,
                        lambda hf=head_factory: run_neural_train_test(
                            Z_train, y_train, Z_test, y_test,
                            hf, train_config, seed=seed,
                        ),
                        extra={
                            "group": "E", "channels": ch_cfg["name"],
                            "context": ctx_cfg["name"],
                            "context_min": ctx_cfg["before"] * 2 + 1,
                            "classifier": clf_cfg["name"],
                        },
                    )
                else:
                    row = _run_and_log(
                        exp_name,
                        lambda cc=clf_cfg["clf_config"]: run_sklearn_train_test(
                            Z_train, y_train, Z_test, y_test, cc,
                        ),
                        extra={
                            "group": "E", "channels": ch_cfg["name"],
                            "context": ctx_cfg["name"],
                            "context_min": ctx_cfg["before"] * 2 + 1,
                            "classifier": clf_cfg["name"],
                        },
                    )
                results.append(row)

            gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _save_group_results(results, "E", "group_e_context", output_dir)
    return results


# ============================================================================
# Group F: Layer x Fusion Strategy (80 experiments)
# ============================================================================

SINGLE_LAYER_HEADS = [
    {"name": "MLP[64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}},
    {"name": "MLP[128]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.5}},
    {"name": "Linear", "type": "linear", "kwargs": {}},
    {"name": "SVM_rbf", "type": "sklearn", "clf_config": {"type": "svm", "kernel": "rbf"}},
]

CONCAT_COMBOS = [
    {"name": "L0+L2", "layers": [0, 2]},
    {"name": "L0+L5", "layers": [0, 5]},
    {"name": "L2+L3", "layers": [2, 3]},
    {"name": "L2+L5", "layers": [2, 5]},
    {"name": "L0+L2+L5", "layers": [0, 2, 5]},
    {"name": "L0+L3+L5", "layers": [0, 3, 5]},
    {"name": "L2+L3+L4", "layers": [2, 3, 4]},
    {"name": "L1+L3+L5", "layers": [1, 3, 5]},
    {"name": "L0+L2+L3+L5", "layers": [0, 2, 3, 5]},
    {"name": "L0+L2+L4+L5", "layers": [0, 2, 4, 5]},
    {"name": "L2+L3+L4+L5", "layers": [2, 3, 4, 5]},
    {"name": "All_L0-L5", "layers": [0, 1, 2, 3, 4, 5]},
]

FUSION_COMBOS = [
    {"name": "Fusion_L2+L3", "layers": [2, 3]},
    {"name": "Fusion_L2+L5", "layers": [2, 5]},
    {"name": "Fusion_L0+L3+L5", "layers": [0, 3, 5]},
    {"name": "Fusion_L2+L3+L4", "layers": [2, 3, 4]},
    {"name": "Fusion_L0+L2+L5", "layers": [0, 2, 5]},
    {"name": "Fusion_L1+L3+L5", "layers": [1, 3, 5]},
    {"name": "Fusion_L0+L2+L4+L5", "layers": [0, 2, 4, 5]},
    {"name": "Fusion_All", "layers": [0, 1, 2, 3, 4, 5]},
]

ATTENTION_COMBOS = [
    {"name": "Attn_L2+L3", "layers": [2, 3]},
    {"name": "Attn_L2+L5", "layers": [2, 5]},
    {"name": "Attn_L0+L3+L5", "layers": [0, 3, 5]},
    {"name": "Attn_L2+L3+L4", "layers": [2, 3, 4]},
    {"name": "Attn_L0+L2+L5", "layers": [0, 2, 5]},
    {"name": "Attn_L0+L2+L4+L5", "layers": [0, 2, 4, 5]},
    {"name": "Attn_L3+L4+L5", "layers": [3, 4, 5]},
    {"name": "Attn_All", "layers": [0, 1, 2, 3, 4, 5]},
]


def run_group_f(
    all_Z_train: dict[int, np.ndarray],
    all_Z_test: dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Sweep layer combinations: single, concat, fusion, attention.

    ~80 experiments total.
    """
    logger.info("=" * 70)
    logger.info("GROUP F: Layer x Fusion Strategy (~80 experiments)")
    logger.info("=" * 70)

    embed_dim = next(iter(all_Z_train.values())).shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    exp_idx = 0

    # Part F1: Single layers x 4 heads = 24
    logger.info("--- Part F1: Single Layer x Head (24 experiments) ---")
    for layer in ALL_LAYERS:
        Z_tr = all_Z_train[layer]
        Z_te = all_Z_test[layer]
        for head_cfg in SINGLE_LAYER_HEADS:
            exp_idx += 1
            exp_name = f"L{layer}|{head_cfg['name']}"
            logger.info("[%d] %s", exp_idx, exp_name)

            if head_cfg["type"] == "sklearn":
                row = _run_and_log(
                    exp_name,
                    lambda zt=Z_tr, ze=Z_te, cc=head_cfg["clf_config"]: (
                        run_sklearn_train_test(zt, y_train, ze, y_test, cc)
                    ),
                    extra={"group": "F", "subgroup": "single", "layer": layer, "head": head_cfg["name"]},
                )
            else:
                def head_factory(hc=head_cfg, ed=embed_dim):
                    return build_head(hc["type"], ed, n_classes, **hc["kwargs"])

                row = _run_and_log(
                    exp_name,
                    lambda zt=Z_tr, ze=Z_te, hf=head_factory: run_neural_train_test(
                        zt, y_train, ze, y_test, hf, train_config, seed=seed,
                    ),
                    extra={"group": "F", "subgroup": "single", "layer": layer, "head": head_cfg["name"]},
                )
            results.append(row)

    # Part F2: Concat combos x 2 heads = 24
    logger.info("--- Part F2: Concatenation (24 experiments) ---")
    for combo in CONCAT_COMBOS:
        Z_tr = np.concatenate([all_Z_train[l] for l in combo["layers"]], axis=1)
        Z_te = np.concatenate([all_Z_test[l] for l in combo["layers"]], axis=1)
        concat_dim = Z_tr.shape[1]

        for hidden in [[min(256, concat_dim // 2), 64], [128, 64]]:
            exp_idx += 1
            h_str = "x".join(str(h) for h in hidden)
            exp_name = f"Concat_{combo['name']}|MLP[{h_str}]"
            logger.info("[%d] %s (dim=%d)", exp_idx, exp_name, concat_dim)

            def head_factory(cd=concat_dim, hd=hidden):
                return build_head("mlp", cd, n_classes, hidden_dims=hd, dropout=0.5)

            row = _run_and_log(
                exp_name,
                lambda zt=Z_tr, ze=Z_te, hf=head_factory: run_neural_train_test(
                    zt, y_train, ze, y_test, hf, train_config, seed=seed,
                ),
                extra={"group": "F", "subgroup": "concat", "layers": combo["layers"]},
            )
            results.append(row)

    # Part F3: Fusion combos = 8
    logger.info("--- Part F3: Learnable Fusion (8 experiments) ---")
    for combo in FUSION_COMBOS:
        exp_idx += 1
        exp_name = combo["name"]
        logger.info("[%d] %s", exp_idx, exp_name)
        nl = len(combo["layers"])

        def head_factory(n_layers=nl):
            return build_head(
                "multi_layer_fusion", embed_dim, n_classes,
                n_layers=n_layers, hidden_dims=[64], dropout=0.5,
            )

        row = _run_and_log(
            exp_name,
            lambda li=combo["layers"], hf=head_factory: run_neural_train_test_multi_layer(
                all_Z_train, all_Z_test, li, y_train, y_test,
                hf, train_config, seed,
            ),
            extra={"group": "F", "subgroup": "fusion", "layers": combo["layers"]},
        )
        results.append(row)

    # Part F4: Attention pool = 8
    logger.info("--- Part F4: Attention Pool (8 experiments) ---")
    for combo in ATTENTION_COMBOS:
        exp_idx += 1
        exp_name = combo["name"]
        logger.info("[%d] %s", exp_idx, exp_name)
        nl = len(combo["layers"])

        def head_factory(n_layers=nl):
            return build_head(
                "attention_pool", embed_dim, n_classes,
                n_layers=n_layers, hidden_dims=[64], dropout=0.5,
            )

        row = _run_and_log(
            exp_name,
            lambda li=combo["layers"], hf=head_factory: run_neural_train_test_multi_layer(
                all_Z_train, all_Z_test, li, y_train, y_test,
                hf, train_config, seed,
            ),
            extra={"group": "F", "subgroup": "attention", "layers": combo["layers"]},
        )
        results.append(row)

    _save_group_results(results, "F", "group_f_layer_fusion", output_dir)
    return results


# ============================================================================
# Group G: Training Hyperparameters (90 experiments)
# ============================================================================

def run_group_g(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    base_train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Sweep training hyperparameters: LR, WD, optimizer, scheduler, batch size, etc.

    ~104 experiments (40+15+15+6+8+4+2+8+6).
    """
    logger.info("=" * 70)
    logger.info("GROUP G: Training Hyperparameters (~104 experiments)")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    exp_idx = 0

    def default_head_factory():
        return build_head("mlp", embed_dim, n_classes, hidden_dims=[64], dropout=0.5)

    # Part G1: LR x WD grid = 40
    logger.info("--- Part G1: LR x WD grid (40 experiments) ---")
    lr_values = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    wd_values = [0.0, 0.001, 0.01, 0.05, 0.1]

    for lr in lr_values:
        for wd in wd_values:
            exp_idx += 1
            exp_name = f"LR={lr:.0e}_WD={wd}"
            logger.info("[%d] %s", exp_idx, exp_name)

            tc = TrainConfig(
                epochs=base_train_config.epochs,
                lr=lr, weight_decay=wd,
                label_smoothing=base_train_config.label_smoothing,
                early_stopping_patience=base_train_config.early_stopping_patience,
                device=base_train_config.device,
            )
            row = _run_and_log(
                exp_name,
                lambda tc_=tc: run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    default_head_factory, tc_, seed=seed,
                ),
                extra={"group": "G", "subgroup": "lr_wd", "lr": lr, "weight_decay": wd},
            )
            results.append(row)

    # Part G2: Optimizer comparison x 3 LRs = 15
    logger.info("--- Part G2: Optimizer comparison (15 experiments) ---")
    optimizers = ["adamw", "adam", "sgd", "rmsprop"]
    opt_lrs = [5e-4, 1e-3, 5e-3]
    # SGD needs higher LR typically
    for opt_name in optimizers:
        for lr in opt_lrs:
            exp_idx += 1
            exp_name = f"{opt_name}_LR={lr:.0e}"
            logger.info("[%d] %s", exp_idx, exp_name)

            tc = TrainConfig(
                epochs=base_train_config.epochs,
                lr=lr, weight_decay=0.01,
                early_stopping_patience=base_train_config.early_stopping_patience,
                device=base_train_config.device,
                optimizer=opt_name,
            )
            row = _run_and_log(
                exp_name,
                lambda tc_=tc: run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    default_head_factory, tc_, seed=seed,
                ),
                extra={"group": "G", "subgroup": "optimizer", "optimizer": opt_name, "lr": lr},
            )
            results.append(row)

    # Part G3: Scheduler comparison x 3 LRs = 15
    logger.info("--- Part G3: Scheduler comparison (15 experiments) ---")
    schedulers = ["cosine", "step", "onecycle", "plateau", "none"]
    for sched in schedulers:
        for lr in opt_lrs:
            exp_idx += 1
            exp_name = f"sched_{sched}_LR={lr:.0e}"
            logger.info("[%d] %s", exp_idx, exp_name)

            tc = TrainConfig(
                epochs=base_train_config.epochs,
                lr=lr, weight_decay=0.01,
                early_stopping_patience=base_train_config.early_stopping_patience,
                device=base_train_config.device,
                scheduler=sched,
            )
            row = _run_and_log(
                exp_name,
                lambda tc_=tc: run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    default_head_factory, tc_, seed=seed,
                ),
                extra={"group": "G", "subgroup": "scheduler", "scheduler": sched, "lr": lr},
            )
            results.append(row)

    # Part G4: Label smoothing = 6
    logger.info("--- Part G4: Label smoothing (6 experiments) ---")
    for ls in [0.0, 0.01, 0.03, 0.05, 0.1, 0.15]:
        exp_idx += 1
        exp_name = f"LS={ls}"
        logger.info("[%d] %s", exp_idx, exp_name)

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=base_train_config.lr, weight_decay=base_train_config.weight_decay,
            label_smoothing=ls,
            early_stopping_patience=base_train_config.early_stopping_patience,
            device=base_train_config.device,
        )
        row = _run_and_log(
            exp_name,
            lambda tc_=tc: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, tc_, seed=seed,
            ),
            extra={"group": "G", "subgroup": "label_smoothing", "label_smoothing": ls},
        )
        results.append(row)

    # Part G5: Epoch/patience combos = 8
    logger.info("--- Part G5: Epoch/patience (8 experiments) ---")
    ep_configs = [
        (100, 15), (100, 30), (200, 20), (200, 30),
        (200, 50), (300, 30), (300, 50), (500, 50),
    ]
    for epochs, patience in ep_configs:
        exp_idx += 1
        exp_name = f"{epochs}ep_{patience}pat"
        logger.info("[%d] %s", exp_idx, exp_name)

        tc = TrainConfig(
            epochs=epochs,
            lr=base_train_config.lr, weight_decay=base_train_config.weight_decay,
            early_stopping_patience=patience,
            device=base_train_config.device,
        )
        row = _run_and_log(
            exp_name,
            lambda tc_=tc: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, tc_, seed=seed,
            ),
            extra={"group": "G", "subgroup": "ep_patience", "epochs": epochs, "patience": patience},
        )
        results.append(row)

    # Part G6: Grad clipping = 4
    logger.info("--- Part G6: Gradient clipping (4 experiments) ---")
    for gc_val in [0.5, 1.0, 5.0, 10.0]:
        exp_idx += 1
        exp_name = f"GradClip={gc_val}"
        logger.info("[%d] %s", exp_idx, exp_name)

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=base_train_config.lr, weight_decay=base_train_config.weight_decay,
            early_stopping_patience=base_train_config.early_stopping_patience,
            device=base_train_config.device,
            grad_clip=gc_val,
        )
        row = _run_and_log(
            exp_name,
            lambda tc_=tc: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, tc_, seed=seed,
            ),
            extra={"group": "G", "subgroup": "grad_clip", "grad_clip": gc_val},
        )
        results.append(row)

    # Part G7: BN vs no-BN = 2
    logger.info("--- Part G7: BatchNorm ablation (2 experiments) ---")
    for use_bn, bn_name in [(True, "BN"), (False, "noBN")]:
        exp_idx += 1
        exp_name = f"MLP[64]-d0.5_{bn_name}"
        logger.info("[%d] %s", exp_idx, exp_name)

        def head_factory(bn=use_bn):
            return build_head("mlp", embed_dim, n_classes,
                              hidden_dims=[64], dropout=0.5, use_batchnorm=bn)

        row = _run_and_log(
            exp_name,
            lambda hf=head_factory: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                hf, base_train_config, seed=seed,
            ),
            extra={"group": "G", "subgroup": "batchnorm", "use_batchnorm": use_bn},
        )
        results.append(row)

    # Part G8: Batch size sweep = 8
    logger.info("--- Part G8: Batch size sweep (8 experiments) ---")
    # With ~8K-10K train samples, mini-batch adds gradient noise regularization
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 0]  # 0 = full-batch
    for bs in batch_sizes:
        exp_idx += 1
        bs_name = "full" if bs == 0 else str(bs)
        exp_name = f"BS={bs_name}"
        logger.info("[%d] %s", exp_idx, exp_name)

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=base_train_config.lr, weight_decay=base_train_config.weight_decay,
            early_stopping_patience=base_train_config.early_stopping_patience,
            device=base_train_config.device,
            batch_size=bs,
        )
        row = _run_and_log(
            exp_name,
            lambda tc_=tc: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, tc_, seed=seed,
            ),
            extra={"group": "G", "subgroup": "batch_size", "batch_size": bs},
        )
        results.append(row)

    # Part G9: Batch size + LR interaction = 6
    logger.info("--- Part G9: Batch size x LR interaction (6 experiments) ---")
    bs_lr_combos = [
        (64, 5e-4), (64, 1e-3), (64, 5e-3),
        (256, 5e-4), (256, 1e-3), (256, 5e-3),
    ]
    for bs, lr in bs_lr_combos:
        exp_idx += 1
        exp_name = f"BS={bs}_LR={lr:.0e}"
        logger.info("[%d] %s", exp_idx, exp_name)

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=lr, weight_decay=base_train_config.weight_decay,
            early_stopping_patience=base_train_config.early_stopping_patience,
            device=base_train_config.device,
            batch_size=bs,
        )
        row = _run_and_log(
            exp_name,
            lambda tc_=tc: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, tc_, seed=seed,
            ),
            extra={"group": "G", "subgroup": "bs_lr", "batch_size": bs, "lr": lr},
        )
        results.append(row)

    _save_group_results(results, "G", "group_g_training_hp", output_dir)
    return results


# ============================================================================
# Group H: Augmentation Deep Dive (70 experiments)
# ============================================================================

def run_group_h(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Systematic augmentation parameter optimization.

    ~70 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP H: Augmentation Deep Dive (~70 experiments)")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    exp_idx = 0

    def default_head_factory():
        return build_head("mlp", embed_dim, n_classes, hidden_dims=[64], dropout=0.5)

    # Part H1: DC param grid = 20
    logger.info("--- Part H1: Distribution Calibration grid (20 experiments) ---")
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for n_synth in [20, 50, 100, 200]:
            exp_idx += 1
            exp_name = f"DC_a{alpha}_n{n_synth}"
            logger.info("[%d] %s", exp_idx, exp_name)

            pretrain_aug = {"strategy": "dc", "alpha": alpha, "n_synthetic": n_synth}
            row = _run_and_log(
                exp_name,
                lambda pa=pretrain_aug: run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    default_head_factory, train_config,
                    pretrain_aug_config=pa, seed=seed,
                ),
                extra={"group": "H", "subgroup": "dc", "dc_alpha": alpha, "dc_n": n_synth},
            )
            results.append(row)

    # Part H2: SMOTE param grid = 9
    logger.info("--- Part H2: SMOTE grid (9 experiments) ---")
    for k in [3, 5, 7]:
        for n_synth in [20, 50, 100]:
            exp_idx += 1
            exp_name = f"SMOTE_k{k}_n{n_synth}"
            logger.info("[%d] %s", exp_idx, exp_name)

            pretrain_aug = {"strategy": "smote", "k": k, "n_synthetic": n_synth}
            row = _run_and_log(
                exp_name,
                lambda pa=pretrain_aug: run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    default_head_factory, train_config,
                    pretrain_aug_config=pa, seed=seed,
                ),
                extra={"group": "H", "subgroup": "smote", "smote_k": k, "smote_n": n_synth},
            )
            results.append(row)

    # Part H3: FroFA strength sweep = 7
    logger.info("--- Part H3: FroFA strength (7 experiments) ---")
    for s in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]:
        exp_idx += 1
        exp_name = f"FroFA_s{s}"
        logger.info("[%d] %s", exp_idx, exp_name)

        epoch_aug = {"strategy": "frofa", "strength": s}
        row = _run_and_log(
            exp_name,
            lambda ea=epoch_aug: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, train_config,
                epoch_aug_config=ea, seed=seed,
            ),
            extra={"group": "H", "subgroup": "frofa", "frofa_strength": s},
        )
        results.append(row)

    # Part H4: Adaptive noise scale = 6
    logger.info("--- Part H4: Adaptive noise (6 experiments) ---")
    for s in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        exp_idx += 1
        exp_name = f"AdaptNoise_s{s}"
        logger.info("[%d] %s", exp_idx, exp_name)

        epoch_aug = {"strategy": "adaptive_noise", "scale": s}
        row = _run_and_log(
            exp_name,
            lambda ea=epoch_aug: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, train_config,
                epoch_aug_config=ea, seed=seed,
            ),
            extra={"group": "H", "subgroup": "adaptive_noise", "noise_scale": s},
        )
        results.append(row)

    # Part H5: Within-class mixup alpha = 6
    logger.info("--- Part H5: Within-class mixup (6 experiments) ---")
    for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        exp_idx += 1
        exp_name = f"WCMixup_a{a}"
        logger.info("[%d] %s", exp_idx, exp_name)

        epoch_aug = {"strategy": "within_class_mixup", "alpha": a}
        row = _run_and_log(
            exp_name,
            lambda ea=epoch_aug: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, train_config,
                epoch_aug_config=ea, seed=seed,
            ),
            extra={"group": "H", "subgroup": "within_class_mixup", "wcm_alpha": a},
        )
        results.append(row)

    # Part H6: Combined strategies = 12
    logger.info("--- Part H6: Combined augmentation (12 experiments) ---")
    combined_configs = [
        {"name": "DC05+FroFA01", "pretrain": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch": {"strategy": "frofa", "strength": 0.1}},
        {"name": "DC03+FroFA01", "pretrain": {"strategy": "dc", "alpha": 0.3, "n_synthetic": 100}, "epoch": {"strategy": "frofa", "strength": 0.1}},
        {"name": "DC05+FroFA005", "pretrain": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch": {"strategy": "frofa", "strength": 0.05}},
        {"name": "DC07+FroFA01", "pretrain": {"strategy": "dc", "alpha": 0.7, "n_synthetic": 30}, "epoch": {"strategy": "frofa", "strength": 0.1}},
        {"name": "DC05+AdaptN01", "pretrain": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch": {"strategy": "adaptive_noise", "scale": 0.1}},
        {"name": "DC05+WCMixup03", "pretrain": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch": {"strategy": "within_class_mixup", "alpha": 0.3}},
        {"name": "SMOTE5+FroFA01", "pretrain": {"strategy": "smote", "k": 5, "n_synthetic": 30}, "epoch": {"strategy": "frofa", "strength": 0.1}},
        {"name": "SMOTE5+FroFA005", "pretrain": {"strategy": "smote", "k": 5, "n_synthetic": 30}, "epoch": {"strategy": "frofa", "strength": 0.05}},
        {"name": "SMOTE3+AdaptN01", "pretrain": {"strategy": "smote", "k": 3, "n_synthetic": 50}, "epoch": {"strategy": "adaptive_noise", "scale": 0.1}},
        {"name": "SMOTE5+WCMixup03", "pretrain": {"strategy": "smote", "k": 5, "n_synthetic": 30}, "epoch": {"strategy": "within_class_mixup", "alpha": 0.3}},
        {"name": "DC05+FroFA02", "pretrain": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch": {"strategy": "frofa", "strength": 0.2}},
        {"name": "DC10+FroFA01", "pretrain": {"strategy": "dc", "alpha": 1.0, "n_synthetic": 50}, "epoch": {"strategy": "frofa", "strength": 0.1}},
    ]
    for combo in combined_configs:
        exp_idx += 1
        exp_name = combo["name"]
        logger.info("[%d] %s", exp_idx, exp_name)

        row = _run_and_log(
            exp_name,
            lambda pa=combo["pretrain"], ea=combo["epoch"]: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, train_config,
                pretrain_aug_config=pa, epoch_aug_config=ea, seed=seed,
            ),
            extra={"group": "H", "subgroup": "combined"},
        )
        results.append(row)

    # Part H7: Legacy augmentations = 10
    logger.info("--- Part H7: Legacy augmentations (10 experiments) ---")
    legacy_configs = [
        {"name": "GaussNoise_s001", "epoch_aug": {"gaussian_noise_sigma": 0.01}},
        {"name": "GaussNoise_s005", "epoch_aug": {"gaussian_noise_sigma": 0.05}},
        {"name": "GaussNoise_s01", "epoch_aug": {"gaussian_noise_sigma": 0.1}},
        {"name": "GaussNoise_s02", "epoch_aug": {"gaussian_noise_sigma": 0.2}},
        {"name": "Mixup_a02", "epoch_aug": {"mixup_alpha": 0.2}},
        {"name": "Mixup_a05", "epoch_aug": {"mixup_alpha": 0.5}},
        {"name": "Mixup_a10", "epoch_aug": {"mixup_alpha": 1.0}},
        {"name": "ChDrop_p01", "epoch_aug": {"channel_drop_p": 0.1, "n_channels": 3, "embed_per_channel": embed_dim // 3}},
        {"name": "ChDrop_p02", "epoch_aug": {"channel_drop_p": 0.2, "n_channels": 3, "embed_per_channel": embed_dim // 3}},
        {"name": "ChDrop_p03", "epoch_aug": {"channel_drop_p": 0.3, "n_channels": 3, "embed_per_channel": embed_dim // 3}},
    ]
    for lcfg in legacy_configs:
        exp_idx += 1
        exp_name = lcfg["name"]
        logger.info("[%d] %s", exp_idx, exp_name)

        row = _run_and_log(
            exp_name,
            lambda ea=lcfg["epoch_aug"]: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, train_config,
                epoch_aug_config=ea, seed=seed,
            ),
            extra={"group": "H", "subgroup": "legacy"},
        )
        results.append(row)

    _save_group_results(results, "H", "group_h_aug_deep", output_dir)
    return results


# ============================================================================
# Group I: TTA + Ensemble + Final (68 experiments)
# ============================================================================

def run_group_i(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """TTA sweep, multi-seed ensemble, hybrid, sklearn baselines.

    ~68 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP I: TTA + Ensemble + Final (~68 experiments)")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []
    exp_idx = 0

    def default_head_factory():
        return build_head("mlp", embed_dim, n_classes, hidden_dims=[64], dropout=0.5)

    # Part I1: TTA sweep = 24
    logger.info("--- Part I1: TTA sweep (24 experiments) ---")
    tta_ks = [3, 5, 10, 20]
    tta_strategies = ["frofa", "adaptive_noise"]
    tta_strengths = [0.05, 0.1, 0.2]

    for k in tta_ks:
        for strat in tta_strategies:
            for strength in tta_strengths:
                exp_idx += 1
                exp_name = f"TTA-{k}_{strat}_s{strength}"
                logger.info("[%d] %s", exp_idx, exp_name)

                row = _run_and_log(
                    exp_name,
                    lambda k_=k, s_=strat, st_=strength: run_tta_train_test(
                        Z_train, y_train, Z_test, y_test,
                        default_head_factory, train_config,
                        tta_k=k_, tta_strategy=s_, tta_strength=st_, seed=seed,
                    ),
                    extra={"group": "I", "subgroup": "tta", "tta_k": k,
                           "tta_strategy": strat, "tta_strength": strength},
                )
                results.append(row)

    # Part I2: Multi-seed ensemble = 8
    logger.info("--- Part I2: Multi-seed ensemble (8 experiments) ---")
    for n_seeds in [3, 5, 7, 10, 15, 20]:
        exp_idx += 1
        exp_name = f"Ensemble_{n_seeds}seed"
        logger.info("[%d] %s", exp_idx, exp_name)

        row = _run_and_log(
            exp_name,
            lambda ns=n_seeds: run_ensemble_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, train_config,
                n_seeds=ns, base_seed=seed,
            ),
            extra={"group": "I", "subgroup": "ensemble", "n_seeds": n_seeds},
        )
        results.append(row)

    # Ensemble + DC augmentation
    for n_seeds in [5, 10]:
        exp_idx += 1
        exp_name = f"Ensemble_{n_seeds}seed+DC"
        logger.info("[%d] %s", exp_idx, exp_name)
        pretrain_aug = {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}

        row = _run_and_log(
            exp_name,
            lambda ns=n_seeds, pa=pretrain_aug: run_ensemble_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, train_config,
                n_seeds=ns, base_seed=seed, pretrain_aug_config=pa,
            ),
            extra={"group": "I", "subgroup": "ensemble_aug", "n_seeds": n_seeds},
        )
        results.append(row)

    # Part I3: Multi-seed stability (same config, different seeds) = 15
    logger.info("--- Part I3: Multi-seed stability (15 experiments) ---")
    stability_configs = [
        ("MLP[64]-d0.5", "mlp", {"hidden_dims": [64], "dropout": 0.5}),
        ("Linear", "linear", {}),
        ("MLP[128,64]-d0.5", "mlp", {"hidden_dims": [128, 64], "dropout": 0.5}),
    ]
    for cfg_name, head_type, head_kwargs in stability_configs:
        for seed_offset in range(5):
            exp_idx += 1
            s = seed + seed_offset * 100
            exp_name = f"stability_{cfg_name}_seed{s}"
            logger.info("[%d] %s", exp_idx, exp_name)

            def head_factory(ht=head_type, kw=head_kwargs):
                return build_head(ht, embed_dim, n_classes, **kw)

            row = _run_and_log(
                exp_name,
                lambda hf=head_factory, s_=s: run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    hf, train_config, seed=s_,
                ),
                extra={"group": "I", "subgroup": "stability",
                       "config": cfg_name, "seed": s},
            )
            results.append(row)

    # Part I4: Sklearn baselines = 6
    logger.info("--- Part I4: Sklearn baselines (6 experiments) ---")
    sklearn_baselines = [
        {"name": "SVM_rbf_default", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
        {"name": "SVM_rbf_C10_g0001", "config": {"type": "svm", "kernel": "rbf", "C": 10.0, "gamma": 0.0001}},
        {"name": "RF", "config": {"type": "random_forest"}},
        {"name": "LogReg", "config": {"type": "logistic_regression"}},
        {"name": "ExtraTrees", "config": {"type": "extra_trees"}},
        {"name": "GradBoost", "config": {"type": "gradient_boosting"}},
    ]
    for clf in sklearn_baselines:
        exp_idx += 1
        exp_name = f"sklearn_{clf['name']}"
        logger.info("[%d] %s", exp_idx, exp_name)

        row = _run_and_log(
            exp_name,
            lambda cc=clf["config"]: run_sklearn_train_test(
                Z_train, y_train, Z_test, y_test, cc,
            ),
            extra={"group": "I", "subgroup": "sklearn_baseline", "classifier": clf["name"]},
        )
        results.append(row)

    # Part I5: Class-weight experiments = 3
    logger.info("--- Part I5: Class weight (3 experiments) ---")
    for cw_name, cw in [("balanced", None), ("1:2", [1.0, 2.0]), ("2:1", [2.0, 1.0])]:
        exp_idx += 1
        exp_name = f"ClassWeight_{cw_name}"
        logger.info("[%d] %s", exp_idx, exp_name)

        tc = TrainConfig(
            epochs=train_config.epochs,
            lr=train_config.lr, weight_decay=train_config.weight_decay,
            early_stopping_patience=train_config.early_stopping_patience,
            device=train_config.device,
            class_weight=cw,
        )
        row = _run_and_log(
            exp_name,
            lambda tc_=tc: run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                default_head_factory, tc_, seed=seed,
            ),
            extra={"group": "I", "subgroup": "class_weight", "class_weight": cw_name},
        )
        results.append(row)

    _save_group_results(results, "I", "group_i_tta_ensemble", output_dir)
    return results


# ============================================================================
# Summary + Visualization
# ============================================================================

def _save_group_results(results: list[dict], group: str, filename: str, output_dir: Path):
    """Save group results to CSV."""
    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved: %s (%d rows)", csv_path, len(df))


def generate_summary(all_results: list[dict], output_dir: Path):
    """Generate combined ranking (AUC-primary) and summary report."""
    df = pd.DataFrame(all_results)
    valid = df.dropna(subset=["auc"]) if "auc" in df.columns else df
    if len(valid) == 0:
        logger.warning("No valid results to summarize")
        return

    ranked = valid.sort_values("auc", ascending=False).head(30)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(tables_dir / "top30_ranking.csv", index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 30 CONFIGURATIONS (AUC-primary):")
    logger.info("=" * 70)
    for i, (_, row) in enumerate(ranked.iterrows()):
        group = row.get("group", "?")
        name = row.get("name", "?")
        auc = row.get("auc", 0) or 0
        eer = row.get("eer", 0) or 0
        acc = row.get("accuracy", 0)
        f1_val = row.get("f1", 0)
        logger.info(
            "  #%2d [%s] %s: AUC=%.4f EER=%.4f Acc=%.4f F1=%.4f",
            i + 1, group, name, auc, eer, acc, f1_val,
        )

    # Save summary JSON
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    per_group = {}
    for g in valid["group"].unique() if "group" in valid.columns else []:
        g_df = valid[valid["group"] == g].sort_values("auc", ascending=False)
        per_group[g] = {
            "count": len(g_df),
            "best": g_df.head(1).to_dict("records")[0] if len(g_df) > 0 else None,
        }

    summary = {
        "total_experiments": len(all_results),
        "successful": len(valid),
        "failed": len(all_results) - len(valid),
        "svm_baseline_auc": 0.9859,
        "top10": ranked.head(10).to_dict("records"),
        "per_group": per_group,
    }
    with open(reports_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: %s", reports_dir / "phase2_summary.json")


def plot_group_bar(results: list[dict], title: str, output_path: Path):
    """Horizontal bar chart of AUC for a group of results."""
    setup_style()
    valid = [r for r in results if r.get("auc") is not None and r.get("error") is None]
    if not valid:
        return

    valid.sort(key=lambda r: r["auc"], reverse=True)
    top = valid[:25]
    names = [r["name"] for r in top]
    aucs = [r["auc"] for r in top]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.4), max(6, len(names) * 0.3)))
    colors = ["#DE8F05" if i == 0 else "#0173B2" for i in range(len(names))]
    bars = ax.barh(range(len(names)), aucs, color=colors)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("AUC")
    ax.set_title(title)
    ax.axvline(x=0.9859, color="red", linestyle="--", linewidth=1, label="SVM baseline (0.9859)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=7,
        )

    save_figure(fig, output_path)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Comprehensive Sweep")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--group", nargs="+",
                        choices=["D", "E", "F", "G", "H", "I"], default=None,
                        help="Run specific group(s) (e.g. --group D E for GPU-1)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Primary transformer layer (default: from config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config or 42)")
    parser.add_argument("--split-date", default=None,
                        help="Override split date (YYYY-MM-DD), e.g. 2026-02-17")
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

    output_dir = Path(cfg.get("output_dir", "results/phase2_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_output(formats=["png"], dpi=200)

    # Data config
    data_cfg = cfg["data"]
    label_column = data_cfg.get("label_column", "occupancy_label")
    default_channels = cfg.get("default_channels", [
        "d620900d_motionSensor", "408981c2_contactSensor",
        "d620900d_temperatureMeasurement",
    ])

    ctx_before = cfg.get("default_context_before", 100)
    ctx_after = cfg.get("default_context_after", 100)
    primary_layer = args.layer if args.layer is not None else cfg.get("default_layer", 2)
    stride = cfg.get("stride", 1)

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

    # Model config
    pretrained = cfg["model"]["pretrained_name"]
    output_token = cfg["model"].get("output_token", "combined")

    groups = args.group if args.group else ["D", "E", "F", "G", "H", "I"]
    split_date = args.split_date if args.split_date else cfg.get("split_date")

    logger.info("=" * 70)
    logger.info("Phase 2 Comprehensive Sweep (v4: unified sensor array)")
    logger.info("  Groups: %s", groups)
    logger.info("  Channels: %s", default_channels)
    logger.info("  Context: %d+1+%d = %dmin (bidirectional)",
                ctx_before, ctx_after, ctx_before + 1 + ctx_after)
    logger.info("  Layer: L%d", primary_layer)
    logger.info("  Split date: %s", split_date or "natural CSV boundary")
    logger.info("  Device: %s", device)
    logger.info("  Seed: %d", seed)
    logger.info("=" * 70)

    # --- Load unified sensor data for non-E groups ---
    needs_default_data = any(g in groups for g in ["D", "F", "G", "H", "I"])

    if needs_default_data:
        logger.info("Loading unified sensor data with default channels...")

        sensor, train_labels, test_labels, ch_names, timestamps = load_unified_split(
            data_cfg["train_csv"], data_cfg["test_csv"],
            label_column=label_column, channels=default_channels,
            split_date=split_date,
        )

        n_train_labeled = (train_labels >= 0).sum()
        n_test_labeled = (test_labels >= 0).sum()
        logger.info("Unified: %d rows, train=%d labeled, test=%d labeled",
                     len(sensor), n_train_labeled, n_test_labeled)

        # Build datasets from unified sensor array with default context
        ds_config = DatasetConfig(
            context_mode="bidirectional",
            context_before=ctx_before,
            context_after=ctx_after,
            stride=stride,
        )
        # Both datasets share the unified sensor array — context windows
        # can cross the split boundary (sensor overlap allowed).
        train_dataset = OccupancyDataset(sensor, train_labels, timestamps, ds_config)
        test_dataset = OccupancyDataset(sensor, test_labels, timestamps, ds_config)

        n_train = len(train_dataset)
        n_test = len(test_dataset)
        logger.info("Datasets: train=%d windows, test=%d windows", n_train, n_test)

        # Determine which layers to extract
        layers_needed = {primary_layer}
        if "F" in groups:
            layers_needed.update(ALL_LAYERS)

        all_Z_train = {}
        all_Z_test = {}
        for l in sorted(layers_needed):
            logger.info("Extracting L%d embeddings...", l)
            model = load_mantis_model(pretrained, l, output_token, device)
            all_Z_train[l] = extract_embeddings(model, train_dataset, device)
            all_Z_test[l] = extract_embeddings(model, test_dataset, device)
            logger.info("  L%d: train=%s, test=%s",
                        l, all_Z_train[l].shape, all_Z_test[l].shape)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        y_train = train_dataset.labels
        y_test = test_dataset.labels

        Z_train_primary = all_Z_train[primary_layer]
        Z_test_primary = all_Z_test[primary_layer]

        logger.info("Primary embeddings (L%d): train=%s, test=%s",
                     primary_layer, Z_train_primary.shape, Z_test_primary.shape)

    # --- Run groups ---
    t_start = time.time()
    all_results = []

    for group in groups:
        logger.info("")
        t_group = time.time()

        if group == "D":
            results = run_group_d(
                Z_train_primary, y_train, Z_test_primary, y_test,
                base_train_config, seed, output_dir,
            )
        elif group == "E":
            results = run_group_e(
                train_csv=data_cfg["train_csv"],
                test_csv=data_cfg["test_csv"],
                label_column=label_column,
                split_date=split_date,
                stride=stride,
                pretrained=pretrained,
                output_token=output_token,
                primary_layer=primary_layer,
                train_config=base_train_config,
                seed=seed,
                output_dir=output_dir,
            )
        elif group == "F":
            results = run_group_f(
                all_Z_train, all_Z_test, y_train, y_test,
                base_train_config, seed, output_dir,
            )
        elif group == "G":
            results = run_group_g(
                Z_train_primary, y_train, Z_test_primary, y_test,
                base_train_config, seed, output_dir,
            )
        elif group == "H":
            results = run_group_h(
                Z_train_primary, y_train, Z_test_primary, y_test,
                base_train_config, seed, output_dir,
            )
        elif group == "I":
            results = run_group_i(
                Z_train_primary, y_train, Z_test_primary, y_test,
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
                f"Group {group}: AUC Ranking",
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
