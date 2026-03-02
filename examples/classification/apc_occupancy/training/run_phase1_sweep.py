#!/usr/bin/env python3
"""Phase 1 Comprehensive Sweep: MantisV2 Frozen Embeddings + sklearn Classifiers.

Designed for 3 parallel GPU servers to find optimal configuration.
Findings from initial round: context window is the dominant factor
(AUC scales from 0.56 at 1min to 0.84 at 241min, no saturation).
Layer choice is negligible (±0.01 AUC range at any given context).

CRITICAL: Uses P4-style separate train/test event files to maintain
the curated ~75:25 train/test ratio. The old split_date approach
created ~50:50 split, invalidating all previous results.

Three sweep groups (run independently on separate GPUs):

  Group A — Context Window Deep Exploration (~51 experiments)
    - 21 symmetric bidirectional + 10 backward-only
      + 16 asymmetric past-heavy + 4 asymmetric future-heavy
    - Fixed: L2, RF, M+C channels, combined token
    - Goal: Map full context window landscape, find saturation point

  Group B — Channel Ablation × Context (~63 experiments)
    - 21 channel combos × 3 context sizes (61min, 121min, 241min)
    - Fixed: L2, RF, combined token
    - Goal: Check if channel importance varies with context length

  Group C — Classifier × Layer at Long Contexts (~54 experiments)
    - 3 classifiers × 6 layers × 3 contexts (121min, 241min, 361min)
    - Fixed: M+C channels, combined token
    - Goal: Find best classifier and layer at long contexts

Usage:
  cd examples/classification/apc_occupancy

  # Run all groups sequentially
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --device cuda

  # Run single group (for parallel GPU servers)
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group A --device cuda
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group B --device cuda
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group C --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Headless matplotlib
import matplotlib as mpl
if not os.environ.get("DISPLAY"):
    mpl.use("Agg")
import matplotlib.pyplot as plt

# Local imports — run from apc_occupancy/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import PreprocessConfig, load_sensor_and_labels
from data.dataset import DatasetConfig, OccupancyDataset
from evaluation.metrics import compute_metrics, compute_wilson_ci, ClassificationMetrics

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

ALL_LAYERS = [0, 1, 2, 3, 4, 5]

# ── Group A: Context Window Deep Exploration ────────────────────────
# Subgroup A1: Symmetric Bidirectional (21 configs)
# Key range: performance scales log-linearly from 1min to 241min+
CONTEXT_SYMMETRIC = [
    {"name": "1min (0+1+0)", "context_before": 0, "context_after": 0},
    {"name": "3min (1+1+1)", "context_before": 1, "context_after": 1},
    {"name": "5min (2+1+2)", "context_before": 2, "context_after": 2},
    {"name": "9min (4+1+4)", "context_before": 4, "context_after": 4},
    {"name": "15min (7+1+7)", "context_before": 7, "context_after": 7},
    {"name": "21min (10+1+10)", "context_before": 10, "context_after": 10},
    {"name": "31min (15+1+15)", "context_before": 15, "context_after": 15},
    {"name": "41min (20+1+20)", "context_before": 20, "context_after": 20},
    {"name": "61min (30+1+30)", "context_before": 30, "context_after": 30},
    {"name": "91min (45+1+45)", "context_before": 45, "context_after": 45},
    {"name": "121min (60+1+60)", "context_before": 60, "context_after": 60},
    {"name": "151min (75+1+75)", "context_before": 75, "context_after": 75},
    {"name": "181min (90+1+90)", "context_before": 90, "context_after": 90},
    {"name": "241min (120+1+120)", "context_before": 120, "context_after": 120},
    {"name": "301min (150+1+150)", "context_before": 150, "context_after": 150},
    {"name": "361min (180+1+180)", "context_before": 180, "context_after": 180},
    {"name": "481min (240+1+240)", "context_before": 240, "context_after": 240},
    {"name": "601min (300+1+300)", "context_before": 300, "context_after": 300},
    {"name": "721min (360+1+360)", "context_before": 360, "context_after": 360},
    {"name": "961min (480+1+480)", "context_before": 480, "context_after": 480},
    {"name": "1441min (720+1+720)", "context_before": 720, "context_after": 720},
]

# Subgroup A2: Backward-Only (10 configs) — no future information
CONTEXT_BACKWARD = [
    {"name": "bw 6min (5+1+0)", "context_before": 5, "context_after": 0},
    {"name": "bw 16min (15+1+0)", "context_before": 15, "context_after": 0},
    {"name": "bw 31min (30+1+0)", "context_before": 30, "context_after": 0},
    {"name": "bw 61min (60+1+0)", "context_before": 60, "context_after": 0},
    {"name": "bw 121min (120+1+0)", "context_before": 120, "context_after": 0},
    {"name": "bw 241min (240+1+0)", "context_before": 240, "context_after": 0},
    {"name": "bw 361min (360+1+0)", "context_before": 360, "context_after": 0},
    {"name": "bw 481min (480+1+0)", "context_before": 480, "context_after": 0},
    {"name": "bw 721min (720+1+0)", "context_before": 720, "context_after": 0},
    {"name": "bw 1441min (1440+1+0)", "context_before": 1440, "context_after": 0},
]

# Subgroup A3: Asymmetric Past-Heavy (16 configs) — practical deployment
CONTEXT_ASYM_PAST = [
    {"name": "asym 30p+5f (30+1+5)", "context_before": 30, "context_after": 5},
    {"name": "asym 60p+5f (60+1+5)", "context_before": 60, "context_after": 5},
    {"name": "asym 60p+10f (60+1+10)", "context_before": 60, "context_after": 10},
    {"name": "asym 60p+30f (60+1+30)", "context_before": 60, "context_after": 30},
    {"name": "asym 120p+5f (120+1+5)", "context_before": 120, "context_after": 5},
    {"name": "asym 120p+10f (120+1+10)", "context_before": 120, "context_after": 10},
    {"name": "asym 120p+30f (120+1+30)", "context_before": 120, "context_after": 30},
    {"name": "asym 120p+60f (120+1+60)", "context_before": 120, "context_after": 60},
    {"name": "asym 240p+10f (240+1+10)", "context_before": 240, "context_after": 10},
    {"name": "asym 240p+30f (240+1+30)", "context_before": 240, "context_after": 30},
    {"name": "asym 240p+60f (240+1+60)", "context_before": 240, "context_after": 60},
    {"name": "asym 240p+120f (240+1+120)", "context_before": 240, "context_after": 120},
    {"name": "asym 360p+30f (360+1+30)", "context_before": 360, "context_after": 30},
    {"name": "asym 360p+60f (360+1+60)", "context_before": 360, "context_after": 60},
    {"name": "asym 480p+60f (480+1+60)", "context_before": 480, "context_after": 60},
    {"name": "asym 720p+60f (720+1+60)", "context_before": 720, "context_after": 60},
]

# Subgroup A4: Asymmetric Future-Heavy (4 configs) — comparison baseline
CONTEXT_ASYM_FUTURE = [
    {"name": "asym 5p+30f (5+1+30)", "context_before": 5, "context_after": 30},
    {"name": "asym 10p+60f (10+1+60)", "context_before": 10, "context_after": 60},
    {"name": "asym 30p+120f (30+1+120)", "context_before": 30, "context_after": 120},
    {"name": "asym 60p+240f (60+1+240)", "context_before": 60, "context_after": 240},
]

# All context configs for Group A
ALL_CONTEXT_CONFIGS = CONTEXT_SYMMETRIC + CONTEXT_BACKWARD + CONTEXT_ASYM_PAST + CONTEXT_ASYM_FUTURE

# ── Group B: Channel Ablation at multiple contexts ──────────────────
# Test at 3 representative context sizes to check for interaction effects
GROUP_B_CONTEXTS = [
    {"name": "61min", "context_before": 30, "context_after": 30},
    {"name": "121min", "context_before": 60, "context_after": 60},
    {"name": "241min", "context_before": 120, "context_after": 120},
]

# 21 channel combinations
CHANNEL_CONFIGS = [
    # Single channels
    {"name": "M only", "channels": ["d620900d_motionSensor"]},
    {"name": "P only", "channels": ["f2e891c6_powerMeter"]},
    {"name": "T1 only", "channels": ["d620900d_temperatureMeasurement"]},
    {"name": "T2 only", "channels": ["ccea734e_temperatureMeasurement"]},
    {"name": "C only", "channels": ["408981c2_contactSensor"]},
    {"name": "E only", "channels": ["408981c2_temperatureMeasurement"]},
    # Pairs
    {"name": "M+C", "channels": ["d620900d_motionSensor", "408981c2_contactSensor"]},
    {"name": "M+T1", "channels": ["d620900d_motionSensor", "d620900d_temperatureMeasurement"]},
    {"name": "M+P", "channels": ["d620900d_motionSensor", "f2e891c6_powerMeter"]},
    {"name": "M+T2", "channels": ["d620900d_motionSensor", "ccea734e_temperatureMeasurement"]},
    {"name": "T1+C", "channels": ["d620900d_temperatureMeasurement", "408981c2_contactSensor"]},
    {"name": "P+C", "channels": ["f2e891c6_powerMeter", "408981c2_contactSensor"]},
    {"name": "M+E", "channels": ["d620900d_motionSensor", "408981c2_temperatureMeasurement"]},
    # Triples
    {"name": "M+C+T1", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "d620900d_temperatureMeasurement"]},
    {"name": "M+C+P", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter"]},
    {"name": "M+P+T1", "channels": ["d620900d_motionSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement"]},
    {"name": "M+C+E", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "408981c2_temperatureMeasurement"]},
    # Quads and more
    {"name": "M+P+T1+T2", "channels": ["d620900d_motionSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement"]},
    {"name": "M+C+P+T1", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement"]},
    {"name": "M+C+P+T1+T2", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement"]},
    {"name": "All 6", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement", "408981c2_temperatureMeasurement"]},
]

# ── Group C: Classifier × Layer at Long Contexts ───────────────────
CLASSIFIER_NAMES = ["random_forest", "svm", "nearest_centroid"]
GROUP_C_CONTEXTS = [
    {"name": "121min", "context_before": 60, "context_after": 60},
    {"name": "241min", "context_before": 120, "context_after": 120},
    {"name": "361min", "context_before": 180, "context_after": 180},
]


# ============================================================================
# Data loading — P4-style (separate train/test event files)
# ============================================================================

def load_train_test_data(
    cfg: dict,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load sensor data with separate train/test labels (P4-style).

    Uses two separate event CSV files for train and test splits,
    maintaining the curated ~75:25 ratio.

    Returns
    -------
    sensor_arr : shape (n_timesteps, n_channels)
    all_labels : shape (n_timesteps,) — merged (train + test, -1 = unlabeled)
    train_labels : shape (n_timesteps,) — train only (-1 elsewhere)
    test_labels : shape (n_timesteps,) — test only (-1 elsewhere)
    ch_names : channel name list
    timestamps : DatetimeIndex
    """
    data_cfg = cfg["data"]
    ch = channels or cfg.get("default_channels")

    base_kwargs = dict(
        sensor_csv=data_cfg["sensor_csv"],
        label_format=data_cfg.get("label_format", "events"),
        initial_occupancy=data_cfg.get("initial_occupancy", 0),
        binarize=data_cfg.get("binarize", True),
        channels=ch,
    )

    # Load sensor + train labels
    train_prep = PreprocessConfig(label_csv=data_cfg["train_label_csv"], **base_kwargs)
    sensor_arr, train_labels, ch_names, timestamps, _ = load_sensor_and_labels(train_prep)

    # Load test labels (same sensor CSV, different events)
    test_prep = PreprocessConfig(label_csv=data_cfg["test_label_csv"], **base_kwargs)
    _, test_labels, _, _, _ = load_sensor_and_labels(test_prep)

    # Resolve overlap: train takes priority
    overlap = (train_labels >= 0) & (test_labels >= 0)
    if overlap.sum() > 0:
        logger.warning(
            "Resolving %d overlapping timesteps (train takes priority)",
            overlap.sum(),
        )
        test_labels[overlap] = -1

    # Merge for dataset construction
    all_labels = np.where(train_labels >= 0, train_labels, test_labels)

    n_train = (train_labels >= 0).sum()
    n_test = (test_labels >= 0).sum()
    n_total = (all_labels >= 0).sum()
    logger.info(
        "P4-style split: train=%d, test=%d, total=%d labeled (%.1f:%.1f ratio)",
        n_train, n_test, n_total,
        100 * n_train / max(n_total, 1), 100 * n_test / max(n_total, 1),
    )

    return sensor_arr, all_labels, train_labels, test_labels, ch_names, timestamps


def get_label_masks(
    dataset: OccupancyDataset,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train/test boolean masks from label arrays.

    Uses the dataset's valid_indices to determine which windows
    belong to train vs test based on the original label arrays.
    """
    valid_idx = dataset._valid_indices
    train_mask = train_labels[valid_idx] >= 0
    test_mask = test_labels[valid_idx] >= 0
    return train_mask, test_mask


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

    Returns shape (n_windows, embed_dim).
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


def build_classifier(name: str, seed: int = 42):
    """Build sklearn classifier by name."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif name == "svm":
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    elif name == "nearest_centroid":
        return NearestCentroid()
    else:
        raise ValueError(f"Unknown classifier: {name}")


def _extract_prob_positive(clf, X_test) -> np.ndarray | None:
    """Extract P(class=1) from classifier, handling edge cases."""
    if not hasattr(clf, "predict_proba"):
        return None
    try:
        proba = clf.predict_proba(X_test)
    except Exception:
        return None

    if hasattr(clf, "classes_"):
        classes = list(clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        else:
            return np.zeros(len(X_test))
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


# ============================================================================
# Evaluation runner
# ============================================================================

def run_train_test_eval(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    clf_name: str,
    seed: int = 42,
) -> ClassificationMetrics:
    """Train classifier and evaluate on test set."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    clf = build_classifier(clf_name, seed)
    clf.fit(Z_train_s, y_train)

    y_pred = clf.predict(Z_test_s)
    y_prob = _extract_prob_positive(clf, Z_test_s)

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics


# ============================================================================
# Group A: Context Window × Layer
# ============================================================================

def run_group_a(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """Deep context window exploration with RF classifier.

    21 symmetric + 10 backward + 16 asym-past + 4 asym-future = 51 configs.
    Fixed: L2 (best layer from initial round), RF, M+C, combined token.
    """
    n_total = len(ALL_CONTEXT_CONFIGS)
    logger.info("=" * 70)
    logger.info("GROUP A: Context Window Deep Exploration (%d experiments)", n_total)
    logger.info("=" * 70)

    results = []
    default_channels = cfg.get("default_channels", ["d620900d_motionSensor", "408981c2_contactSensor"])
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    default_layer = cfg.get("default_layer", 2)

    # Load sensor data with P4-style train/test split (shared across all context configs)
    sensor_arr, all_labels, train_labels, test_labels, ch_names, timestamps = (
        load_train_test_data(cfg, channels=default_channels)
    )

    for i, ctx_cfg in enumerate(ALL_CONTEXT_CONFIGS, 1):
        ctx_name = ctx_cfg["name"]
        ctx_before = ctx_cfg["context_before"]
        ctx_after = ctx_cfg["context_after"]
        total_min = ctx_before + 1 + ctx_after
        logger.info("--- [%d/%d] Context: %s (total %dmin) ---", i, n_total, ctx_name, total_min)
        t0 = time.time()

        try:
            ds_config = DatasetConfig(
                context_mode="bidirectional",
                context_before=ctx_before,
                context_after=ctx_after,
                stride=cfg.get("stride", 1),
            )
            dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
            train_mask, test_mask = get_label_masks(dataset, train_labels, test_labels)

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                logger.warning("Skipping %s: empty train or test split", ctx_name)
                continue

            trainer = load_mantis_model(pretrained, default_layer, "combined", device)
            Z = extract_embeddings(trainer, dataset, device)
            del trainer

            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            metrics = run_train_test_eval(Z_train, y_train, Z_test, y_test, "random_forest", seed)
            elapsed = time.time() - t0

            # Determine subgroup for analysis
            if ctx_after == 0:
                subgroup = "backward"
            elif ctx_before == ctx_after:
                subgroup = "symmetric"
            elif ctx_before > ctx_after:
                subgroup = "asym_past"
            else:
                subgroup = "asym_future"

            row = {
                "group": "A",
                "subgroup": subgroup,
                "context": ctx_name,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "total_context_min": total_min,
                "layer": default_layer,
                "channels": "+".join([c.split("_")[0][0].upper() for c in default_channels]),
                "classifier": "RF",
                "output_token": "combined",
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "f1_macro": metrics.f1_macro,
                "auc": metrics.roc_auc if not np.isnan(metrics.roc_auc) else None,
                "eer": metrics.eer if not np.isnan(metrics.eer) else None,
                "ci_lower": metrics.ci_lower,
                "ci_upper": metrics.ci_upper,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "embed_dim": Z.shape[1],
                "time_s": round(elapsed, 1),
            }
            results.append(row)
            logger.info(
                "    Acc=%.4f  F1=%.4f  AUC=%.4f  train=%d test=%d (%.1fs)",
                metrics.accuracy, metrics.f1,
                metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.0,
                int(train_mask.sum()), int(test_mask.sum()),
                elapsed,
            )
        except Exception as e:
            logger.error("    FAILED: %s", e)
            results.append({
                "group": "A", "context": ctx_name,
                "context_before": ctx_before, "context_after": ctx_after,
                "layer": default_layer, "error": str(e),
            })

    # Save results
    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_a_context_deep.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_a_context_deep.csv", len(df))

    return results


# ============================================================================
# Group B: Channel Ablation × Layer
# ============================================================================

def run_group_b(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """Channel ablation at multiple context sizes.

    21 channel combos × 3 contexts (61min, 121min, 241min) = 63 experiments.
    Fixed: L2 (best), RF, combined token.
    Goal: Check if channel importance varies with context length.
    """
    n_total = len(CHANNEL_CONFIGS) * len(GROUP_B_CONTEXTS)
    logger.info("=" * 70)
    logger.info("GROUP B: Channel Ablation × Context (%d experiments)", n_total)
    logger.info("=" * 70)

    results = []
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    default_layer = cfg.get("default_layer", 2)
    exp_idx = 0

    for ctx_cfg in GROUP_B_CONTEXTS:
        ctx_name = ctx_cfg["name"]
        ctx_before = ctx_cfg["context_before"]
        ctx_after = ctx_cfg["context_after"]
        logger.info("=== Context: %s (%d+1+%d) ===", ctx_name, ctx_before, ctx_after)

        for ch_cfg in CHANNEL_CONFIGS:
            ch_name = ch_cfg["name"]
            channels = ch_cfg["channels"]
            exp_idx += 1
            logger.info("  [%d/%d] %s | %s", exp_idx, n_total, ctx_name, ch_name)
            t0 = time.time()

            try:
                # Load data with specific channels (P4-style split)
                sensor_arr, all_labels, train_labels, test_labels, ch_names_loaded, timestamps = (
                    load_train_test_data(cfg, channels=channels)
                )

                ds_config = DatasetConfig(
                    context_mode="bidirectional",
                    context_before=ctx_before,
                    context_after=ctx_after,
                    stride=cfg.get("stride", 1),
                )
                dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
                train_mask, test_mask = get_label_masks(dataset, train_labels, test_labels)

                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    logger.warning("Skipping %s|%s: empty split", ctx_name, ch_name)
                    continue

                trainer = load_mantis_model(pretrained, default_layer, "combined", device)
                Z = extract_embeddings(trainer, dataset, device)
                del trainer

                Z_train = Z[train_mask]
                y_train = dataset.labels[train_mask]
                Z_test = Z[test_mask]
                y_test = dataset.labels[test_mask]

                metrics = run_train_test_eval(Z_train, y_train, Z_test, y_test, "random_forest", seed)
                elapsed = time.time() - t0

                row = {
                    "group": "B",
                    "channels": ch_name,
                    "n_channels": len(channels),
                    "layer": default_layer,
                    "context": f"{ctx_before}+1+{ctx_after}",
                    "context_name": ctx_name,
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                    "classifier": "RF",
                    "accuracy": metrics.accuracy,
                    "f1": metrics.f1,
                    "f1_macro": metrics.f1_macro,
                    "auc": metrics.roc_auc if not np.isnan(metrics.roc_auc) else None,
                    "eer": metrics.eer if not np.isnan(metrics.eer) else None,
                    "ci_lower": metrics.ci_lower,
                    "ci_upper": metrics.ci_upper,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "embed_dim": Z.shape[1],
                    "time_s": round(elapsed, 1),
                }
                results.append(row)
                logger.info(
                    "    Acc=%.4f  F1=%.4f  AUC=%.4f  train=%d test=%d (%.1fs)",
                    metrics.accuracy, metrics.f1,
                    metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.0,
                    int(train_mask.sum()), int(test_mask.sum()),
                    elapsed,
                )
            except Exception as e:
                logger.error("    FAILED: %s", e)
                results.append({
                    "group": "B", "channels": ch_name, "context_name": ctx_name,
                    "context_before": ctx_before, "context_after": ctx_after,
                    "error": str(e),
                })

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_b_channel_context.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_b_channel_context.csv", len(df))

    return results


# ============================================================================
# Group C: Classifier × Output Token × Target Seq Len
# ============================================================================

def run_group_c(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """Classifier × Layer sweep at long contexts.

    3 classifiers × 6 layers × 3 contexts (121, 241, 361 min) = 54 experiments.
    Fixed: M+C channels, combined token.
    Goal: Find best classifier/layer at contexts where it matters.
    """
    n_total = len(CLASSIFIER_NAMES) * len(ALL_LAYERS) * len(GROUP_C_CONTEXTS)
    logger.info("=" * 70)
    logger.info("GROUP C: Classifier × Layer at Long Contexts (%d experiments)", n_total)
    logger.info("=" * 70)

    results = []
    default_channels = cfg.get("default_channels", ["d620900d_motionSensor", "408981c2_contactSensor"])
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)

    # Load sensor data with P4-style split (shared across all combos)
    sensor_arr, all_labels, train_labels, test_labels, ch_names, timestamps = (
        load_train_test_data(cfg, channels=default_channels)
    )

    exp_idx = 0
    for ctx_cfg in GROUP_C_CONTEXTS:
        ctx_name = ctx_cfg["name"]
        ctx_before = ctx_cfg["context_before"]
        ctx_after = ctx_cfg["context_after"]
        logger.info("=== Context: %s (%d+1+%d) ===", ctx_name, ctx_before, ctx_after)

        ds_config = DatasetConfig(
            context_mode="bidirectional",
            context_before=ctx_before,
            context_after=ctx_after,
            stride=cfg.get("stride", 1),
        )

        try:
            dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
            train_mask, test_mask = get_label_masks(dataset, train_labels, test_labels)
        except Exception as e:
            logger.error("Dataset build failed for %s: %s", ctx_name, e)
            continue

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning("Skipping %s: empty split", ctx_name)
            continue

        for layer in ALL_LAYERS:
            # Extract embeddings once per context × layer
            logger.info("  --- L%d ---", layer)
            try:
                trainer = load_mantis_model(pretrained, layer, "combined", device)
                Z = extract_embeddings(trainer, dataset, device)
                del trainer
            except Exception as e:
                logger.error("  Embedding extraction failed (L%d): %s", layer, e)
                for clf_name in CLASSIFIER_NAMES:
                    results.append({
                        "group": "C", "classifier": clf_name, "layer": layer,
                        "context_name": ctx_name, "error": str(e),
                    })
                continue

            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_name in CLASSIFIER_NAMES:
                exp_idx += 1
                exp_name = f"{ctx_name} | L{layer} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_train_test_eval(Z_train, y_train, Z_test, y_test, clf_name, seed)
                    elapsed = time.time() - t0

                    row = {
                        "group": "C",
                        "classifier": clf_name,
                        "layer": layer,
                        "context_name": ctx_name,
                        "context": f"{ctx_before}+1+{ctx_after}",
                        "context_before": ctx_before,
                        "context_after": ctx_after,
                        "channels": "+".join([c.split("_")[0][0].upper() for c in default_channels]),
                        "output_token": "combined",
                        "accuracy": metrics.accuracy,
                        "f1": metrics.f1,
                        "f1_macro": metrics.f1_macro,
                        "auc": metrics.roc_auc if not np.isnan(metrics.roc_auc) else None,
                        "eer": metrics.eer if not np.isnan(metrics.eer) else None,
                        "ci_lower": metrics.ci_lower,
                        "ci_upper": metrics.ci_upper,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "embed_dim": Z.shape[1],
                        "time_s": round(elapsed, 1),
                    }
                    results.append(row)
                    logger.info(
                        "  [%d/%d] %s: Acc=%.4f  F1=%.4f  AUC=%.4f  train=%d test=%d (%.1fs)",
                        exp_idx, n_total, exp_name,
                        metrics.accuracy, metrics.f1,
                        metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.0,
                        int(train_mask.sum()), int(test_mask.sum()),
                        elapsed,
                    )
                except Exception as e:
                    logger.error("  %s FAILED: %s", exp_name, e)
                    results.append({
                        "group": "C", "classifier": clf_name, "layer": layer,
                        "context_name": ctx_name, "error": str(e),
                    })

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_c_classifier_layer.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_c_classifier_layer.csv", len(df))

    return results


# ============================================================================
# Visualization Generation
# ============================================================================

def generate_visualizations(all_results: list[dict], output_dir: Path):
    """Generate sweep-level analysis plots from Phase 1 experiment results.

    Produces: Top-K bar chart, context performance curve, classifier comparison.
    """
    from visualization.curves import plot_context_performance, plot_sweep_bar
    from visualization.style import save_figure, setup_style

    df = pd.DataFrame(all_results)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    if len(df) == 0 or "accuracy" not in df.columns:
        logger.warning("No valid results for visualization")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Top-K AUC bar chart ---
    try:
        if "auc" in df.columns and df["auc"].notna().any():
            fig = plot_sweep_bar(
                df, group_col="experiment", metric_col="auc",
                hue_col="classifier", title="Phase 1 — Top 20 by AUC",
                top_k=20, output_path=plots_dir / "sweep_top20_auc",
            )
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "sweep_top20_auc.png")
    except Exception:
        logger.warning("Failed to plot sweep AUC bar chart", exc_info=True)

    # --- 2. Context performance curve ---
    try:
        bidir = df[
            df["context"].str.contains("bidir", case=False, na=False)
            & df["auc"].notna()
        ].copy() if "context" in df.columns else df.iloc[:0].copy()

        if len(bidir) > 0 and "total_context_min" in bidir.columns:
            clf_groups = {}
            for clf_name in bidir["classifier"].unique():
                clf_df = bidir[bidir["classifier"] == clf_name]
                avg = clf_df.groupby("total_context_min")["auc"].mean().sort_index()
                clf_groups[clf_name] = (avg.index.tolist(), avg.values.tolist())

            eer_avg = bidir.groupby("total_context_min")["eer"].mean().sort_index()

            fig = plot_context_performance(
                context_mins=eer_avg.index.tolist(),
                auc_values=None,
                eer_values=eer_avg.values.tolist(),
                classifier_groups=clf_groups,
                output_path=plots_dir / "context_performance",
            )
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "context_performance.png")
    except Exception:
        logger.warning("Failed to plot context performance curve", exc_info=True)

    # --- 3. Per-group best AUC ---
    try:
        if "group" in df.columns and "auc" in df.columns:
            setup_style()
            groups = sorted(df["group"].unique())
            best_aucs = []
            best_labels = []
            for grp in groups:
                grp_df = df[df["group"] == grp]
                if grp_df["auc"].notna().any():
                    best_aucs.append(grp_df["auc"].max())
                    best_labels.append(f"Group {grp}")
                else:
                    best_aucs.append(0)
                    best_labels.append(f"Group {grp} (no AUC)")

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#0173B2", "#DE8F05", "#029E73"]
            bars = ax.bar(best_labels, best_aucs,
                          color=[colors[i % len(colors)] for i in range(len(groups))],
                          edgecolor="white")
            for bar, val in zip(bars, best_aucs):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("AUC")
            ax.set_title("Best AUC per Group — Phase 1")
            ax.set_ylim(0.5, 1.0)
            ax.grid(axis="y", alpha=0.3)
            save_figure(fig, plots_dir / "group_best_auc")
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "group_best_auc.png")
    except Exception:
        logger.warning("Failed to plot per-group AUC comparison", exc_info=True)

    logger.info("Visualization complete. Plots saved to: %s", plots_dir)


# ============================================================================
# Summary + Ranking
# ============================================================================

def generate_summary(all_results: list[dict], output_dir: Path):
    """Generate combined ranking and summary report.

    Primary ranking: AUC (threshold-independent, best for binary classification).
    Secondary: EER (lower is better), then accuracy.
    """
    df = pd.DataFrame(all_results)
    if "error" in df.columns:
        valid = df[df["error"].isna()].copy()
    else:
        valid = df.copy()
    if "accuracy" not in valid.columns or len(valid) == 0:
        logger.warning("No valid results to summarize")
        return

    valid = valid.dropna(subset=["accuracy"])

    # Split: AUC-available vs AUC-unavailable
    has_auc = valid[valid["auc"].notna()].copy() if "auc" in valid.columns else valid.iloc[:0].copy()
    no_auc = valid[valid["auc"].isna()].copy() if "auc" in valid.columns else valid.copy()

    if len(has_auc) > 0:
        sort_keys = ["auc"]
        ascending = [False]
        if "eer" in has_auc.columns:
            sort_keys.append("eer")
            ascending.append(True)
        sort_keys.append("accuracy")
        ascending.append(False)
        has_auc = has_auc.sort_values(sort_keys, ascending=ascending)

    if len(no_auc) > 0:
        no_auc = no_auc.sort_values("accuracy", ascending=False)

    ranked = pd.concat([has_auc, no_auc], ignore_index=True).head(20)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(tables_dir / "top20_ranking.csv", index=False)

    # Save full ranked results
    full_ranked = pd.concat([has_auc, no_auc], ignore_index=True)
    full_ranked.to_csv(tables_dir / "all_results_ranked.csv", index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 20 CONFIGURATIONS (Primary: AUC, Secondary: EER):")
    logger.info("=" * 70)
    for i, (_, row) in enumerate(ranked.iterrows()):
        group = row.get("group", "?")
        ctx = row.get("context", "?")
        layer = row.get("layer", "?")
        ch = row.get("channels", "?")
        clf = row.get("classifier", "?")
        token = row.get("output_token", "?")
        acc = row.get("accuracy", 0)
        auc_val = row.get("auc")
        eer_val = row.get("eer")
        n_train = row.get("n_train", "?")
        n_test = row.get("n_test", "?")
        auc_str = f"AUC={auc_val:.4f}" if auc_val is not None else "AUC=N/A"
        eer_str = f" EER={eer_val:.4f}" if eer_val is not None else ""
        logger.info(
            "  #%2d [%s] %s | L%s | %s | %s | %s: %s%s Acc=%.4f (train=%s,test=%s)",
            i + 1, group, ctx, layer, ch, clf, token, auc_str, eer_str, acc,
            n_train, n_test,
        )

    # Save summary JSON
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_experiments": len(all_results),
        "successful": len(valid),
        "failed": len(all_results) - len(valid),
        "with_auc": len(has_auc),
        "without_auc": len(no_auc),
        "split_method": "P4-style (separate train/test event files)",
        "top5": ranked.head(5).to_dict("records"),
        "metric_note": "Primary: AUC (threshold-independent), Secondary: EER",
    }
    with open(reports_dir / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: %s", reports_dir / "phase1_summary.json")


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Comprehensive Sweep")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--group", choices=["A", "B", "C"], default=None,
                        help="Run specific group only (A, B, or C)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override stride (default from config)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.stride is not None:
        cfg["stride"] = args.stride

    output_dir = Path(cfg.get("output_dir", "results/phase1_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    logger.info("Device: %s", device)
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", output_dir)
    logger.info("Split method: P4-style (separate train/test event files)")

    t_start = time.time()
    all_results = []

    groups = [args.group] if args.group else ["A", "B", "C"]

    for group in groups:
        if group == "A":
            results = run_group_a(cfg, device, output_dir)
        elif group == "B":
            results = run_group_b(cfg, device, output_dir)
        elif group == "C":
            results = run_group_c(cfg, device, output_dir)
        else:
            continue
        all_results.extend(results)

    generate_summary(all_results, output_dir)
    generate_visualizations(all_results, output_dir)

    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Done! Total time: %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
