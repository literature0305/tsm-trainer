"""Gap-Filling Sweep for Enter/Leave N=105 — Classifier Diversity + Channel Ablation.

Addresses identified experimental gaps from the comprehensive analysis:

  Gap 1: N=105 Multi-Classifier Sweep
    - Test SVM_rbf, LogReg, kNN, ExtraTrees, GradBoost, NearestCentroid
    - On optimal config: M+C, L2/L3, 2+1+2 bidirectional
    - Compare against RF baseline (96.19%)

  Gap 2: N=105 Channel Ablation
    - 15 channel combinations on cleaned N=105 data
    - Using optimal context (2+1+2) and layers (L2, L3)
    - Verify M+C optimality on cleaned data

  Gap 3: Context x Classifier Interaction
    - Top 4 classifiers x 5 context windows x 2 layers
    - Detect if non-RF classifiers prefer different contexts

Three groups for parallel GPU execution:

  Group J — Multi-Classifier Sweep (24 experiments, ~10 min)
      6 classifiers x 2 layers (L2, L3) x 2 contexts (2+1+2, 3+1+3)

  Group K — Channel Ablation N=105 (90 experiments, ~25 min)
      15 channel combos x 2 layers (L2, L3) x 3 classifiers (RF, SVM, LogReg)

  Group L — Context x Classifier Interaction (40 experiments, ~15 min)
      4 classifiers x 5 contexts x 2 layers (L2, L3)

Usage:
    cd examples/classification/apc_enter_leave

    # Terminal 1:
    python training/run_n105_gap_sweep.py \\
        --config training/configs/enter-leave-phase2.yaml --group J --device cuda

    # Terminal 2:
    python training/run_n105_gap_sweep.py \\
        --config training/configs/enter-leave-phase2.yaml --group K --device cuda

    # Terminal 3:
    python training/run_n105_gap_sweep.py \\
        --config training/configs/enter-leave-phase2.yaml --group L --device cuda

    # All groups sequentially:
    python training/run_n105_gap_sweep.py \\
        --config training/configs/enter-leave-phase2.yaml --group all --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from training.run_event_detection import (
    load_config,
    load_data,
    load_mantis_model,
    extract_all_embeddings,
)
from data.dataset import EventDataset, build_dataset_config
from evaluation.metrics import (
    EventClassificationMetrics,
    aggregate_cv_predictions,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

CHANNEL_MAP = {
    "M": "d620900d_motionSensor",
    "C": "408981c2_contactSensor",
    "T1": "d620900d_temperatureMeasurement",
    "T2": "ccea734e_temperatureMeasurement",
    "P": "f2e891c6_powerMeter",
}


def _make_classifier(name: str):
    """Create sklearn classifier by name."""
    classifiers = {
        "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "LogReg": LogisticRegression(max_iter=2000, random_state=42),
        "kNN_5": KNeighborsClassifier(n_neighbors=5),
        "kNN_7": KNeighborsClassifier(n_neighbors=7),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "NC": NearestCentroid(),
    }
    return classifiers[name]


CONTEXT_CONFIGS = [
    {"name": "1+1+1", "before": 1, "after": 1},
    {"name": "2+1+2", "before": 2, "after": 2},
    {"name": "3+1+3", "before": 3, "after": 3},
    {"name": "4+1+4", "before": 4, "after": 4},
    {"name": "5+1+5", "before": 5, "after": 5},
]

# Channel combinations for ablation
CHANNEL_COMBOS = [
    # Univariate (5)
    ["M"], ["C"], ["T1"], ["T2"], ["P"],
    # Pair (10)
    ["M", "C"], ["M", "T1"], ["M", "T2"], ["M", "P"],
    ["C", "T1"], ["C", "T2"], ["C", "P"],
    ["T1", "T2"], ["T1", "P"], ["T2", "P"],
    # Triple (5 — physically meaningful)
    ["M", "C", "T1"], ["M", "C", "T2"], ["M", "C", "P"],
    ["M", "T1", "T2"], ["C", "T1", "T2"],
]


# ============================================================================
# Embedding extraction
# ============================================================================

def _extract_embeddings(raw_cfg, include_none, device, layer, ctx_before,
                        ctx_after, channels=None):
    """Extract MantisV2 embeddings for a given config."""
    cfg = copy.deepcopy(raw_cfg)
    cfg["dataset"] = {
        "context_mode": "bidirectional",
        "context_before": ctx_before,
        "context_after": ctx_after,
    }
    if channels is not None:
        cfg["data"]["channels"] = [CHANNEL_MAP[k] for k in channels]

    sensor_array, sensor_ts, event_ts, event_labels, ch_names, class_names = (
        load_data(cfg, include_none=include_none)
    )
    ds_cfg = build_dataset_config(cfg["dataset"])
    dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    network, model = load_mantis_model(pretrained_name, layer, output_token, device)
    Z = extract_all_embeddings(model, dataset)
    del model, network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    labels = dataset.event_labels
    return Z, labels, class_names


# ============================================================================
# LOOCV evaluation
# ============================================================================

def run_sklearn_loocv(Z, y, clf_name, class_names):
    """Run Leave-One-Out Cross-Validation with a given sklearn classifier."""
    n = len(y)
    all_preds = np.zeros(n, dtype=y.dtype)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Z_train, y_train = Z[mask], y[mask]
        Z_test = Z[i:i+1]

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train)
        Z_test_s = scaler.transform(Z_test)

        clf = _make_classifier(clf_name)
        clf.fit(Z_train_s, y_train)
        all_preds[i] = clf.predict(Z_test_s)[0]

    metrics = aggregate_cv_predictions(y, all_preds, class_names)
    return metrics, all_preds


# ============================================================================
# Group J: Multi-Classifier Sweep
# ============================================================================

def run_group_j(raw_cfg, device, output_dir):
    """Group J: 6 classifiers x 2 layers x 2 contexts = 24 experiments."""
    logger.info("=" * 70)
    logger.info("GROUP J: Multi-Classifier Sweep on N=105")
    logger.info("=" * 70)

    classifiers = ["RF", "SVM_rbf", "LogReg", "kNN_5", "kNN_7", "ExtraTrees", "GradBoost", "NC"]
    layers = [2, 3]
    contexts = [
        {"name": "2+1+2", "before": 2, "after": 2},
        {"name": "3+1+3", "before": 3, "after": 3},
    ]

    results = []
    total = len(classifiers) * len(layers) * len(contexts)
    exp_idx = 0

    for ctx in contexts:
        for layer in layers:
            Z, y, class_names = _extract_embeddings(
                raw_cfg, include_none=True, device=device, layer=layer,
                ctx_before=ctx["before"], ctx_after=ctx["after"],
            )

            for clf_name in classifiers:
                exp_idx += 1
                logger.info(
                    "[J %d/%d] %s | L%d | %s",
                    exp_idx, total, clf_name, layer, ctx["name"],
                )
                t0 = time.time()
                metrics, preds = run_sklearn_loocv(Z, y, clf_name, class_names)
                elapsed = time.time() - t0

                row = {
                    "group": "J",
                    "classifier": clf_name,
                    "layer": layer,
                    "context": ctx["name"],
                    "channels": "M+C",
                    "accuracy": metrics.accuracy,
                    "f1_macro": metrics.f1_macro,
                    "n_correct": int(metrics.accuracy * len(y)),
                    "n_total": len(y),
                    "time_s": round(elapsed, 1),
                }
                results.append(row)
                logger.info(
                    "  -> Acc=%.4f | F1m=%.4f | %.1fs",
                    metrics.accuracy, metrics.f1_macro, elapsed,
                )

    _save_results(results, output_dir / "group_j_results.csv")
    return results


# ============================================================================
# Group K: Channel Ablation N=105
# ============================================================================

def run_group_k(raw_cfg, device, output_dir):
    """Group K: 15+ channel combos x 2 layers x 3 classifiers = ~90 experiments."""
    logger.info("=" * 70)
    logger.info("GROUP K: Channel Ablation on N=105 (2+1+2 context)")
    logger.info("=" * 70)

    classifiers = ["RF", "SVM_rbf", "LogReg"]
    layers = [2, 3]
    ctx_before, ctx_after = 2, 2

    results = []
    total = len(CHANNEL_COMBOS) * len(layers) * len(classifiers)
    exp_idx = 0

    for layer in layers:
        for ch_combo in CHANNEL_COMBOS:
            ch_label = "+".join(ch_combo)
            try:
                Z, y, class_names = _extract_embeddings(
                    raw_cfg, include_none=True, device=device, layer=layer,
                    ctx_before=ctx_before, ctx_after=ctx_after,
                    channels=ch_combo,
                )
            except Exception as e:
                logger.warning("  Skip %s L%d: %s", ch_label, layer, e)
                continue

            for clf_name in classifiers:
                exp_idx += 1
                logger.info(
                    "[K %d/%d] %s | L%d | %s",
                    exp_idx, total, ch_label, layer, clf_name,
                )
                t0 = time.time()
                metrics, preds = run_sklearn_loocv(Z, y, clf_name, class_names)
                elapsed = time.time() - t0

                row = {
                    "group": "K",
                    "channels": ch_label,
                    "n_channels": len(ch_combo),
                    "classifier": clf_name,
                    "layer": layer,
                    "context": "2+1+2",
                    "accuracy": metrics.accuracy,
                    "f1_macro": metrics.f1_macro,
                    "n_correct": int(metrics.accuracy * len(y)),
                    "n_total": len(y),
                    "time_s": round(elapsed, 1),
                }
                results.append(row)
                logger.info(
                    "  -> Acc=%.4f | F1m=%.4f | %.1fs",
                    metrics.accuracy, metrics.f1_macro, elapsed,
                )

    _save_results(results, output_dir / "group_k_results.csv")
    return results


# ============================================================================
# Group L: Context x Classifier Interaction
# ============================================================================

def run_group_l(raw_cfg, device, output_dir):
    """Group L: 4 classifiers x 5 contexts x 2 layers = 40 experiments."""
    logger.info("=" * 70)
    logger.info("GROUP L: Context x Classifier Interaction on N=105")
    logger.info("=" * 70)

    classifiers = ["RF", "SVM_rbf", "LogReg", "ExtraTrees"]
    layers = [2, 3]

    results = []
    total = len(classifiers) * len(CONTEXT_CONFIGS) * len(layers)
    exp_idx = 0

    for ctx in CONTEXT_CONFIGS:
        for layer in layers:
            Z, y, class_names = _extract_embeddings(
                raw_cfg, include_none=True, device=device, layer=layer,
                ctx_before=ctx["before"], ctx_after=ctx["after"],
            )

            for clf_name in classifiers:
                exp_idx += 1
                logger.info(
                    "[L %d/%d] %s | L%d | %s",
                    exp_idx, total, clf_name, layer, ctx["name"],
                )
                t0 = time.time()
                metrics, preds = run_sklearn_loocv(Z, y, clf_name, class_names)
                elapsed = time.time() - t0

                row = {
                    "group": "L",
                    "classifier": clf_name,
                    "layer": layer,
                    "context": ctx["name"],
                    "channels": "M+C",
                    "accuracy": metrics.accuracy,
                    "f1_macro": metrics.f1_macro,
                    "n_correct": int(metrics.accuracy * len(y)),
                    "n_total": len(y),
                    "time_s": round(elapsed, 1),
                }
                results.append(row)
                logger.info(
                    "  -> Acc=%.4f | F1m=%.4f | %.1fs",
                    metrics.accuracy, metrics.f1_macro, elapsed,
                )

    _save_results(results, output_dir / "group_l_results.csv")
    return results


# ============================================================================
# Utilities
# ============================================================================

def _save_results(results, filepath):
    """Save experiment results to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to save to %s", filepath)
        return

    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Saved %d results to %s", len(results), filepath)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gap-filling sweep for Enter/Leave N=105")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--group", type=str, default="all", choices=["J", "K", "L", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/n105_gap_sweep")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw_cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = args.group
    all_results = []

    if groups in ("J", "all"):
        results = run_group_j(raw_cfg, args.device, output_dir)
        all_results.extend(results)

    if groups in ("K", "all"):
        results = run_group_k(raw_cfg, args.device, output_dir)
        all_results.extend(results)

    if groups in ("L", "all"):
        results = run_group_l(raw_cfg, args.device, output_dir)
        all_results.extend(results)

    # Save combined results
    if all_results:
        _save_results(all_results, output_dir / "all_results.csv")

    # Summary
    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE: %d experiments", len(all_results))
    if all_results:
        best = max(all_results, key=lambda r: r["accuracy"])
        logger.info(
            "Best: %s | %s | L%d | %s | Acc=%.4f",
            best.get("classifier", ""),
            best.get("channels", "M+C"),
            best.get("layer", 0),
            best.get("context", ""),
            best["accuracy"],
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
