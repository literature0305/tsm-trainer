"""Combined-Optimal Sweep for Occupancy Neural Head Fine-Tuning.

Addresses identified experimental gaps from comprehensive analysis:

  Gap 1: Combined-Optimal (all best components together)
    - Best head: MLP[256,128]-d0.5 (AUC), MLP[128,64]-d0.5 (Acc)
    - Best layer: L5 (neural AUC optimal)
    - Best HP: LS=0.1, no-BN, LR=0.01
    - Best aug: Mixup alpha=0.5

  Gap 2: Best AUC Head (MLP[256,128]) on L5
    - Group D tested on L2 default only
    - Group F tested L5 with MLP[64] only

  Gap 3: Optimized HP with Head Re-ranking
    - Apply best HP (LS=0.1, no-BN, LR=0.01) across all heads

Three groups for parallel GPU execution:

  Group M — Combined-Optimal (30 experiments, ~15 min)
      Best head x Best layer x Best HP x Best aug combinations

  Group N — Head Re-ranking with Optimal HP (36 experiments, ~10 min)
      6 heads x 6 layers with LS=0.1 + no-BN + LR=0.01

  Group O — Extended Channel Verification (24 experiments, ~20 min)
      4 channel combos x 3 classifiers x 2 layers

Usage:
    cd examples/classification/apc_occupancy

    # Terminal 1:
    python training/run_combined_optimal_sweep.py \\
        --config training/configs/occupancy-phase2.yaml --group M --device cuda

    # Terminal 2:
    python training/run_combined_optimal_sweep.py \\
        --config training/configs/occupancy-phase2.yaml --group N --device cuda

    # Terminal 3:
    python training/run_combined_optimal_sweep.py \\
        --config training/configs/occupancy-phase2.yaml --group O --device cuda

    # All groups:
    python training/run_combined_optimal_sweep.py \\
        --config training/configs/occupancy-phase2.yaml --group all --device cuda
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Headless matplotlib
import matplotlib as mpl
if not os.environ.get("DISPLAY"):
    mpl.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import PreprocessConfig, load_occupancy_data
from data.dataset import DatasetConfig, OccupancyDataset
from evaluation.metrics import compute_metrics, ClassificationMetrics
from training.heads import build_head
from training.augmentation import apply_augmentation

logger = logging.getLogger(__name__)

# ============================================================================
# Channel Map
# ============================================================================

CHANNEL_MAP = {
    "M": "d620900d_motionSensor",
    "C": "408981c2_contactSensor",
    "T1": "d620900d_temperatureMeasurement",
    "T2": "ccea734e_temperatureMeasurement",
    "P": "f2e891c6_powerMeter",
}

ALL_LAYERS = [0, 1, 2, 3, 4, 5]


# ============================================================================
# Training Config
# ============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    early_stopping_patience: int = 30
    augmentation: dict | None = None
    device: str = "cpu"
    use_batchnorm: bool = False  # BN removal found beneficial


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
    """Extract frozen embeddings for all windows in the dataset."""
    X, _ = dataset.get_numpy_arrays()  # (N, C, L)
    n_samples, n_channels, seq_len = X.shape

    all_embeddings = []
    for ch in range(n_channels):
        X_ch = X[:, [ch], :]
        Z_ch = model.transform(X_ch)
        all_embeddings.append(Z_ch)

    Z = np.concatenate(all_embeddings, axis=-1)

    if np.isnan(Z).any():
        n_nan = np.isnan(Z).sum()
        logger.warning("Found %d NaN values, replacing with 0", n_nan)
        Z = np.nan_to_num(Z, nan=0.0)

    return Z


def load_config(path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Neural training loop
# ============================================================================

def train_head(head, Z_train, y_train, config: TrainConfig, n_classes: int):
    """Train a classification head on embedding data."""
    device = torch.device(config.device)
    head = head.to(device)
    head.train()

    Z_train = Z_train.to(device)
    y_train = y_train.to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        if config.augmentation is not None:
            Z_aug, y_aug = apply_augmentation(Z_train, y_train, config.augmentation)
        else:
            Z_aug, y_aug = Z_train, y_train

        logits = head(Z_aug)
        loss = loss_fn(logits, y_aug)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    head.eval()
    return head


def evaluate_neural(Z_train, y_train, Z_test, y_test, head_config, train_config, seed=42):
    """Train head on train set, evaluate on test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    Z_tr = torch.tensor(Z_train_s, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    Z_te = torch.tensor(Z_test_s, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    input_dim = Z_tr.shape[1]
    n_classes = len(np.unique(y_train))

    head = build_head(head_config, input_dim, n_classes)
    head = train_head(head, Z_tr, y_tr, train_config, n_classes)

    device = torch.device(train_config.device)
    with torch.no_grad():
        logits = head(Z_te.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    y_test_np = y_te.numpy()
    acc = float(np.mean(preds == y_test_np))

    # AUC (binary)
    try:
        auc = float(roc_auc_score(y_test_np, probs[:, 1]))
    except Exception:
        auc = 0.0

    return {"accuracy": acc, "auc": auc, "preds": preds, "probs": probs}


def evaluate_sklearn(Z_train, y_train, Z_test, y_test, clf_name):
    """Evaluate sklearn classifier."""
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    classifiers = {
        "SVM_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "LogReg": LogisticRegression(max_iter=2000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    clf = classifiers[clf_name]
    clf.fit(Z_train_s, y_train)
    preds = clf.predict(Z_test_s)
    probs = clf.predict_proba(Z_test_s)

    acc = float(np.mean(preds == y_test))
    try:
        auc = float(roc_auc_score(y_test, probs[:, 1]))
    except Exception:
        auc = 0.0

    return {"accuracy": acc, "auc": auc}


# ============================================================================
# Data loading
# ============================================================================

def load_data_and_embeddings(cfg, channels, layer, device):
    """Load data, create dataset, extract embeddings."""
    cfg = {**cfg}  # shallow copy
    channel_names = [CHANNEL_MAP[k] for k in channels]
    cfg_data = cfg.get("data", {})

    pp_cfg = PreprocessConfig(
        sensor_csv=cfg_data["sensor_csv"],
        events_csv=cfg_data["events_csv"],
        column_names_csv=cfg_data.get("column_names_csv", ""),
        channels=channel_names,
    )
    sensor_array, sensor_ts, occ_ts, occ_labels = load_occupancy_data(pp_cfg)

    ds_cfg_dict = cfg.get("dataset", {})
    ds_cfg = DatasetConfig(
        context_mode=ds_cfg_dict.get("context_mode", "bidirectional"),
        context_before=ds_cfg_dict.get("context_before", 125),
        context_after=ds_cfg_dict.get("context_after", 125),
    )
    dataset = OccupancyDataset(sensor_array, sensor_ts, occ_ts, occ_labels, ds_cfg)

    model_cfg = cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    model = load_mantis_model(pretrained_name, layer, output_token, device)
    Z = extract_embeddings(model, dataset, device)

    _, y = dataset.get_numpy_arrays()
    y = y.astype(np.int64)

    # Split by date
    split_date = cfg.get("evaluation", {}).get("split_date", "2026-02-15")
    train_mask = dataset.timestamps < np.datetime64(split_date)
    test_mask = ~train_mask

    Z_train, y_train = Z[train_mask], y[train_mask]
    Z_test, y_test = Z[test_mask], y[test_mask]

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Z_train, y_train, Z_test, y_test


# ============================================================================
# Group M: Combined-Optimal
# ============================================================================

def run_group_m(cfg, device, output_dir):
    """Group M: Combined-optimal neural experiments."""
    logger.info("=" * 70)
    logger.info("GROUP M: Combined-Optimal Neural Sweep")
    logger.info("=" * 70)

    results = []

    # Best components from individual groups
    heads = [
        {"name": "MLP[64]-d0.5", "type": "mlp", "hidden_dims": [64], "dropout": 0.5},
        {"name": "MLP[128]-d0.5", "type": "mlp", "hidden_dims": [128], "dropout": 0.5},
        {"name": "MLP[128,64]-d0.5", "type": "mlp", "hidden_dims": [128, 64], "dropout": 0.5},
        {"name": "MLP[256,128]-d0.5", "type": "mlp", "hidden_dims": [256, 128], "dropout": 0.5},
        {"name": "Linear", "type": "linear"},
    ]

    layers = [2, 5]  # L2 = sklearn optimal, L5 = neural optimal

    hp_configs = [
        {"name": "default_HP", "lr": 1e-3, "ls": 0.0, "bn": True},
        {"name": "optimal_HP", "lr": 1e-2, "ls": 0.1, "bn": False},
    ]

    aug_configs = [
        {"name": "no_aug", "aug": None},
        {"name": "mixup_a05", "aug": {"strategy": "mixup", "mixup_alpha": 0.5}},
    ]

    total = len(heads) * len(layers) * len(hp_configs) * len(aug_configs)
    exp_idx = 0

    for layer in layers:
        Z_train, y_train, Z_test, y_test = load_data_and_embeddings(
            cfg, channels=["M", "C", "T1"], layer=layer, device=device,
        )

        for head_cfg in heads:
            for hp in hp_configs:
                for aug in aug_configs:
                    exp_idx += 1
                    exp_name = f"{head_cfg['name']}|L{layer}|{hp['name']}|{aug['name']}"
                    logger.info("[M %d/%d] %s", exp_idx, total, exp_name)

                    t0 = time.time()
                    train_config = TrainConfig(
                        lr=hp["lr"],
                        label_smoothing=hp["ls"],
                        augmentation=aug["aug"],
                        device=device,
                        use_batchnorm=hp["bn"],
                    )

                    head_cfg_copy = {k: v for k, v in head_cfg.items() if k != "name"}
                    if not hp["bn"] and head_cfg_copy.get("type") == "mlp":
                        head_cfg_copy["use_batchnorm"] = False

                    result = evaluate_neural(
                        Z_train, y_train, Z_test, y_test,
                        head_cfg_copy, train_config,
                    )
                    elapsed = time.time() - t0

                    row = {
                        "group": "M",
                        "head": head_cfg["name"],
                        "layer": layer,
                        "hp": hp["name"],
                        "augmentation": aug["name"],
                        "auc": round(result["auc"], 4),
                        "accuracy": round(result["accuracy"], 4),
                        "time_s": round(elapsed, 1),
                    }
                    results.append(row)
                    logger.info(
                        "  -> AUC=%.4f | Acc=%.4f | %.1fs",
                        result["auc"], result["accuracy"], elapsed,
                    )

    _save_results(results, output_dir / "group_m_results.csv")
    return results


# ============================================================================
# Group N: Head Re-ranking with Optimal HP
# ============================================================================

def run_group_n(cfg, device, output_dir):
    """Group N: 6 heads x 6 layers with optimal HP."""
    logger.info("=" * 70)
    logger.info("GROUP N: Head Re-ranking with Optimal HP")
    logger.info("=" * 70)

    results = []

    heads = [
        {"name": "Linear", "type": "linear"},
        {"name": "MLP[64]-d0.5", "type": "mlp", "hidden_dims": [64], "dropout": 0.5},
        {"name": "MLP[128]-d0.5", "type": "mlp", "hidden_dims": [128], "dropout": 0.5},
        {"name": "MLP[128,64]-d0.5", "type": "mlp", "hidden_dims": [128, 64], "dropout": 0.5},
        {"name": "MLP[256,128]-d0.5", "type": "mlp", "hidden_dims": [256, 128], "dropout": 0.5},
        {"name": "MLP[256]-d0.5", "type": "mlp", "hidden_dims": [256], "dropout": 0.5},
    ]

    total = len(heads) * len(ALL_LAYERS)
    exp_idx = 0

    for layer in ALL_LAYERS:
        Z_train, y_train, Z_test, y_test = load_data_and_embeddings(
            cfg, channels=["M", "C", "T1"], layer=layer, device=device,
        )

        for head_cfg in heads:
            exp_idx += 1
            exp_name = f"{head_cfg['name']}|L{layer}|optimal_HP"
            logger.info("[N %d/%d] %s", exp_idx, total, exp_name)

            t0 = time.time()
            train_config = TrainConfig(
                lr=1e-2,
                label_smoothing=0.1,
                device=device,
                use_batchnorm=False,
            )

            head_cfg_copy = {k: v for k, v in head_cfg.items() if k != "name"}
            if head_cfg_copy.get("type") == "mlp":
                head_cfg_copy["use_batchnorm"] = False

            result = evaluate_neural(
                Z_train, y_train, Z_test, y_test,
                head_cfg_copy, train_config,
            )
            elapsed = time.time() - t0

            row = {
                "group": "N",
                "head": head_cfg["name"],
                "layer": layer,
                "auc": round(result["auc"], 4),
                "accuracy": round(result["accuracy"], 4),
                "time_s": round(elapsed, 1),
            }
            results.append(row)
            logger.info(
                "  -> AUC=%.4f | Acc=%.4f | %.1fs",
                result["auc"], result["accuracy"], elapsed,
            )

    _save_results(results, output_dir / "group_n_results.csv")
    return results


# ============================================================================
# Group O: Extended Channel Verification
# ============================================================================

def run_group_o(cfg, device, output_dir):
    """Group O: Test T2 and P additions on optimal config."""
    logger.info("=" * 70)
    logger.info("GROUP O: Extended Channel Verification")
    logger.info("=" * 70)

    results = []

    channel_combos = [
        ["M", "C", "T1"],           # baseline (3ch)
        ["M", "C", "T1", "T2"],     # + T2 (4ch)
        ["M", "C", "T1", "P"],      # + P (4ch)
        ["M", "C", "T1", "T2", "P"],  # all 5ch
    ]

    classifiers = ["SVM_rbf", "LogReg", "RF"]
    layers = [2, 5]

    total = len(channel_combos) * len(classifiers) * len(layers)
    exp_idx = 0

    for layer in layers:
        for ch_combo in channel_combos:
            ch_label = "+".join(ch_combo)
            try:
                Z_train, y_train, Z_test, y_test = load_data_and_embeddings(
                    cfg, channels=ch_combo, layer=layer, device=device,
                )
            except Exception as e:
                logger.warning("  Skip %s L%d: %s", ch_label, layer, e)
                continue

            for clf_name in classifiers:
                exp_idx += 1
                logger.info(
                    "[O %d/%d] %s | L%d | %s",
                    exp_idx, total, ch_label, layer, clf_name,
                )
                t0 = time.time()
                result = evaluate_sklearn(
                    Z_train, y_train, Z_test, y_test, clf_name,
                )
                elapsed = time.time() - t0

                row = {
                    "group": "O",
                    "channels": ch_label,
                    "n_channels": len(ch_combo),
                    "classifier": clf_name,
                    "layer": layer,
                    "auc": round(result["auc"], 4),
                    "accuracy": round(result["accuracy"], 4),
                    "time_s": round(elapsed, 1),
                }
                results.append(row)
                logger.info(
                    "  -> AUC=%.4f | Acc=%.4f | %.1fs",
                    result["auc"], result["accuracy"], elapsed,
                )

    _save_results(results, output_dir / "group_o_results.csv")
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
    parser = argparse.ArgumentParser(
        description="Combined-optimal sweep for Occupancy neural head",
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--group", type=str, default="all", choices=["M", "N", "O", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/combined_optimal_sweep")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = args.group
    all_results = []

    if groups in ("M", "all"):
        results = run_group_m(cfg, args.device, output_dir)
        all_results.extend(results)

    if groups in ("N", "all"):
        results = run_group_n(cfg, args.device, output_dir)
        all_results.extend(results)

    if groups in ("O", "all"):
        results = run_group_o(cfg, args.device, output_dir)
        all_results.extend(results)

    # Save combined results
    if all_results:
        _save_results(all_results, output_dir / "all_results.csv")

    # Summary
    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE: %d experiments", len(all_results))
    if all_results:
        best = max(all_results, key=lambda r: r.get("auc", 0))
        logger.info(
            "Best AUC: %s | L%s | AUC=%.4f | Acc=%.4f",
            best.get("head", best.get("classifier", "")),
            best.get("layer", ""),
            best.get("auc", 0),
            best.get("accuracy", 0),
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
