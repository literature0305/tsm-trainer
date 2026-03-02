"""Occupancy detection dataset with sliding window construction.

Converts preprocessed (n_timesteps, n_channels) arrays into sliding window
samples shaped (n_channels, seq_len) suitable for MantisV2 input.

Supports two context modes:

**Backward mode** (legacy):
    Each window covers [t - seq_len + 1, t] — the preceding ``seq_len``
    timesteps.  The label comes from the *last* timestep of the window.

**Bidirectional mode** (default, recommended):
    Each window covers [t - context_before, t + context_after] — past and
    future context centered on the prediction timestep t.  Better captures
    state-transition patterns visible in sensor data.

MantisV2 constraints:
  - The final input length must be a multiple of 32 (patch_size).
  - MantisV2 per-patch normalization needs >= 2 patches (minimum 64).
  - In backward mode: seq_len must be a multiple of 32; auto-interp to 64
    if < 64.
  - In bidirectional mode: effective window is auto-interpolated to at
    least 64 timesteps (via target_seq_len).

Two construction APIs:
  - Legacy: separate train/test arrays (``create_datasets``).
  - P4 pipeline: full sensor array + date-based split (``create_datasets_from_splits``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Maximum future context (look-ahead) in minutes.
# Constraint: real-time deployment scenarios allow at most ~100 min of
# future information.  Configs requesting more will be rejected.
MAX_CONTEXT_AFTER = 100


@dataclass
class DatasetConfig:
    """Configuration for sliding window dataset construction.

    Supports two context modes:

    **Backward mode** (``context_mode="backward"``):
        Window covers ``seq_len`` timesteps before the prediction point:
        [t - seq_len + 1, t].  seq_len must be a multiple of 32.

    **Bidirectional mode** (``context_mode="bidirectional"``):
        Window covers past and future context: [t - context_before, t + context_after].
        Total window = context_before + 1 + context_after timesteps.
        Auto-interpolated to MantisV2-compatible length (>= 64, multiple of 32).

    Attributes:
        context_mode: ``"backward"`` or ``"bidirectional"``.
        seq_len: Backward window length (backward mode only).  Must be multiple
            of 32.  Default 288 = 24 h at 5-min bins.
        context_before: Minutes before prediction point (bidirectional).
        context_after: Minutes after prediction point (bidirectional).
        stride: Step between consecutive windows.  Default 1.
        target_seq_len: If set, interpolate each window to this length.
            Must be multiple of 32.  Auto-computed for bidirectional if None.
    """

    context_mode: str = "bidirectional"
    seq_len: int = 288
    context_before: int = 4
    context_after: int = 4
    stride: int = 1
    target_seq_len: int | None = None

    @property
    def effective_seq_len(self) -> int:
        """Actual number of timesteps in each raw context window."""
        if self.context_mode == "bidirectional":
            return self.context_before + 1 + self.context_after
        return self.seq_len

    def describe(self) -> str:
        """Human-readable description of the context window."""
        if self.context_mode == "bidirectional":
            eff = self.effective_seq_len
            tgt = self.target_seq_len
            desc = (
                f"bidirectional: {self.context_before}+1+{self.context_after}"
                f"={eff} timesteps"
            )
            if tgt is not None and tgt != eff:
                desc += f" -> interp to {tgt}"
            return desc
        tgt = self.target_seq_len
        desc = f"backward: {self.seq_len} timesteps"
        if tgt is not None and tgt != self.seq_len:
            desc += f" -> interp to {tgt}"
        return desc

    def __post_init__(self):
        if self.context_mode not in ("backward", "bidirectional"):
            raise ValueError(
                f"context_mode must be 'backward' or 'bidirectional', "
                f"got {self.context_mode!r}"
            )

        _min_mantis_len = 64  # 2 × patch_size (avoids NaN in per-patch norm)

        if self.context_mode == "backward":
            if self.seq_len % 32 != 0:
                raise ValueError(
                    f"seq_len must be a multiple of 32 for MantisV2, "
                    f"got {self.seq_len}"
                )
            if self.target_seq_len is None and self.seq_len < _min_mantis_len:
                self.target_seq_len = _min_mantis_len
        else:  # bidirectional
            if self.context_before < 0 or self.context_after < 0:
                raise ValueError(
                    f"context_before and context_after must be >= 0, "
                    f"got before={self.context_before}, after={self.context_after}"
                )
            if self.context_after > MAX_CONTEXT_AFTER:
                raise ValueError(
                    f"context_after={self.context_after} exceeds "
                    f"MAX_CONTEXT_AFTER={MAX_CONTEXT_AFTER}. "
                    f"Real-time deployment limits future look-ahead."
                )
            eff = self.effective_seq_len
            if self.target_seq_len is None:
                if eff < _min_mantis_len:
                    self.target_seq_len = _min_mantis_len
                elif eff % 32 != 0:
                    self.target_seq_len = ((eff + 31) // 32) * 32

        if self.target_seq_len is not None and self.target_seq_len % 32 != 0:
            raise ValueError(
                f"target_seq_len must be a multiple of 32, "
                f"got {self.target_seq_len}"
            )


def build_dataset_config(ds_cfg: dict) -> DatasetConfig:
    """Build DatasetConfig from YAML dict with smart mode detection.

    Detection logic:
      - If ``context_mode`` is explicit -> use it.
      - Elif ``context_before`` or ``context_after`` present -> bidirectional.
      - Elif ``seq_len`` present -> backward.
      - Else -> bidirectional with defaults.
    """
    if "context_mode" in ds_cfg:
        mode = ds_cfg["context_mode"]
    elif "context_before" in ds_cfg or "context_after" in ds_cfg:
        mode = "bidirectional"
    elif "seq_len" in ds_cfg:
        mode = "backward"
    else:
        mode = "bidirectional"

    if mode == "backward":
        return DatasetConfig(
            context_mode="backward",
            seq_len=ds_cfg.get("seq_len", 288),
            stride=ds_cfg.get("stride", 1),
            target_seq_len=ds_cfg.get("target_seq_len"),
        )
    return DatasetConfig(
        context_mode="bidirectional",
        context_before=ds_cfg.get("context_before", 4),
        context_after=ds_cfg.get("context_after", 4),
        stride=ds_cfg.get("stride", 1),
        target_seq_len=ds_cfg.get("target_seq_len"),
    )


class OccupancyDataset(Dataset):
    """Sliding window dataset for occupancy detection.

    Supports backward and bidirectional context modes.

    Parameters
    ----------
    sensor_array : np.ndarray
        Shape (n_timesteps, n_channels).  Float32 sensor readings.
    label_array : np.ndarray
        Shape (n_timesteps,).  Int64 labels.  -1 = unlabeled (skipped).
    timestamps : pd.DatetimeIndex or None
        Timestamps for each row.  Required for date-based train/test split.
    config : DatasetConfig, optional
        Window construction parameters.
    """

    def __init__(
        self,
        sensor_array: np.ndarray,
        label_array: np.ndarray,
        timestamps: pd.DatetimeIndex | None = None,
        config: DatasetConfig | None = None,
    ):
        if config is None:
            config = DatasetConfig()

        self.config = config
        n_timesteps, n_channels = sensor_array.shape
        self.n_channels = n_channels

        if n_timesteps != len(label_array):
            raise ValueError(
                f"sensor_array ({n_timesteps}) and label_array "
                f"({len(label_array)}) must have the same length"
            )

        # Find valid target indices: labeled AND within context bounds
        labeled_mask = label_array >= 0
        labeled_indices = np.where(labeled_mask)[0]

        if config.context_mode == "bidirectional":
            ctx_before = config.context_before
            ctx_after = config.context_after
            # Valid: enough room for both past and future context
            valid_mask = (
                (labeled_indices >= ctx_before)
                & (labeled_indices + ctx_after < n_timesteps)
            )
            valid_indices = labeled_indices[valid_mask]
            n_boundary = int(labeled_mask.sum()) - len(valid_indices)
            if n_boundary > 0:
                logger.info(
                    "Boundary skip: %d/%d labeled timesteps excluded "
                    "(insufficient context for %d+1+%d window)",
                    n_boundary, int(labeled_mask.sum()),
                    ctx_before, ctx_after,
                )
        else:
            min_idx = config.seq_len - 1
            valid_indices = labeled_indices[labeled_indices >= min_idx]
            n_boundary = int(labeled_mask.sum()) - len(valid_indices)
            if n_boundary > 0:
                logger.info(
                    "Boundary skip: %d/%d labeled timesteps excluded "
                    "(insufficient context for seq_len=%d)",
                    n_boundary, int(labeled_mask.sum()), config.seq_len,
                )

        # Apply stride
        if config.stride > 1:
            valid_indices = valid_indices[::config.stride]

        if len(valid_indices) == 0:
            raise ValueError(
                f"No valid windows: {labeled_mask.sum()} labeled timesteps, "
                f"config={config.describe()}"
            )

        # Pre-extract all windows (valid_indices guarantees in-bounds)
        windows = []

        if config.context_mode == "bidirectional":
            for idx in valid_indices:
                start = idx - ctx_before
                end = idx + ctx_after + 1  # exclusive
                window = sensor_array[start:end]  # (eff_len, n_channels)
                windows.append(window.T)  # (n_channels, eff_len)
        else:
            seq_len = config.seq_len
            for idx in valid_indices:
                start = idx - seq_len + 1
                window = sensor_array[start:idx + 1]
                windows.append(window.T)

        self.windows = np.stack(windows).astype(np.float32)
        self.labels = label_array[valid_indices].astype(np.int64)
        self._valid_indices = valid_indices

        # Store timestamps for date-based splitting
        if timestamps is not None:
            self.timestamps = timestamps
            self.prediction_timestamps = timestamps[valid_indices]
        else:
            self.timestamps = None
            self.prediction_timestamps = None

        n_occ = (self.labels == 1).sum()
        n_emp = (self.labels == 0).sum()
        logger.info(
            "OccupancyDataset: %d windows (%s, stride=%d, channels=%d), "
            "class dist: {0: %d, 1: %d}",
            len(self.labels), config.describe(), config.stride, n_channels,
            n_emp, n_occ,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Return (window, label) pair.

        Returns
        -------
        window : np.ndarray
            Shape (n_channels, effective_seq_len) or (n_channels, target_seq_len).
        label : int
            Binary label (0=empty, 1=occupied).
        """
        window = self.windows[idx]

        if self.config.target_seq_len is not None:
            tensor = torch.from_numpy(window).unsqueeze(0)  # (1, C, L)
            tensor = F.interpolate(
                tensor, size=self.config.target_seq_len,
                mode="linear", align_corners=False,
            )
            window = tensor.squeeze(0).numpy()

        return window, int(self.labels[idx])

    def get_numpy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all data as numpy arrays for sklearn-style training.

        Returns
        -------
        X : np.ndarray, shape (n_windows, n_channels, effective_seq_len)
        y : np.ndarray, shape (n_windows,)
        """
        if self.config.target_seq_len is not None:
            tensor = torch.from_numpy(self.windows)
            tensor = F.interpolate(
                tensor, size=self.config.target_seq_len,
                mode="linear", align_corners=False,
            )
            return tensor.numpy(), self.labels.copy()

        return self.windows.copy(), self.labels.copy()

    def get_train_test_split(
        self,
        split_date: str | pd.Timestamp,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return boolean masks for train/test split by date.

        Parameters
        ----------
        split_date : str or pd.Timestamp
            Samples with prediction timestamp < split_date go to train,
            >= split_date go to test.

        Returns
        -------
        train_mask, test_mask : np.ndarray of bool
            Boolean masks over dataset indices.
        """
        if self.prediction_timestamps is None:
            raise ValueError(
                "Cannot split by date: timestamps were not provided "
                "during dataset construction."
            )
        split_ts = pd.Timestamp(split_date)
        train_mask = self.prediction_timestamps < split_ts
        test_mask = self.prediction_timestamps >= split_ts
        n_train = train_mask.sum()
        n_test = test_mask.sum()
        logger.info(
            "Date split at %s: train=%d, test=%d",
            split_ts.strftime("%Y-%m-%d"), n_train, n_test,
        )
        return np.array(train_mask), np.array(test_mask)

    def get_day_groups(self) -> np.ndarray:
        """Return calendar day index per sample (for Leave-One-Day-Out CV).

        Returns
        -------
        day_groups : np.ndarray, shape (n_windows,)
            Integer day index in chronological order.
        """
        if self.prediction_timestamps is None:
            raise ValueError("Timestamps required for day grouping.")
        dates = self.prediction_timestamps.date
        unique_dates = sorted(set(dates))
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        return np.array([date_to_idx[d] for d in dates], dtype=np.int64)


def create_datasets_from_splits(
    sensor_array: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    config: DatasetConfig | None = None,
    timestamps: pd.DatetimeIndex | None = None,
) -> tuple[OccupancyDataset, OccupancyDataset]:
    """Create train/test datasets from split label arrays (P4 pipeline).

    Both label arrays must be aligned to sensor_array, with -1 for
    timesteps outside the split's labeled region.
    """
    if config is None:
        config = DatasetConfig()

    train_ds = OccupancyDataset(sensor_array, train_labels, timestamps, config)
    test_ds = OccupancyDataset(sensor_array, test_labels, timestamps, config)

    logger.info(
        "Created P4 datasets: train=%d, test=%d windows "
        "(shared sensor: %d timesteps, %d channels)",
        len(train_ds), len(test_ds),
        sensor_array.shape[0], sensor_array.shape[1],
    )
    return train_ds, test_ds


def create_datasets(
    train_sensor: np.ndarray,
    train_labels: np.ndarray,
    test_sensor: np.ndarray,
    test_labels: np.ndarray,
    config: DatasetConfig | None = None,
) -> tuple[OccupancyDataset, OccupancyDataset]:
    """Create train/test datasets from separate arrays (legacy API)."""
    if config is None:
        config = DatasetConfig()

    train_ds = OccupancyDataset(train_sensor, train_labels, config=config)
    test_ds = OccupancyDataset(test_sensor, test_labels, config=config)

    logger.info(
        "Created datasets: train=%d, test=%d windows",
        len(train_ds), len(test_ds),
    )
    return train_ds, test_ds
