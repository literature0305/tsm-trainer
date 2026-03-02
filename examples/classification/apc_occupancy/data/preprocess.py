"""Data preprocessing for SmartThings sensor data.

Supports two label formats:
  - "counts": Per-timestep occupancy_counts.csv (time, occupancy)
  - "events": Event-based occupancy_events CSV (time, Status, Head-count, At-home count)

The preprocessing pipeline:
  1. Load sensor CSV (full time range)
  2. Load label CSV (per-timestep counts or event-based)
  3. For events: replay ENTER/LEAVE events to generate per-timestep labels
  4. Align sensor data to label timestamps
  5. Filter/select channels, handle NaN
  6. Optionally inject hour-of-day cyclical features
  7. Return (sensor_array, label_array, channel_names, timestamps)

Extended mode (load_sensor_and_labels):
  Loads the full sensor array separately from labels, allowing context
  windows to reference sensor data outside the labeled time range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing.

    Attributes:
        sensor_csv: Path to merged sensor data CSV.
        label_csv: Path to label CSV (counts or events format).
        label_format: Label file format. "counts" for per-timestep
            occupancy_counts.csv, "events" for event-based CSV.
        initial_occupancy: Initial occupancy state for event-based labels.
            Used for timesteps before the first event.
        nan_threshold: Drop channels with NaN fraction above this value.
        channels: Explicit list of channel names to use. If None, all
            channels passing nan_threshold are used.
        exclude_channels: Channels to always exclude (substring match).
        binarize: If True, convert occupancy count to binary (>0 -> 1).
        add_time_features: If True, append hour_sin and hour_cos channels
            derived from the timestamp of each row.
    """
    sensor_csv: str | Path = ""
    label_csv: str | Path = ""
    label_format: str = "counts"  # "counts" or "events"
    initial_occupancy: int = 0
    nan_threshold: float = 0.5
    channels: list[str] | None = None
    exclude_channels: list[str] = field(default_factory=list)
    binarize: bool = True
    add_time_features: bool = False


# ---------------------------------------------------------------------------
# Sensor loading
# ---------------------------------------------------------------------------

def _load_sensor_data(path: str | Path) -> pd.DataFrame:
    """Load sensor CSV with timestamp index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index.name = "time"
    df = df.sort_index()
    logger.info(
        "Loaded sensor data: %d rows, %d channels, range [%s, %s]",
        len(df), len(df.columns), df.index.min(), df.index.max(),
    )
    return df


# ---------------------------------------------------------------------------
# Label loading — counts format (legacy)
# ---------------------------------------------------------------------------

def _load_label_counts(path: str | Path, binarize: bool = True) -> pd.Series:
    """Load per-timestep occupancy_counts.csv."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index.name = "time"
    df = df.sort_index()

    occupancy = df["occupancy"]
    if binarize:
        occupancy = (occupancy > 0).astype(np.int64)
        logger.info(
            "Loaded labels (counts, binarized): %d rows, "
            "class dist: {0: %d, 1: %d}",
            len(occupancy), (occupancy == 0).sum(), (occupancy == 1).sum(),
        )
    else:
        logger.info(
            "Loaded labels (counts, raw): %d rows, range [%d, %d]",
            len(occupancy), occupancy.min(), occupancy.max(),
        )
    return occupancy


# ---------------------------------------------------------------------------
# Label loading — events format (new)
# ---------------------------------------------------------------------------

def _load_label_events(
    path: str | Path,
    sensor_timestamps: pd.DatetimeIndex,
    initial_occupancy: int = 0,
    binarize: bool = True,
) -> pd.Series:
    """Convert event-based label CSV to per-timestep labels.

    Replays ENTER_HOME/LEAVE_HOME events chronologically and assigns
    an occupancy state to each sensor timestep.

    Parameters
    ----------
    path : str or Path
        Path to event CSV with columns: time, Status, Head-count, At-home count.
    sensor_timestamps : pd.DatetimeIndex
        Timestamps from the sensor data to align labels to.
    initial_occupancy : int
        Occupancy count before the first event.
    binarize : bool
        If True, convert counts to binary (>0 -> 1).

    Returns
    -------
    pd.Series
        Per-timestep labels indexed by sensor_timestamps.
        Timesteps after the last event are marked as -1 (unlabeled).
    """
    events_df = pd.read_csv(path, parse_dates=["time"])
    events_df = events_df.sort_values("time").reset_index(drop=True)

    # Determine the occupancy count column name (flexible naming)
    count_col = None
    for candidate in ["At-home count", "At_home_count", "at_home_count", "occupancy"]:
        if candidate in events_df.columns:
            count_col = candidate
            break

    # Filter out NONE events if Status column exists
    if "Status" in events_df.columns:
        events_df = events_df[events_df["Status"] != "NONE"].copy()

    if count_col is None:
        # If no count column, infer from Status
        logger.info("No count column found; inferring from Status column")
        state = initial_occupancy
        counts = []
        for _, row in events_df.iterrows():
            status = str(row.get("Status", "")).upper()
            if "ENTER" in status:
                state += 1
            elif "LEAVE" in status:
                state = max(0, state - 1)
            counts.append(state)
        events_df["_count"] = counts
        count_col = "_count"

    # Assign labels to each sensor timestep
    event_times = events_df["time"].values
    event_counts = events_df[count_col].values.astype(np.int64)

    labels = np.full(len(sensor_timestamps), -1, dtype=np.int64)
    current_count = initial_occupancy

    ts_array = sensor_timestamps.values
    event_idx = 0
    n_events = len(event_times)

    for i, ts in enumerate(ts_array):
        # Advance event pointer to the latest event at or before this timestep
        while event_idx < n_events and event_times[event_idx] <= ts:
            current_count = int(event_counts[event_idx])
            event_idx += 1

        # Only label timesteps within the event range [first_event, last_event]
        if n_events > 0 and ts >= event_times[0]:
            if event_idx >= n_events and ts > event_times[-1]:
                # After last event: mark as unlabeled
                labels[i] = -1
            else:
                labels[i] = current_count
        # Before first event: unlabeled (not enough info to assign a label)

    if binarize:
        # Keep -1 as unlabeled, binarize the rest
        labeled_mask = labels >= 0
        labels[labeled_mask] = (labels[labeled_mask] > 0).astype(np.int64)

    n_labeled = (labels >= 0).sum()
    n_unlabeled = (labels == -1).sum()
    n_occupied = (labels == 1).sum()
    n_empty = (labels == 0).sum()

    logger.info(
        "Loaded labels (events): %d total, %d labeled "
        "(occupied=%d, empty=%d), %d unlabeled",
        len(labels), n_labeled, n_occupied, n_empty, n_unlabeled,
    )

    return pd.Series(labels, index=sensor_timestamps, name="occupancy")


# ---------------------------------------------------------------------------
# Channel filtering and NaN handling
# ---------------------------------------------------------------------------

def _filter_channels(
    df: pd.DataFrame,
    nan_threshold: float,
    channels: list[str] | None,
    exclude_channels: list[str],
) -> pd.DataFrame:
    """Filter sensor channels by NaN fraction and explicit lists."""
    # Drop non-numeric columns first (e.g., string-valued sensor states)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = sorted(set(df.columns) - set(numeric_cols))
    if non_numeric:
        logger.info(
            "Dropped %d non-numeric channels: %s", len(non_numeric), non_numeric,
        )
        df = df[numeric_cols]

    if exclude_channels:
        cols_before = set(df.columns)
        for pattern in exclude_channels:
            matched = [c for c in df.columns if pattern in c]
            df = df.drop(columns=matched, errors="ignore")
        dropped = cols_before - set(df.columns)
        if dropped:
            logger.info("Excluded channels: %s", sorted(dropped))

    if channels is not None:
        available = [c for c in channels if c in df.columns]
        missing = [c for c in channels if c not in df.columns]
        if missing:
            logger.warning("Requested channels not found: %s", missing)
        df = df[available]
        logger.info("Using %d explicitly specified channels", len(available))
        return df

    nan_fracs = df.isna().mean()
    keep = nan_fracs[nan_fracs <= nan_threshold].index.tolist()
    dropped_cols = nan_fracs[nan_fracs > nan_threshold].index.tolist()

    if dropped_cols:
        logger.info(
            "Dropped %d channels exceeding %.0f%% NaN threshold: %s",
            len(dropped_cols), nan_threshold * 100, dropped_cols,
        )

    df = df[keep]
    logger.info("Retained %d channels (NaN threshold=%.2f)", len(keep), nan_threshold)
    return df


def _fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill -> backward-fill -> zero-fill remaining NaN."""
    n_nan_before = df.isna().sum().sum()
    df = df.ffill().bfill().fillna(0.0)
    n_nan_after = df.isna().sum().sum()
    logger.info(
        "NaN fill: %d -> %d (filled %d values)",
        n_nan_before, n_nan_after, n_nan_before - n_nan_after,
    )
    return df


# ---------------------------------------------------------------------------
# Time feature injection
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hour_sin and hour_cos cyclical features as extra columns.

    Encodes hour-of-day as two cyclical features using sine/cosine
    transformation, ensuring continuity at midnight (23:55 -> 00:00).

    The features range from -1 to +1:
      hour_sin = sin(2*pi*hour/24)
      hour_cos = cos(2*pi*hour/24)
    """
    hour_frac = df.index.hour + df.index.minute / 60.0
    hour_rad = 2.0 * np.pi * hour_frac / 24.0
    df = df.copy()
    df["hour_sin"] = np.sin(hour_rad).astype(np.float32)
    df["hour_cos"] = np.cos(hour_rad).astype(np.float32)
    logger.info("Added time features: hour_sin, hour_cos")
    return df


# ---------------------------------------------------------------------------
# Public API — legacy (backward-compatible)
# ---------------------------------------------------------------------------

def load_and_preprocess(
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load and preprocess sensor + label data (legacy API).

    Labels and sensors are inner-joined on timestamp. For the new P4
    pipeline with separate train/test labels, use load_sensor_and_labels()
    instead.

    Returns
    -------
    sensor_array : np.ndarray, shape (n_timesteps, n_channels)
    label_array : np.ndarray, shape (n_timesteps,)
    channel_names : list[str]
    timestamps : pd.DatetimeIndex
    """
    sensor_df = _load_sensor_data(config.sensor_csv)

    if config.label_format == "events":
        labels = _load_label_events(
            config.label_csv, sensor_df.index,
            initial_occupancy=config.initial_occupancy,
            binarize=config.binarize,
        )
        # Keep only labeled rows for legacy mode
        labeled_mask = labels >= 0
        sensor_df = sensor_df.loc[labeled_mask]
        labels = labels.loc[labeled_mask]
    else:
        labels = _load_label_counts(config.label_csv, binarize=config.binarize)
        # Inner join on timestamp
        common_idx = sensor_df.index.intersection(labels.index)
        if len(common_idx) == 0:
            raise ValueError(
                f"No overlapping timestamps between sensor data "
                f"[{sensor_df.index.min()}, {sensor_df.index.max()}] "
                f"and labels [{labels.index.min()}, {labels.index.max()}]"
            )
        sensor_df = sensor_df.loc[common_idx]
        labels = labels.loc[common_idx]

    logger.info(
        "Aligned %d rows (range: %s to %s)",
        len(sensor_df), sensor_df.index.min(), sensor_df.index.max(),
    )

    # Channel filtering
    sensor_df = _filter_channels(
        sensor_df, config.nan_threshold, config.channels, config.exclude_channels,
    )
    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    # NaN handling
    sensor_df = _fill_nan(sensor_df)

    # Time features (appended after NaN fill so they are never NaN)
    if config.add_time_features:
        sensor_df = _add_time_features(sensor_df)

    sensor_array = sensor_df.values.astype(np.float32)
    label_array = labels.values.astype(np.int64)
    channel_names = list(sensor_df.columns)
    timestamps = sensor_df.index

    logger.info(
        "Final preprocessed data: %d timesteps, %d channels, "
        "label dist: {0: %d, 1: %d}",
        len(sensor_array), len(channel_names),
        (label_array == 0).sum(), (label_array == 1).sum(),
    )

    return sensor_array, label_array, channel_names, timestamps


# ---------------------------------------------------------------------------
# Public API — new (P4 pipeline)
# ---------------------------------------------------------------------------

def load_sensor_and_labels(
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex, pd.DatetimeIndex]:
    """Load full sensor data and aligned labels separately.

    Unlike load_and_preprocess(), this function returns the FULL sensor
    array (not trimmed to label range), allowing context windows to
    reference sensor data outside the labeled time range.

    Returns
    -------
    sensor_array : np.ndarray, shape (n_sensor_timesteps, n_channels)
        Full sensor data (entire CSV time range).
    label_array : np.ndarray, shape (n_sensor_timesteps,)
        Per-timestep labels aligned to sensor timestamps.
        -1 = unlabeled (outside label range or after last event).
    channel_names : list[str]
    sensor_timestamps : pd.DatetimeIndex
        Full sensor time range.
    labeled_timestamps : pd.DatetimeIndex
        Timestamps where label >= 0 (for dataset construction).
    """
    sensor_df = _load_sensor_data(config.sensor_csv)

    # Generate per-timestep labels aligned to sensor timestamps
    if config.label_format == "events":
        labels = _load_label_events(
            config.label_csv, sensor_df.index,
            initial_occupancy=config.initial_occupancy,
            binarize=config.binarize,
        )
    else:
        raw_labels = _load_label_counts(config.label_csv, binarize=config.binarize)
        # Map counts to sensor timestamps; default -1 for unmatched
        labels = pd.Series(
            np.full(len(sensor_df), -1, dtype=np.int64),
            index=sensor_df.index, name="occupancy",
        )
        common_idx = sensor_df.index.intersection(raw_labels.index)
        labels.loc[common_idx] = raw_labels.loc[common_idx].values

    # Channel filtering (on full sensor data)
    sensor_df = _filter_channels(
        sensor_df, config.nan_threshold, config.channels, config.exclude_channels,
    )
    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    # NaN handling
    sensor_df = _fill_nan(sensor_df)

    # Time features
    if config.add_time_features:
        sensor_df = _add_time_features(sensor_df)

    sensor_array = sensor_df.values.astype(np.float32)
    label_array = labels.values.astype(np.int64)
    channel_names = list(sensor_df.columns)
    sensor_timestamps = sensor_df.index
    labeled_timestamps = sensor_timestamps[label_array >= 0]

    n_labeled = (label_array >= 0).sum()
    n_unlabeled = (label_array == -1).sum()

    logger.info(
        "Full sensor: %d timesteps, %d channels. "
        "Labels: %d labeled, %d unlabeled. "
        "Labeled range: [%s, %s]",
        len(sensor_array), len(channel_names),
        n_labeled, n_unlabeled,
        labeled_timestamps.min() if len(labeled_timestamps) > 0 else "N/A",
        labeled_timestamps.max() if len(labeled_timestamps) > 0 else "N/A",
    )

    return sensor_array, label_array, channel_names, sensor_timestamps, labeled_timestamps


# ---------------------------------------------------------------------------
# Public API — v3 (label-embedded CSVs from prepare_split.py)
# ---------------------------------------------------------------------------

def load_labeled_csv(
    csv_path: str | Path,
    label_column: str = "occupancy_label",
    channels: list[str] | None = None,
    nan_threshold: float = 0.5,
    exclude_channels: list[str] | None = None,
    add_time_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load pre-split sensor CSV with embedded occupancy labels.

    Reads CSVs generated by ``data/prepare_split.py``, where each row
    contains sensor values and a pre-computed occupancy label:

        time, sensor1, sensor2, ..., sensorN, occupancy_label

    Label convention:
      -1 = Unlabeled (before first event or after last event)
       0 = Empty     (at-home count == 0)
       1 = Occupied  (at-home count > 0)

    Parameters
    ----------
    csv_path : str or Path
        Path to the label-embedded CSV.
    label_column : str
        Name of the label column (default: ``"occupancy_label"``).
    channels : list[str] or None
        Explicit list of sensor channel names to use.
    nan_threshold : float
        Drop channels with NaN fraction above this value.
    exclude_channels : list[str] or None
        Channel name substrings to exclude.
    add_time_features : bool
        If True, append hour_sin and hour_cos columns.

    Returns
    -------
    sensor_array : np.ndarray, shape (n_timesteps, n_channels)
    label_array : np.ndarray, shape (n_timesteps,)
    channel_names : list[str]
    timestamps : pd.DatetimeIndex
    """
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time",
                     low_memory=False)
    df = df.sort_index()

    if label_column not in df.columns:
        raise ValueError(
            f"Label column {label_column!r} not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Separate labels from sensor data
    label_array = df[label_column].values.astype(np.int64)
    sensor_df = df.drop(columns=[label_column])

    n_labeled = (label_array >= 0).sum()
    n_occ = (label_array == 1).sum()
    n_emp = (label_array == 0).sum()
    logger.info(
        "Loaded labeled CSV: %d rows, %d sensor columns, "
        "%d labeled (occupied=%d, empty=%d), range [%s, %s]",
        len(sensor_df), len(sensor_df.columns),
        n_labeled, n_occ, n_emp,
        sensor_df.index.min(), sensor_df.index.max(),
    )

    # Channel filtering
    sensor_df = _filter_channels(
        sensor_df, nan_threshold, channels, exclude_channels or [],
    )
    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    # NaN handling
    sensor_df = _fill_nan(sensor_df)

    # Time features
    if add_time_features:
        sensor_df = _add_time_features(sensor_df)

    sensor_array = sensor_df.values.astype(np.float32)
    channel_names = list(sensor_df.columns)
    timestamps = sensor_df.index

    logger.info(
        "Final: %d timesteps, %d channels",
        len(sensor_array), len(channel_names),
    )
    return sensor_array, label_array, channel_names, timestamps


# ---------------------------------------------------------------------------
# Public API — v4 (unified sensor array + date-based label split)
# ---------------------------------------------------------------------------


def load_unified_split(
    train_csv: str | Path,
    test_csv: str | Path,
    label_column: str = "occupancy_label",
    channels: list[str] | None = None,
    nan_threshold: float = 0.5,
    split_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load train + test CSVs into a unified sensor array with date-based label split.

    Concatenates both CSVs into one continuous sensor timeline so that
    context windows can cross the train/test boundary.  Label arrays are
    split by *split_date*: timesteps before the date get train labels,
    timesteps at or after get test labels.  The other region is set to -1.

    If *split_date* is ``None``, the boundary between the two CSVs is used
    (i.e. the first timestamp of the test CSV).

    Parameters
    ----------
    train_csv, test_csv : str or Path
        Paths to the label-embedded CSVs from ``prepare_split.py``.
    label_column : str
        Name of the label column.
    channels : list[str] or None
        Sensor channels to include.
    nan_threshold : float
        Drop channels with NaN fraction above this.
    split_date : str or None
        ``"YYYY-MM-DD"`` or ``"YYYY-MM-DD HH:MM"``.  Overrides the
        natural boundary between the two CSVs.

    Returns
    -------
    sensor_array : np.ndarray, shape (N, C)
        Full unified sensor timeline.
    train_labels : np.ndarray, shape (N,)
        Labels for train region; -1 elsewhere.
    test_labels : np.ndarray, shape (N,)
        Labels for test region; -1 elsewhere.
    channel_names : list[str]
    timestamps : pd.DatetimeIndex
    """
    # Load both CSVs
    tr_df = pd.read_csv(train_csv, parse_dates=["time"], index_col="time",
                        low_memory=False).sort_index()
    te_df = pd.read_csv(test_csv, parse_dates=["time"], index_col="time",
                        low_memory=False).sort_index()

    # Concatenate and deduplicate (keep first occurrence)
    full_df = pd.concat([tr_df, te_df])
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    full_df = full_df.sort_index()

    # Extract labels
    if label_column not in full_df.columns:
        raise ValueError(
            f"Label column {label_column!r} not found. "
            f"Available: {list(full_df.columns)}"
        )
    all_labels = full_df[label_column].values.astype(np.int64)
    sensor_df = full_df.drop(columns=[label_column])

    # Determine split index
    if split_date is not None:
        split_dt = pd.Timestamp(split_date)
    else:
        split_dt = te_df.index.min()  # natural boundary

    split_idx = int(np.searchsorted(full_df.index, split_dt))

    # Create masked label arrays
    train_labels = all_labels.copy()
    train_labels[split_idx:] = -1  # mask test region

    test_labels = all_labels.copy()
    test_labels[:split_idx] = -1  # mask train region

    # Channel filtering
    sensor_df = _filter_channels(
        sensor_df, nan_threshold, channels, [],
    )
    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    sensor_df = _fill_nan(sensor_df)
    sensor_array = sensor_df.values.astype(np.float32)
    channel_names = list(sensor_df.columns)
    timestamps = full_df.index

    # Stats
    tr_labeled = (train_labels >= 0).sum()
    te_labeled = (test_labels >= 0).sum()
    tr_occ = (train_labels == 1).sum()
    te_occ = (test_labels == 1).sum()
    logger.info(
        "Unified split at %s (%d/%d rows): "
        "train %d labeled (occ=%d, emp=%d, %.1f%% occ), "
        "test %d labeled (occ=%d, emp=%d, %.1f%% occ)",
        split_dt.strftime("%Y-%m-%d %H:%M"), split_idx, len(full_df),
        tr_labeled, tr_occ, tr_labeled - tr_occ,
        100 * tr_occ / max(tr_labeled, 1),
        te_labeled, te_occ, te_labeled - te_occ,
        100 * te_occ / max(te_labeled, 1),
    )
    return sensor_array, train_labels, test_labels, channel_names, timestamps


# ---------------------------------------------------------------------------
# Public API — v2 (single events file + date-based split)
# ---------------------------------------------------------------------------

def load_occupancy_data(
    config: PreprocessConfig,
    split_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load sensor data with per-timestep occupancy labels from a single events file.

    This is the recommended v2 API that uses a single (cleaned) events CSV
    instead of separate train/test event files.  When ``split_date`` is
    provided, it returns train_labels and test_labels as a second/third
    element instead of a single label array.

    Parameters
    ----------
    config : PreprocessConfig
        Must have ``label_format="events"`` and ``label_csv`` pointing to
        the cleaned events file.
    split_date : str or None
        If given (e.g. ``"2026-02-15"``), returns separate train/test label
        arrays.  Train = labeled timesteps before split_date;
        test = labeled timesteps on or after split_date.

    Returns
    -------
    If split_date is None:
        sensor_array, label_array, channel_names, timestamps

    If split_date is given:
        sensor_array, train_labels, test_labels, channel_names, timestamps
    """
    sensor_df = _load_sensor_data(config.sensor_csv)

    if config.label_format != "events":
        raise ValueError(
            f"load_occupancy_data requires label_format='events', "
            f"got {config.label_format!r}"
        )

    labels = _load_label_events(
        config.label_csv, sensor_df.index,
        initial_occupancy=config.initial_occupancy,
        binarize=config.binarize,
    )

    # Channel filtering
    sensor_df = _filter_channels(
        sensor_df, config.nan_threshold, config.channels, config.exclude_channels,
    )
    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    # NaN handling
    sensor_df = _fill_nan(sensor_df)

    # Time features
    if config.add_time_features:
        sensor_df = _add_time_features(sensor_df)

    sensor_array = sensor_df.values.astype(np.float32)
    label_array = labels.values.astype(np.int64)
    channel_names = list(sensor_df.columns)
    timestamps = sensor_df.index

    n_labeled = (label_array >= 0).sum()
    n_occ = (label_array == 1).sum()
    n_emp = (label_array == 0).sum()
    logger.info(
        "Loaded: %d sensor timesteps (%d channels), %d labeled "
        "(occupied=%d, empty=%d)",
        len(sensor_array), len(channel_names), n_labeled, n_occ, n_emp,
    )

    if split_date is None:
        return sensor_array, label_array, channel_names, timestamps

    # Date-based train/test split
    split_ts = pd.Timestamp(split_date)
    train_labels = label_array.copy()
    test_labels = label_array.copy()

    # Train: keep only labeled timesteps BEFORE split_date
    test_mask = timestamps >= split_ts
    train_labels[test_mask] = -1

    # Test: keep only labeled timesteps ON or AFTER split_date
    train_mask = timestamps < split_ts
    test_labels[train_mask] = -1

    n_train = (train_labels >= 0).sum()
    n_test = (test_labels >= 0).sum()
    logger.info(
        "Date split at %s: train=%d labeled, test=%d labeled",
        split_ts.strftime("%Y-%m-%d"), n_train, n_test,
    )

    return sensor_array, train_labels, test_labels, channel_names, timestamps
