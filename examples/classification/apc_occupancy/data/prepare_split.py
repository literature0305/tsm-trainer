#!/usr/bin/env python3
"""Prepare train/test sensor CSVs with pre-computed occupancy labels.

Single source of truth for creating labeled split files.  Eliminates
ambiguity by:

  1. Computing occupancy labels GLOBALLY (all events across full timeline)
  2. Splitting at a configurable timestamp (midnight or exact time)
  3. Embedding labels directly into each CSV

Label convention:
  -1 = Unlabeled (before first event or after last event)
   0 = Empty     (binarized: at-home count == 0)
   1 = Occupied  (binarized: at-home count > 0)

Generalizable to any sensor resampling interval (1-min, 30-sec, 10-sec).

Output files:
  {base}_train_with_occupancy_label.csv
  {base}_test_with_occupancy_label.csv

Each row: time, sensor1, sensor2, ..., sensorN, occupancy_label

Usage:
  cd examples/classification/apc_occupancy

  # Default split (midnight Feb 17)
  python data/prepare_split.py \\
      --sensor-csv /path/to/sensor_interval1.csv \\
      --events-csv /path/to/occupancy_events.csv \\
      --output-dir /path/to/output/

  # Custom split date (midnight boundary)
  python data/prepare_split.py \\
      --sensor-csv /path/to/sensor_interval1.csv \\
      --events-csv /path/to/occupancy_events.csv \\
      --split-date 2026-02-16 \\
      --output-dir /path/to/output/

  # Exact split time (e.g. 07:00 to align with daily rhythm)
  python data/prepare_split.py \\
      --sensor-csv /path/to/sensor_interval1.csv \\
      --events-csv /path/to/occupancy_events.csv \\
      --split-time "2026-02-17 07:00:00" \\
      --output-dir /path/to/output/

  # Analyze mode: compare multiple split points (no output files)
  python data/prepare_split.py \\
      --sensor-csv /path/to/sensor_interval1.csv \\
      --events-csv /path/to/occupancy_events.csv \\
      --analyze

  # Headerless sensor CSV (read column names from reference file)
  python data/prepare_split.py \\
      --sensor-csv /path/to/sensor_interval1_no_header.csv \\
      --events-csv /path/to/occupancy_events.csv \\
      --columns-from /path/to/sensor_5min_with_header.csv \\
      --output-dir /path/to/output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Core: Occupancy Label Computation
# ============================================================================

def compute_occupancy_labels(
    sensor_timestamps: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    binarize: bool = True,
) -> np.ndarray:
    """Replay occupancy events to compute per-timestep labels.

    Events are processed GLOBALLY across the full sensor timeline.
    NONE events (Head-count=0) are skipped as they don't change state.

    The At-home count column from the events CSV is used directly as the
    ground truth.  For timesteps between events, the label is carried
    forward from the most recent event (deterministic state).

    Parameters
    ----------
    sensor_timestamps : DatetimeIndex
        Full sensor timeline (any interval).
    events_df : DataFrame
        Must have columns: time, Status, At-home count.
    binarize : bool
        If True, labels are 0 (empty) / 1 (occupied) / -1 (unlabeled).
        If False, labels are raw At-home count / -1 (unlabeled).

    Returns
    -------
    labels : ndarray of shape (n_timesteps,)
    """
    # Filter state-changing events only (skip NONE)
    state_events = (
        events_df[events_df["Status"] != "NONE"]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )

    n_ts = len(sensor_timestamps)
    labels = np.full(n_ts, -1, dtype=np.int64)

    if len(state_events) == 0:
        return labels

    # Resolve count column (flexible naming)
    count_col = None
    for candidate in ["At-home count", "At_home_count", "at_home_count"]:
        if candidate in state_events.columns:
            count_col = candidate
            break
    if count_col is None:
        raise ValueError(
            "Events CSV must have an 'At-home count' column. "
            f"Found columns: {list(state_events.columns)}"
        )

    event_times = state_events["time"].values.astype("datetime64[ns]")
    event_counts = state_events[count_col].values.astype(np.int64)
    n_events = len(event_times)
    first_event = event_times[0]
    last_event = event_times[-1]

    ts_array = sensor_timestamps.values.astype("datetime64[ns]")
    current_count = 0  # initial (only used before first event → stays -1)
    event_idx = 0

    for i in range(n_ts):
        ts = ts_array[i]

        # Advance event pointer to consume all events at-or-before this ts
        while event_idx < n_events and event_times[event_idx] <= ts:
            current_count = int(event_counts[event_idx])
            event_idx += 1

        # Label only within [first_event, last_event]
        if ts >= first_event:
            if event_idx >= n_events and ts > last_event:
                labels[i] = -1  # after last event → unknown
            else:
                labels[i] = current_count

    if binarize:
        labeled_mask = labels >= 0
        labels[labeled_mask] = (labels[labeled_mask] > 0).astype(np.int64)

    return labels


# ============================================================================
# I/O Helpers
# ============================================================================

def load_sensor_csv(
    path: str | Path,
    columns_from: str | Path | None = None,
) -> pd.DataFrame:
    """Load sensor CSV, handling both with-header and without-header cases.

    Auto-detection logic:
      1. Read first row → check if first column name is parseable as timestamp
      2. If yes → headerless CSV → use --columns-from or auto-generate names
      3. If no  → CSV has a proper header

    Returns DataFrame with DatetimeIndex named 'time'.
    """
    path = Path(path)

    # Peek at first row to detect header
    peek = pd.read_csv(path, nrows=0, low_memory=False)
    first_col_name = peek.columns[0]
    has_header = True
    try:
        pd.Timestamp(first_col_name)
        has_header = False  # first "column name" is actually a data value
    except (ValueError, TypeError):
        pass

    if has_header:
        df = pd.read_csv(
            path, parse_dates=["time"], index_col="time", low_memory=False,
        )
    elif columns_from is not None:
        ref = pd.read_csv(columns_from, nrows=0)
        col_names = list(ref.columns)
        if len(col_names) != peek.shape[1]:
            raise ValueError(
                f"Column count mismatch: reference has {len(col_names)} "
                f"columns but sensor CSV has {peek.shape[1]} columns"
            )
        df = pd.read_csv(
            path, header=None, names=col_names,
            parse_dates=["time"], index_col="time", low_memory=False,
        )
    else:
        # Auto-generate column names
        n_cols = peek.shape[1]
        col_names = ["time"] + [f"sensor_{i}" for i in range(1, n_cols)]
        print(
            f"WARNING: Sensor CSV has no header and --columns-from not "
            f"specified. Using auto-generated column names ({n_cols} columns).",
            file=sys.stderr,
        )
        df = pd.read_csv(
            path, header=None, names=col_names,
            parse_dates=["time"], index_col="time", low_memory=False,
        )

    df = df.sort_index()
    return df


def load_events_csv(path: str | Path) -> pd.DataFrame:
    """Load occupancy events CSV."""
    df = pd.read_csv(path, parse_dates=["time"])
    required = {"time", "Status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing required columns: {missing}")
    return df.sort_values("time").reset_index(drop=True)


# ============================================================================
# Split and Save
# ============================================================================

def split_and_save(
    sensor_df: pd.DataFrame,
    labels: np.ndarray,
    split_point: str | pd.Timestamp,
    output_dir: str | Path,
    base_name: str,
    label_column: str = "occupancy_label",
) -> tuple[Path, Path]:
    """Split labeled sensor data at split_point and save CSVs.

    Parameters
    ----------
    split_point : str or pd.Timestamp
        Any parseable timestamp. Can be a date ("2026-02-17" → midnight)
        or an exact time ("2026-02-17 07:00:00").

    Returns (train_path, test_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add label column (as last column)
    df = sensor_df.copy()
    df[label_column] = labels

    split_dt = pd.Timestamp(split_point)
    train_df = df[df.index < split_dt]
    test_df = df[df.index >= split_dt]

    if len(train_df) == 0:
        raise ValueError(f"Train set is empty with split_point={split_point}")
    if len(test_df) == 0:
        raise ValueError(f"Test set is empty with split_point={split_point}")

    train_path = output_dir / f"{base_name}_train_with_occupancy_label.csv"
    test_path = output_dir / f"{base_name}_test_with_occupancy_label.csv"

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)

    return train_path, test_path


# ============================================================================
# Validation & Statistics
# ============================================================================

def print_stats(
    sensor_df: pd.DataFrame,
    labels: np.ndarray,
    split_point: str | pd.Timestamp,
    events_df: pd.DataFrame,
):
    """Print comprehensive statistics about the split."""
    split_dt = pd.Timestamp(split_point)
    timestamps = sensor_df.index

    tr_mask = timestamps < split_dt
    te_mask = timestamps >= split_dt

    tr_labels = labels[tr_mask]
    te_labels = labels[te_mask]

    state_events = events_df[events_df["Status"] != "NONE"]
    tr_events = state_events[state_events["time"] < split_dt]
    te_events = state_events[state_events["time"] >= split_dt]

    # Detect sensor interval
    if len(timestamps) > 1:
        diffs = np.diff(timestamps.values.astype("int64")) / 1e9  # seconds
        median_interval = np.median(diffs)
        interval_str = f"{median_interval:.0f}s"
        if median_interval >= 60:
            interval_str = f"{median_interval/60:.0f}min"
    else:
        interval_str = "unknown"

    def _stats(lab, name):
        n = len(lab)
        labeled = lab[lab >= 0]
        unlabeled = (lab == -1).sum()
        occ = (labeled == 1).sum() if len(labeled) > 0 else 0
        emp = (labeled == 0).sum() if len(labeled) > 0 else 0
        print(f"  {name}:")
        print(f"    Total rows:       {n:,}")
        print(f"    Labeled:          {len(labeled):,} ({100*len(labeled)/n:.1f}%)")
        print(f"    Unlabeled (-1):   {unlabeled:,} ({100*unlabeled/n:.1f}%)")
        if len(labeled) > 0:
            print(f"    Occupied (1):     {occ:,} ({100*occ/len(labeled):.1f}%)")
            print(f"    Empty (0):        {emp:,} ({100*emp/len(labeled):.1f}%)")
        return len(labeled)

    print()
    print("=" * 65)
    print("  OCCUPANCY TRAIN/TEST SPLIT REPORT")
    print("=" * 65)
    print()
    print(f"  Sensor interval:    {interval_str}")
    print(f"  Sensor time range:  {timestamps[0]} ~ {timestamps[-1]}")
    print(f"  Total sensor rows:  {len(timestamps):,}")
    # Format split label: "YYYY-MM-DD (midnight)" or exact timestamp
    if split_dt == split_dt.normalize():
        split_label = f"{split_dt.strftime('%Y-%m-%d')} (midnight)"
    else:
        split_label = str(split_dt)
    print(f"  Split point:        {split_label}")
    print(f"  Split day-of-week:  {split_dt.day_name()}")
    print()

    # Event summary
    print(f"  State-changing events:")
    print(f"    Total:  {len(state_events):,}")
    print(f"    Train:  {len(tr_events):,}")
    print(f"    Test:   {len(te_events):,}")
    if len(tr_events) > 0 and len(te_events) > 0:
        gap = (te_events["time"].iloc[0] - tr_events["time"].iloc[-1])
        gap_hours = gap.total_seconds() / 3600
        print(f"    Gap:    {gap_hours:.1f}h "
              f"({tr_events['time'].iloc[-1].strftime('%m/%d %H:%M')} → "
              f"{te_events['time'].iloc[0].strftime('%m/%d %H:%M')})")
    print()

    n_tr = _stats(tr_labels, "Train")
    print()
    n_te = _stats(te_labels, "Test")
    print()

    total_labeled = n_tr + n_te
    print(f"  Labeled ratio:      "
          f"train {100*n_tr/total_labeled:.1f}% / "
          f"test {100*n_te/total_labeled:.1f}%")
    print(f"  Total labeled:      {total_labeled:,} / {len(timestamps):,} "
          f"({100*total_labeled/len(timestamps):.1f}%)")

    # Context window safety check
    print()
    print("  Context window safety (test set):")
    te_timestamps = timestamps[te_mask]
    te_labeled_idx = np.where(te_labels >= 0)[0]
    if len(te_labeled_idx) > 0:
        first_labeled_offset = te_labeled_idx[0]
        last_labeled_offset = te_labeled_idx[-1]
        rows_after_last = len(te_labels) - 1 - last_labeled_offset
        print(f"    Sensor rows before first labeled: {first_labeled_offset:,}")
        print(f"    Sensor rows after last labeled:   {rows_after_last:,}")
        for ctx in [5, 30, 60, 120, 360, 720]:
            usable = np.sum(
                (te_labeled_idx >= ctx)
                & (te_labeled_idx + ctx < len(te_labels))
            )
            pct = 100 * usable / len(te_labeled_idx) if len(te_labeled_idx) > 0 else 0
            flag = " ✓" if pct > 90 else " ⚠" if pct > 50 else " ✗"
            print(f"    ctx={ctx:>4d} ({ctx*2+1:>5d} window): "
                  f"{usable:,}/{len(te_labeled_idx):,} usable "
                  f"({pct:.1f}%){flag}")

    # Validation checks
    print()
    print("  Validation:")
    errors = []

    # Check labels sum
    all_labeled = labels[labels >= 0]
    if len(all_labeled) != total_labeled:
        errors.append(f"Label count mismatch: {len(all_labeled)} vs {total_labeled}")

    # Check no labels outside event range
    event_first = state_events["time"].min()
    event_last = state_events["time"].max()
    labeled_ts = timestamps[labels >= 0]
    if len(labeled_ts) > 0:
        if labeled_ts[0] < event_first:
            errors.append(
                f"Labels exist before first event: {labeled_ts[0]} < {event_first}"
            )
        if labeled_ts[-1] > event_last:
            errors.append(
                f"Labels exist after last event: {labeled_ts[-1]} > {event_last}"
            )

    # Check train/test don't overlap
    if tr_mask.sum() + te_mask.sum() != len(timestamps):
        errors.append("Train/test split doesn't cover all timestamps")

    if errors:
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("    ✓ All checks passed")

    print()
    print("=" * 65)


# ============================================================================
# Split Analysis
# ============================================================================

def analyze_split_points(
    sensor_df: pd.DataFrame,
    labels: np.ndarray,
    events_df: pd.DataFrame,
    candidates: list[str] | None = None,
):
    """Analyze multiple split point candidates with per-day statistics.

    Prints:
      1. Per-day breakdown: labeled count, occupied/empty counts, occupancy rate
      2. Comparison table for each candidate split point:
         train/test sizes, ratio, class balance
      3. Context window safety for each candidate
    """
    timestamps = sensor_df.index

    if candidates is None:
        candidates = [
            "2026-02-15 00:00:00",
            "2026-02-16 00:00:00",
            "2026-02-16 07:00:00",
            "2026-02-17 00:00:00",
            "2026-02-17 07:00:00",
            "2026-02-18 00:00:00",
        ]

    # --- Per-day breakdown ---
    print()
    print("=" * 80)
    print("  PER-DAY STATISTICS (occupancy labels)")
    print("=" * 80)
    print()
    print(f"  {'Date':<12} {'Day':<5} {'Total':>7} {'Labeled':>8} "
          f"{'Occ(1)':>8} {'Emp(0)':>8} {'Occ%':>7}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    dates = sorted(set(timestamps.date))
    for d in dates:
        day_mask = np.array([dt.date() == d for dt in timestamps.to_pydatetime()])
        day_labels = labels[day_mask]
        n_total = len(day_labels)
        labeled = day_labels[day_labels >= 0]
        n_labeled = len(labeled)
        n_occ = int((labeled == 1).sum())
        n_emp = int((labeled == 0).sum())
        occ_pct = 100.0 * n_occ / n_labeled if n_labeled > 0 else 0.0
        dow = pd.Timestamp(d).strftime("%a")
        print(f"  {str(d):<12} {dow:<5} {n_total:>7,} {n_labeled:>8,} "
              f"{n_occ:>8,} {n_emp:>8,} {occ_pct:>6.1f}%")

    total_labeled = int((labels >= 0).sum())
    total_occ = int((labels == 1).sum())
    total_emp = int((labels == 0).sum())
    print(f"  {'TOTAL':<12} {'':5} {len(labels):>7,} {total_labeled:>8,} "
          f"{total_occ:>8,} {total_emp:>8,} "
          f"{100*total_occ/total_labeled:.1f}%")

    # --- Split point comparison ---
    print()
    print("=" * 80)
    print("  SPLIT POINT COMPARISON")
    print("=" * 80)
    print()
    print(f"  {'Split Point':<24} {'Tr_Lab':>8} {'Te_Lab':>8} "
          f"{'Ratio':>8} {'Tr_Occ%':>8} {'Te_Occ%':>8} "
          f"{'|ΔOcc%|':>8}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    best_split = None
    best_balance = float("inf")

    for sp_str in candidates:
        sp = pd.Timestamp(sp_str)
        tr_mask = timestamps < sp
        te_mask = timestamps >= sp

        tr_labels = labels[tr_mask]
        te_labels = labels[te_mask]

        tr_labeled = tr_labels[tr_labels >= 0]
        te_labeled = te_labels[te_labels >= 0]

        n_tr = len(tr_labeled)
        n_te = len(te_labeled)
        total = n_tr + n_te

        if total == 0 or n_tr == 0 or n_te == 0:
            print(f"  {sp_str:<24} {'SKIP':>8} {'':>8} {'':>8} {'':>8} {'':>8}")
            continue

        tr_occ_pct = 100.0 * (tr_labeled == 1).sum() / n_tr
        te_occ_pct = 100.0 * (te_labeled == 1).sum() / n_te
        delta_occ = abs(tr_occ_pct - te_occ_pct)
        ratio_str = f"{100*n_tr/total:.0f}:{100*n_te/total:.0f}"

        print(f"  {sp_str:<24} {n_tr:>8,} {n_te:>8,} "
              f"{ratio_str:>8} {tr_occ_pct:>7.1f}% {te_occ_pct:>7.1f}% "
              f"{delta_occ:>7.1f}%")

        # Track best balance (smallest |ΔOcc%|, with minimum test set size)
        if n_te >= 1000 and delta_occ < best_balance:
            best_balance = delta_occ
            best_split = sp_str

    print()
    if best_split:
        print(f"  → Best balanced split (min |ΔOcc%|, test≥1000): {best_split} "
              f"(|ΔOcc%|={best_balance:.1f}%)")

    # --- Context window safety for top candidates ---
    print()
    print("=" * 80)
    print("  CONTEXT WINDOW SAFETY (test set usability)")
    print("=" * 80)
    print()

    ctx_sizes = [5, 30, 60, 120, 200, 360]
    header_parts = [f"  {'Split Point':<24}"]
    for ctx in ctx_sizes:
        header_parts.append(f"ctx={ctx:>3d}")
    print("  ".join(header_parts))
    print(f"  {'-'*24} " + "  ".join(["-" * 7] * len(ctx_sizes)))

    for sp_str in candidates:
        sp = pd.Timestamp(sp_str)
        te_mask = timestamps >= sp
        te_labels = labels[te_mask]
        te_labeled_idx = np.where(te_labels >= 0)[0]

        if len(te_labeled_idx) == 0:
            continue

        parts = [f"  {sp_str:<24}"]
        for ctx in ctx_sizes:
            usable = int(np.sum(
                (te_labeled_idx >= ctx)
                & (te_labeled_idx + ctx < len(te_labels))
            ))
            pct = 100 * usable / len(te_labeled_idx)
            parts.append(f"{pct:>5.0f}%{'✓' if pct > 90 else '⚠' if pct > 50 else '✗'}")
        print("  ".join(parts))

    print()
    print("=" * 80)
    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare train/test sensor CSVs with occupancy labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sensor-csv", required=True,
        help="Path to sensor CSV (any resampling interval)",
    )
    parser.add_argument(
        "--events-csv", required=True,
        help="Path to occupancy events CSV (time, Status, At-home count)",
    )
    parser.add_argument(
        "--split-date", default="2026-02-17",
        help="Split at midnight of this date (default: 2026-02-17). "
             "Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--split-time", default=None,
        help="Split at exact timestamp (overrides --split-date). "
             "Format: 'YYYY-MM-DD HH:MM:SS', e.g. '2026-02-17 07:00:00'",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze mode: compare multiple split points, no output files",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save output CSVs (required unless --analyze)",
    )
    parser.add_argument(
        "--columns-from", default=None,
        help="Reference CSV to read column names from (for headerless input)",
    )
    parser.add_argument(
        "--label-column", default="occupancy_label",
        help="Name of the label column in output (default: occupancy_label)",
    )
    parser.add_argument(
        "--no-binarize", action="store_true",
        help="Keep raw At-home count instead of binarizing to 0/1",
    )
    args = parser.parse_args()

    # Validate args
    if not args.analyze and args.output_dir is None:
        parser.error("--output-dir is required when not in --analyze mode")

    print(f"Loading sensor data: {args.sensor_csv}")
    sensor_df = load_sensor_csv(args.sensor_csv, columns_from=args.columns_from)
    print(f"  Loaded {len(sensor_df):,} rows, {len(sensor_df.columns)} columns")
    print(f"  Time range: {sensor_df.index[0]} ~ {sensor_df.index[-1]}")

    print(f"\nLoading events: {args.events_csv}")
    events_df = load_events_csv(args.events_csv)
    state_events = events_df[events_df["Status"] != "NONE"]
    print(f"  Loaded {len(events_df)} events ({len(state_events)} state-changing)")
    print(f"  Event range: {events_df['time'].min()} ~ {events_df['time'].max()}")

    print(f"\nComputing occupancy labels (binarize={not args.no_binarize})...")
    labels = compute_occupancy_labels(
        sensor_df.index,
        events_df,
        binarize=not args.no_binarize,
    )
    n_labeled = (labels >= 0).sum()
    print(f"  Labeled: {n_labeled:,} / {len(labels):,} "
          f"({100*n_labeled/len(labels):.1f}%)")

    # --- Analyze mode: compare split points and exit ---
    if args.analyze:
        analyze_split_points(sensor_df, labels, events_df)
        return

    # --- Split mode ---
    # Determine split point: --split-time overrides --split-date
    if args.split_time:
        split_point = args.split_time
        split_label = f"exact time ({args.split_time})"
    else:
        split_point = args.split_date
        split_label = f"midnight {args.split_date}"

    # Derive base name from sensor CSV filename
    sensor_stem = Path(args.sensor_csv).stem
    base_name = sensor_stem

    print(f"\nSplitting at {split_label}...")
    train_path, test_path = split_and_save(
        sensor_df, labels,
        split_point=split_point,
        output_dir=args.output_dir,
        base_name=base_name,
        label_column=args.label_column,
    )
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")

    print_stats(sensor_df, labels, split_point, events_df)

    print(f"\nOutput files:")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
