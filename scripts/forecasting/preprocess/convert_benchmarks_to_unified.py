#!/usr/bin/env python3
"""
convert_benchmarks_to_unified.py — Benchmark datasets → unified HF Arrow format
=================================================================================

Converts the four benchmark directories to the unified format used by the
training pipeline, so inspect_dataset.py can produce train/eval statistics
on a common footing.

UNIFIED OUTPUT FORMAT
---------------------
  target    : Sequence(float64)                univariate  (length,)
              Sequence(Sequence(float64))      multivariate (n_variates, length)
  id        : string
  timestamp : Sequence(timestamp[ms])

SOURCE FORMATS HANDLED
-----------------------
  chronos   Arrow Dataset
              columns: timestamp + named variable columns (e.g. HUFL, HULL, …)
              → each row → one multivariate sample

  fev_bench Parquet (per dataset/freq, multiple schema patterns)
    Type A   target + id + timestamp                   → already standard
    Type B   id + timestamp + target_0 … target_N      → stack numbered cols
    Type C   id + timestamp + HUFL … OT               → stack all named cols
    Type D   id + timestamp + target + covariates      → use only target col
    (covariate columns: numeric lists kept only when no explicit target col)

  gift_eval Arrow Dataset (per dataset/freq)
              columns: item_id, start(scalar), freq(scalar), target
              → timestamp reconstructed via pd.date_range(start, freq=freq)

  ltsf      CSV files: date + variable columns
              → entire CSV = one multivariate sample

TARGET COLUMN SELECTION RULES
-------------------------------
  Priority 1 – explicit  : any column named 'target' or matching 'target_\\d+'
  Priority 2 – numeric   : all List[float32/64] columns not in EXCLUDE_COLS
  EXCLUDE_COLS = {id, item_id, timestamp, start, freq, subset, type, split}

OUTPUT STRUCTURE
----------------
  {output_root}/
  ├── chronos/{dataset_name}/
  ├── fev_bench/{dataset_name}__{freq}/    (double-underscore separator)
  ├── gift_eval/{dataset_name}__{freq}/
  └── ltsf/{csv_stem}/

  Each leaf directory is an HF Arrow DatasetDict with split "train".

USAGE
-----
  python convert_benchmarks_to_unified.py \\
      --benchmarks_root /group-volume/ts-dataset/benchmarks \\
      --output_root     /group-volume/ts-dataset/benchmarks_unified \\
      [--benchmarks chronos fev_bench gift_eval ltsf]   # default: all four
      [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Columns to always skip when detecting target columns
_EXCLUDE_COLS: frozenset[str] = frozenset({
    "id", "item_id", "timestamp", "start", "freq",
    "subset", "type", "split", "source",
})


# ──────────────────────────────────────────────────────────────────────────────
# Column detection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _target_cols_from_hf_features(col_names: list[str], features) -> list[str]:
    """Return target column names from HuggingFace dataset features."""
    import datasets as hf

    # Priority 1 – explicit 'target' / 'target_N'
    explicit = [n for n in col_names if n == "target" or re.fullmatch(r"target_\d+", n)]
    if explicit:
        return explicit

    # Priority 2 – all Sequence[numeric] columns not in _EXCLUDE_COLS
    result = []
    for col in col_names:
        if col in _EXCLUDE_COLS:
            continue
        feat = features.get(col)
        if feat is None:
            continue
        if isinstance(feat, hf.Sequence):
            inner = feat.feature
            if isinstance(inner, hf.Value) and ("float" in inner.dtype or "int" in inner.dtype):
                result.append(col)
    return result


def _target_cols_from_pa_schema(schema: pa.Schema) -> list[str]:
    """Return target column names from PyArrow schema."""
    col_names = schema.names

    # Priority 1 – explicit 'target' / 'target_N'
    explicit = [n for n in col_names if n == "target" or re.fullmatch(r"target_\d+", n)]
    if explicit:
        return explicit

    # Priority 2 – all List[float/int] columns not in _EXCLUDE_COLS
    result = []
    for field in schema:
        if field.name in _EXCLUDE_COLS:
            continue
        if isinstance(field.type, (pa.ListType, pa.LargeListType)):
            vt = field.type.value_type
            if pa.types.is_floating(vt) or pa.types.is_integer(vt):
                result.append(field.name)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Row conversion
# ──────────────────────────────────────────────────────────────────────────────

def _build_unified_row(
    row: dict,
    target_cols: list[str],
    sid: str,
    timestamp: list,
) -> tuple[list, str, list] | None:
    """Convert one source row to (target, id, timestamp) or None on failure."""
    arrays = []
    for col in target_cols:
        try:
            arr = np.asarray(row[col], dtype=np.float64).ravel()
        except Exception as exc:
            logger.debug("    col '%s' conversion failed: %s", col, exc)
            continue
        if arr.ndim != 1 or len(arr) == 0:
            continue
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(arr)

    if not arrays:
        return None

    # Ensure all arrays have the same length (use minimum)
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]

    Y = np.stack(arrays)   # (n_variates, length)
    target = Y[0].tolist() if Y.shape[0] == 1 else Y.tolist()

    # Truncate / pad timestamp to match length
    ts = list(timestamp)[:min_len]

    return target, sid, ts


# ──────────────────────────────────────────────────────────────────────────────
# Arrow output
# ──────────────────────────────────────────────────────────────────────────────

def _save_arrow(
    targets: list,
    ids: list[str],
    timestamps: list,
    output_dir: Path,
) -> int:
    """Save as HF Arrow DatasetDict({'train': Dataset}). Returns number of rows."""
    import datasets as hf

    if not targets:
        logger.warning("    No rows — skipping %s", output_dir.name)
        return 0

    first = targets[0]
    is_multi = isinstance(first[0], list)

    if is_multi:
        target_feat = hf.Sequence(hf.Sequence(hf.Value("float64")))
    else:
        target_feat = hf.Sequence(hf.Value("float64"))

    features = hf.Features({
        "target":    target_feat,
        "id":        hf.Value("string"),
        "timestamp": hf.Sequence(hf.Value("timestamp[ms]")),
    })
    ds = hf.Dataset.from_dict(
        {"target": targets, "id": ids, "timestamp": timestamps},
        features=features,
    )
    hf.DatasetDict({"train": ds}).save_to_disk(str(output_dir))

    n_var = len(first) if is_multi else 1
    length = len(first[0]) if is_multi else len(first)
    logger.info(
        "    ✓ %s  rows=%d  dim=%d  len=%d",
        output_dir.name, len(ds), n_var, length,
    )
    return len(ds)


# ──────────────────────────────────────────────────────────────────────────────
# Chronos benchmark converter
# ──────────────────────────────────────────────────────────────────────────────

def convert_chronos(src_root: Path, out_root: Path, overwrite: bool) -> None:
    """Convert chronos Arrow datasets (wide-column format) to unified format."""
    import datasets as hf

    ds_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
    logger.info("chronos: %d datasets", len(ds_dirs))
    total_rows = 0

    for ds_dir in ds_dirs:
        out_dir = out_root / ds_dir.name
        if out_dir.exists() and not overwrite:
            logger.info("  [skip] %s", ds_dir.name)
            continue

        try:
            raw = hf.load_from_disk(str(ds_dir))
            ds = raw["train"] if isinstance(raw, hf.DatasetDict) else raw
            target_cols = _target_cols_from_hf_features(ds.column_names, ds.features)

            if not target_cols:
                logger.warning("  [WARN] %s: no target columns detected", ds_dir.name)
                continue

            logger.info("  %s: %d rows, cols=%s", ds_dir.name, len(ds), target_cols)

            targets, ids, timestamps = [], [], []
            for i in range(len(ds)):
                row = ds[i]
                ts = list(row.get("timestamp", []))
                sid = str(row.get("id", f"{ds_dir.name}_{i:04d}"))
                result = _build_unified_row(row, target_cols, sid, ts)
                if result:
                    t, s, p = result
                    targets.append(t); ids.append(s); timestamps.append(p)

            out_dir.mkdir(parents=True, exist_ok=True)
            total_rows += _save_arrow(targets, ids, timestamps, out_dir)

        except Exception as exc:
            logger.error("  [ERROR] %s: %s", ds_dir.name, exc, exc_info=True)

    logger.info("chronos done — %d total rows written", total_rows)


# ──────────────────────────────────────────────────────────────────────────────
# fev_bench converter
# ──────────────────────────────────────────────────────────────────────────────

def convert_fev_bench(src_root: Path, out_root: Path, overwrite: bool) -> None:
    """Convert fev_bench Parquet files to unified format.

    Iterates over {src_root}/{dataset}/{freq}/train-*.parquet.
    Output key: {dataset}__{freq}  (double underscore).
    """
    ds_names = sorted(d.name for d in src_root.iterdir() if d.is_dir())
    logger.info("fev_bench: %d top-level datasets", len(ds_names))
    total_rows = 0

    for ds_name in ds_names:
        ds_dir = src_root / ds_name
        freq_dirs = sorted(d for d in ds_dir.iterdir() if d.is_dir())

        for freq_dir in freq_dirs:
            freq = freq_dir.name
            out_key = f"{ds_name}__{freq}"
            out_dir = out_root / out_key

            if out_dir.exists() and not overwrite:
                logger.info("  [skip] %s", out_key)
                continue

            parquet_files = sorted(freq_dir.glob("*.parquet"))
            if not parquet_files:
                logger.warning("  [WARN] %s: no parquet files", out_key)
                continue

            try:
                table = pa.concat_tables([pq.read_table(str(f)) for f in parquet_files])
                schema = table.schema
                col_names = schema.names
                target_cols = _target_cols_from_pa_schema(schema)

                if not target_cols:
                    logger.warning("  [WARN] %s: no target columns", out_key)
                    continue

                logger.info(
                    "  %s: %d rows, cols=%s", out_key, len(table), target_cols
                )

                data = table.to_pydict()
                n_rows = len(table)
                targets, ids, timestamps = [], [], []

                for i in range(n_rows):
                    row = {c: data[c][i] for c in col_names}
                    ts_raw = row.get("timestamp")
                    ts = list(ts_raw) if ts_raw is not None else []
                    id_val = str(row.get("id", f"{out_key}_{i:06d}"))
                    result = _build_unified_row(row, target_cols, id_val, ts)
                    if result:
                        t, s, p = result
                        targets.append(t); ids.append(s); timestamps.append(p)

                out_dir.mkdir(parents=True, exist_ok=True)
                total_rows += _save_arrow(targets, ids, timestamps, out_dir)

            except Exception as exc:
                logger.error("  [ERROR] %s: %s", out_key, exc, exc_info=True)

    logger.info("fev_bench done — %d total rows written", total_rows)


# ──────────────────────────────────────────────────────────────────────────────
# gift_eval converter
# ──────────────────────────────────────────────────────────────────────────────

def convert_gift_eval(src_root: Path, out_root: Path, overwrite: bool) -> None:
    """Convert gift_eval Arrow datasets to unified format.

    Schema: item_id (string), start (timestamp scalar), freq (string), target (Sequence[float]).
    Timestamp is reconstructed via pd.date_range(start, periods=len(target), freq=freq).
    Output key: {dataset}__{freq}.
    """
    import datasets as hf

    ds_names = sorted(d.name for d in src_root.iterdir() if d.is_dir() and d.name != "README.md")
    logger.info("gift_eval: %d top-level datasets", len(ds_names))
    total_rows = 0

    for ds_name in ds_names:
        ds_dir = src_root / ds_name
        freq_dirs = sorted(d for d in ds_dir.iterdir() if d.is_dir())

        for freq_dir in freq_dirs:
            freq = freq_dir.name
            out_key = f"{ds_name}__{freq}"
            out_dir = out_root / out_key

            if out_dir.exists() and not overwrite:
                logger.info("  [skip] %s", out_key)
                continue

            try:
                raw = hf.load_from_disk(str(freq_dir))
                ds = raw["train"] if isinstance(raw, hf.DatasetDict) else raw
                logger.info("  %s: %d rows", out_key, len(ds))

                targets, ids, timestamps = [], [], []

                for i in range(len(ds)):
                    row = ds[i]
                    target_raw = row.get("target")
                    if target_raw is None or len(target_raw) == 0:
                        continue

                    arr = np.asarray(target_raw, dtype=np.float64)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    target = arr.tolist()

                    # Reconstruct timestamp from start + freq
                    start = row.get("start")
                    row_freq = str(row.get("freq", freq))
                    try:
                        ts_index = pd.date_range(
                            start=pd.Timestamp(start),
                            periods=len(arr),
                            freq=row_freq,
                        )
                        ts_list = ts_index.to_pydatetime().tolist()
                    except Exception as ts_exc:
                        logger.debug("    %s row %d: timestamp failed (%s)", out_key, i, ts_exc)
                        ts_list = []  # inspect_dataset.py tolerates empty timestamp

                    sid = str(row.get("item_id", f"{out_key}_{i:06d}"))
                    targets.append(target)
                    ids.append(sid)
                    timestamps.append(ts_list)

                out_dir.mkdir(parents=True, exist_ok=True)
                total_rows += _save_arrow(targets, ids, timestamps, out_dir)

            except Exception as exc:
                logger.error("  [ERROR] %s: %s", out_key, exc, exc_info=True)

    logger.info("gift_eval done — %d total rows written", total_rows)


# ──────────────────────────────────────────────────────────────────────────────
# LTSF converter
# ──────────────────────────────────────────────────────────────────────────────

def convert_ltsf(src_root: Path, out_root: Path, overwrite: bool) -> None:
    """Convert LTSF CSV files to unified format.

    Each CSV is one long multivariate time series.
    Columns: date + variable columns (all become target variates).
    One output row per CSV file.
    """
    csv_files = sorted(src_root.glob("*.csv"))
    logger.info("ltsf: %d CSV files", len(csv_files))
    total_rows = 0

    for csv_path in csv_files:
        out_dir = out_root / csv_path.stem
        if out_dir.exists() and not overwrite:
            logger.info("  [skip] %s", csv_path.stem)
            continue

        try:
            df = pd.read_csv(csv_path)

            # Find timestamp column (named 'date' or first non-numeric column)
            date_col = None
            for col in df.columns:
                if col.lower() in ("date", "datetime", "timestamp", "time"):
                    date_col = col
                    break
            if date_col is None:
                date_col = df.columns[0]
                logger.warning("  %s: no 'date' column found, using '%s'", csv_path.name, date_col)

            try:
                ts_series = pd.to_datetime(df[date_col])
                ts_list = ts_series.dt.to_pydatetime().tolist()
            except Exception:
                ts_list = []

            var_cols = [c for c in df.columns if c != date_col]
            if not var_cols:
                logger.warning("  %s: no variable columns", csv_path.name)
                continue

            # Build 2D target (n_variates, length)
            values = df[var_cols].values.astype(np.float64).T  # (n_variates, T)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            target = values[0].tolist() if values.shape[0] == 1 else values.tolist()

            logger.info(
                "  %s: 1 row, dim=%d, length=%d, vars=%s",
                csv_path.stem, len(var_cols), values.shape[1], var_cols,
            )

            out_dir.mkdir(parents=True, exist_ok=True)
            total_rows += _save_arrow(
                [target], [csv_path.stem], [ts_list], out_dir
            )

        except Exception as exc:
            logger.error("  [ERROR] %s: %s", csv_path.name, exc, exc_info=True)

    logger.info("ltsf done — %d total rows written", total_rows)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert benchmark datasets to unified HF Arrow format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--benchmarks_root",
        default="/group-volume/ts-dataset/benchmarks",
        help="Root directory containing chronos/, fev_bench/, gift_eval/, ltsf/",
    )
    p.add_argument(
        "--output_root",
        default="/group-volume/ts-dataset/benchmarks_unified",
        help="Root output directory (default: /group-volume/ts-dataset/benchmarks_unified)",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=["chronos", "fev_bench", "gift_eval", "ltsf"],
        choices=["chronos", "fev_bench", "gift_eval", "ltsf"],
        help="Which benchmark types to convert (default: all four)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-converted datasets (default: skip existing)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks_root = Path(args.benchmarks_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Benchmark → Unified Arrow Converter")
    logger.info("  Source : %s", benchmarks_root)
    logger.info("  Output : %s", output_root)
    logger.info("  Types  : %s", args.benchmarks)
    logger.info("  Overwrite: %s", args.overwrite)
    logger.info("=" * 60)

    converters = {
        "chronos":   (convert_chronos,   "chronos"),
        "fev_bench": (convert_fev_bench, "fev_bench"),
        "gift_eval": (convert_gift_eval, "gift_eval"),
        "ltsf":      (convert_ltsf,      "ltsf"),
    }

    for bench in args.benchmarks:
        fn, subdir = converters[bench]
        src = benchmarks_root / subdir
        out = output_root / subdir
        out.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            logger.warning("[SKIP] %s — source not found: %s", bench, src)
            continue
        logger.info("")
        logger.info("── %s ────────────────────────────────────", bench.upper())
        fn(src, out, args.overwrite)

    logger.info("")
    logger.info("=" * 60)
    logger.info("All done. Output at: %s", output_root)
    logger.info("Run inspect_dataset.py on any subdirectory for statistics.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
