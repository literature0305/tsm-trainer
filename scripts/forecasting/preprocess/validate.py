"""Stage 3: Validate downloaded and converted datasets.

Each validate_* function attempts to load the dataset and checks:
  - Existence and non-emptiness
  - Expected field / column structure
  - Prints sample statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.forecasting.preprocess.configs import DatasetSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    name: str
    kind: str
    ok: bool
    message: str
    stats: dict | None = None

    def log(self) -> None:
        status = "OK" if self.ok else "FAIL"
        logger.info(f"  [Stage 3 {status}] {self.kind}/{self.name}: {self.message}")
        if self.stats:
            for k, v in self.stats.items():
                logger.info(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def validate(spec: "DatasetSpec", root: Path) -> ValidationResult:
    """Validate a dataset based on its kind."""
    if spec.kind == "chronos_valid":
        return validate_chronos_valid(spec, root)
    elif spec.kind == "chronos_train":
        return validate_chronos_train(spec, root)
    elif spec.kind == "fev-bench":
        return validate_fev_bench(root)
    elif spec.kind == "gift-eval":
        return validate_gift_eval(root)
    elif spec.kind == "ltsf":
        return validate_ltsf(spec, root)
    else:
        return ValidationResult(
            name=spec.name, kind=spec.kind, ok=False,
            message=f"Unknown kind: {spec.kind!r}",
        )


# ---------------------------------------------------------------------------
# chronos_valid
# ---------------------------------------------------------------------------

def validate_chronos_valid(spec: "DatasetSpec", root: Path) -> ValidationResult:
    """Validate a single benchmark Arrow dataset (plain Dataset)."""
    path = root / "benchmarks" / "chronos" / spec.name
    name, kind = spec.name, "chronos_valid"

    if not path.exists():
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"Path does not exist: {path}")

    try:
        import datasets as hf_datasets
        ds = hf_datasets.load_from_disk(str(path))
    except Exception as e:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"load_from_disk failed: {e}")

    if len(ds) == 0:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message="Dataset is empty")

    # Check for a time-series column (usually "target" or a Sequence column)
    seq_cols = [
        col for col, feat in ds.features.items()
        if hasattr(feat, "feature")  # datasets.Sequence has .feature attribute
    ]

    # Sample statistics
    sample_n = min(10, len(ds))
    sample = ds.select(range(sample_n))
    lengths = {}
    for col in seq_cols[:3]:
        vals = sample[col]
        try:
            lens = [len(v) for v in vals]
            lengths[col] = f"len: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.0f}"
        except Exception:
            lengths[col] = "?"

    stats = {
        "rows": len(ds),
        "columns": ds.column_names,
        "sequence_cols": seq_cols,
        **lengths,
    }
    return ValidationResult(
        name=name, kind=kind, ok=True,
        message=f"{len(ds):,} rows, {len(ds.column_names)} columns",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# chronos_train
# ---------------------------------------------------------------------------

def validate_chronos_train(spec: "DatasetSpec", root: Path) -> ValidationResult:
    """Validate a training corpus (DatasetDict with 'train' split)."""
    path = root / "chronos_datasets" / spec.name
    name, kind = spec.name, "chronos_train"

    if not path.exists():
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"Path does not exist: {path}")

    try:
        import datasets as hf_datasets
        ds_dict = hf_datasets.load_from_disk(str(path))
    except Exception as e:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"load_from_disk failed: {e}")

    if "train" not in ds_dict:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"Missing 'train' split. Keys: {list(ds_dict.keys())}")

    ds = ds_dict["train"]
    if len(ds) == 0:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message="'train' split is empty")

    seq_cols = [
        col for col, feat in ds.features.items()
        if hasattr(feat, "feature")
    ]

    stats = {
        "rows": len(ds),
        "columns": ds.column_names,
        "sequence_cols": seq_cols,
    }
    return ValidationResult(
        name=name, kind=kind, ok=True,
        message=f"{len(ds):,} rows in 'train' split",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# fev-bench
# ---------------------------------------------------------------------------

def validate_fev_bench(root: Path) -> ValidationResult:
    """Validate fev-bench snapshot (counts Parquet files)."""
    path = root / "benchmarks" / "fev_bench"
    name, kind = "fev_bench", "fev-bench"

    if not path.exists():
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"Path does not exist: {path}")

    parquet_files = list(path.rglob("*.parquet"))
    if not parquet_files:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message="No Parquet files found")

    total_size = sum(f.stat().st_size for f in parquet_files)
    size_mb = total_size / (1024 * 1024)

    stats = {
        "parquet_files": len(parquet_files),
        "size_mb": f"{size_mb:.1f}",
    }

    # Optionally verify via fev library
    fev_ok = _try_fev_load(path)
    if fev_ok is not None:
        stats["fev_task_load"] = "OK" if fev_ok else "FAILED"

    return ValidationResult(
        name=name, kind=kind, ok=True,
        message=f"{len(parquet_files)} Parquet files, {size_mb:.1f} MB",
        stats=stats,
    )


def _try_fev_load(data_path: Path) -> bool | None:
    """Attempt to load one fev task. Returns None if fev not installed."""
    try:
        import fev
        import os
        old_cache = os.environ.get("HF_DATASETS_CACHE")
        os.environ["HF_DATASETS_CACHE"] = str(data_path)
        try:
            tasks_url = (
                "https://github.com/autogluon/fev/raw/refs/heads/main/"
                "benchmarks/fev_bench/tasks.yaml"
            )
            benchmark = fev.Benchmark.from_yaml(tasks_url)
            if benchmark.tasks:
                task = benchmark.tasks[0]
                for window in task.iter_windows():
                    _ = window.get_input_data()
                    break
            return True
        except Exception:
            return False
        finally:
            if old_cache is None:
                os.environ.pop("HF_DATASETS_CACHE", None)
            else:
                os.environ["HF_DATASETS_CACHE"] = old_cache
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# gift-eval
# ---------------------------------------------------------------------------

def validate_gift_eval(root: Path) -> ValidationResult:
    """Validate gift-eval snapshot (counts files and size)."""
    path = root / "benchmarks" / "gift_eval"
    name, kind = "gift_eval", "gift-eval"

    if not path.exists():
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"Path does not exist: {path}")

    files = [f for f in path.rglob("*") if f.is_file()]
    if not files:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message="No files found")

    total_size = sum(f.stat().st_size for f in files)
    size_mb = total_size / (1024 * 1024)
    stats = {
        "files": len(files),
        "size_mb": f"{size_mb:.1f}",
    }
    return ValidationResult(
        name=name, kind=kind, ok=True,
        message=f"{len(files)} files, {size_mb:.1f} MB",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# ltsf
# ---------------------------------------------------------------------------

def validate_ltsf(spec: "DatasetSpec", root: Path) -> ValidationResult:
    """Validate a single LTSF CSV file."""
    csv_path = root / "benchmarks" / "ltsf" / f"{spec.name}.csv"
    name, kind = spec.name, "ltsf"

    if not csv_path.exists():
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"CSV not found: {csv_path}")

    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=5)
        # Count total rows without loading full file
        with open(csv_path) as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header

        size_mb = csv_path.stat().st_size / (1024 * 1024)
        expected = spec.meta  # may contain n_vars, freq, horizons

        stats: dict = {
            "rows": total_rows,
            "cols": len(df.columns),
            "size_mb": f"{size_mb:.1f}",
            "columns_sample": df.columns.tolist()[:8],
        }

        ok = total_rows > 0 and len(df.columns) > 1
        msg = f"{total_rows} rows × {len(df.columns)} cols, {size_mb:.1f} MB"
        return ValidationResult(name=name, kind=kind, ok=ok, message=msg, stats=stats)

    except Exception as e:
        return ValidationResult(name=name, kind=kind, ok=False,
                                message=f"pd.read_csv failed: {e}")
