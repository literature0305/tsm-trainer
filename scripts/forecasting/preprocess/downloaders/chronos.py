"""Chronos dataset downloader: HF → Parquet (Stage 1) and Parquet → Arrow (Stage 2).

Handles both:
  - chronos_valid: benchmark evaluation datasets
  - chronos_train: training corpora

Stage 1 (download):
  Saves raw Parquet to:
    chronos_valid  →  {root}/benchmarks_raw/chronos/{name}/train-*.parquet
    chronos_train  →  {root}/chronos_datasets_raw/{name}/train-*.parquet

Stage 2 (convert):
  Reads Parquet, saves Arrow to:
    chronos_valid  →  {root}/benchmarks/chronos/{name}/       (plain Dataset)
    chronos_train  →  {root}/chronos_datasets/{name}/         (DatasetDict)
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.forecasting.preprocess.configs import DatasetSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raw_dir(root: Path, spec: "DatasetSpec") -> Path:
    """Return the raw Parquet directory for this spec."""
    if spec.kind == "chronos_valid":
        return root / "benchmarks_raw" / "chronos" / spec.name
    else:  # chronos_train
        return root / "chronos_datasets_raw" / spec.name


def _out_dir(root: Path, spec: "DatasetSpec") -> Path:
    """Return the final Arrow output directory for this spec."""
    if spec.kind == "chronos_valid":
        return root / "benchmarks" / "chronos" / spec.name
    else:  # chronos_train
        return root / "chronos_datasets" / spec.name


def _collect_parquet_files(parquet_dir: Path) -> list[str]:
    return sorted(str(f) for f in parquet_dir.glob("train-*.parquet"))


def _check_arrow_ready(path: Path, require_dict: bool = False) -> bool:
    if require_dict:
        return (path / "dataset_dict.json").exists() and (path / "train").exists()
    return (path / "dataset_info.json").exists() or (path / "dataset_dict.json").exists()


def _fmt_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


# ---------------------------------------------------------------------------
# ETT CSV fallback (for autogluon/chronos_datasets_extra incompatibility)
# ---------------------------------------------------------------------------

ETT_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
_ETT_GROUPS = {
    "ETTh": ["ETTh1", "ETTh2"],
    "ETTm": ["ETTm1", "ETTm2"],
}
_ETT_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
)


def _is_ett_group(name: str) -> bool:
    return name in _ETT_GROUPS


def _download_ett_csv_to_arrow(name: str, parquet_dir: Path) -> bool:
    """Download ETT CSVs and save as a Parquet file (ETT group datasets).

    Works for name="ETTh" (→ ETTh1, ETTh2) or name="ETTm" (→ ETTm1, ETTm2).
    Falls back to individual CSVs if the group name isn't recognized.
    """
    import pandas as pd

    subsets = _ETT_GROUPS.get(name)
    if subsets is None:
        # Single dataset, e.g. "ETTh1"
        subsets = [name]

    all_timestamps: list[list] = []
    all_data: dict[str, list] = {col: [] for col in ETT_COLUMNS}

    for subset_name in subsets:
        url = f"{_ETT_BASE_URL}/{subset_name}.csv"
        logger.info(f"    [{subset_name}] Downloading CSV from {url}")

        df = None

        # Strategy 1: urllib
        try:
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
            if proxy:
                ph = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
                opener = urllib.request.build_opener(ph)
            else:
                opener = urllib.request.build_opener()
            req = urllib.request.Request(url, headers={"User-Agent": "tsm-trainer/1.0"})
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                opener.retrieve(url, tmp_path)
            except AttributeError:
                with opener.open(req, timeout=300) as resp:
                    tmp_path_obj = Path(tmp_path)
                    tmp_path_obj.write_bytes(resp.read())
            df = pd.read_csv(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            logger.info(f"    [{subset_name}] OK — {len(df)} rows")
        except Exception as e:
            logger.warning(f"    [{subset_name}] urllib failed: {e}")

        # Strategy 2: pandas read_csv directly
        if df is None:
            try:
                df = pd.read_csv(url)
                logger.info(f"    [{subset_name}] OK (pandas) — {len(df)} rows")
            except Exception as e:
                logger.warning(f"    [{subset_name}] pandas failed: {e}")

        # Strategy 3: huggingface_hub
        if df is None:
            try:
                from huggingface_hub import hf_hub_download
                local_path = hf_hub_download(
                    repo_id="zhouhaoyi/ETDataset",
                    filename=f"ETT-small/{subset_name}.csv",
                    repo_type="dataset",
                )
                df = pd.read_csv(local_path)
                logger.info(f"    [{subset_name}] OK (hf_hub) — {len(df)} rows")
            except Exception as e:
                logger.warning(f"    [{subset_name}] hf_hub failed: {e}")

        if df is None:
            logger.error(f"    [{subset_name}] All download strategies failed")
            return False

        # Find date column
        date_col = next(
            (c for c in ["date", "Date", "datetime", "timestamp"] if c in df.columns),
            None,
        )
        if date_col is None:
            logger.error(f"    [{subset_name}] No date column found")
            return False

        all_timestamps.append(pd.to_datetime(df[date_col]).tolist())
        missing = [c for c in ETT_COLUMNS if c not in df.columns]
        if missing:
            logger.error(f"    [{subset_name}] Missing columns: {missing}")
            return False
        for col in ETT_COLUMNS:
            all_data[col].append(df[col].tolist())

    # Build Arrow dataset and save as Parquet
    import datasets as hf_datasets

    feature_dict = {"timestamp": hf_datasets.Sequence(hf_datasets.Value("timestamp[s]"))}
    for col in ETT_COLUMNS:
        feature_dict[col] = hf_datasets.Sequence(hf_datasets.Value("float64"))

    data_dict = {"timestamp": all_timestamps, **all_data}
    features = hf_datasets.Features(feature_dict)
    ds = hf_datasets.Dataset.from_dict(data_dict, features=features)

    parquet_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = parquet_dir / "train-00000-of-00001.parquet"
    ds.to_parquet(str(out_parquet))
    logger.info(f"  Saved ETT Parquet: {out_parquet} ({len(ds)} rows)")
    return True


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------

def download(
    spec: "DatasetSpec",
    root: Path,
    force: bool = False,
    resume: bool = False,
    retries: int = 3,
    retry_backoff: float = 5.0,
    dry_run: bool = False,
) -> bool:
    """Download dataset from HuggingFace and save raw Parquet.

    Returns True on success (or skip-because-exists).
    """
    import time as time_mod

    parquet_dir = _raw_dir(root, spec)

    # Skip if already done
    existing = _collect_parquet_files(parquet_dir)
    if existing and not force:
        logger.info(f"  [Stage 1 SKIP] {spec.name} — raw Parquet exists ({len(existing)} files)")
        return True

    if dry_run:
        logger.info(f"  [Stage 1 DRY RUN] Would download {spec.hf_repo}/{spec.name}")
        return True

    # ETT group datasets require special handling
    if _is_ett_group(spec.name):
        logger.info(f"  [Stage 1] {spec.name} — ETT CSV fallback")
        return _download_ett_csv_to_arrow(spec.name, parquet_dir)

    # Standard HF download with retry
    import datasets as hf_datasets

    trust_remote = spec.hf_repo == "autogluon/chronos_datasets_extra"

    for attempt in range(1, retries + 1):
        logger.info(
            f"  [Stage 1] {spec.name} — downloading from {spec.hf_repo} "
            f"(attempt {attempt}/{retries})"
        )
        start = time_mod.time()
        try:
            ds = hf_datasets.load_dataset(
                spec.hf_repo,
                spec.name,
                split="train",
                trust_remote_code=trust_remote,
            )
            parquet_dir.mkdir(parents=True, exist_ok=True)
            out_path = parquet_dir / "train-00000-of-00001.parquet"
            logger.info(f"    Loaded {len(ds):,} rows, saving Parquet …")
            ds.to_parquet(str(out_path))
            elapsed = time_mod.time() - start
            logger.info(f"    OK: {len(ds):,} rows → {out_path} ({elapsed:.1f}s)")
            return True
        except Exception as e:
            elapsed = time_mod.time() - start
            logger.warning(f"    Attempt {attempt} FAILED ({elapsed:.1f}s): {e}")
            if attempt < retries:
                logger.info(f"    Retrying in {retry_backoff}s …")
                time_mod.sleep(retry_backoff)

    # Final fallback for ETT-style individual datasets
    if spec.name.startswith("ETT"):
        logger.info(f"  [Stage 1] Trying ETT CSV fallback for {spec.name} …")
        return _download_ett_csv_to_arrow(spec.name, parquet_dir)

    logger.error(f"  [Stage 1 FAILED] {spec.name} — all {retries} attempts exhausted")
    return False


# ---------------------------------------------------------------------------
# Stage 2: Convert (Parquet → Arrow)
# ---------------------------------------------------------------------------

def convert(
    spec: "DatasetSpec",
    root: Path,
    num_proc: int = 1,
    max_series: int | None = None,
    seed: int = 42,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Convert raw Parquet to HuggingFace Arrow dataset.

    Returns True on success (or skip-because-exists).
    """
    import datasets as hf_datasets

    parquet_dir = _raw_dir(root, spec)
    out_path = _out_dir(root, spec)

    # Check if already done
    require_dict = (spec.kind == "chronos_train")
    if _check_arrow_ready(out_path, require_dict=require_dict) and not force:
        logger.info(f"  [Stage 2 SKIP] {spec.name} — Arrow already exists at {out_path}")
        return True

    # Collect raw Parquet files
    parquet_files = _collect_parquet_files(parquet_dir)
    if not parquet_files:
        logger.error(
            f"  [Stage 2 FAIL] {spec.name} — no Parquet files in {parquet_dir}. "
            f"Run Stage 1 first."
        )
        return False

    if dry_run:
        logger.info(
            f"  [Stage 2 DRY RUN] Would convert {len(parquet_files)} Parquet(s) → {out_path}"
        )
        return True

    logger.info(
        f"  [Stage 2] {spec.name} — {len(parquet_files)} Parquet file(s) → {out_path}"
    )
    start = time.time()
    try:
        ds = hf_datasets.load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            split="train",
        )
        logger.info(f"    Loaded {len(ds):,} rows, columns: {ds.column_names}")

        # Optional subsample (training corpora)
        if max_series and len(ds) > max_series:
            import numpy as np
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(ds))[:max_series].tolist()
            indices.sort()
            ds = ds.select(indices)
            logger.info(f"    Subsampled to {len(ds):,} rows (max_series={max_series})")

        out_path.mkdir(parents=True, exist_ok=True)

        if spec.kind == "chronos_train":
            ds_dict = hf_datasets.DatasetDict({"train": ds})
            ds_dict.save_to_disk(str(out_path), num_proc=num_proc if num_proc > 1 else None)
        else:
            ds.save_to_disk(str(out_path), num_proc=num_proc if num_proc > 1 else None)

        elapsed = time.time() - start
        size = _fmt_size(out_path)
        logger.info(f"    OK: {len(ds):,} rows → {out_path} ({size}, {elapsed:.1f}s)")
        return True

    except Exception as e:
        logger.error(f"  [Stage 2 FAILED] {spec.name}: {e}")
        return False
