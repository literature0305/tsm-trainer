"""LTSF dataset downloader: URL or HF → CSV.

Stage 1: Download CSV files → {root}/benchmarks/ltsf/{name}.csv
Stage 2: No-op (LTSF adapter reads CSV directly with pd.read_csv).
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
# Dataset URL registry
# ---------------------------------------------------------------------------

_ETT_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
)

LTSF_DOWNLOAD_URLS: dict[str, str | None] = {
    "ETTh1": f"{_ETT_BASE_URL}/ETTh1.csv",
    "ETTh2": f"{_ETT_BASE_URL}/ETTh2.csv",
    "ETTm1": f"{_ETT_BASE_URL}/ETTm1.csv",
    "ETTm2": f"{_ETT_BASE_URL}/ETTm2.csv",
    "Weather": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/weather.csv",
    "Exchange": None,       # requires manual download
    "Traffic": None,        # requires manual download (~160 MB)
    "Electricity": None,    # requires manual download (~120 MB)
    "ILI": None,            # requires manual download (CDC ILI data)
}

# HuggingFace community-hosted fallbacks (thuml organisation)
LTSF_HF_FALLBACK: dict[str, str] = {
    "Weather": "thuml/weather",
    "Exchange": "thuml/exchange_rate",
    "Traffic": "thuml/traffic",
    "Electricity": "thuml/electricity",
    "ILI": "thuml/illness",
}


def _out_path(root: Path, name: str) -> Path:
    return root / "benchmarks" / "ltsf" / f"{name}.csv"


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
    """Download a single LTSF CSV file."""
    name = spec.name
    csv_path = _out_path(root, name)

    if csv_path.exists() and not force:
        logger.info(f"  [Stage 1 SKIP] ltsf/{name} — CSV exists at {csv_path}")
        return True

    url = LTSF_DOWNLOAD_URLS.get(name)

    if dry_run:
        source = url or f"HF: {LTSF_HF_FALLBACK.get(name, 'manual')}"
        logger.info(f"  [Stage 1 DRY RUN] Would download ltsf/{name} from {source}")
        return True

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ok = False

    # Try direct URL first
    if url:
        for attempt in range(1, retries + 1):
            ok = _download_url(url, csv_path)
            if ok:
                break
            if attempt < retries:
                logger.info(f"    Retrying in {retry_backoff}s …")
                time.sleep(retry_backoff)

    # Try HuggingFace fallback
    if not ok and name in LTSF_HF_FALLBACK:
        ok = _download_from_hf(LTSF_HF_FALLBACK[name], csv_path)

    if not ok:
        logger.warning(
            f"  [Stage 1 WARN] ltsf/{name} — automatic download failed.\n"
            f"    Manual download required. Place CSV at: {csv_path}"
        )

    return ok


def _download_url(url: str, save_path: Path) -> bool:
    """Download CSV from a direct URL."""
    logger.info(f"    Downloading {url} …")
    start = time.time()
    try:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy:
            ph = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
            opener = urllib.request.build_opener(ph)
        else:
            opener = urllib.request.build_opener()

        req = urllib.request.Request(url, headers={"User-Agent": "tsm-trainer/1.0"})
        with opener.open(req, timeout=300) as resp:
            data = resp.read()

        save_path.write_bytes(data)
        elapsed = time.time() - start
        size_mb = len(data) / (1024 * 1024)
        logger.info(f"    OK: {save_path.name} — {size_mb:.1f} MB ({elapsed:.1f}s)")
        return True
    except Exception as e:
        logger.warning(f"    URL download failed: {e}")
        return False


def _download_from_hf(dataset_id: str, save_path: Path) -> bool:
    """Download dataset from HuggingFace and save as CSV."""
    logger.info(f"    HF fallback: {dataset_id} …")
    start = time.time()
    try:
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset(dataset_id, split="train")
        df = ds.to_pandas()
        df.to_csv(save_path, index=False)
        elapsed = time.time() - start
        logger.info(
            f"    OK (HF): {save_path.name} — {len(df)} rows, "
            f"{len(df.columns)} cols ({elapsed:.1f}s)"
        )
        return True
    except ImportError:
        logger.warning("    datasets library not installed; cannot use HF fallback")
        return False
    except Exception as e:
        logger.warning(f"    HF fallback failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Stage 2: Convert (no-op)
# ---------------------------------------------------------------------------

def convert(
    spec: "DatasetSpec",
    root: Path,
    **kwargs,
) -> bool:
    """No-op — LTSF adapter reads CSV directly."""
    csv_path = _out_path(root, spec.name)
    logger.info(
        f"  [Stage 2 SKIP] ltsf/{spec.name} — no conversion needed.\n"
        f"    CSV available at: {csv_path}"
    )
    return True
