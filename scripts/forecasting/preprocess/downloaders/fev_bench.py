"""fev-bench downloader: HuggingFace snapshot_download.

Stage 1: Download autogluon/fev_datasets → {root}/benchmarks/fev_bench/
Stage 2: No-op (fev library manages its own data access).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.forecasting.preprocess.configs import DatasetSpec

logger = logging.getLogger(__name__)

FEV_DATASETS_REPO = "autogluon/fev_datasets"


def _out_dir(root: Path) -> Path:
    return root / "benchmarks" / "fev_bench"


def _is_downloaded(out_dir: Path) -> bool:
    """Check if snapshot was already downloaded (has any parquet files)."""
    if not out_dir.exists():
        return False
    return any(out_dir.rglob("*.parquet"))


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------

def download(
    spec: "DatasetSpec",
    root: Path,
    force: bool = False,
    resume: bool = False,
    retries: int = 3,
    dry_run: bool = False,
) -> bool:
    """Download fev_datasets snapshot from HuggingFace."""
    out_dir = _out_dir(root)

    if _is_downloaded(out_dir) and not force:
        logger.info(f"  [Stage 1 SKIP] fev-bench — snapshot exists at {out_dir}")
        return True

    if dry_run:
        logger.info(f"  [Stage 1 DRY RUN] Would download {FEV_DATASETS_REPO} → {out_dir}")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False

    logger.info(f"  [Stage 1] fev-bench — downloading {FEV_DATASETS_REPO} → {out_dir}")
    logger.info("    This may take a while (80+ dataset configs) …")
    out_dir.mkdir(parents=True, exist_ok=True)

    import time
    for attempt in range(1, retries + 1):
        start = time.time()
        try:
            snapshot_download(
                FEV_DATASETS_REPO,
                repo_type="dataset",
                local_dir=str(out_dir),
            )
            elapsed = time.time() - start
            logger.info(f"    OK: {out_dir} ({elapsed:.0f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"    Attempt {attempt} FAILED ({elapsed:.1f}s): {e}")
            if attempt < retries:
                import time as t
                t.sleep(5.0)

    logger.error(f"  [Stage 1 FAILED] fev-bench — all {retries} attempts exhausted")
    return False


# ---------------------------------------------------------------------------
# Stage 2: Convert (no-op)
# ---------------------------------------------------------------------------

def convert(
    spec: "DatasetSpec",
    root: Path,
    **kwargs,
) -> bool:
    """No-op — fev library auto-manages data access."""
    out_dir = _out_dir(root)
    logger.info(
        f"  [Stage 2 SKIP] fev-bench — no conversion needed.\n"
        f"    Set HF_DATASETS_CACHE={out_dir} when running fev evaluations."
    )
    return True
