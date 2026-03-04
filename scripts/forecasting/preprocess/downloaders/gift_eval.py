"""GIFT-Eval downloader: HuggingFace snapshot_download.

Stage 1: Download Salesforce/GiftEval → {root}/benchmarks/gift_eval/
Stage 2: No-op (gift_eval library uses GIFT_EVAL env var for data access).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.forecasting.preprocess.configs import DatasetSpec

logger = logging.getLogger(__name__)

GIFT_EVAL_REPO = "Salesforce/GiftEval"


def _out_dir(root: Path) -> Path:
    return root / "benchmarks" / "gift_eval"


def _is_downloaded(out_dir: Path) -> bool:
    """Check if snapshot was already downloaded (has any files)."""
    if not out_dir.exists():
        return False
    file_count = sum(1 for f in out_dir.rglob("*") if f.is_file())
    return file_count > 10  # expect many files in a real download


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
    """Download GIFT-Eval snapshot from HuggingFace."""
    out_dir = _out_dir(root)

    if _is_downloaded(out_dir) and not force:
        logger.info(f"  [Stage 1 SKIP] gift-eval — snapshot exists at {out_dir}")
        return True

    if dry_run:
        logger.info(f"  [Stage 1 DRY RUN] Would download {GIFT_EVAL_REPO} → {out_dir}")
        return True

    logger.info(f"  [Stage 1] gift-eval — downloading {GIFT_EVAL_REPO} → {out_dir}")
    logger.info("    This may take a while (dataset is several GB) …")
    out_dir.mkdir(parents=True, exist_ok=True)

    import time
    for attempt in range(1, retries + 1):
        start = time.time()
        ok = _download_via_snapshot(out_dir)
        elapsed = time.time() - start
        if ok:
            logger.info(f"    OK: {out_dir} ({elapsed:.0f}s)")
            return True
        logger.warning(f"    Attempt {attempt} FAILED ({elapsed:.1f}s)")
        if attempt == 1:
            logger.info("    Trying huggingface-cli fallback …")
            ok = _download_via_cli(out_dir)
            if ok:
                return True
        if attempt < retries:
            time.sleep(5.0)

    logger.error(f"  [Stage 1 FAILED] gift-eval — all attempts exhausted")
    return False


def _download_via_snapshot(out_dir: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(GIFT_EVAL_REPO, repo_type="dataset", local_dir=str(out_dir))
        return True
    except ImportError:
        logger.error("huggingface_hub not installed")
        return False
    except Exception as e:
        logger.warning(f"    snapshot_download failed: {e}")
        return False


def _download_via_cli(out_dir: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "huggingface-cli", "download",
                GIFT_EVAL_REPO,
                "--repo-type=dataset",
                f"--local-dir={out_dir}",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            return True
        logger.warning(f"    CLI failed: {result.stderr[:200]}")
        return False
    except FileNotFoundError:
        logger.warning("    huggingface-cli not found")
        return False
    except Exception as e:
        logger.warning(f"    CLI error: {e}")
        return False


# ---------------------------------------------------------------------------
# Stage 2: Convert (no-op)
# ---------------------------------------------------------------------------

def convert(
    spec: "DatasetSpec",
    root: Path,
    **kwargs,
) -> bool:
    """No-op — gift_eval library uses GIFT_EVAL environment variable."""
    out_dir = _out_dir(root)
    logger.info(
        f"  [Stage 2 SKIP] gift-eval — no conversion needed.\n"
        f"    Set GIFT_EVAL={out_dir} when running gift-eval evaluations."
    )
    return True
