#!/usr/bin/env python3
"""Unified data-pipeline CLI: download, convert, and validate datasets.

Prepares chronos/fev-bench/gift-eval/ltsf data in training- and
evaluation-ready formats in a single config-driven pipeline.

Stages
------
  Stage 0 — Environment check (disk space, packages, env vars)
  Stage 1 — Download raw data (HF → Parquet or CSV snapshot)
  Stage 2 — Convert to target format (Parquet → Arrow / no-op)
  Stage 3 — Validate output (load + field checks + stats)

Running all stages (default when --stage is omitted):
  python download_and_prepare.py --root_dir /group-volume/ts-dataset --datasets all

Examples
--------
  # Benchmark validation datasets only (chronos-lite config, Stage 2 only)
  python download_and_prepare.py \\
      --root_dir /group-volume/ts-dataset \\
      --datasets chronos_valid \\
      --config_dir scripts/forecasting/evaluation/configs \\
      --subset chronos-lite \\
      --stage 2

  # Full pipeline — dry run
  python download_and_prepare.py \\
      --root_dir /group-volume/ts-dataset \\
      --datasets all \\
      --dry_run

  # Training corpora — convert only, resume interrupted run
  python download_and_prepare.py \\
      --root_dir /group-volume/ts-dataset \\
      --datasets chronos_train \\
      --stage 2 \\
      --resume \\
      --num_proc 4 \\
      --max_series 1000000

  # Validate chronos benchmark + training datasets
  python download_and_prepare.py \\
      --root_dir /group-volume/ts-dataset \\
      --datasets chronos_train chronos_valid \\
      --stage 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest helpers (resume support)
# ---------------------------------------------------------------------------

_MANIFEST_DIR_NAME = ".pipeline_manifest"


def _manifest_path(root: Path, name: str) -> Path:
    return root / _MANIFEST_DIR_NAME / f"{name}.json"


def _load_manifest(root: Path, name: str) -> dict:
    p = _manifest_path(root, name)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_manifest(root: Path, name: str, data: dict) -> None:
    p = _manifest_path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def _is_stage_done(root: Path, name: str, stage: int) -> bool:
    manifest = _load_manifest(root, name)
    return bool(manifest.get(f"stage{stage}_done"))


def _mark_stage_done(root: Path, name: str, stage: int, extra: dict | None = None) -> None:
    manifest = _load_manifest(root, name)
    manifest[f"stage{stage}_done"] = True
    if extra:
        manifest.update(extra)
    _save_manifest(root, name, manifest)


# ---------------------------------------------------------------------------
# Stage 0: Environment check
# ---------------------------------------------------------------------------

def check_environment(args: argparse.Namespace) -> None:
    """Check disk space, Python packages, and environment variables."""
    logger.info("=" * 65)
    logger.info("  Stage 0: Environment check")
    logger.info("=" * 65)

    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Disk space
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    logger.info(f"  Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total at {root}")
    if free_gb < 10:
        logger.warning(f"  Low disk space: {free_gb:.1f} GB — downloads may fail!")

    # Key packages
    _check_pkg("datasets", required=True)
    _check_pkg("huggingface_hub", required=True)
    _check_pkg("yaml", import_as="yaml", required=True)
    _check_pkg("pandas")
    _check_pkg("numpy")
    _check_pkg("fev", optional=True)

    # Env vars
    for var in ["HF_TOKEN", "HF_HOME", "HTTPS_PROXY", "HTTP_PROXY"]:
        val = os.environ.get(var, "")
        if val:
            display = (val[:4] + "****") if "TOKEN" in var else (val[:30] + "...")
            logger.info(f"  {var}={display}")

    logger.info("  Environment check complete.\n")


def _check_pkg(pkg: str, import_as: str | None = None, required: bool = False,
               optional: bool = False) -> None:
    mod = import_as or pkg
    try:
        __import__(mod)
        logger.info(f"  [OK] {pkg}")
    except ImportError:
        if required:
            logger.error(f"  [MISSING] {pkg} — required! Install with: pip install {pkg}")
        elif optional:
            logger.info(f"  [OPTIONAL] {pkg} not installed (some features unavailable)")
        else:
            logger.warning(f"  [MISSING] {pkg}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _valid_stage(val: str) -> int | None:
    if val.lower() in ("all", "none", ""):
        return None
    try:
        s = int(val)
        if s in (0, 1, 2, 3):
            return s
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"--stage must be 0, 1, 2, 3, or 'all'; got {val!r}"
    )


def parse_args() -> argparse.Namespace:
    default_config_dir = str(
        Path(__file__).parent.parent / "evaluation" / "configs"
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core options
    parser.add_argument(
        "--root_dir", required=True,
        help="Root directory for all datasets (e.g., /group-volume/ts-dataset)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all", "chronos_train", "chronos_valid", "fev-bench", "gift-eval", "ltsf"],
        metavar="KIND",
        help=(
            "Which dataset kinds to process. "
            "Choices: all | chronos_train | chronos_valid | fev-bench | gift-eval | ltsf"
        ),
    )
    parser.add_argument(
        "--config_dir", default=default_config_dir,
        help="Directory containing evaluation YAML configs (default: %(default)s)",
    )
    parser.add_argument(
        "--subset", default=None,
        help=(
            "Comma-separated name patterns or config-file stems to filter datasets. "
            "E.g. 'chronos-lite', 'm4_*,nn5', 'ETTh1,ETTm1'"
        ),
    )
    parser.add_argument(
        "--stage", default=None, type=_valid_stage,
        metavar="{0,1,2,3,all}",
        help="Which stage to run (0=env, 1=download, 2=convert, 3=validate). "
             "Default: run all stages 0→3.",
    )

    # Download options
    parser.add_argument("--force", action="store_true",
                        help="Re-download/re-convert even if output already exists")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages already recorded as done in the manifest")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of download retries per dataset (default: %(default)s)")
    parser.add_argument("--retry_backoff", type=float, default=5.0,
                        help="Seconds between retry attempts (default: %(default)s)")

    # Convert options
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Parallel workers for Arrow conversion (default: %(default)s)")
    parser.add_argument("--max_series", type=int, default=None,
                        help="Max series to keep per training corpus (random sample)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --max_series sampling (default: %(default)s)")

    # Misc
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without actually doing it")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    root = Path(args.root_dir)

    # Determine which stages to run
    run_all = args.stage is None
    run_stages = (
        {0, 1, 2, 3} if run_all else {args.stage}
    )

    # ------------------------------------------------------------------
    # Stage 0: Environment check
    # ------------------------------------------------------------------
    if 0 in run_stages:
        check_environment(args)

    # ------------------------------------------------------------------
    # Config scan
    # ------------------------------------------------------------------
    from scripts.forecasting.preprocess.configs import ConfigScanner

    logger.info("=" * 65)
    logger.info("  Config scan")
    logger.info(f"  config_dir : {args.config_dir}")
    logger.info(f"  datasets   : {args.datasets}")
    logger.info(f"  subset     : {args.subset or '(all)'}")
    logger.info("=" * 65)

    scanner = ConfigScanner()
    specs = scanner.scan(
        config_dir=args.config_dir,
        datasets_filter=args.datasets,
        subset_pattern=args.subset,
    )

    if not specs:
        logger.warning("No datasets matched the given filters. Nothing to do.")
        return

    logger.info(f"  Found {len(specs)} dataset(s):")
    for s in specs:
        logger.info(f"    [{s.kind}] {s.name}")
    logger.info("")

    # ------------------------------------------------------------------
    # Import downloader dispatcher
    # ------------------------------------------------------------------
    from scripts.forecasting.preprocess.downloaders import get_downloader

    results: dict[str, dict[int, bool]] = {}

    # ------------------------------------------------------------------
    # Stage 1: Download
    # ------------------------------------------------------------------
    if 1 in run_stages:
        logger.info("=" * 65)
        logger.info("  Stage 1: Download (HF → raw Parquet / snapshot)")
        logger.info("=" * 65)

        for spec in specs:
            name = spec.name
            if args.resume and _is_stage_done(root, name, 1):
                logger.info(f"  [SKIP] {name} — Stage 1 done (resume mode)")
                results.setdefault(name, {})[1] = True
                continue

            downloader = get_downloader(spec.kind)
            ok = downloader.download(
                spec=spec,
                root=root,
                force=args.force,
                resume=args.resume,
                retries=args.retries,
                dry_run=args.dry_run,
                **(_ltsf_extra(args) if spec.kind == "ltsf" else {}),
            )
            results.setdefault(name, {})[1] = ok
            if ok and not args.dry_run:
                _mark_stage_done(root, name, 1)

        _log_stage_summary(specs, results, stage=1)

    # ------------------------------------------------------------------
    # Stage 2: Convert
    # ------------------------------------------------------------------
    if 2 in run_stages:
        logger.info("=" * 65)
        logger.info("  Stage 2: Convert (Parquet / CSV → Arrow)")
        logger.info("=" * 65)

        for spec in specs:
            name = spec.name
            if args.resume and _is_stage_done(root, name, 2):
                logger.info(f"  [SKIP] {name} — Stage 2 done (resume mode)")
                results.setdefault(name, {})[2] = True
                continue

            downloader = get_downloader(spec.kind)
            ok = downloader.convert(
                spec=spec,
                root=root,
                num_proc=args.num_proc,
                max_series=args.max_series,
                seed=args.seed,
                force=args.force,
                dry_run=args.dry_run,
            )
            results.setdefault(name, {})[2] = ok
            if ok and not args.dry_run:
                _mark_stage_done(root, name, 2)

        _log_stage_summary(specs, results, stage=2)

    # ------------------------------------------------------------------
    # Stage 3: Validate
    # ------------------------------------------------------------------
    if 3 in run_stages:
        logger.info("=" * 65)
        logger.info("  Stage 3: Validate")
        logger.info("=" * 65)

        from scripts.forecasting.preprocess.validate import validate

        all_ok = True
        for spec in specs:
            result = validate(spec, root)
            result.log()
            results.setdefault(spec.name, {})[3] = result.ok
            if not result.ok:
                all_ok = False

        logger.info("")
        if all_ok:
            logger.info("  All datasets validated OK.")
        else:
            logger.warning("  Some datasets failed validation — check logs above.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 65)
    if args.dry_run:
        logger.info("  DRY RUN complete — no files were written.")
    else:
        logger.info("  Pipeline complete.")
        _log_final_summary(specs, results, run_stages)
    logger.info("=" * 65)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ltsf_extra(args: argparse.Namespace) -> dict:
    """Extra kwargs only used by the ltsf downloader."""
    return {"retry_backoff": args.retry_backoff}


def _log_stage_summary(specs, results: dict, stage: int) -> None:
    ok_count = sum(1 for s in specs if results.get(s.name, {}).get(stage))
    logger.info(f"\n  Stage {stage} summary: {ok_count}/{len(specs)} OK\n")


def _log_final_summary(specs, results: dict, run_stages: set) -> None:
    logger.info("")
    for spec in specs:
        sr = results.get(spec.name, {})
        parts = [f"S{s}={'OK' if sr.get(s) else 'FAIL'}" for s in sorted(run_stages) if s != 0]
        logger.info(f"  {spec.name}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
