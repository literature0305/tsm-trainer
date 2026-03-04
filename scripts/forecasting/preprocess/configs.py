"""Dataset registry and config scanner for the download-and-prepare pipeline.

DatasetSpec: immutable description of a single dataset to download/convert.
ConfigScanner: reads evaluation YAML configs and returns a list of DatasetSpec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHRONOS_TRAIN_CORPORA: list[str] = [
    "training_corpus_tsmixup_10m",
    "training_corpus_kernel_synth_1m",
]

DEFAULT_HF_REPO = "autogluon/chronos_datasets"
ETT_HF_REPO = "autogluon/chronos_datasets_extra"

# YAML file names that signal fev/gift/ltsf benchmarks (not chronos-style lists)
_SPECIAL_YAML_NAMES = {"fev-bench.yaml", "gift-eval.yaml", "ltsf.yaml"}


# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    kind: Literal["chronos_train", "chronos_valid", "fev-bench", "gift-eval", "ltsf"]
    hf_repo: str
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"DatasetSpec(name={self.name!r}, kind={self.kind!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ett(name: str) -> bool:
    """Return True for ETTm* / ETTh* dataset names."""
    return name.startswith("ETTm") or name.startswith("ETTh")


def _matches_any(value: str, patterns: list[str]) -> bool:
    """Return True if value matches any of the (glob-capable) patterns."""
    for pat in patterns:
        if fnmatch(value, pat) or pat == value or pat in value:
            return True
    return False


def _parse_patterns(pattern_str: str | None) -> list[str] | None:
    """Split comma-separated pattern string into a list, or None if not given."""
    if not pattern_str:
        return None
    return [p.strip() for p in pattern_str.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# ConfigScanner
# ---------------------------------------------------------------------------

class ConfigScanner:
    """Scan evaluation YAML configs and produce DatasetSpec lists."""

    def scan(
        self,
        config_dir: str | Path,
        datasets_filter: list[str] | None = None,
        subset_pattern: str | None = None,
    ) -> list[DatasetSpec]:
        """Return DatasetSpecs matching the given filters.

        Parameters
        ----------
        config_dir:
            Directory containing evaluation YAML configs
            (e.g., scripts/forecasting/evaluation/configs/).
        datasets_filter:
            Which kinds to include, e.g. ["chronos_valid", "fev-bench"].
            None or ["all"] means all kinds.
        subset_pattern:
            Comma-separated names / glob patterns applied to dataset names
            *and* config file stems.  E.g. "chronos-lite", "m4_*,nn5",
            "ETTh1,ETTm1".  None means include everything.
        """
        config_dir = Path(config_dir)
        all_kinds = {"chronos_train", "chronos_valid", "fev-bench", "gift-eval", "ltsf"}

        if datasets_filter is None or "all" in datasets_filter:
            active_kinds = all_kinds
        else:
            active_kinds = set(datasets_filter) & all_kinds

        patterns = _parse_patterns(subset_pattern)
        specs: list[DatasetSpec] = []

        if "chronos_train" in active_kinds:
            specs.extend(self._scan_chronos_train(patterns))

        if "chronos_valid" in active_kinds:
            specs.extend(self._scan_chronos_valid(config_dir, patterns))

        if "fev-bench" in active_kinds:
            if (config_dir / "fev-bench.yaml").exists():
                specs.append(DatasetSpec(
                    name="fev_bench",
                    kind="fev-bench",
                    hf_repo="autogluon/fev_datasets",
                ))

        if "gift-eval" in active_kinds:
            if (config_dir / "gift-eval.yaml").exists():
                specs.append(DatasetSpec(
                    name="gift_eval",
                    kind="gift-eval",
                    hf_repo="Salesforce/GiftEval",
                ))

        if "ltsf" in active_kinds:
            ltsf_yaml = config_dir / "ltsf.yaml"
            if ltsf_yaml.exists():
                specs.extend(self._scan_ltsf(ltsf_yaml, patterns))

        return specs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_chronos_train(self, patterns: list[str] | None) -> list[DatasetSpec]:
        out = []
        for corpus in CHRONOS_TRAIN_CORPORA:
            if patterns and not _matches_any(corpus, patterns):
                continue
            out.append(DatasetSpec(
                name=corpus,
                kind="chronos_train",
                hf_repo=DEFAULT_HF_REPO,
            ))
        return out

    def _scan_chronos_valid(
        self,
        config_dir: Path,
        patterns: list[str] | None,
    ) -> list[DatasetSpec]:
        """Parse all chronos-style list YAMLs for benchmark datasets."""
        seen: set[str] = set()
        specs: list[DatasetSpec] = []

        yaml_files = sorted(config_dir.glob("*.yaml"))
        for yaml_path in yaml_files:
            if yaml_path.name in _SPECIAL_YAML_NAMES:
                continue

            # Load content; skip non-list YAMLs (fev/gift/ltsf use dict format)
            try:
                with open(yaml_path) as f:
                    content = yaml.safe_load(f)
            except Exception:
                continue
            if not isinstance(content, list):
                continue

            config_stem = yaml_path.stem  # e.g. "chronos-lite"

            # Determine whether the config file itself matches the pattern.
            # If a pattern like "chronos-lite" is given, we include all datasets
            # in chronos-lite.yaml regardless of individual dataset name.
            config_matches = patterns is None or _matches_any(config_stem, patterns)

            for entry in content:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name:
                    continue
                if name in seen:
                    continue

                # Include if config file matches OR individual dataset name matches
                ds_matches = patterns is None or _matches_any(name, patterns)
                if not config_matches and not ds_matches:
                    continue

                seen.add(name)

                # Determine hf_repo
                hf_repo = entry.get("hf_repo")
                if hf_repo is None:
                    hf_repo = ETT_HF_REPO if _is_ett(name) else DEFAULT_HF_REPO

                specs.append(DatasetSpec(
                    name=name,
                    kind="chronos_valid",
                    hf_repo=hf_repo,
                    meta={k: v for k, v in entry.items() if k not in ("name", "hf_repo")},
                ))

        return specs

    def _scan_ltsf(
        self,
        yaml_path: Path,
        patterns: list[str] | None,
    ) -> list[DatasetSpec]:
        """Extract LTSF dataset entries from ltsf.yaml."""
        try:
            with open(yaml_path) as f:
                content = yaml.safe_load(f)
        except Exception:
            return []
        if not isinstance(content, dict):
            return []

        specs = []
        for entry in content.get("datasets", []):
            if isinstance(entry, dict):
                name = entry.get("name", "")
                meta = {k: v for k, v in entry.items() if k != "name"}
            elif isinstance(entry, str):
                name = entry
                meta = {}
            else:
                continue

            if not name:
                continue
            if patterns and not _matches_any(name, patterns):
                continue

            specs.append(DatasetSpec(
                name=name,
                kind="ltsf",
                hf_repo="",
                meta=meta,
            ))
        return specs
