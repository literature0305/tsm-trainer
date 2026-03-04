"""Downloader modules for each dataset kind."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.forecasting.preprocess.configs import DatasetSpec


def get_downloader(kind: str):
    """Return the downloader module for a given dataset kind."""
    if kind in ("chronos_train", "chronos_valid"):
        from . import chronos
        return chronos
    elif kind == "fev-bench":
        from . import fev_bench
        return fev_bench
    elif kind == "gift-eval":
        from . import gift_eval
        return gift_eval
    elif kind == "ltsf":
        from . import ltsf
        return ltsf
    else:
        raise ValueError(f"Unknown dataset kind: {kind!r}")
