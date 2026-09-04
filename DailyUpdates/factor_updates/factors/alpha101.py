#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WorldQuant Alpha101 thin BaseFactor wrappers.

Each wrapper calls ``extract_alpha`` for its own id. The engine caches the wide
OHLCV panel for the shared market DataFrame, but does not materialize all 101
result columns at once. Industry maps load from ``YG_QUANT_DB_PATH`` / default db.
"""

from __future__ import annotations

from typing import List, Optional, Type

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

from ._alpha101_engine import IMPLEMENTED_ALPHAS, extract_alpha

_ALPHA101_DEPENDENCIES: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "pct_chg",
    "total_mv",
]


def _make_alpha_class(alpha_id: int) -> Type[BaseFactor]:
    """Build a concrete BaseFactor subclass for one alpha id."""

    class _AlphaFactor(BaseFactor):
        def __init__(self, _id: int = alpha_id):
            self.alpha_id = _id
            self.name = f"alpha{_id:03d}"
            self.description = f"WorldQuant Alpha#{_id:03d}"
            self.dependencies = list(_ALPHA101_DEPENDENCIES)
            super().__init__()

        def calculate(
            self, data: pd.DataFrame, start_date: Optional[str] = None
        ) -> pd.DataFrame:
            return extract_alpha(data, self.alpha_id, start_date=start_date)

    _AlphaFactor.__name__ = f"Alpha{alpha_id:03d}Factor"
    _AlphaFactor.__qualname__ = f"Alpha{alpha_id:03d}Factor"
    return _AlphaFactor


# One BaseFactor subclass per implemented alpha for FactorUpdater discovery.
# Nested builder class is not left as a discoverable module attribute.
for _aid in IMPLEMENTED_ALPHAS:
    _cls = _make_alpha_class(_aid)
    globals()[_cls.__name__] = _cls

del _aid, _cls

__all__ = [f"Alpha{n:03d}Factor" for n in IMPLEMENTED_ALPHAS]
