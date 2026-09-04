#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容封装：转调 Universes catalog。新代码请直接 from Universes.catalog import get。"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from Universes.catalog import get as get_universe
from Universes.rules import delist_on as _delist_on
from Universes.rules import iso_day

UNIVERSE_INDEX_CODES = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz1000": "000852.SH",
}

__all__ = ["UNIVERSE_INDEX_CODES", "UniverseFilter", "iso_day"]


class UniverseFilter:
    def __init__(self, storage, min_list_days: int = 60):
        self.storage = storage
        self.min_list_days = int(min_list_days)

    def build_mask(
        self,
        dates: pd.Index,
        symbols: pd.Index,
        open_panel: pd.DataFrame,
        universe: str = "all",
        require_next_open: bool = False,
    ) -> pd.DataFrame:
        """asof 日的可选股票池。默认不看 T+1 开盘。"""
        return get_universe(universe).build_mask(
            dates,
            symbols,
            open_panel,
            self.storage,
            require_next_open=require_next_open,
            min_list_days=self.min_list_days,
        )

    def delist_on(self, symbols: Sequence[str]):
        return _delist_on(symbols, self.storage)
