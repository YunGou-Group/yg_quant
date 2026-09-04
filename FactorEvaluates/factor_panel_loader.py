#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子面板加载者：从 BinStorage 读成日期 × 股票宽表。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from DailyUpdates.storage import BinStorage
from yg_quant_repo import default_factor_dir


class FactorPanelLoader:
    def __init__(self, factor_dir: Optional[str] = None):
        root = factor_dir or str(default_factor_dir())
        self.storage = BinStorage(root)
        self.root = Path(self.storage.root)

    def list_factors(self) -> list:
        return self.storage.list_stored_factor_names()

    def calendar(self) -> list:
        return self.storage.read_calendar()

    def load(
        self,
        factor_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.storage.read_factor_panel(
            factor_name,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

    def load_many(
        self,
        factor_names: Sequence[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        return self.storage.read_factor_panels(
            factor_names,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )
