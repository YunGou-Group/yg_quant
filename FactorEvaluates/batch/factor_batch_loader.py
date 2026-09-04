#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按批读取因子，对齐到共享日期/股票轴。"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..factor_panel_loader import FactorPanelLoader
from .shared_panel_store import SharedPanelStore


class FactorBatchLoader:
    def __init__(self, store: SharedPanelStore, loader: FactorPanelLoader, batch_size: int = 32):
        self.store = store
        self.loader = loader
        self.batch_size = max(1, int(batch_size))

    def chunks(self, names: Sequence[str]) -> List[List[str]]:
        return [
            list(names[i : i + self.batch_size])
            for i in range(0, len(names), self.batch_size)
        ]

    def load_cube(self, chunk: Sequence[str]) -> np.ndarray:
        """把一批因子对齐到共享的 (日期, 股票) 轴上。"""
        dates = self.store.dates
        symbols = self.store.symbols
        date_index = pd.Index(dates)
        symbol_index = pd.Index(symbols)
        chunk = list(chunk)
        panels = self.loader.load_many(
            chunk,
            start_date=dates[0],
            end_date=dates[-1],
            symbols=symbols,
        )
        cube = np.full(
            (len(dates), len(symbols), len(chunk)), np.nan, dtype=np.float32
        )
        for j, name in enumerate(chunk):
            panel = panels.get(name)
            if panel is None or panel.empty:
                continue
            if not isinstance(panel.index, pd.Index) or panel.index.dtype != object:
                panel = panel.copy()
                panel.index = pd.to_datetime(panel.index).strftime("%Y-%m-%d")
            if list(panel.columns[:1]) and str(panel.columns[0]) != str(symbols[0]):
                panel = panel.copy()
                panel.columns = [str(c) for c in panel.columns]
            aligned = panel.reindex(index=date_index, columns=symbol_index)
            cube[:, :, j] = aligned.to_numpy(dtype=np.float32, copy=False)
        return cube

    def iter_batches(
        self, names: Sequence[str]
    ) -> Iterable[Tuple[List[str], np.ndarray]]:
        for chunk in self.chunks(names):
            yield chunk, self.load_cube(chunk)
