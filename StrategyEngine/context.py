#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交易日快照：策略当天唯一输入。回测与以后的 QMT 共用。"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PanelView(Protocol):
    """截面数据源。回测是 PanelStore，实盘可以换成 QMT 快照。"""

    symbols: List[str]
    mask: np.ndarray
    dates: List[str]

    def market_row(self, field: str, t: int) -> np.ndarray: ...

    def factor_row(self, name: str, t: int) -> np.ndarray: ...

    def panel_for(self, field: str) -> np.ndarray: ...


class DayContext:
    def __init__(
        self,
        *,
        store: PanelView,
        t: int,
        asof: str,
        execute_on: Optional[str],
        position: np.ndarray,
        cash: float,
        value: float,
    ):
        self._store = store
        self._t = int(t)
        self.asof = str(asof)
        self.execute_on = execute_on
        self.symbols: List[str] = store.symbols
        self.position = np.asarray(position, dtype=np.float64)
        self.cash = float(cash)
        self.value = float(value)

    def universe(self) -> np.ndarray:
        return self._store.mask[self._t].copy()

    def market(self, field: str) -> np.ndarray:
        return self._store.market_row(field, self._t)

    def factor(self, name: str) -> np.ndarray:
        return self._store.factor_row(name, self._t)

    def history(self, field: str, lookback: int) -> np.ndarray:
        """窗口右端为 asof（含当天），不会读到未来。"""
        n = int(lookback)
        if n < 1:
            raise ValueError("lookback 必须为正整数")
        panel = self._store.panel_for(field)
        t1 = self._t + 1
        t0 = max(0, t1 - n)
        sl = panel[t0:t1]
        if sl.shape[0] == n:
            return sl.copy()
        pad = np.full((n - sl.shape[0], sl.shape[1]), np.nan, dtype=np.float64)
        return np.vstack([pad, sl])
