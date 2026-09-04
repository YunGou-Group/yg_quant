#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""远期收益计算者。契约 B：open[T+1+N] / open[T+1] - 1。唯一允许 shift open。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd


# 与分层净值路径同一上下限：坏价/漏复权会造成 100× 伪收益，不截断会把整张图拉爆。
FWD_RETURN_CLIP = (-0.999, 10.0)


class ReturnCalculator:
    ASOF = "close_T"
    LABEL_TEMPLATE = "open_T+1_to_open_T+1+{n}"

    def __init__(self, open_panel: pd.DataFrame, lru_size: int = 2):
        if open_panel is None or open_panel.empty:
            raise ValueError("open 面板不能为空")
        if lru_size < 1:
            raise ValueError("lru_size 至少为 1")
        frame = open_panel.copy()
        frame.index = pd.to_datetime(frame.index).strftime("%Y-%m-%d")
        frame = frame.sort_index()
        self.open = frame.astype("float64")
        self._lru_size = int(lru_size)
        self._cache: "OrderedDict[int, pd.DataFrame]" = OrderedDict()

    @property
    def dates(self) -> pd.Index:
        return self.open.index

    @property
    def symbols(self) -> pd.Index:
        return self.open.columns

    @classmethod
    def label(cls, horizon: int) -> str:
        return cls.LABEL_TEMPLATE.format(n=int(horizon))

    def get(self, horizon: int) -> pd.DataFrame:
        n = int(horizon)
        if n < 1:
            raise ValueError(f"horizon 必须为正整数，收到 {horizon!r}")
        cached = self._cache.get(n)
        if cached is not None:
            self._cache.move_to_end(n)
            return cached
        computed = self._compute(n)
        self._cache[n] = computed
        while len(self._cache) > self._lru_size:
            self._cache.popitem(last=False)
        return computed

    def metadata(self, horizon: int) -> Tuple[str, str]:
        return self.label(horizon), self.ASOF

    def _compute(self, n: int) -> pd.DataFrame:
        entry = self.open.shift(-1)
        exit_px = self.open.shift(-(1 + n))
        ret = exit_px / entry - 1.0
        ret = ret.replace([np.inf, -np.inf], np.nan)
        ret = ret.where(entry.notna() & exit_px.notna() & (entry != 0))
        lo, hi = FWD_RETURN_CLIP
        return ret.clip(lower=lo, upper=hi)
