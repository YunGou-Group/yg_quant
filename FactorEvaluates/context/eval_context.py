#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次评估的对齐面板与中间量（数据包，不是执行者）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from ..return_calculator import ReturnCalculator


@dataclass
class EvalContext:
    factor: pd.DataFrame
    fwd_ret: pd.DataFrame
    universe_mask: Optional[pd.DataFrame]
    horizon: int
    label: str
    asof: str = ReturnCalculator.ASOF
    intermediates: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        factor: pd.DataFrame,
        open_panel: pd.DataFrame,
        horizon: int,
        universe_mask: Optional[pd.DataFrame] = None,
        calculator: Optional[ReturnCalculator] = None,
    ) -> "EvalContext":
        n = int(horizon)
        if n < 1:
            raise ValueError(f"horizon 必须为正整数，收到 {horizon!r}")
        engine = calculator if calculator is not None else ReturnCalculator(open_panel)
        fwd_ret = engine.get(n)
        factor_aligned, fwd_aligned, mask_aligned = cls._align(
            factor, fwd_ret, universe_mask
        )
        return cls(
            factor=factor_aligned,
            fwd_ret=fwd_aligned,
            universe_mask=mask_aligned,
            horizon=n,
            label=ReturnCalculator.label(n),
            asof=ReturnCalculator.ASOF,
        )

    def masked_factor(self) -> pd.DataFrame:
        if self.universe_mask is None:
            return self.factor
        return self.factor.where(self.universe_mask)

    def masked_fwd_ret(self) -> pd.DataFrame:
        if self.universe_mask is None:
            return self.fwd_ret
        return self.fwd_ret.where(self.universe_mask)

    @staticmethod
    def _as_date_index(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
        out.columns = [str(col) for col in out.columns]
        return out.sort_index()

    @classmethod
    def _align(
        cls,
        factor: pd.DataFrame,
        fwd_ret: pd.DataFrame,
        universe_mask: Optional[pd.DataFrame] = None,
    ) -> tuple:
        factor_n = cls._as_date_index(factor)
        fwd_n = cls._as_date_index(fwd_ret)
        dates = factor_n.index.intersection(fwd_n.index)
        symbols = factor_n.columns.intersection(fwd_n.columns)
        if len(dates) == 0 or len(symbols) == 0:
            raise ValueError("因子面板与远期收益没有共同的日期或股票")
        factor_a = factor_n.loc[dates, symbols]
        fwd_a = fwd_n.loc[dates, symbols]
        mask_a = None
        if universe_mask is not None:
            mask_n = cls._as_date_index(universe_mask).reindex(
                index=dates, columns=symbols
            )
            mask_a = mask_n.astype("boolean").fillna(False).astype(bool)
        return factor_a, fwd_a, mask_a
