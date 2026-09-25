#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五福 × Alpha088：同一套池子、走弱、最短持仓、止损防御，score 换成 Kakushadze #088。

不用 25 日 W² 动量。Alpha088 在 ETF 截面上按原文公式计算（截面 rank，IC 在 A 股库里约 +0.06）：
  min(
    rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)), 8)),
    Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3)
  )
adv60 用 60 日均量。需要 high/low，预热 100 日。不覆盖 wufu / wufu_ns。

用法：
  python -m StrategyEngine --mode backtest --strategy wufu_a088 --start 2023-01-01
"""

from __future__ import annotations

import logging
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from StrategyEngine.allocators import Allocator
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import n_field, rebalance_field
from Strategies.rebalance_schedule import RebalanceSpec, parse_rebalance, spec_from_cli
from Strategies.wufu import (
    EtfMetrics,
    VOLUME_THRESHOLD,
    passed_loss_filter,
    passed_ma_filter,
    volume_ratio_daily,
)
from Strategies.wufu_ns import (
    DEFAULT_N,
    KEEP_RATIO,
    MIN_HOLD_DAYS,
    STOP_DEFENSE_DAYS,
    STOP_RATIO,
    WufuXinChun,
)

logger = logging.getLogger("Strategies")

DEFAULT_REBALANCE = "daily"
WARMUP_DAYS = 100
ALPHA_NAME = "alpha088"


def alpha088_wide(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """与 `_alpha101_engine.Alphas.alpha088` 同一公式，不拉行业中性面板。"""
    from DailyUpdates.factor_updates.factors._alpha101_engine import (
        correlation,
        decay_linear,
        rank,
        sma,
        ts_rank,
    )

    p1 = rank(
        decay_linear((rank(open_) + rank(low)) - (rank(high) + rank(close)), 8)
    )
    adv60 = sma(volume, 60)
    p2 = ts_rank(
        decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7),
        3,
    )
    return p1.where(p1 <= p2, p2)


class WufuAlpha088(WufuXinChun):
    name = "wufu_a088"

    @classmethod
    def from_cli(cls, args, allocator):
        n = args.n if getattr(args, "n", None) is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return cls(n=n, allocator=allocator, rebalance=spec)

    @classmethod
    def cli_fields(cls):
        return [n_field(DEFAULT_N), rebalance_field(DEFAULT_REBALANCE)]

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        del args
        return {
            "asset_class": "etf",
            "market_fields": ("close", "high", "low", "vol", "amount"),
            "factors": (),
        }

    @classmethod
    def warmup(cls, args) -> int:
        del args
        return WARMUP_DAYS

    @classmethod
    def run_tag(cls, args) -> str:
        n = args.n if getattr(args, "n", None) is not None else DEFAULT_N
        spec = parse_rebalance(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return f"wufu_a088_h{int(n)}{spec.tag_suffix()}"

    def __init__(
        self,
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = DEFAULT_REBALANCE,
        names: Optional[Mapping[str, str]] = None,
        index_close: Optional[Mapping[str, pd.Series]] = None,
        list_dates: Optional[Mapping[str, str]] = None,
        alpha088: Optional[pd.DataFrame] = None,
    ):
        super().__init__(
            n=n,
            allocator=allocator,
            rebalance=rebalance,
            names=names,
            index_close=index_close,
            list_dates=list_dates,
        )
        self._alpha: Optional[pd.DataFrame] = (
            alpha088.copy() if alpha088 is not None else None
        )
        self._asof = ""

    def score(self, ctx: DayContext) -> TargetHoldings:
        self._asof = ctx.asof
        return super().score(ctx)

    def _ensure(self, ctx: DayContext) -> None:
        super()._ensure(ctx)
        if self._alpha is not None:
            return
        store = ctx._store
        need = ("open", "high", "low", "close", "vol")
        missing = [f for f in need if f not in getattr(store, "market", {})]
        if missing:
            logger.warning("alpha088 缺行情字段 %s，当日 score 为空", missing)
            self._alpha = pd.DataFrame(
                np.nan, index=list(store.dates), columns=list(store.symbols)
            )
            return
        dates = [str(d) for d in store.dates]
        symbols = list(store.symbols)

        def _frame(field: str) -> pd.DataFrame:
            return pd.DataFrame(
                np.asarray(store.panel_for(field), dtype=np.float64),
                index=dates,
                columns=symbols,
            )

        logger.info("计算 ETF 截面 Alpha088，%s 日 × %s 只", len(dates), len(symbols))
        self._alpha = alpha088_wide(
            _frame("open"),
            _frame("high"),
            _frame("low"),
            _frame("close"),
            _frame("vol"),
        )
        self._alpha.index = dates
        self._alpha.columns = symbols

    def _metrics(
        self,
        symbols: Sequence[str],
        pool: np.ndarray,
        universe: np.ndarray,
        closes: np.ndarray,
        vols: np.ndarray,
    ) -> List[EtfMetrics]:
        scores = self._score_row(symbols)
        rows: List[EtfMetrics] = []
        today_close = closes[-1]
        today_vol = vols[-1]
        past_vol = vols[:-1]
        for i, in_pool in enumerate(pool):
            if not in_pool or not universe[i]:
                continue
            px = closes[:, i]
            if not np.isfinite(today_close[i]) or today_close[i] <= 0:
                continue
            score = float(scores[i]) if scores is not None else float("nan")
            if not np.isfinite(score):
                continue
            vol_ratio = volume_ratio_daily(past_vol[:, i], float(today_vol[i]))
            rows.append(
                EtfMetrics(
                    symbol=str(symbols[i]),
                    name=self._names.get(str(symbols[i]), str(symbols[i])),
                    score=score,
                    r2=1.0,
                    volume_ratio=vol_ratio,
                    passed_momentum=True,
                    passed_r2=True,
                    passed_ma=passed_ma_filter(px),
                    passed_volume=vol_ratio is not None and vol_ratio < VOLUME_THRESHOLD,
                    passed_loss=passed_loss_filter(px),
                )
            )
        return rows

    def _score_row(self, symbols: Sequence[str]) -> Optional[np.ndarray]:
        if self._alpha is None or not self._asof:
            return None
        if self._asof not in self._alpha.index:
            return np.full(len(symbols), np.nan, dtype=np.float64)
        row = self._alpha.loc[self._asof]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row.reindex([str(s) for s in symbols]).to_numpy(dtype=np.float64)


# 父类常量再导出，避免测试/调用方从本模块找最短持仓等。
__all__ = [
    "ALPHA_NAME",
    "DEFAULT_N",
    "DEFAULT_REBALANCE",
    "KEEP_RATIO",
    "MIN_HOLD_DAYS",
    "STOP_DEFENSE_DAYS",
    "STOP_RATIO",
    "WARMUP_DAYS",
    "WufuAlpha088",
    "alpha088_wide",
]
