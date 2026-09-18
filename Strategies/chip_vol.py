#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""昨日低位筹码沉积做备选，当日 vol_rel_ma5 市场分位做触发。

asof 日 T（收盘可见，T+1 开盘成交）：
1. 备选：T-1 的 chip_low_deposit 截面 Top K（默认 50）
2. 触发：T 日 vol_rel_ma5 在可交易股票里的截面分位 >= 0.8（当日最强的一截低位放量），
   且因子值 > 0（仍要求落在低位放量一侧，避免熊市里买「相对没那么差」的高位放量）
3. 在触发子集里按当日 vol_rel_ma5 取最多 n 只等权；无人触发则空仓

分位随当天市场截面浮动，不是一条固定阈值。筹码只用到昨天，触发用当天收盘量价。

用法：
  python -m StrategyEngine --mode backtest --strategy chip_vol --start 2023-01-01
"""

from __future__ import annotations

import numpy as np

from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

CHIP_FACTOR = "chip_low_deposit"
TRIGGER_FACTOR = "vol_rel_ma5"
CANDIDATE_N = 50
DEFAULT_N = 10
DEFAULT_REBALANCE = "daily"
TRIGGER_PCT = 0.80


def chip_pool(chip_yest: np.ndarray, eligible: np.ndarray, candidate_n: int) -> np.ndarray:
    """昨日筹码 Top K，返回 bool 掩码。"""
    ok = np.asarray(eligible, dtype=bool) & np.isfinite(chip_yest)
    k = int(min(max(1, int(candidate_n)), int(ok.sum())))
    if k <= 0:
        return np.zeros(len(chip_yest), dtype=bool)
    ranked = np.where(ok, np.asarray(chip_yest, dtype=np.float64), -np.inf)
    idx = np.argpartition(ranked, -k)[-k:]
    out = np.zeros(len(chip_yest), dtype=bool)
    out[idx] = ok[idx]
    return out


def market_pct(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask 有效样本上的截面分位，最低 0、最高 1。"""
    out = np.full(len(values), np.nan, dtype=np.float64)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(values)
    n = int(ok.sum())
    if n <= 0:
        return out
    x = np.asarray(values, dtype=np.float64)[ok]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    if n == 1:
        out[ok] = 1.0
    else:
        out[ok] = ranks / float(n - 1)
    return out


def trigger_picks(
    vol_today: np.ndarray,
    pool: np.ndarray,
    pct: np.ndarray,
    hold_n: int,
    pct_floor: float = TRIGGER_PCT,
) -> np.ndarray:
    """备选 ∩ 市场分位达标 ∩ 因子为正，再按因子取最多 hold_n 只。"""
    fired = (
        np.asarray(pool, dtype=bool)
        & np.isfinite(vol_today)
        & (vol_today > 0)
        & np.isfinite(pct)
        & (pct >= float(pct_floor))
    )
    k = int(min(max(1, int(hold_n)), int(fired.sum())))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    ranked = np.where(fired, np.asarray(vol_today, dtype=np.float64), -np.inf)
    idx = np.argpartition(ranked, -k)[-k:]
    idx = idx[fired[idx]]
    idx = idx[np.argsort(-ranked[idx], kind="mergesort")]
    return np.asarray(idx[:k], dtype=np.int64)


class ChipVolTrigger(Strategy):
    name = "chip_vol"

    @classmethod
    def from_cli(cls, args, allocator):
        n = args.n if args.n is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return cls(n=n, candidate_n=CANDIDATE_N, allocator=allocator, rebalance=spec)

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        return {
            "factors": [CHIP_FACTOR, TRIGGER_FACTOR],
            "market_fields": ("close", "up_limit", "down_limit"),
        }

    @classmethod
    def warmup(cls, args) -> int:
        return 5

    @classmethod
    def run_tag(cls, args) -> str:
        n = args.n if args.n is not None else DEFAULT_N
        spec = parse_rebalance(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return f"chip_vol_n{n}_c{CANDIDATE_N}_p{int(TRIGGER_PCT * 100)}{spec.tag_suffix()}"

    def __init__(
        self,
        n: int = DEFAULT_N,
        candidate_n: int = CANDIDATE_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = DEFAULT_REBALANCE,
    ):
        self.n = max(1, int(n))
        self.candidate_n = max(1, int(candidate_n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self._gate = RebalanceGate(self.rebalance)

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held
        chip_hist = ctx.history(CHIP_FACTOR, 2)
        chip_yest = np.asarray(chip_hist[0], dtype=np.float64)
        vol_today = np.asarray(ctx.factor(TRIGGER_FACTOR), dtype=np.float64)
        eligible = ctx.universe() & self._tradable(ctx)
        pool = chip_pool(chip_yest, eligible, self.candidate_n)
        pct = market_pct(vol_today, eligible)
        picks = trigger_picks(vol_today, pool, pct, self.n)
        if picks.size == 0:
            return self._gate.remember(
                TargetHoldings.from_array(
                    ctx.asof, ctx.execute_on, ctx.symbols, np.zeros(len(ctx.symbols))
                )
            )
        weights = self.allocator.size(ctx, picks)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )

    def _tradable(self, ctx: DayContext) -> np.ndarray:
        n = len(ctx.symbols)
        try:
            close = ctx.market("close")
            up = ctx.market("up_limit")
            down = ctx.market("down_limit")
        except KeyError:
            return np.ones(n, dtype=bool)
        has = np.isfinite(up) & np.isfinite(down) & np.isfinite(close)
        return ~has | ((close != up) & (close != down))
