#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五福动量 × Alpha088：按 12-1 动量多空近 5 日累计切换。

日度信号是昨截面 style_momentum 高低 20% 今日收益之差（12-1 赢家减输家）。
切换用这列的 5 日累计：> 0 → wufu_ns；< 0 → wufu_a088。
止损/候选不足仍买银华日利。不覆盖 wufu_ns / wufu_a088。

用法：
  python -m StrategyEngine --mode backtest --strategy wufu_mix --start 2023-01-01
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from StrategyEngine.allocators import Allocator
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import n_field, rebalance_field
from Strategies.rebalance_schedule import RebalanceSpec, parse_rebalance, spec_from_cli
from Strategies.wufu import (
    LIQ_DAYS,
    LOOKBACK_DAYS,
    VOLUME_LOOKBACK,
    WeakState,
    apply_filters,
    avg_daily_yuan,
    liquidity_threshold,
    select_targets,
    step_weak_period,
    weak_vote_need,
)
from Strategies.wufu_a088 import WufuAlpha088
from Strategies.wufu_ns import (
    KEEP_RATIO,
    MIN_HOLD_DAYS,
    WufuXinChun,
    need_defense,
    step_replace,
    _count_days,
)

logger = logging.getLogger("Strategies")

DEFAULT_N = 5
DEFAULT_REBALANCE = "daily"
MOM_LS_FRAC = 0.2
MOM_LS_WINDOW = 5
MOM_LS_MIN_PERIODS = 5
MOM_LS_MIN_STOCKS = 50
LegName = str


def daily_momentum_ls(
    style: pd.DataFrame,
    close: pd.DataFrame,
    *,
    frac: float = MOM_LS_FRAC,
    min_stocks: int = MOM_LS_MIN_STOCKS,
) -> pd.Series:
    """昨截面动量风格高低组，今日收益多空。风格滞后一天，不含当日。"""
    style_n = style.copy()
    close_n = close.copy()
    style_n.index = pd.to_datetime(style_n.index).strftime("%Y-%m-%d")
    close_n.index = pd.to_datetime(close_n.index).strftime("%Y-%m-%d")
    style_n, close_n = style_n.align(close_n, join="inner", axis=0)
    style_n, close_n = style_n.align(close_n, join="inner", axis=1)
    ret = close_n.pct_change()
    lagged = style_n.shift(1)
    s = lagged.to_numpy(dtype=np.float64)
    r = ret.to_numpy(dtype=np.float64)
    out = np.full(s.shape[0], np.nan, dtype=np.float64)
    lo_q, hi_q = float(frac), 1.0 - float(frac)
    need = max(int(min_stocks), 10)
    for t in range(s.shape[0]):
        ok = np.isfinite(s[t]) & np.isfinite(r[t])
        if int(ok.sum()) < need:
            continue
        x = s[t, ok]
        y = r[t, ok]
        lo, hi = np.quantile(x, [lo_q, hi_q])
        top = y[x >= hi]
        bot = y[x <= lo]
        if top.size == 0 or bot.size == 0:
            continue
        out[t] = float(top.mean() - bot.mean())
    return pd.Series(out, index=lagged.index, name="momentum_ls")


def rolling_style_perf(
    daily: pd.Series,
    window: int = MOM_LS_WINDOW,
    min_periods: int = MOM_LS_MIN_PERIODS,
) -> pd.Series:
    return pd.to_numeric(daily, errors="coerce").rolling(
        window=int(window), min_periods=int(min_periods)
    ).sum()


def leg_from_perf(value: Optional[float], current: Optional[str] = None) -> LegName:
    """动量风格 5 日累计为正用 ns，为负用 a088；缺失则维持当前腿。"""
    cur = current if current in {"mom", "a088"} else "mom"
    if value is None or not np.isfinite(value) or float(value) == 0.0:
        return cur
    return "mom" if float(value) > 0.0 else "a088"


class WufuMix(WufuAlpha088):
    name = "wufu_mix"

    @classmethod
    def from_cli(cls, args, allocator):
        n = args.n if getattr(args, "n", None) is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return cls(n=n, allocator=allocator, rebalance=spec)

    @classmethod
    def cli_fields(cls):
        return [n_field(DEFAULT_N), rebalance_field(DEFAULT_REBALANCE)]

    @classmethod
    def run_tag(cls, args) -> str:
        n = args.n if getattr(args, "n", None) is not None else DEFAULT_N
        spec = parse_rebalance(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return f"wufu_mix_h{int(n)}{spec.tag_suffix()}"

    def __init__(
        self,
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = DEFAULT_REBALANCE,
        names=None,
        index_close=None,
        list_dates=None,
        alpha088=None,
        mom_style_perf: Optional[pd.Series] = None,
    ):
        super().__init__(
            n=n,
            allocator=allocator,
            rebalance=rebalance,
            names=names,
            index_close=index_close,
            list_dates=list_dates,
            alpha088=alpha088,
        )
        self._leg: LegName = "mom"
        self._mom_style_perf: Optional[pd.Series] = (
            None if mom_style_perf is None else pd.to_numeric(mom_style_perf, errors="coerce")
        )
        self._mom_style_injected = mom_style_perf is not None

    def score(self, ctx: DayContext) -> TargetHoldings:
        self._asof = ctx.asof
        self._ensure(ctx)
        if self._weak is None:
            self._weak = WeakState()
        universe = ctx.universe()
        closes = np.asarray(ctx.history("close", LOOKBACK_DAYS + 1), dtype=np.float64)
        opens = np.asarray(ctx.history("open", 1), dtype=np.float64)
        vols = np.asarray(ctx.history("vol", VOLUME_LOOKBACK + 1), dtype=np.float64)
        amounts = np.asarray(ctx.history("amount", LIQ_DAYS), dtype=np.float64)
        holdings = [ctx.symbols[i] for i, w in enumerate(ctx.position) if w > 1e-9]
        self._sync_entry(ctx.asof, holdings, opens, closes)
        stopped = self._stop_hit(holdings, closes)
        if stopped:
            self._defense_until = ctx.asof
            logger.info("调仓 %s 止损，进入防御", ctx.asof)
        if self._in_defense(ctx.asof, ctx._store.dates) or stopped:
            return self._emit(ctx, self._defensive_or_empty(ctx, universe, closes[-1]))

        below, above, n_valid = self._weak_counts(ctx.asof)
        need = weak_vote_need(n_valid)
        self._weak = step_weak_period(
            self._weak, ctx.asof, ctx._store.dates, below, above, need=need
        )

        desired = self._desired_leg(ctx.asof)
        switched = desired != self._leg
        perf = self._style_perf_at(ctx.asof)
        if switched:
            logger.info(
                "调仓 %s 动量风格5日累计=%s，%s → %s",
                ctx.asof,
                "na" if perf is None else f"{perf:.4f}",
                self._leg,
                desired,
            )
        if not switched:
            skipped = self._gate.skip(ctx)
            if skipped is not None:
                return skipped
            hold_days = _count_days(ctx._store.dates, self._hold_start, ctx.asof)
            if holdings and hold_days < MIN_HOLD_DAYS:
                logger.info(
                    "调仓 %s 最短持仓 %s/%s，续持 %s",
                    ctx.asof,
                    hold_days,
                    MIN_HOLD_DAYS,
                    holdings,
                )
                return self._emit(ctx, holdings[: self.n])

        threshold = liquidity_threshold(amounts)
        avg_money = {ctx.symbols[i]: float(v) for i, v in enumerate(avg_daily_yuan(amounts))}
        pool = self._pool(ctx.asof, ctx.symbols, universe, avg_money, threshold)
        self._leg = desired
        if desired == "a088":
            rows = WufuAlpha088._metrics(self, ctx.symbols, pool, universe, closes, vols)
        else:
            rows = WufuXinChun._metrics(self, ctx.symbols, pool, universe, closes, vols)
        filtered = apply_filters(rows, is_weak=self._weak.is_weak)
        if need_defense(filtered):
            logger.info("调仓 %s 腿=%s 候选弱/少，防御", ctx.asof, self._leg)
            return self._emit(ctx, self._defensive_or_empty(ctx, universe, closes[-1]))
        ratio = 1.0 if self._weak.is_weak else KEEP_RATIO
        targets = select_targets(filtered, holdings, self.n, ratio)
        if not switched:
            targets = step_replace(holdings, targets, self.n)
        if not targets:
            return self._emit(ctx, self._defensive_or_empty(ctx, universe, closes[-1]))
        if switched:
            logger.info("调仓 %s 动量风格切到 %s 目标=%s", ctx.asof, self._leg, targets)
        return self._emit(ctx, targets)

    def _style_perf_at(self, asof: str) -> Optional[float]:
        series = self._mom_style_perf
        if series is None or series.empty:
            return None
        work = series.copy()
        work.index = pd.Index([str(i)[:10] for i in work.index])
        work = pd.to_numeric(work[work.index <= str(asof)], errors="coerce").dropna()
        if work.empty:
            return None
        return float(work.iloc[-1])

    def _desired_leg(self, asof: str) -> LegName:
        return leg_from_perf(self._style_perf_at(asof), self._leg)

    def _ensure(self, ctx: DayContext) -> None:
        super()._ensure(ctx)
        if self._mom_style_injected or self._mom_style_perf is not None:
            return
        try:
            self._mom_style_perf = _load_momentum_style_perf(
                start=str(ctx._store.dates[0]),
                end=str(ctx._store.dates[-1]),
            )
        except Exception:
            logger.exception("动量风格收益加载失败，mix 默认维持动量腿")
            self._mom_style_perf = pd.Series(dtype="float64")

    def _emit(self, ctx: DayContext, targets: Sequence[str]) -> TargetHoldings:
        picks = np.asarray([self._index[s] for s in targets if s in self._index], dtype=np.int64)
        if picks.size == 0:
            return self._gate.remember(
                TargetHoldings.from_array(
                    ctx.asof, ctx.execute_on, ctx.symbols, np.zeros(len(ctx.symbols))
                )
            )
        weights = self.allocator.size(ctx, picks)
        logger.info(
            "调仓 %s 腿=%s %s 目标=%s",
            ctx.asof,
            self._leg,
            "走弱" if self._weak is not None and self._weak.is_weak else "正常",
            [ctx.symbols[i] for i in picks],
        )
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )


def _load_momentum_style_perf(start: str, end: str) -> pd.Series:
    from FactorEvaluates.factor_panel_loader import FactorPanelLoader
    from FactorEvaluates.market_panel_loader import MarketPanelLoader

    style = FactorPanelLoader().load("style_momentum", start_date=start, end_date=end)
    close = MarketPanelLoader().load_field("close", start_date=start, end_date=end)
    if style is None or style.empty or close is None or close.empty:
        raise ValueError("style_momentum 或 close 为空")
    return rolling_style_perf(daily_momentum_ls(style, close))

