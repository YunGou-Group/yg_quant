#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五福闹新春日频：在 wufu 池子与 W² 动量上，按手册补震荡滤波、最短持仓、止损防御。

相对 QMT/PTrade 简述的日频替代（不覆盖 wufu）：
- 13:10 选股/买卖 → asof 收盘给目标，引擎 T+1 开盘成交
- 震荡期拉普拉斯/高斯 → 线性 R² 低时先高斯平滑再回归，更低则叠加拉普拉斯曲率
- 最短持仓 5 日、默认周五调仓（可改 daily / 5），止损仍每天检查
- 分步换仓：得分比例 0.8 续持，调仓日最多换 1 只

用法：
  python -m StrategyEngine --mode backtest --strategy wufu_ns --start 2023-01-01
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from FactorEvaluates.market_panel_loader import MarketPanelLoader
from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy, n_field, rebalance_field
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli
from Strategies.wufu import (
    DEFENSIVE,
    FIXED_POOL,
    GLOBAL_POOL,
    LIQ_DAYS,
    LOOKBACK_DAYS,
    MAX_SCORE,
    MIN_SCORE,
    R2_THRESHOLD,
    VOLUME_LOOKBACK,
    VOLUME_THRESHOLD,
    WARMUP_DAYS,
    WEAK_INDEXES,
    WEAK_MA,
    EtfMetrics,
    WeakState,
    apply_filters,
    avg_daily_yuan,
    dynamic_sector_pool,
    index_vs_ma,
    liquidity_threshold,
    momentum_score,
    passed_loss_filter,
    passed_ma_filter,
    select_targets,
    step_weak_period,
    volume_ratio_daily,
    weak_vote_need,
)

logger = logging.getLogger("Strategies")

DEFAULT_N = 1
DEFAULT_REBALANCE = "weekly"
CHOPPY_R2 = 0.35
LAPLACE_R2 = 0.18
GAUSS_SIGMA = 1.5
LAPLACE_GAIN = 8.0
MIN_HOLD_DAYS = 5
KEEP_RATIO = 0.8
STOP_RATIO = 0.95
STOP_DEFENSE_DAYS = 3
MIN_CANDIDATES = 2
WEAK_SCORE = 0.3
MIN_LIST_DAYS = 60


def gaussian_smooth(series: np.ndarray, sigma: float = GAUSS_SIGMA) -> np.ndarray:
    y = np.asarray(series, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= float(kernel.sum())
    padded = np.pad(y, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def discrete_laplacian(series: np.ndarray) -> np.ndarray:
    y = np.asarray(series, dtype=np.float64)
    out = np.zeros_like(y)
    if y.size < 3:
        return out
    out[1:-1] = y[:-2] - 2.0 * y[1:-1] + y[2:]
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def momentum_adaptive(
    price_series: np.ndarray, lookback_days: int = LOOKBACK_DAYS
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """趋势市走线性 W²；R² 偏低改高斯；再低叠加拉普拉斯曲率。"""
    linear = momentum_score(price_series, lookback_days)
    score_l, ann_l, r2_l = linear
    if score_l is None or r2_l is None:
        return None, None, None, "none"
    r2 = float(r2_l)
    if r2 >= CHOPPY_R2:
        return score_l, ann_l, r2_l, "linear"
    series = np.asarray(price_series, dtype=np.float64)
    series = series[np.isfinite(series) & (series > 0)]
    need = int(lookback_days) + 1
    if series.size < need:
        return score_l, ann_l, r2_l, "linear"
    log_px = np.log(series[-need:])
    smooth = gaussian_smooth(log_px, GAUSS_SIGMA)
    score_g, ann_g, r2_g = momentum_score(np.exp(smooth), lookback_days)
    if score_g is None:
        return score_l, ann_l, r2_l, "linear"
    if r2 >= LAPLACE_R2:
        return score_g, ann_g, r2_g, "gauss"
    lap = discrete_laplacian(smooth)
    curve = float(lap[-2]) if lap.size >= 2 else 0.0
    mixed = float(score_g) + LAPLACE_GAIN * curve
    mixed = min(float(MAX_SCORE), mixed)
    return mixed, ann_g, r2_g, "laplace"


def need_defense(
    filtered: Sequence[EtfMetrics],
    *,
    min_n: int = MIN_CANDIDATES,
    weak: float = WEAK_SCORE,
) -> bool:
    if not filtered:
        return True
    best = max(float(m.score) for m in filtered)
    return len(filtered) < int(min_n) and best < float(weak)


def listing_age_days(list_date: object, asof: str) -> Optional[int]:
    raw = str(list_date or "").strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return None
    try:
        start = pd.Timestamp(raw)
        end = pd.Timestamp(asof)
    except Exception:
        return None
    return int((end - start).days)


def is_seasoned(code: str, asof: str, list_dates: Mapping[str, object], min_days: int = MIN_LIST_DAYS) -> bool:
    age = listing_age_days(list_dates.get(code), asof)
    if age is None:
        return True
    return age >= int(min_days)


def step_replace(holdings: Sequence[str], targets: Sequence[str], n: int) -> List[str]:
    """每天最多换入 1 只，避免一次全清。"""
    hold_n = max(1, int(n))
    wanted = [str(s) for s in targets][:hold_n]
    held = [str(s) for s in holdings]
    if not wanted or not held:
        return wanted
    keep = [s for s in held if s in wanted]
    add = [s for s in wanted if s not in held]
    if keep and add and len(keep) >= hold_n:
        return (keep[:-1] + add[:1])[:hold_n]
    if keep:
        out = keep[:]
        for sym in add:
            if len(out) >= hold_n:
                break
            out.append(sym)
        return out[:hold_n]
    return add[:hold_n]


def _load_etf_meta() -> Tuple[Dict[str, str], Dict[str, str]]:
    from DailyUpdates.storage import SQLiteStorage
    from yg_quant_repo import default_db_path

    basic = SQLiteStorage(str(default_db_path())).read_etf_basic()
    names: Dict[str, str] = {}
    listed: Dict[str, str] = {}
    if basic is None or basic.empty:
        return names, listed
    for row in basic.itertuples(index=False):
        code = str(row.symbol)
        names[code] = str(getattr(row, "name", "") or "")
        listed[code] = str(getattr(row, "list_date", "") or "")
    return names, listed


def _count_days(dates: Sequence[str], start: Optional[str], asof: str) -> int:
    if not start:
        return 0
    return sum(1 for day in dates if start <= day <= asof)


class WufuXinChun(Strategy):
    name = "wufu_ns"

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
            "market_fields": ("close", "vol", "amount"),
            "factors": (),
        }

    @classmethod
    def cost_kwargs(cls, args=None) -> dict:
        del args
        return {
            "commission": 0.0001,
            "stamp": 0.0,
            "slippage": 0.00258,
            "min_commission": 5.0,
        }

    @classmethod
    def warmup(cls, args) -> int:
        del args
        return WARMUP_DAYS

    @classmethod
    def run_tag(cls, args) -> str:
        n = args.n if getattr(args, "n", None) is not None else DEFAULT_N
        spec = parse_rebalance(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return f"wufu_ns_h{int(n)}{spec.tag_suffix()}"

    def __init__(
        self,
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = DEFAULT_REBALANCE,
        names: Optional[Mapping[str, str]] = None,
        index_close: Optional[Mapping[str, pd.Series]] = None,
        list_dates: Optional[Mapping[str, str]] = None,
    ):
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self._gate = RebalanceGate(self.rebalance)
        self._names: Dict[str, str] = dict(names) if names is not None else {}
        self._list_dates: Dict[str, str] = dict(list_dates) if list_dates is not None else {}
        self._index_close: Dict[str, pd.Series] = dict(index_close) if index_close is not None else {}
        self._injected = names is not None
        self._list_injected = list_dates is not None
        self._index_injected = index_close is not None
        self._ready = False
        self._weak = None
        self._index: Dict[str, int] = {}
        self._weak_index_warned = False
        self._hold_start: Optional[str] = None
        self._entry_px: float = 0.0
        self._last_held: List[str] = []
        self._defense_until: Optional[str] = None

    def score(self, ctx: DayContext) -> TargetHoldings:
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
            logger.info("调仓 %s 止损，进入 %s 日防御", ctx.asof, STOP_DEFENSE_DAYS)
        in_defense = self._in_defense(ctx.asof, ctx._store.dates)
        if in_defense or stopped:
            return self._emit(ctx, self._defensive_or_empty(ctx, universe, closes[-1]))

        below, above, n_valid = self._weak_counts(ctx.asof)
        need = weak_vote_need(n_valid)
        self._weak = step_weak_period(
            self._weak, ctx.asof, ctx._store.dates, below, above, need=need
        )

        skipped = self._gate.skip(ctx)
        if skipped is not None:
            return skipped

        hold_days = _count_days(ctx._store.dates, self._hold_start, ctx.asof)
        if holdings and hold_days < MIN_HOLD_DAYS:
            logger.info("调仓 %s 最短持仓 %s/%s，续持 %s", ctx.asof, hold_days, MIN_HOLD_DAYS, holdings)
            return self._emit(ctx, holdings[: self.n])

        threshold = liquidity_threshold(amounts)
        avg_money = {ctx.symbols[i]: float(v) for i, v in enumerate(avg_daily_yuan(amounts))}
        pool = self._pool(ctx.asof, ctx.symbols, universe, avg_money, threshold)
        rows = self._metrics(ctx.symbols, pool, universe, closes, vols)
        filtered = apply_filters(rows, is_weak=self._weak.is_weak)
        if need_defense(filtered):
            logger.info("调仓 %s 候选弱/少，防御", ctx.asof)
            return self._emit(ctx, self._defensive_or_empty(ctx, universe, closes[-1]))

        ratio = 1.0 if self._weak.is_weak else KEEP_RATIO
        targets = select_targets(filtered, holdings, self.n, ratio)
        targets = step_replace(holdings, targets, self.n)
        if not targets:
            return self._emit(ctx, self._defensive_or_empty(ctx, universe, closes[-1]))
        return self._emit(ctx, targets)

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
            "调仓 %s %s 目标=%s",
            ctx.asof,
            "走弱" if self._weak is not None and self._weak.is_weak else "正常",
            [ctx.symbols[i] for i in picks],
        )
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )

    def _sync_entry(
        self,
        asof: str,
        holdings: Sequence[str],
        opens: np.ndarray,
        closes: np.ndarray,
    ) -> None:
        current = [str(s) for s in holdings]
        if current != self._last_held:
            self._hold_start = asof if current else None
            self._entry_px = 0.0
            if current:
                loc = self._index.get(current[0])
                if loc is not None:
                    open_px = float(opens[-1, loc]) if opens.ndim == 2 else float("nan")
                    close_px = float(closes[-1, loc])
                    self._entry_px = open_px if np.isfinite(open_px) and open_px > 0 else close_px
            self._last_held = current

    def _stop_hit(self, holdings: Sequence[str], closes: np.ndarray) -> bool:
        if not holdings or self._entry_px <= 0:
            return False
        loc = self._index.get(str(holdings[0]))
        if loc is None:
            return False
        last = float(closes[-1, loc])
        if not np.isfinite(last) or last <= 0:
            return False
        return last < self._entry_px * STOP_RATIO

    def _in_defense(self, asof: str, dates: Sequence[str]) -> bool:
        if not self._defense_until:
            return False
        elapsed = _count_days(dates, self._defense_until, asof)
        if elapsed > STOP_DEFENSE_DAYS:
            self._defense_until = None
            return False
        return True

    def _defensive_or_empty(
        self, ctx: DayContext, universe: np.ndarray, today_close: np.ndarray
    ) -> List[str]:
        loc = self._index.get(DEFENSIVE)
        if loc is not None and universe[loc] and np.isfinite(today_close[loc]):
            return [DEFENSIVE]
        return []

    def _ensure(self, ctx: DayContext) -> None:
        if self._ready:
            return
        self._weak = WeakState()
        self._index = {str(sym): i for i, sym in enumerate(ctx.symbols)}
        if not self._injected or not self._list_injected:
            try:
                names, listed = _load_etf_meta()
            except Exception:
                logger.warning("etf_basic 加载失败，动态池将为空")
                names, listed = {}, {}
            if not self._injected:
                self._names = names
            if not self._list_injected:
                self._list_dates = listed
        if not self._index_injected:
            loader = MarketPanelLoader()
            loaded: Dict[str, pd.Series] = {}
            for label, symbol in WEAK_INDEXES:
                try:
                    loaded[label] = loader.load_index_close(symbol)
                except Exception:
                    loaded[label] = pd.Series(dtype="float64")
            self._index_close = loaded
        self._ready = True

    def _weak_counts(self, asof: str) -> Tuple[int, int, int]:
        above = 0
        below = 0
        ready = 0
        missing: List[str] = []
        for label, _symbol in WEAK_INDEXES:
            series = self._index_close.get(label)
            flag = index_vs_ma(series, asof, WEAK_MA)
            if flag is True:
                above += 1
                ready += 1
            elif flag is False:
                below += 1
                ready += 1
            elif series is None or getattr(series, "empty", True):
                missing.append(label)
            else:
                work = pd.to_numeric(series, errors="coerce")
                work.index = pd.to_datetime(work.index).strftime("%Y-%m-%d")
                work = work[work.index <= asof].dropna()
                if len(work) < WEAK_MA:
                    missing.append(label)
                else:
                    ready += 1
        if missing and not self._weak_index_warned:
            self._weak_index_warned = True
            logger.warning(
                "走弱指数历史不足 %s，有效 %s/4",
                missing,
                ready,
            )
        return below, above, ready

    def _pool(
        self,
        asof: str,
        symbols: Sequence[str],
        universe: np.ndarray,
        avg_money: Mapping[str, float],
        threshold: float,
    ) -> np.ndarray:
        listed = {
            str(s)
            for s, ok in zip(symbols, universe)
            if ok and is_seasoned(str(s), asof, self._list_dates)
        }

        def liquid(codes: Iterable[str]) -> List[str]:
            return [
                c
                for c in codes
                if c in listed and float(avg_money.get(c, 0.0) or 0.0) > threshold
            ]

        if self._weak is not None and self._weak.is_weak:
            picked = liquid(GLOBAL_POOL) or [c for c in GLOBAL_POOL if c in listed]
        else:
            fixed = liquid(FIXED_POOL)
            names = {code: self._names.get(code, "") for code in listed if self._names.get(code)}
            dynamic = dynamic_sector_pool(names, avg_money, threshold)
            picked = sorted(set(fixed) | set(dynamic))
        out = np.zeros(len(symbols), dtype=bool)
        for code in picked:
            loc = self._index.get(code)
            if loc is not None:
                out[loc] = True
        return out

    def _metrics(
        self,
        symbols: Sequence[str],
        pool: np.ndarray,
        universe: np.ndarray,
        closes: np.ndarray,
        vols: np.ndarray,
    ) -> List[EtfMetrics]:
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
            score, _ann, r2, _kind = momentum_adaptive(px, LOOKBACK_DAYS)
            if score is None or r2 is None:
                continue
            vol_ratio = volume_ratio_daily(past_vol[:, i], float(today_vol[i]))
            rows.append(
                EtfMetrics(
                    symbol=str(symbols[i]),
                    name=self._names.get(str(symbols[i]), str(symbols[i])),
                    score=float(score),
                    r2=float(r2),
                    volume_ratio=vol_ratio,
                    passed_momentum=MIN_SCORE <= float(score) <= MAX_SCORE,
                    passed_r2=float(r2) > R2_THRESHOLD,
                    passed_ma=passed_ma_filter(px),
                    passed_volume=vol_ratio is not None and vol_ratio < VOLUME_THRESHOLD,
                    passed_loss=passed_loss_filter(px),
                )
            )
        return rows
