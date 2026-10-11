#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五福连续排序：当天可交易池 + 年化斜率 × R²，广度不够则空仓。

不使用写死的 FIXED_POOL / GLOBAL_POOL，也不使用分数上下限、放量、近端跌幅、
均线、高斯和拉普拉斯这些硬关。R² 门槛只决定「有没有足够多的直线趋势」：
不够就整仓现金，够了再按连续分数取前 n 只等权。

用法：
  python -m StrategyEngine --mode backtest --strategy wufu_rank --start 2023-01-01
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from FactorEvaluates.market_panel_loader import MarketPanelLoader
from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy, n_field, rebalance_field
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli
from Strategies.wufu import (
    LIQ_DAYS,
    LOOKBACK_DAYS,
    R2_THRESHOLD,
    WEAK_INDEXES,
    WEAK_MA,
    WeakState,
    avg_daily_yuan,
    index_vs_ma,
    liquidity_threshold,
    momentum_score,
    step_weak_period,
    weak_vote_need,
)

logger = logging.getLogger("Strategies")

DEFAULT_N = 5
DEFAULT_REBALANCE = "weekly"
SEASON_DAYS = 60
CORR_DAYS = 60
CORR_CUT = 0.90
MIN_CORR_OBS = 40
WARMUP_DAYS = CORR_DAYS + LOOKBACK_DAYS
COMMODITY_WORDS = ("黄金", "白银", "原油", "豆粕")


@dataclass(frozen=True)
class RankRow:
    symbol: str
    score: float
    ann: float
    r2: float


def iso_day(raw: object) -> str:
    text = str(raw or "").strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return ""
    if len(text) >= 10 and text[4] == "-":
        return text[:10]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def cash_bond_reit(name: str, fund_type: str) -> bool:
    """货币、债券、REIT。现金流主题是股票策略，不在这里排除。"""
    text = str(name or "")
    kind = str(fund_type or "")
    upper = text.upper()
    if "REIT" in upper or "不动产" in text:
        return True
    if "债券" in kind or "货币" in kind:
        return True
    if "货币" in text:
        return True
    if "现金流" in text:
        return False
    return "现金" in text or "债" in text


def offshore_or_commodity(name: str) -> bool:
    text = str(name or "")
    if "QDII" in text.upper():
        return True
    return any(word in text for word in COMMODITY_WORDS)


def season_ok(
    dates: Sequence[str],
    list_iso: str,
    asof: str,
    close_col: np.ndarray,
    min_days: int = SEASON_DAYS,
) -> bool:
    need = int(min_days)
    if list_iso:
        age = sum(1 for day in dates if list_iso <= day <= asof)
        return age >= need
    finite = np.isfinite(np.asarray(close_col, dtype=np.float64)).sum()
    return int(finite) >= need


def pairwise_corr(rets: np.ndarray, min_obs: int = MIN_CORR_OBS) -> np.ndarray:
    """列之间的收益相关。重叠样本不足的一对记 0，不当成克隆。"""
    x = np.asarray(rets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        return np.zeros((x.shape[1] if x.ndim == 2 else 0, 0), dtype=np.float64)
    finite = np.isfinite(x)
    count = finite.T @ finite.astype(np.float64)
    count_col = finite.sum(axis=0).astype(np.float64)
    sums = np.where(finite, x, 0.0).sum(axis=0)
    means = np.divide(sums, count_col, out=np.zeros(x.shape[1], dtype=np.float64), where=count_col > 0)
    centered = np.where(finite, x - means, 0.0)
    var = np.sum(centered * centered, axis=0)
    cov = centered.T @ centered
    denom = np.sqrt(var[:, None] * var[None, :])
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
    corr[count < int(min_obs)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def keep_liquid_clones(
    rets: np.ndarray,
    money: np.ndarray,
    *,
    corr_cut: float = CORR_CUT,
    min_obs: int = MIN_CORR_OBS,
) -> np.ndarray:
    """相关不低于 corr_cut 的只留成交额更大的一只。"""
    k = int(np.asarray(money).shape[0])
    mask = np.zeros(k, dtype=bool)
    if k == 0:
        return mask
    corr = pairwise_corr(rets, min_obs=min_obs)
    order = np.argsort(-np.asarray(money, dtype=np.float64), kind="mergesort")
    kept: List[int] = []
    cut = float(corr_cut)
    for j in order:
        if any(float(corr[int(j), prev]) >= cut for prev in kept):
            continue
        kept.append(int(j))
        mask[int(j)] = True
    return mask


def select_symbols(rows: Sequence[RankRow], n: int, breadth_r2: float = R2_THRESHOLD) -> List[str]:
    """上行趋势按分数从高到低取 n 只。R² 过线的只数不足 n 时整仓空着。"""
    hold_n = max(1, int(n))
    up = [
        row
        for row in rows
        if row.ann > 0.0 and row.r2 > 0.0 and np.isfinite(row.score)
    ]
    breadth = sum(1 for row in up if row.r2 > float(breadth_r2))
    if breadth < hold_n:
        return []
    up.sort(key=lambda row: (-row.score, row.symbol))
    return [row.symbol for row in up[:hold_n]]


def _load_etf_meta() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    from DailyUpdates.storage import SQLiteStorage
    from yg_quant_repo import default_db_path

    basic = SQLiteStorage(str(default_db_path())).read_etf_basic()
    names: Dict[str, str] = {}
    listed: Dict[str, str] = {}
    kinds: Dict[str, str] = {}
    if basic is None or basic.empty:
        return names, listed, kinds
    has_type = "fund_type" in basic.columns
    for row in basic.itertuples(index=False):
        code = str(row.symbol)
        names[code] = str(getattr(row, "name", "") or "")
        listed[code] = iso_day(getattr(row, "list_date", ""))
        kinds[code] = str(getattr(row, "fund_type", "") or "") if has_type else ""
    return names, listed, kinds


class WufuRank(Strategy):
    name = "wufu_rank"

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
            "market_fields": ("close", "amount"),
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
        return f"wufu_rank_h{int(n)}{spec.tag_suffix()}"

    def __init__(
        self,
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = DEFAULT_REBALANCE,
        names: Optional[Mapping[str, str]] = None,
        list_dates: Optional[Mapping[str, str]] = None,
        fund_types: Optional[Mapping[str, str]] = None,
        index_close: Optional[Mapping[str, pd.Series]] = None,
    ):
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self._gate = RebalanceGate(self.rebalance)
        self._names = dict(names) if names is not None else {}
        self._list_dates = dict(list_dates) if list_dates is not None else {}
        self._fund_types = dict(fund_types) if fund_types is not None else {}
        self._index_close = dict(index_close) if index_close is not None else {}
        self._meta_injected = names is not None
        self._index_injected = index_close is not None
        self._ready = False
        self._weak = WeakState()
        self._index: Dict[str, int] = {}
        self._weak_index_warned = False

    def score(self, ctx: DayContext) -> TargetHoldings:
        self._ensure(ctx)
        skipped = self._gate.skip(ctx)
        if skipped is not None:
            return skipped
        below, above, n_valid = self._weak_counts(ctx.asof)
        self._weak = step_weak_period(
            self._weak,
            ctx.asof,
            ctx._store.dates,
            below,
            above,
            need=weak_vote_need(n_valid),
        )
        universe = ctx.universe()
        closes = np.asarray(ctx.history("close", CORR_DAYS + 1), dtype=np.float64)
        amounts = np.asarray(ctx.history("amount", LIQ_DAYS), dtype=np.float64)
        pool = self._pool_index(ctx.asof, ctx.symbols, ctx._store.dates, universe, closes, amounts)
        rows = self._rows(ctx.symbols, pool, closes)
        targets = select_symbols(rows, self.n, R2_THRESHOLD)
        breadth = sum(1 for row in rows if row.ann > 0.0 and row.r2 > R2_THRESHOLD)
        logger.info(
            "调仓 %s %s 池=%s 广度=%s 目标=%s",
            ctx.asof,
            "走弱" if self._weak.is_weak else "正常",
            int(pool.sum()),
            breadth,
            targets or "现金",
        )
        return self._emit(ctx, targets)

    def _emit(self, ctx: DayContext, targets: Sequence[str]) -> TargetHoldings:
        picks = np.asarray([self._index[s] for s in targets if s in self._index], dtype=np.int64)
        if picks.size == 0:
            weights = np.zeros(len(ctx.symbols), dtype=np.float64)
        else:
            weights = self.allocator.size(ctx, picks)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )

    def _ensure(self, ctx: DayContext) -> None:
        if self._ready:
            return
        self._index = {str(sym): i for i, sym in enumerate(ctx.symbols)}
        if not self._meta_injected:
            try:
                names, listed, kinds = _load_etf_meta()
            except Exception:
                logger.warning("etf_basic 加载失败，候选池将为空")
                names, listed, kinds = {}, {}, {}
            self._names = names
            self._list_dates = listed
            self._fund_types = kinds
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
                ready += 1
        if missing and not self._weak_index_warned:
            self._weak_index_warned = True
            logger.warning("走弱指数历史不足 %s，有效 %s/4", missing, ready)
        return below, above, ready

    def _pool_index(
        self,
        asof: str,
        symbols: Sequence[str],
        dates: Sequence[str],
        universe: np.ndarray,
        closes: np.ndarray,
        amounts: np.ndarray,
    ) -> np.ndarray:
        threshold = liquidity_threshold(amounts)
        money = avg_daily_yuan(amounts)
        chosen: List[int] = []
        monies: List[float] = []
        weak = bool(self._weak.is_weak)
        for i, sym in enumerate(symbols):
            code = str(sym)
            if not universe[i]:
                continue
            if not season_ok(dates, self._list_dates.get(code, ""), asof, closes[:, i]):
                continue
            yuan = float(money[i]) if i < money.shape[0] else 0.0
            if not np.isfinite(yuan) or yuan <= threshold:
                continue
            name = self._names.get(code, "")
            if cash_bond_reit(name, self._fund_types.get(code, "")):
                continue
            if weak and not offshore_or_commodity(name):
                continue
            chosen.append(i)
            monies.append(yuan)
        if not chosen:
            return np.zeros(len(symbols), dtype=bool)
        rets = _returns(closes[:, chosen])
        keep = keep_liquid_clones(rets, np.asarray(monies, dtype=np.float64))
        out = np.zeros(len(symbols), dtype=bool)
        for flag, idx in zip(keep, chosen):
            if flag:
                out[idx] = True
        return out

    def _rows(self, symbols: Sequence[str], pool: np.ndarray, closes: np.ndarray) -> List[RankRow]:
        rows: List[RankRow] = []
        for i, flagged in enumerate(pool):
            if not flagged:
                continue
            score, ann, r2 = momentum_score(closes[:, i], LOOKBACK_DAYS)
            if score is None or ann is None or r2 is None:
                continue
            rows.append(
                RankRow(symbol=str(symbols[i]), score=float(score), ann=float(ann), r2=float(r2))
            )
        return rows


def _returns(closes: np.ndarray) -> np.ndarray:
    px = np.asarray(closes, dtype=np.float64)
    if px.shape[0] < 2:
        return np.full((0, px.shape[1]), np.nan, dtype=np.float64)
    prev = px[:-1]
    out = np.full_like(px[1:], np.nan)
    ok = np.isfinite(px[1:]) & np.isfinite(prev) & (prev > 0)
    np.divide(px[1:], prev, out=out, where=ok)
    out[ok] -= 1.0
    return out
