#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小市值：行业 MA20 选时 + 财务/市值筛选；周五收盘给名单，下个交易日（通常周一）开盘成交。仓位由 allocator 决定（默认等权）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from Universes.names import NameHistory
from Universes.rules import is_restricted_name

from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.strategy import Strategy
from Strategies.rebalance_schedule import week_end_asofs

logger = logging.getLogger("Strategies")

MA_PERIOD = 20
CANDIDATE_N = 20
HOLD_N = 6
MIN_LISTING_DAYS = 400
ROE_THRESHOLD = 0.15
ROA_THRESHOLD = 0.10
INDUSTRY_SRC = "SW2021"
UNIVERSE_INDEX_CODES = ("399101.SZ", "000688.SH")
ANTI_TAIL_HIGH_RATIO = 0.70
ANTI_TAIL_SINGLE_DROP = 0.05
ANTI_TAIL_CUM_DROP_2W = 0.10

# 申万一级（不含 801980，避免重复「农林牧渔」干扰选时）
INDUSTRY_CODES: Tuple[str, ...] = (
    "801010.SI",
    "801030.SI",
    "801040.SI",
    "801050.SI",
    "801080.SI",
    "801110.SI",
    "801120.SI",
    "801130.SI",
    "801140.SI",
    "801150.SI",
    "801160.SI",
    "801170.SI",
    "801180.SI",
    "801200.SI",
    "801210.SI",
    "801230.SI",
    "801710.SI",
    "801720.SI",
    "801730.SI",
    "801740.SI",
    "801750.SI",
    "801760.SI",
    "801770.SI",
    "801780.SI",
    "801790.SI",
    "801880.SI",
    "801890.SI",
    "801950.SI",
    "801960.SI",
    "801970.SI",
)

TROUBLE_MAKER_INDUSTRIES = frozenset(
    {
        # "801780.SI",  # 银行
        # "801050.SI",  # 有色
        # "801040.SI",  # 钢铁
        # "801950.SI",  # 煤炭
        # "801180.SI",  # 房地产
    }
)


@dataclass
class FinRow:
    ann: str
    end: str
    flag: str
    roe: float
    roa: float


@dataclass
class Span:
    """一段有效期内的取值。end 为空串表示至今仍然有效。"""

    start: str
    end: str
    value: str

    def covers(self, asof: str) -> bool:
        if self.start and asof < self.start:
            return False
        if self.end and asof > self.end:
            return False
        return True


@dataclass
class SmallCapData:
    # symbol -> 按 start 升序的行业归属区间（申万调样历史）
    industry_spans: Dict[str, List[Span]] = field(default_factory=dict)
    industry_name: Dict[str, str] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    name_history: NameHistory = field(default_factory=NameHistory)
    list_date: Dict[str, str] = field(default_factory=dict)
    financial: Dict[str, List[FinRow]] = field(default_factory=dict)
    # 每个指数一份 {trade_date: members}，查询时逐指数 as-of 再并集
    universe_snaps: List[Dict[str, Set[str]]] = field(default_factory=list)

    def industry_asof(self, symbol: str, asof: str) -> str:
        for span in self.industry_spans.get(symbol, ()):
            if span.covers(asof):
                return span.value
        return ""

    def name_asof(self, symbol: str, asof: str) -> Optional[str]:
        """asof 当日的证券简称；没有历史记录时返回 None。"""
        return self.name_history.name_asof(symbol, asof)


class SmallCapStrategy(Strategy):
    """周五收盘给目标，T+1 开盘成交；优势行业广度选时，小市值财务过滤。"""

    name = "small_cap"

    @classmethod
    def from_cli(cls, args, allocator):
        return cls(
            candidate_n=args.n if args.n is not None else CANDIDATE_N,
            hold_n=args.hold,
            anti_tail=args.anti_tail,
            allocator=allocator,
        )

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        return {"market_fields": ("close", "up_limit", "down_limit", "total_mv")}

    @classmethod
    def run_tag(cls, args) -> str:
        return "small_cap_anti" if args.anti_tail else "small_cap"

    @classmethod
    def warmup(cls, args) -> int:
        """行业 MA 选时需要 asof 前的收盘，不能跟仓位插件的 lookback 绑死。"""
        return MA_PERIOD

    def __init__(
        self,
        *,
        candidate_n: int = CANDIDATE_N,
        hold_n: int = HOLD_N,
        ma_period: int = MA_PERIOD,
        min_listing_days: int = MIN_LISTING_DAYS,
        roe_threshold: float = ROE_THRESHOLD,
        roa_threshold: float = ROA_THRESHOLD,
        anti_tail: bool = False,
        skip_smallest: bool = True,
        allocator: Allocator | None = None,
        data: Optional[SmallCapData] = None,
    ):
        self.candidate_n = max(1, int(candidate_n))
        self.hold_n = max(1, int(hold_n))
        self.ma_period = max(2, int(ma_period))
        self.min_listing_days = max(0, int(min_listing_days))
        self.roe_threshold = float(roe_threshold)
        self.roa_threshold = float(roa_threshold)
        self.anti_tail = bool(anti_tail)
        self.skip_smallest = bool(skip_smallest)
        self.allocator: Allocator = allocator or EqualWeight()
        self._data = data
        self._held: Optional[Dict[str, float]] = None
        self._rebalance: Set[str] = set()
        self._ratio_hist: List[Tuple[str, Dict[str, float]]] = []
        self._ready = False
        self._not_bj: Optional[np.ndarray] = None
        self._list_ord: Optional[np.ndarray] = None
        # 行业归属与证券简称都随时间变化，按 asof 现算并缓存最近一天
        self._codes_cache: Tuple[str, Optional[np.ndarray]] = ("", None)
        self._name_cache: Tuple[str, Optional[np.ndarray]] = ("", None)
        self._symbols_arr: Optional[np.ndarray] = None
        self._fin_ann: Optional[List[Optional[np.ndarray]]] = None
        self._fin_end: Optional[List[Optional[np.ndarray]]] = None
        self._fin_flag: Optional[List[Optional[np.ndarray]]] = None
        self._fin_roe: Optional[List[Optional[np.ndarray]]] = None
        self._fin_roa: Optional[List[Optional[np.ndarray]]] = None

    def score(self, ctx: DayContext) -> Optional[TargetHoldings]:
        self._ensure(ctx)
        if ctx.asof not in self._rebalance:
            if not self._held:
                return None
            return TargetHoldings(ctx.asof, ctx.execute_on, dict(self._held))
        # 样本末日无 T+1 开盘：仍算信号并打印应买入名单，但不改持仓、不成交。
        signal_only = ctx.execute_on is None
        ratios = self._industry_ratios(ctx)
        if not signal_only:
            self._ratio_hist.append((ctx.asof, ratios))
        top = self._top_industry(ratios)
        if top is None:
            logger.info("调仓 %s 无优势行业（MA20 广度为空）", ctx.asof)
            return self._finish_rebalance(ctx, np.zeros(len(ctx.symbols)), signal_only)
        if top in TROUBLE_MAKER_INDUSTRIES:
            logger.info("调仓 %s 搅屎棍空仓 行业=%s", ctx.asof, top)
            return self._finish_rebalance(ctx, np.zeros(len(ctx.symbols)), signal_only)
        if self.anti_tail and self._anti_tail_hit(top):
            logger.info("调仓 %s 防尾声空仓 行业=%s", ctx.asof, top)
            return self._finish_rebalance(ctx, np.zeros(len(ctx.symbols)), signal_only)
        picks = self._pick(ctx)
        if picks.size == 0:
            logger.info("调仓 %s 无标的 行业=%s %s", ctx.asof, top, self._data.industry_name.get(top, ""))
            return self._finish_rebalance(ctx, np.zeros(len(ctx.symbols)), signal_only)
        names = [ctx.symbols[i] for i in picks]
        weights = self.allocator.size(ctx, picks)
        if signal_only:
            logger.info(
                "调仓 %s 样本末日信号（不成交）行业=%s %s 应买入 %s",
                ctx.asof,
                top,
                self._data.industry_name.get(top, ""),
                sorted(names),
            )
            return self._hold_without_trade(ctx)
        logger.info("调仓 %s 行业=%s 持仓 %s", ctx.asof, top, sorted(names))
        return self._commit(ctx, weights)

    def _finish_rebalance(
        self, ctx: DayContext, weights: np.ndarray, signal_only: bool
    ) -> TargetHoldings:
        if signal_only:
            held = sorted(self._held.keys()) if self._held else []
            logger.info("调仓 %s 样本末日信号（不成交）维持持仓 %s", ctx.asof, held)
            return self._hold_without_trade(ctx)
        return self._commit(ctx, weights)

    def _hold_without_trade(self, ctx: DayContext) -> Optional[TargetHoldings]:
        if not self._held:
            return None
        return TargetHoldings(ctx.asof, None, dict(self._held))

    def _commit(self, ctx: DayContext, weights: np.ndarray) -> TargetHoldings:
        target = _target(ctx, weights)
        self._held = dict(target.weights)
        return target

    def _ensure(self, ctx: DayContext) -> None:
        if self._ready:
            return
        dates = list(ctx._store.run_dates)
        self._rebalance = week_end_asofs(dates)
        if self._data is None:
            self._data = load_small_cap_data(ctx.symbols)
        self._bind_symbol_cache(ctx.symbols)
        self._ready = True

    def _bind_symbol_cache(self, symbols: Sequence[str]) -> None:
        names = [str(s) for s in symbols]
        self._symbols_arr = np.array(names, dtype=object)
        self._not_bj = np.array([not _is_bj(name) for name in names], dtype=bool)
        ords = np.full(len(names), -1, dtype=np.int64)
        for i, name in enumerate(names):
            listed = _parse_day(self._data.list_date.get(name))
            if listed is not None:
                ords[i] = listed.toordinal()
        self._list_ord = ords
        self._fin_ann = []
        self._fin_end = []
        self._fin_flag = []
        self._fin_roe = []
        self._fin_roa = []
        for name in names:
            rows = self._data.financial.get(name) or []
            if not rows:
                self._fin_ann.append(None)
                self._fin_end.append(None)
                self._fin_flag.append(None)
                self._fin_roe.append(None)
                self._fin_roa.append(None)
                continue
            ordered = sorted(rows, key=lambda r: (r.ann, r.end or "", 0 if str(r.flag) == "1" else 1))
            self._fin_ann.append(
                np.array([int(r.ann.replace("-", "")[:8]) for r in ordered], dtype=np.int64)
            )
            self._fin_end.append(np.array([r.end or "9999-99-99" for r in ordered], dtype=object))
            self._fin_flag.append(
                np.array([0 if str(r.flag) == "1" else 1 for r in ordered], dtype=np.int8)
            )
            self._fin_roe.append(np.array([r.roe for r in ordered], dtype=np.float64))
            self._fin_roa.append(np.array([r.roa for r in ordered], dtype=np.float64))

    def _codes_asof(self, asof: str, symbols: Sequence[str]) -> np.ndarray:
        """asof 当日的申万一级归属，走调样历史而不是当前快照。"""
        cached_day, cached = self._codes_cache
        if cached is not None and cached_day == asof and len(cached) == len(symbols):
            return cached
        codes = np.array(
            [self._data.industry_asof(str(sym), asof) for sym in symbols],
            dtype=object,
        )
        self._codes_cache = (asof, codes)
        return codes

    def _industry_ratios(self, ctx: DayContext) -> Dict[str, float]:
        codes = self._codes_asof(ctx.asof, ctx.symbols)
        hist = ctx.history("close", self.ma_period)
        finite = np.isfinite(hist)
        count = finite.sum(axis=0)
        with np.errstate(all="ignore"):
            ma = np.nansum(np.where(finite, hist, 0.0), axis=0) / np.maximum(count, 1)
        close = hist[-1]
        ok = (count >= self.ma_period) & np.isfinite(close) & np.isfinite(ma)
        above = ok & (close > ma)
        ratios: Dict[str, float] = {}
        for code in INDUSTRY_CODES:
            mask = codes == code
            denom = int((ok & mask).sum())
            if denom <= 0:
                continue
            ratios[code] = float((above & mask).sum()) / denom
        return ratios

    def _top_industry(self, ratios: Dict[str, float]) -> Optional[str]:
        best_code = None
        best = -1.0
        for code in INDUSTRY_CODES:
            ratio = ratios.get(code)
            if ratio is None:
                continue
            if ratio > best:
                best = ratio
                best_code = code
        return best_code

    def _anti_tail_hit(self, industry: str) -> bool:
        series = [item[1].get(industry) for item in self._ratio_hist]
        if not series or series[-1] is None:
            return False
        r0 = series[-1]
        r1 = series[-2] if len(series) >= 2 else None
        r2 = series[-3] if len(series) >= 3 else None
        if (
            r1 is not None
            and r0 < r1
            and r1 >= ANTI_TAIL_HIGH_RATIO
            and (r1 - r0) >= ANTI_TAIL_SINGLE_DROP
        ):
            return True
        if (
            r1 is not None
            and r2 is not None
            and r0 < r1 < r2
            and (r2 - r0) >= ANTI_TAIL_CUM_DROP_2W
        ):
            return True
        return False

    def _pick(self, ctx: DayContext) -> np.ndarray:
        steps: List[Tuple[str, int]] = []
        live = ctx.universe() & np.isfinite(ctx.market("close"))
        steps.append(("行情", int(live.sum())))
        live = live & (self._not_bj if self._not_bj is not None else ~np.array([_is_bj(s) for s in ctx.symbols], dtype=bool))
        steps.append(("非北交", int(live.sum())))
        live = live & self._name_ok(ctx.asof, ctx.symbols)
        steps.append(("名称", int(live.sum())))
        live = live & self._listed_ok(ctx.asof, ctx.symbols)
        steps.append(("上市", int(live.sum())))
        live = live & self._not_limit(ctx)
        steps.append(("未触板", int(live.sum())))
        live = live & self._in_universe(ctx.asof, ctx.symbols)
        steps.append(("成分", int(live.sum())))
        live = live & self._financial_ok(ctx.asof, ctx.symbols)
        steps.append(("财务", int(live.sum())))
        try:
            mv = ctx.market("total_mv")
        except KeyError:
            logger.info("调仓 %s 无市值字段", ctx.asof)
            return np.zeros(0, dtype=int)
        live = live & np.isfinite(mv) & (mv > 0)
        steps.append(("市值", int(live.sum())))
        idx = np.flatnonzero(live)
        if idx.size == 0:
            logger.info(
                "调仓 %s 筛选为空 %s",
                ctx.asof,
                " ".join(f"{name}={count}" for name, count in steps),
            )
            return idx
        order = idx[np.argsort(mv[idx], kind="mergesort")]
        order = order[: self.candidate_n]
        if self.skip_smallest:
            order = order[1 : self.hold_n]
        else:
            order = order[: self.hold_n]
        return order

    def _name_ok(self, asof: str, symbols: Sequence[str]) -> np.ndarray:
        """按 asof 当日的证券简称判 ST/退市；没有历史名称则不排除（不用当前简称）。"""
        cached_day, cached = self._name_cache
        if cached is not None and cached_day == asof and len(cached) == len(symbols):
            return cached
        out = np.ones(len(symbols), dtype=bool)
        for i, symbol in enumerate(symbols):
            text = self._data.name_asof(str(symbol), asof)
            if text is None:
                continue
            if is_restricted_name(text):
                out[i] = False
        self._name_cache = (asof, out)
        return out

    def _listed_ok(self, asof: str, symbols: Sequence[str]) -> np.ndarray:
        asof_dt = _parse_day(asof)
        if asof_dt is None:
            return np.zeros(len(symbols), dtype=bool)
        if self._list_ord is None or len(self._list_ord) != len(symbols):
            out = np.zeros(len(symbols), dtype=bool)
            for i, name in enumerate(symbols):
                listed = _parse_day(self._data.list_date.get(name))
                if listed is None:
                    continue
                out[i] = (asof_dt - listed).days >= self.min_listing_days
            return out
        return (self._list_ord >= 0) & (
            (asof_dt.toordinal() - self._list_ord) >= self.min_listing_days
        )

    def _not_limit(self, ctx: DayContext) -> np.ndarray:
        n = len(ctx.symbols)
        try:
            close = ctx.market("close")
            up = ctx.market("up_limit")
            down = ctx.market("down_limit")
        except KeyError:
            return np.ones(n, dtype=bool)
        has = np.isfinite(up) & np.isfinite(down) & np.isfinite(close)
        return has & (close != up) & (close != down)

    def _in_universe(self, asof: str, symbols: Sequence[str]) -> np.ndarray:
        pool = _universe_asof(self._data.universe_snaps, asof)
        if self._symbols_arr is not None and len(self._symbols_arr) == len(symbols):
            return np.isin(self._symbols_arr, list(pool))
        return np.array([s in pool for s in symbols], dtype=bool)

    def _financial_ok(self, asof: str, symbols: Sequence[str]) -> np.ndarray:
        n = len(symbols)
        out = np.zeros(n, dtype=bool)
        asof_i = _asof_int(asof)
        if asof_i is None:
            return out
        if self._fin_ann is None or len(self._fin_ann) != n:
            for i, name in enumerate(symbols):
                row = _latest_fin(self._data.financial.get(name), asof)
                if row is None or not np.isfinite(row.roe) or not np.isfinite(row.roa):
                    continue
                out[i] = row.roe > self.roe_threshold and row.roa > self.roa_threshold
            return out
        for i, anns in enumerate(self._fin_ann):
            if anns is None or anns.size == 0:
                continue
            j = int(np.searchsorted(anns, asof_i, side="right")) - 1
            if j < 0:
                continue
            latest = int(anns[j])
            j = int(np.searchsorted(anns, latest, side="right")) - 1
            roe = float(self._fin_roe[i][j])
            roa = float(self._fin_roa[i][j])
            if not np.isfinite(roe) or not np.isfinite(roa):
                continue
            out[i] = roe > self.roe_threshold and roa > self.roa_threshold
        return out


def load_small_cap_data(
    symbols: Optional[Sequence[str]] = None,
    storage=None,
) -> SmallCapData:
    if storage is None:
        from DailyUpdates.storage import SQLiteStorage
        from yg_quant_repo import default_db_path

        storage = SQLiteStorage(str(default_db_path()))
    logger.info("加载小市值辅助表：行业 / 名称 / 财务 / 指数成分")
    data = SmallCapData()
    try:
        # is_new=None 拿全量调样历史（含已调出的记录），按 in_date/out_date 做 as-of
        members = storage.read_industry_members(src=INDUSTRY_SRC, level="l1", is_new=None)
        if members is None or members.empty:
            members = storage.read_industry_members(src=None, level="l1", is_new=None)
        if members is not None and not members.empty:
            work = members.dropna(subset=["symbol", "industry_code"]).copy()
            work["symbol"] = work["symbol"].astype(str)
            work["industry_code"] = work["industry_code"].astype(str)
            if INDUSTRY_CODES:
                work = work[work["industry_code"].isin(INDUSTRY_CODES)]
            for rec in work.itertuples(index=False):
                code = str(rec.industry_code)
                data.industry_spans.setdefault(str(rec.symbol), []).append(
                    Span(
                        _iso_day(getattr(rec, "in_date", "")),
                        _iso_day(getattr(rec, "out_date", "")),
                        code,
                    )
                )
                if getattr(rec, "industry_name", None):
                    data.industry_name[code] = str(rec.industry_name)
            for spans in data.industry_spans.values():
                spans.sort(key=lambda s: s.start)
    except Exception:
        logger.warning("小市值：行业成分加载失败", exc_info=True)
    try:
        basic = storage.read_stock_basic()
        if basic is not None and not basic.empty:
            for rec in basic.itertuples(index=False):
                raw = str(rec.symbol)
                data.names[raw] = str(getattr(rec, "name", "") or "")
                data.list_date[raw] = str(getattr(rec, "list_date", "") or "")
    except Exception:
        logger.warning("小市值：stock_basic 加载失败", exc_info=True)
    try:
        data.name_history = NameHistory.from_storage(storage)
        if data.name_history.empty:
            logger.warning(
                "小市值：namechange 表为空，ST 判定跳过历史简称（不会用当前名）"
            )
    except Exception:
        logger.warning("小市值：namechange 加载失败，跳过历史简称", exc_info=True)
    try:
        fin = storage.read_financial_indicator()
        data.financial = _index_financial(fin)
    except Exception:
        logger.warning("小市值：财务加载失败", exc_info=True)
    try:
        for code in UNIVERSE_INDEX_CODES:
            part = storage.read_index_constituents(index_code=code)
            snap = _constituent_snap(part)
            if snap:
                data.universe_snaps.append(snap)
    except Exception:
        logger.warning("小市值：指数成分加载失败", exc_info=True)
    if symbols:
        wanted = set(str(s) for s in symbols)
        data.industry_spans = {
            k: v for k, v in data.industry_spans.items() if k in wanted
        }
    logger.info(
        "小市值辅助表就绪：行业 %s 行业归属标的 %s，名称历史 %s，财务标的 %s，成分日 %s",
        len(data.industry_name),
        len(data.industry_spans),
        len(data.name_history.spans),
        len(data.financial),
        sum(len(s) for s in data.universe_snaps),
    )
    return data


def _index_financial(frame: Optional[pd.DataFrame]) -> Dict[str, List[FinRow]]:
    out: Dict[str, List[FinRow]] = {}
    if frame is None or frame.empty:
        return out
    work = frame.copy()
    if "roe" not in work.columns or "roa" not in work.columns:
        return out
    work["symbol"] = work["symbol"].astype(str)
    work["ann_date"] = pd.to_datetime(work["ann_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "end_date" in work.columns:
        work["end_date"] = pd.to_datetime(work["end_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        work["end_date"] = ""
    if "update_flag" not in work.columns:
        work["update_flag"] = ""
    work["roe"] = pd.to_numeric(work["roe"], errors="coerce")
    work["roa"] = pd.to_numeric(work["roa"], errors="coerce")
    work = work.dropna(subset=["symbol", "ann_date"])
    for rec in work.itertuples(index=False):
        out.setdefault(str(rec.symbol), []).append(
            FinRow(
                ann=str(rec.ann_date),
                end=str(getattr(rec, "end_date", "") or ""),
                flag=str(getattr(rec, "update_flag", "") or ""),
                roe=float(rec.roe) if np.isfinite(rec.roe) else float("nan"),
                roa=float(rec.roa) if np.isfinite(rec.roa) else float("nan"),
            )
        )
    for rows in out.values():
        rows.sort(key=lambda r: (r.ann, r.end, r.flag))
    return out


def _latest_fin(rows: Optional[List[FinRow]], asof: str) -> Optional[FinRow]:
    if not rows:
        return None
    best: Optional[FinRow] = None
    best_key = None
    asof_dt = _parse_day(asof)
    for row in rows:
        if not row.ann or row.ann > asof:
            continue
        ann_dt = _parse_day(row.ann)
        if asof_dt is None or ann_dt is None:
            days = 10**9
        else:
            days = (asof_dt - ann_dt).days
        flag_rank = 0 if str(row.flag) == "1" else 1
        end_i = _asof_int(row.end) or 0
        key = (days, flag_rank, -end_i)
        if best_key is None or key < best_key:
            best_key = key
            best = row
    return best


def _constituent_snap(frame: Optional[pd.DataFrame]) -> Dict[str, Set[str]]:
    if frame is None or frame.empty:
        return {}
    work = frame.copy()
    work["day"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work = work.dropna(subset=["day"])
    work["symbol"] = work["symbol"].astype(str)
    by_date: Dict[str, Set[str]] = {}
    for day, group in work.groupby("day"):
        by_date[str(day)] = set(group["symbol"])
    return by_date


def _universe_asof(snaps: Sequence[Dict[str, Set[str]]], asof: str) -> Set[str]:
    """每个指数取 asof 当日或之前最近一期，再并集。"""
    out: Set[str] = set()
    for by_date in snaps:
        if not by_date:
            continue
        best = None
        for key in sorted(by_date):
            if key <= asof:
                best = key
            else:
                break
        if best is not None:
            out |= by_date[best]
    return out


def _is_bj(symbol: str) -> bool:
    text = str(symbol).upper()
    if text.startswith("BJ") or text.endswith(".BJ"):
        return True
    digits = "".join(ch for ch in text if ch.isdigit())
    code = digits[-6:] if len(digits) >= 6 else digits
    if code.startswith("688"):
        return False
    return code.startswith(("4", "8"))


def _asof_int(value) -> Optional[int]:
    parsed = _parse_day(value)
    if parsed is None:
        return None
    return parsed.year * 10000 + parsed.month * 100 + parsed.day


def _iso_day(value) -> str:
    """YYYYMMDD / ISO / 空 → YYYY-MM-DD 或空串，便于与 asof 直接字符串比较。"""
    parsed = _parse_day(value)
    return "" if parsed is None else parsed.strftime("%Y-%m-%d")


def _parse_day(value) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip().replace("-", "")[:8]
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return None


def _target(ctx: DayContext, weights: np.ndarray) -> TargetHoldings:
    return TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
