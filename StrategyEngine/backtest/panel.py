#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预加载 open/close/因子/股票池，按日 O(1) 切片。不装评估用远期收益。"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import logging
import numpy as np
import pandas as pd

from Universes.catalog import build_mask as build_universe_mask
from Universes.rules import delist_on, iso_day

from FactorEvaluates.factor_panel_loader import FactorPanelLoader
from FactorEvaluates.market_panel_loader import MarketPanelLoader

from ..context import DayContext


class PanelStore:
    def __init__(
        self,
        *,
        dates: Sequence[str],
        symbols: Sequence[str],
        market: Mapping[str, np.ndarray],
        mask: np.ndarray,
        factors: Optional[Mapping[str, np.ndarray]] = None,
        active_start: Optional[str] = None,
        delist_on: Optional[Sequence[str]] = None,
    ):
        self.dates = [str(d) for d in dates]
        self.symbols = [str(s) for s in symbols]
        n_dates = len(self.dates)
        n_stocks = len(self.symbols)
        self.market: Dict[str, np.ndarray] = {}
        for name, arr in market.items():
            panel = np.asarray(arr, dtype=np.float64)
            if panel.shape != (n_dates, n_stocks):
                raise ValueError(f"行情 {name} 形状 {panel.shape}，期望 {(n_dates, n_stocks)}")
            self.market[str(name)] = panel
        if "open" not in self.market:
            raise ValueError("PanelStore 必须包含 open")
        self.factors: Dict[str, np.ndarray] = {}
        for name, arr in (factors or {}).items():
            panel = np.asarray(arr, dtype=np.float64)
            if panel.shape != (n_dates, n_stocks):
                raise ValueError(f"因子 {name} 形状 {panel.shape}，期望 {(n_dates, n_stocks)}")
            self.factors[str(name)] = panel
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (n_dates, n_stocks):
            raise ValueError(f"mask 形状 {mask_arr.shape}，期望 {(n_dates, n_stocks)}")
        self.mask = mask_arr
        if delist_on is None:
            self.delist_on = np.full(n_stocks, "", dtype="U10")
        else:
            arr = np.asarray(list(delist_on), dtype="U10").reshape(-1)
            if arr.size != n_stocks:
                raise ValueError(f"delist_on 长度 {arr.size}，期望 {n_stocks}")
            self.delist_on = arr
        self._index = {date: i for i, date in enumerate(self.dates)}
        self.active_start = str(active_start) if active_start else (self.dates[0] if self.dates else "")
        self.active_index = 0
        if self.dates and self.active_start:
            for i, date in enumerate(self.dates):
                if date >= self.active_start:
                    self.active_index = i
                    break
            else:
                raise ValueError(f"面板没有 {self.active_start} 及之后的交易日")

    @property
    def run_dates(self) -> List[str]:
        """净值 / 调仓日历，不含 allocator 预热段。"""
        return self.dates[self.active_index :]

    @classmethod
    def from_arrays(
        cls,
        dates: Sequence[str],
        symbols: Sequence[str],
        *,
        open: np.ndarray,
        close: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        factors: Optional[Mapping[str, np.ndarray]] = None,
        extra_market: Optional[Mapping[str, np.ndarray]] = None,
        active_start: Optional[str] = None,
        delist_on: Optional[Sequence[str]] = None,
    ) -> "PanelStore":
        market: Dict[str, np.ndarray] = {"open": np.asarray(open, dtype=np.float64)}
        if close is not None:
            market["close"] = np.asarray(close, dtype=np.float64)
        for name, arr in (extra_market or {}).items():
            market[str(name)] = np.asarray(arr, dtype=np.float64)
        n_dates, n_stocks = market["open"].shape
        if mask is None:
            mask = np.isfinite(market["open"])
        return cls(
            dates=dates,
            symbols=symbols,
            market=market,
            mask=mask,
            factors=factors,
            active_start=active_start,
            delist_on=delist_on,
        )

    @classmethod
    def load(
        cls,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        universe: str = "all",
        factors: Optional[Sequence[str]] = None,
        market_fields: Sequence[str] = ("close", "up_limit", "down_limit"),
        warmup: int = 0,
        market_loader: Optional[MarketPanelLoader] = None,
        factor_loader: Optional[FactorPanelLoader] = None,
        adjust: Optional[str] = None,
        asset_class: str = "stock",
    ) -> "PanelStore":
        kind = str(asset_class or "stock").strip().lower()
        if kind in {"etf", "fund"}:
            return cls._load_etf(
                start=start,
                end=end,
                market_fields=market_fields,
                warmup=warmup,
                market_loader=market_loader,
            )
        market_loader = market_loader or MarketPanelLoader()
        log = logging.getLogger("StrategyEngine")
        exec_adjust = "none" if adjust is None else str(adjust).strip().lower()
        log.info("成交价未复权 adjust=%s（因子 bin 仍按其计算口径）", exec_adjust)
        fetch_start = _warmup_start(market_loader, start, warmup)
        if warmup and fetch_start != start:
            log.info("预热 %s 个交易日，加载 %s ~ %s，回测从 %s 起", warmup, fetch_start or "库起点", end or "库终点", start)
        else:
            log.info("加载 open %s ~ %s", start or "库起点", end or "库终点")
        open_panel = market_loader.load_open(
            start_date=fetch_start, end_date=end, adjust=exec_adjust
        )
        if open_panel.empty:
            raise ValueError("open 面板为空")
        open_panel = open_panel.copy()
        open_panel.index = pd.to_datetime(open_panel.index).strftime("%Y-%m-%d")
        if fetch_start:
            open_panel = open_panel.loc[open_panel.index >= fetch_start]
        if end:
            open_panel = open_panel.loc[open_panel.index <= end]
        if open_panel.empty:
            raise ValueError("指定区间内没有交易日")
        if start and not (open_panel.index >= start).any():
            raise ValueError(f"预热后仍没有 {start} 及之后的交易日")
        dates = [str(d) for d in open_panel.index]
        symbols = [str(c) for c in open_panel.columns]
        log.info("交易日 %s 天 × 股票 %s 只", len(dates), len(symbols))
        market: Dict[str, np.ndarray] = {
            "open": open_panel.to_numpy(dtype=np.float64),
        }
        for field in market_fields:
            name = str(field)
            if name == "open":
                continue
            log.info("加载 %s", name)
            try:
                panel = market_loader.load_field(
                    name, start_date=fetch_start, end_date=end, adjust=exec_adjust
                )
            except ValueError:
                log.warning("行情字段 %s 不在库中，跳过", name)
                continue
            if panel is None or panel.empty:
                log.warning("行情字段 %s 为空，跳过", name)
                continue
            aligned = panel.copy()
            aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
            aligned = aligned.reindex(index=open_panel.index, columns=open_panel.columns)
            market[name] = aligned.to_numpy(dtype=np.float64)
        log.info("构建股票池 mask")
        mask_frame = build_universe_mask(
            universe,
            open_panel.index,
            open_panel.columns,
            open_panel,
            market_loader.storage,
        )
        delist_dates = delist_on(symbols, market_loader.storage)
        factor_panels: Dict[str, np.ndarray] = {}
        names = [str(n) for n in (factors or ()) if n]
        if names:
            log.info("加载因子 %s", names)
            loader = factor_loader or FactorPanelLoader()
            loaded = loader.load_many(names, start_date=dates[0], end_date=dates[-1])
            for name in names:
                panel = loaded.get(name)
                if panel is None or panel.empty:
                    raise ValueError(f"因子 {name} 没有数据")
                aligned = panel.copy()
                aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
                aligned = aligned.reindex(index=open_panel.index, columns=open_panel.columns)
                factor_panels[name] = aligned.to_numpy(dtype=np.float64)
        log.info("面板就绪")
        return cls(
            dates=dates,
            symbols=symbols,
            market=market,
            mask=mask_frame.to_numpy(dtype=bool),
            factors=factor_panels,
            active_start=start,
            delist_on=delist_dates,
        )

    @classmethod
    def _load_etf(
        cls,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        market_fields: Sequence[str] = ("close", "vol", "amount"),
        warmup: int = 0,
        market_loader: Optional[MarketPanelLoader] = None,
    ) -> "PanelStore":
        """场内基金面板：etf_data 成交价，上市/退市 mask，不走 A 股 Universes。"""
        market_loader = market_loader or MarketPanelLoader()
        log = logging.getLogger("StrategyEngine")
        fetch_start = _warmup_start(market_loader, start, warmup)
        if warmup and fetch_start != start:
            log.info("预热 %s 个交易日，加载 ETF %s ~ %s，回测从 %s 起", warmup, fetch_start or "库起点", end or "库终点", start)
        else:
            log.info("加载 ETF open %s ~ %s", start or "库起点", end or "库终点")
        open_panel = market_loader.load_etf_open(start_date=fetch_start, end_date=end)
        if open_panel.empty:
            raise ValueError("ETF open 面板为空")
        open_panel = open_panel.copy()
        open_panel.index = pd.to_datetime(open_panel.index).strftime("%Y-%m-%d")
        if fetch_start:
            open_panel = open_panel.loc[open_panel.index >= fetch_start]
        if end:
            open_panel = open_panel.loc[open_panel.index <= end]
        if open_panel.empty:
            raise ValueError("指定区间内没有 ETF 交易日")
        if start and not (open_panel.index >= start).any():
            raise ValueError(f"预热后仍没有 {start} 及之后的交易日")
        dates = [str(d) for d in open_panel.index]
        symbols = [str(c) for c in open_panel.columns]
        log.info("交易日 %s 天 × ETF %s 只", len(dates), len(symbols))
        market: Dict[str, np.ndarray] = {
            "open": open_panel.to_numpy(dtype=np.float64),
        }
        for field in market_fields:
            name = str(field)
            if name == "open":
                continue
            log.info("加载 ETF %s", name)
            try:
                panel = market_loader.load_etf_field(name, start_date=fetch_start, end_date=end)
            except ValueError:
                log.warning("ETF 字段 %s 不在库中，跳过", name)
                continue
            if panel is None or panel.empty:
                log.warning("ETF 字段 %s 为空，跳过", name)
                continue
            aligned = panel.copy()
            aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
            aligned = aligned.reindex(index=open_panel.index, columns=open_panel.columns)
            market[name] = aligned.to_numpy(dtype=np.float64)
        mask = _etf_listing_mask(dates, symbols, market_loader.storage, market["open"])
        delist_dates = _etf_delist_on(symbols, market_loader.storage)
        close_arr = market.get("close")
        if close_arr is None:
            try:
                close_panel = market_loader.load_etf_field(
                    "close", start_date=fetch_start, end_date=end
                )
            except ValueError:
                close_panel = None
            if close_panel is not None and not close_panel.empty:
                aligned = close_panel.copy()
                aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
                aligned = aligned.reindex(index=open_panel.index, columns=open_panel.columns)
                close_arr = aligned.to_numpy(dtype=np.float64)
                market["close"] = close_arr
        pre_arr = None
        try:
            pre_panel = market_loader.load_etf_field(
                "pre_close", start_date=fetch_start, end_date=end
            )
        except ValueError:
            pre_panel = None
        if pre_panel is not None and not pre_panel.empty:
            aligned = pre_panel.copy()
            aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
            aligned = aligned.reindex(index=open_panel.index, columns=open_panel.columns)
            pre_arr = aligned.to_numpy(dtype=np.float64)
        if close_arr is not None and pre_arr is not None:
            factor = etf_qfq_factor(close_arr, pre_arr)
            n_jump = int((~np.isclose(factor[:-1], factor[1:], rtol=1e-9, atol=0)).sum())
            if n_jump:
                log.info("ETF 份额折算前复权，面板内 %s 次跳变", n_jump)
            _apply_etf_qfq(market, factor)
        log.info("ETF 面板就绪")
        return cls(
            dates=dates,
            symbols=symbols,
            market=market,
            mask=mask,
            factors={},
            active_start=start,
            delist_on=delist_dates,
        )

    def date_index(self, date: str) -> int:
        try:
            return self._index[str(date)]
        except KeyError as exc:
            raise KeyError(f"未知交易日 {date}") from exc

    def panel_for(self, field: str) -> np.ndarray:
        key = str(field)
        if key in self.market:
            return self.market[key]
        if key in self.factors:
            return self.factors[key]
        raise KeyError(self._missing_message(key))

    def market_row(self, field: str, t: int) -> np.ndarray:
        key = str(field)
        panel = self.market.get(key)
        if panel is None:
            raise KeyError(self._missing_message(key, kind="行情"))
        return panel[t].copy()

    def factor_row(self, name: str, t: int) -> np.ndarray:
        key = str(name)
        panel = self.factors.get(key)
        if panel is None:
            raise KeyError(self._missing_message(key, kind="因子"))
        return panel[t].copy()

    def day_context(
        self,
        t: int,
        *,
        position: np.ndarray,
        cash: float,
        value: float,
        execute_on: Optional[str],
    ) -> DayContext:
        return DayContext(
            store=self,
            t=t,
            asof=self.dates[t],
            execute_on=execute_on,
            position=position,
            cash=cash,
            value=value,
        )

    def _missing_message(self, name: str, kind: str = "字段") -> str:
        market = sorted(self.market)
        factors = sorted(self.factors)
        return (
            f"本次回测未装载{kind} {name!r}。"
            f"已装行情 {market or '无'}，因子 {factors or '无'}"
        )


_ETF_QFQ_MIN_STEP = 0.15
_ETF_PRICE_FIELDS = frozenset({"open", "high", "low", "close", "pre_close"})
_ETF_VOLUME_FIELDS = frozenset({"vol"})


def etf_qfq_factor(
    close: np.ndarray,
    pre_close: np.ndarray,
    *,
    min_step: float = _ETF_QFQ_MIN_STEP,
) -> np.ndarray:
    """用次日 pre_close / 当日 close 累积前复权因子，末日为 1。

    ETF 份额折算会把价格腰斩、股数加倍；Tushare 已改 pre_close，但成交价仍是折算后的。
    不复权则持仓市值在折算日蒸发。
    """
    close_arr = np.asarray(close, dtype=np.float64)
    pre_arr = np.asarray(pre_close, dtype=np.float64)
    if close_arr.shape != pre_arr.shape:
        raise ValueError(f"close {close_arr.shape} 与 pre_close {pre_arr.shape} 不一致")
    n_dates, n_symbols = close_arr.shape
    factor = np.ones((n_dates, n_symbols), dtype=np.float64)
    if n_dates < 2:
        return factor
    with np.errstate(divide="ignore", invalid="ignore"):
        step = pre_arr[1:] / close_arr[:-1]
    good = np.isfinite(step) & (close_arr[:-1] > 0) & (pre_arr[1:] > 0)
    jumped = good & (step > 0.05) & (step < 20.0)
    jumped &= np.abs(np.log(np.clip(step, 1e-12, 1e12))) >= np.log(1.0 + float(min_step))
    acc = np.ones(n_symbols, dtype=np.float64)
    for t in range(n_dates - 2, -1, -1):
        acc = np.where(jumped[t], acc * step[t], acc)
        factor[t] = acc
    return factor


def _apply_etf_qfq(market: Dict[str, np.ndarray], factor: np.ndarray) -> None:
    for name, panel in list(market.items()):
        arr = np.asarray(panel, dtype=np.float64)
        if arr.shape != factor.shape:
            continue
        if name in _ETF_PRICE_FIELDS:
            market[name] = arr * factor
        elif name in _ETF_VOLUME_FIELDS:
            with np.errstate(divide="ignore", invalid="ignore"):
                market[name] = arr / factor


def _warmup_start(market_loader: MarketPanelLoader, start: Optional[str], warmup: int) -> Optional[str]:
    """在 start 之前再取 warmup 个交易日，供 ctx.history 使用。"""
    if not start or int(warmup) <= 0:
        return start
    calendar = market_loader.storage.list_trade_dates(end_date=start)
    if len(calendar) <= int(warmup):
        return calendar[0] if calendar else start
    return calendar[-int(warmup)]


def _etf_delist_on(symbols: Sequence[str], storage) -> np.ndarray:
    n = len(symbols)
    out = np.full(n, "", dtype="U10")
    try:
        basic = storage.read_etf_basic()
    except Exception:
        return out
    if basic is None or basic.empty or "delist_date" not in basic.columns:
        return out
    series = (
        basic.drop_duplicates("symbol").set_index("symbol")["delist_date"].map(iso_day)
    )
    mapped = series.reindex([str(s) for s in symbols])
    text = mapped.fillna("").astype(str).to_numpy()
    out[:] = np.where(np.asarray([len(x) == 10 for x in text]), text, "")
    return out


def _etf_listing_mask(
    dates: Sequence[str],
    symbols: Sequence[str],
    storage,
    open_panel: np.ndarray,
) -> np.ndarray:
    """已上市、未到退市日、且当日有开盘。"""
    mask = np.isfinite(np.asarray(open_panel, dtype=np.float64))
    try:
        basic = storage.read_etf_basic()
    except Exception:
        return mask
    if basic is None or basic.empty:
        return mask
    work = basic.drop_duplicates("symbol").set_index("symbol")
    list_dates = work["list_date"].map(iso_day) if "list_date" in work.columns else None
    delist_dates = work["delist_date"].map(iso_day) if "delist_date" in work.columns else None
    day_arr = np.asarray([str(d) for d in dates], dtype="U10")
    for j, symbol in enumerate(symbols):
        listed = ""
        delisted = ""
        if list_dates is not None and symbol in list_dates.index:
            listed = str(list_dates.loc[symbol] or "")
            if listed == "nan":
                listed = ""
        if delist_dates is not None and symbol in delist_dates.index:
            delisted = str(delist_dates.loc[symbol] or "")
            if delisted == "nan":
                delisted = ""
        if len(listed) == 10:
            mask[:, j] &= day_arr >= listed
        if len(delisted) == 10:
            mask[:, j] &= day_arr < delisted
    return mask
