#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用与回测相同的策略 + allocator 算出最新目标仓，再换成 QMT 手数。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from DailyUpdates.storage import SQLiteStorage
from yg_quant_repo import default_db_path

from ..backtest.engine import Engine
from ..backtest.panel import PanelStore
from ..holdings import TargetHoldings
from ..strategy import Strategy
from .bridge import QmtOrderPlan, plan_from_target


def next_session(asof: str, calendar: Optional[Sequence[str]] = None) -> Optional[str]:
    day = _ymd(asof)
    if calendar is None:
        calendar = SQLiteStorage(str(default_db_path())).list_trade_dates(start_date=asof)
    for item in calendar:
        if _ymd(item) > day:
            return str(item)[:10]
    return None


def latest_target(
    store: PanelStore,
    strategy: Strategy,
    *,
    capital: float,
) -> TargetHoldings:
    """从回测起点扫到最新交易日，留下最后一次非空目标（周调仓中间日沿用上一次）。"""
    n = len(store.dates)
    t0 = int(store.active_index)
    if n <= t0:
        raise ValueError("面板没有可评分的交易日")
    cash = float(capital)
    last: Optional[TargetHoldings] = None
    n_stocks = len(store.symbols)
    empty = np.zeros(n_stocks, dtype=np.float64)
    for t in range(t0, n):
        execute_on = store.dates[t + 1] if t + 1 < n else next_session(store.dates[t])
        ctx = store.day_context(
            t,
            position=empty,
            cash=cash,
            value=cash,
            execute_on=execute_on,
        )
        raw = strategy.score(ctx)
        if raw is None:
            continue
        last = Engine._as_target(raw, ctx.asof, execute_on, ctx.symbols, n_stocks)
    if last is None:
        last = TargetHoldings.from_array(
            store.dates[-1],
            next_session(store.dates[-1]),
            store.symbols,
            empty,
        )
    if last.execute_on is None:
        last = TargetHoldings(
            asof=last.asof,
            execute_on=next_session(last.asof),
            weights=last.weights,
        )
    return last


def prices_on(store: PanelStore, asof: str) -> Dict[str, float]:
    t = store.date_index(asof)
    row = None
    close = store.market.get("close")
    if close is not None:
        row = np.asarray(close[t], dtype=np.float64)
    opens = np.asarray(store.market["open"][t], dtype=np.float64)
    if row is None:
        row = opens
    else:
        row = np.where(np.isfinite(row) & (row > 0), row, opens)
    return {
        str(symbol): float(price)
        for symbol, price in zip(store.symbols, row)
        if np.isfinite(price) and price > 0
    }


def plan_live(
    store: PanelStore,
    strategy: Strategy,
    *,
    capital: float,
    strategy_name: str,
    allocator: str,
    account: str = "",
    lot_size: int = 100,
    params: Optional[Dict[str, Any]] = None,
) -> QmtOrderPlan:
    target = latest_target(store, strategy, capital=capital)
    if not target.execute_on:
        raise ValueError(
            f"实盘计划缺少 execute_on（asof={target.asof}）："
            "本地交易日历看不到下一交易日，拒绝写出无日期 JSON"
        )
    return plan_from_target(
        target,
        prices_on(store, target.asof),
        capital=capital,
        strategy=strategy_name,
        allocator=allocator,
        account=account,
        lot_size=lot_size,
        params=params,
    )


def _ymd(value: str) -> str:
    return str(value).replace("-", "")[:8]
