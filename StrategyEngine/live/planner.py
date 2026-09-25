#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用与回测相同的策略 + allocator 算出最新目标仓，再换成 QMT 手数。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from DailyUpdates.storage import SQLiteStorage
from yg_quant_repo import default_db_path

from ..backtest.engine import Engine
from ..backtest.panel import PanelStore
from ..holdings import TargetHoldings
from ..strategy import Strategy
from .bridge import QmtOrderPlan, plan_from_target


def next_session(
    asof: str,
    calendar: Optional[Sequence[str]] = None,
    *,
    storage=None,
) -> str:
    """asof 收盘后的下一开市日，只读本地 trade_calendar，不绑数据商。

    未来开市日由日常数据更新按当前 data_source 写入。表里没有下一开市日就报错，
    绝不在策略层猜周末/节假日。
    """
    if calendar is not None:
        found = _first_after(asof, calendar)
        if found:
            return found
        raise ValueError(
            f"给定交易日历没有 {asof} 之后的开市日，拒绝猜测节假日"
        )
    if storage is None:
        storage = SQLiteStorage(str(default_db_path()))
    found = _first_after(asof, storage.list_trade_dates(start_date=asof))
    if found:
        return found
    raise ValueError(
        f"本地 trade_calendar 没有 {asof} 之后的开市日。"
        "请先跑数据更新（会按当前数据源写入交易所已公布的未来开市日），"
        "不要在策略层猜测节假日。"
    )


def _first_after(asof: str, calendar: Sequence[str]) -> Optional[str]:
    day = _ymd(asof)
    for item in calendar or ():
        if _ymd(item) > day:
            return _iso(_ymd(item))
    return None


def _iso(ymd8: str) -> str:
    text = str(ymd8).replace("-", "")[:8]
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def latest_target(
    store: PanelStore,
    strategy: Strategy,
    *,
    capital: float,
) -> TargetHoldings:
    """从回测起点扫到最新交易日，留下最后一次目标（周调仓中间日沿用上一次）。"""
    return _replay_live(store, strategy, capital=capital)[0]


def _replay_live(
    store: PanelStore,
    strategy: Strategy,
    *,
    capital: float,
) -> Tuple[TargetHoldings, Dict[str, float]]:
    """回放到最后一根已收盘 K 线。

    返回 (下一开市日目标, 末日已成交权重)。末日目标要等 execute_on 才成交，
    不能算进已成交仓，否则 keep 日会把持仓再写成买单。
    """
    n = len(store.dates)
    t0 = int(store.active_index)
    if n <= t0:
        raise ValueError("面板没有可评分的交易日")
    cash = float(capital)
    last: Optional[TargetHoldings] = None
    n_stocks = len(store.symbols)
    filled = np.zeros(n_stocks, dtype=np.float64)
    for t in range(t0, n):
        execute_on = store.dates[t + 1] if t + 1 < n else next_session(store.dates[t])
        ctx = store.day_context(
            t,
            position=filled,
            cash=cash,
            value=cash,
            execute_on=execute_on,
        )
        raw = strategy.score(ctx)
        if raw is None:
            continue
        last = Engine._as_target(raw, ctx.asof, execute_on, ctx.symbols, n_stocks)
        if t + 1 < n:
            filled = last.aligned(store.symbols)
    if last is None:
        last = TargetHoldings.from_array(
            store.dates[-1],
            next_session(store.dates[-1]),
            store.symbols,
            filled,
        )
    if not last.execute_on:
        nxt = next_session(last.asof)
        if not nxt:
            raise ValueError(f"实盘计划缺少 execute_on（asof={last.asof}）")
        last = TargetHoldings(
            asof=last.asof,
            execute_on=nxt,
            weights=last.weights,
        )
    current = {
        str(symbol): float(weight)
        for symbol, weight in zip(store.symbols, filled)
        if float(weight) > 1e-9
    }
    return last, current


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
    target, current = _replay_live(store, strategy, capital=capital)
    if not target.execute_on:
        target = TargetHoldings(
            asof=target.asof,
            execute_on=next_session(target.asof),
            weights=target.weights,
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
        current=current,
    )


def _ymd(value: str) -> str:
    return str(value).replace("-", "")[:8]
