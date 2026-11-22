#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分红入账：明细汇总，以及除权日前后的现金、股数和净值。"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from StrategyEngine.backtest.dividend import events_to_schedule, load_dividend_schedule
from StrategyEngine.backtest.engine import Engine
from StrategyEngine.backtest.panel import PanelStore
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy


class _Book(Strategy):
    def __init__(self, plan):
        self.plan = {str(day): dict(weights) for day, weights in plan.items()}

    def score(self, ctx):
        return TargetHoldings(ctx.asof, ctx.execute_on, self.plan.get(ctx.asof, {}))


def _run(dates, symbols, opens, plan, dividends=None, delist=None, cash=10_000.0):
    panel = PanelStore.from_arrays(
        dates,
        symbols,
        open=np.asarray(opens, dtype=np.float64),
        delist_on=delist,
        dividends=dividends,
    )
    engine = Engine(
        panel,
        initial_cash=cash,
        commission=0.0,
        stamp=0.0,
        slippage=0.0,
        min_commission=0.0,
    )
    return engine.run(_Book(plan))


def _row(**kwargs):
    base = {
        "symbol": "SH600000",
        "ex_date": "2024-06-19",
        "div_proc": "实施",
        "cash_div_tax": 0.0,
        "stk_bo_rate": np.nan,
        "stk_co_rate": np.nan,
        "stk_div": 0.0,
    }
    base.update(kwargs)
    return base


def test_schedule_keeps_implemented_and_sums_same_day():
    frame = pd.DataFrame(
        [
            _row(cash_div_tax=0.4, stk_bo_rate=0.1),
            _row(cash_div_tax=0.6, stk_co_rate=0.2, stk_div=9.0),
            _row(div_proc="预案", cash_div_tax=5.0),
            _row(div_proc="股东大会通过", cash_div_tax=5.0),
            _row(symbol="SH600001", cash_div_tax=3.0),
            _row(ex_date="19990101", cash_div_tax=8.0),
            _row(cash_div_tax=-1.0, stk_bo_rate=-0.5),
        ]
    )
    book = events_to_schedule(frame, ["2024-06-19"], ["SH600000"])
    idx, cash, mult = book["2024-06-19"]
    assert idx.tolist() == [0]
    assert cash[0] == pytest.approx(1.0)
    assert mult[0] == pytest.approx(1.3)


def test_schedule_uses_total_only_when_components_empty():
    frame = pd.DataFrame(
        [
            _row(stk_div=0.5),
            _row(symbol="SH600001", stk_bo_rate=0.2, stk_div=0.9),
        ]
    )
    book = events_to_schedule(frame, ["2024-06-19"], ["SH600000", "SH600001"])
    _, _, mult = book["2024-06-19"]
    assert mult.tolist() == pytest.approx([1.5, 1.2])


def test_cash_dividend_offsets_ex_date_drop():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, ["SH600000"])
    plan = {day: {"SH600000": 1.0} for day in dates}
    result = _run(dates, ["SH600000"], [[10], [10], [9], [9]], plan, book)
    assert result.nav == pytest.approx([10_000.0] * 4)
    assert result.cash[1] == pytest.approx(0.0)
    assert result.cash[2] == pytest.approx(1_000.0)
    assert result.cash[3] == pytest.approx(1_000.0)


def test_bonus_issue_offsets_split():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(stk_bo_rate=1.0)]), dates, ["SH600000"])
    plan = {day: {"SH600000": 1.0} for day in dates}
    result = _run(dates, ["SH600000"], [[10], [10], [5]], plan, book, cash=10_000.0)
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(0.0)


def test_cash_and_bonus_use_pre_adjustment_shares():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(
        pd.DataFrame([_row(cash_div_tax=1.0, stk_co_rate=1.0)]),
        dates,
        ["SH600000"],
    )
    plan = {day: {"SH600000": 1.0} for day in dates}
    # 除权价 = (20 - 1) / 2
    result = _run(dates, ["SH600000"], [[20], [20], [9.5]], plan, book, cash=20_000.0)
    assert result.nav[-1] == pytest.approx(20_000.0)
    assert result.cash[-1] == pytest.approx(1_000.0)


def test_shares_bought_on_ex_date_are_not_entitled():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(
        pd.DataFrame([_row(cash_div_tax=1.0, stk_bo_rate=1.0)]),
        dates,
        ["SH600000"],
    )
    plan = {"2024-06-18": {"SH600000": 1.0}}
    result = _run(dates, ["SH600000"], [[10], [10], [9]], plan, book, cash=9_000.0)
    assert result.cash[-1] == pytest.approx(0.0)
    assert result.nav[-1] == pytest.approx(9_000.0)


def test_dividend_cash_can_be_spent_on_a_later_rebalance():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20"]
    symbols = ["SH600000", "SH600001"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, symbols)
    opens = np.array(
        [
            [10.0, 10.0],
            [10.0, 10.0],
            [9.0, 10.0],
            [9.0, 10.0],
        ]
    )
    plan = {
        "2024-06-17": {"SH600000": 1.0},
        "2024-06-18": {"SH600000": 1.0},
        "2024-06-19": {"SH600001": 1.0},
        "2024-06-20": {"SH600001": 1.0},
    }
    result = _run(dates, symbols, opens, plan, book)
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(0.0)
    assert result.weights[-1, 1] == pytest.approx(1.0)


def test_delist_on_ex_date_sells_after_the_dividend():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, ["SH600000"])
    plan = {day: {"SH600000": 1.0} for day in dates}
    result = _run(
        dates,
        ["SH600000"],
        [[10], [10], [9]],
        plan,
        book,
        delist=["2024-06-19"],
    )
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(10_000.0)
    assert result.forced_fills[-1].reason == ["delist"]


def test_price_drop_without_dividend_reduces_nav():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    plan = {day: {"SH600000": 1.0} for day in dates}
    result = _run(dates, ["SH600000"], [[10], [10], [9]], plan, dividends={})
    assert result.nav[-1] == pytest.approx(9_000.0)
    assert result.cash[-1] == pytest.approx(0.0)


def test_flat_book_does_not_receive_cash():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, ["SH600000"])
    result = _run(dates, ["SH600000"], [[10], [10], [9]], {}, book)
    assert result.nav == pytest.approx([10_000.0, 10_000.0, 10_000.0])
    assert result.cash[-1] == pytest.approx(10_000.0)


def test_loader_reads_sqlite_and_skips_missing_table():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    path = Path(handle.name)
    try:
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE instruments (symbol TEXT)")
        connection.commit()
        connection.close()
        store = SimpleNamespace(db_path=path)
        assert load_dividend_schedule(store, ["2024-06-19"], ["SH600000"]) == {}
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE stock_dividend (
                symbol TEXT, end_date TEXT, div_proc TEXT, cash_div_tax REAL,
                stk_bo_rate REAL, stk_co_rate REAL, stk_div REAL, ex_date TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO stock_dividend VALUES (?,?,?,?,?,?,?,?)",
            ("SH600000", "2023-12-31", "实施", 1.25, None, 0.4, 0.4, "2024-06-19"),
        )
        connection.execute(
            "INSERT INTO stock_dividend VALUES (?,?,?,?,?,?,?,?)",
            ("SH600000", "2023-12-31", "预案", 9.0, None, None, 0.0, "2024-06-19"),
        )
        connection.commit()
        connection.close()
        book = load_dividend_schedule(store, ["2024-06-19"], ["SH600000"])
        _, cash, mult = book["2024-06-19"]
        assert cash[0] == pytest.approx(1.25)
        assert mult[0] == pytest.approx(1.4)
    finally:
        path.unlink(missing_ok=True)


def test_recorded_moutai_dividend_matches_the_table():
    path = Path(__file__).resolve().parents[2] / "data" / "yg_quant.db"
    if not path.is_file():
        pytest.skip("没有行情库")
    book = load_dividend_schedule(
        SimpleNamespace(db_path=path),
        ["2024-06-19"],
        ["SH600519"],
    )
    if "2024-06-19" not in book:
        pytest.skip("库里没有茅台该除权日")
    _, cash, mult = book["2024-06-19"]
    assert cash[0] == pytest.approx(30.876)
    assert mult[0] == pytest.approx(1.0)
