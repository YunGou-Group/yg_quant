#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分红的边界和库内对账。记账恒等式必须精确成立，不靠价格碰巧贴合。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from StrategyEngine.backtest.dividend import events_to_schedule, load_dividend_schedule
from StrategyEngine.backtest.engine import Engine
from StrategyEngine.backtest.panel import PanelStore
from StrategyEngine.backtest.test_dividend import _row, _run
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy

_DB = Path(__file__).resolve().parents[2] / "data" / "yg_quant.db"


class _Hold(Strategy):
    def __init__(self, symbol: str):
        self.symbol = symbol

    def score(self, ctx):
        return TargetHoldings(ctx.asof, ctx.execute_on, {self.symbol: 1.0})


def test_sell_on_ex_date_keeps_the_dividend():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0, stk_bo_rate=1.0)]), dates, ["SH600000"])
    # 除权价 = (10 - 1) / 2。除权日开盘把翻倍后的股数卖掉。
    plan = {"2024-06-17": {"SH600000": 1.0}, "2024-06-18": {}, "2024-06-19": {}}
    result = _run(dates, ["SH600000"], [[10], [10], [4.5]], plan, book)
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(10_000.0)
    assert result.weights[-1, 0] == pytest.approx(0.0)


def test_limit_down_blocks_the_sale_but_not_the_dividend():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    symbols = ["SH600000"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, symbols)
    panel = PanelStore.from_arrays(
        dates,
        symbols,
        open=np.array([[10.0], [10.0], [9.0]]),
        extra_market={"down_limit": np.array([[np.nan], [np.nan], [9.0]])},
        dividends=book,
    )
    result = Engine(
        panel,
        initial_cash=10_000.0,
        commission=0.0,
        stamp=0.0,
        slippage=0.0,
        min_commission=0.0,
    ).run(
        type(
            "_Plan",
            (Strategy,),
            {
                "score": lambda self, ctx: TargetHoldings(
                    ctx.asof,
                    ctx.execute_on,
                    {"SH600000": 1.0} if ctx.asof == "2024-06-17" else {},
                )
            },
        )()
    )
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(1_000.0)
    assert result.weights[-1, 0] == pytest.approx(0.9)
    sold = result.fills[-1]
    assert sold is not None
    assert sold.reason[0] == "limit_down"


def test_two_ex_dates_stack():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20"]
    frame = pd.DataFrame(
        [
            _row(ex_date="2024-06-19", cash_div_tax=1.0),
            _row(ex_date="2024-06-20", cash_div_tax=1.0),
        ]
    )
    book = events_to_schedule(frame, dates, ["SH600000"])
    plan = {day: {"SH600000": 1.0} for day in dates}
    result = _run(dates, ["SH600000"], [[10], [10], [9], [8]], plan, book)
    assert result.nav == pytest.approx([10_000.0] * 4)
    assert result.cash[-1] == pytest.approx(2_000.0)


def test_half_weight_only_credits_shares_held():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, ["SH600000"])
    plan = {day: {"SH600000": 0.5} for day in dates}
    result = _run(dates, ["SH600000"], [[10], [10], [9]], plan, book)
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(5_500.0)


def test_other_symbols_dividend_is_ignored():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    symbols = ["SH600000", "SH600001"]
    book = events_to_schedule(
        pd.DataFrame([_row(symbol="SH600001", cash_div_tax=4.0)]),
        dates,
        symbols,
    )
    plan = {day: {"SH600000": 1.0} for day in dates}
    opens = [[10.0, 10.0], [10.0, 10.0], [9.0, 6.0]]
    result = _run(dates, symbols, opens, plan, book)
    assert result.nav[-1] == pytest.approx(9_000.0)
    assert result.cash[-1] == pytest.approx(0.0)


def test_total_ratio_drives_the_engine_when_components_are_empty():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(stk_div=1.0)]), dates, ["SH600000"])
    plan = {day: {"SH600000": 1.0} for day in dates}
    result = _run(dates, ["SH600000"], [[10], [10], [5]], plan, book)
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(0.0)


def test_last_day_ex_date_is_marked():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, ["SH600000"])
    plan = {day: {"SH600000": 1.0} for day in dates}
    result = _run(dates, ["SH600000"], [[10], [10], [9]], plan, book)
    assert result.dates[-1] == "2024-06-19"
    assert result.nav[-1] == pytest.approx(10_000.0)
    assert result.cash[-1] == pytest.approx(1_000.0)


def test_fresh_start_on_ex_date_has_no_overnight_shares():
    dates = ["2024-06-18", "2024-06-19", "2024-06-20"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, ["SH600000"])
    panel = PanelStore.from_arrays(
        dates,
        ["SH600000"],
        open=np.array([[10.0], [9.0], [9.0]]),
        active_start="2024-06-19",
        dividends=book,
    )
    result = Engine(
        panel,
        initial_cash=9_000.0,
        commission=0.0,
        stamp=0.0,
        slippage=0.0,
        min_commission=0.0,
    ).run(_Hold("SH600000"))
    assert result.dates == ["2024-06-19", "2024-06-20"]
    assert result.cash[0] == pytest.approx(9_000.0)
    assert result.cash[-1] == pytest.approx(0.0)
    assert result.nav[-1] == pytest.approx(9_000.0)


def test_schedule_accepts_compact_dates_and_padded_status():
    frame = pd.DataFrame(
        [
            _row(ex_date="20240619", div_proc=" 实施 ", cash_div_tax=1.5),
            _row(ex_date="20240619", div_proc="预披露", cash_div_tax=8.0),
        ]
    )
    book = events_to_schedule(frame, ["2024-06-19"], ["SH600000"])
    _, cash, mult = book["2024-06-19"]
    assert cash[0] == pytest.approx(1.5)
    assert mult[0] == pytest.approx(1.0)


def test_missing_database_is_an_empty_book():
    assert load_dividend_schedule(SimpleNamespace(db_path=None), ["2024-06-19"], ["SH600000"]) == {}
    assert load_dividend_schedule(SimpleNamespace(db_path="Z:/no/such.db"), ["2024-06-19"], ["SH600000"]) == {}
    assert load_dividend_schedule(SimpleNamespace(db_path=_DB), [], ["SH600000"]) == {}


def test_commission_is_not_charged_on_the_dividend():
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    book = events_to_schedule(pd.DataFrame([_row(cash_div_tax=1.0)]), dates, ["SH600000"])
    panel = PanelStore.from_arrays(
        dates,
        ["SH600000"],
        open=np.array([[10.0], [10.0], [9.0]]),
        dividends=book,
    )
    result = Engine(
        panel,
        initial_cash=10_000.0,
        commission=0.001,
        stamp=0.0,
        slippage=0.0,
        min_commission=0.0,
    ).run(_Hold("SH600000"))
    # 买入被佣金挤到 900 股，之后每股仍足额派 1 元。
    assert result.cash[-1] == pytest.approx(991.0 + 900.0)
    assert result.nav[-1] == pytest.approx(900.0 * 9.0 + 991.0 + 900.0)


def test_random_ex_rights_prices_preserve_nav():
    rng = np.random.default_rng(0)
    dates = ["2024-06-17", "2024-06-18", "2024-06-19"]
    for _ in range(40):
        pre = float(rng.uniform(8.0, 40.0))
        cash_ps = float(rng.uniform(0.0, pre * 0.15))
        ratio = float(rng.choice([0.0, 0.1, 0.3, 0.5, 1.0]))
        if cash_ps == 0.0 and ratio == 0.0:
            continue
        ex = (pre - cash_ps) / (1.0 + ratio)
        initial = pre * 1000.0
        frame = pd.DataFrame([_row(cash_div_tax=cash_ps, stk_bo_rate=ratio)])
        book = events_to_schedule(frame, dates, ["SH600000"])
        plan = {day: {"SH600000": 1.0} for day in dates}
        result = _run(dates, ["SH600000"], [[pre], [pre], [ex]], plan, book, cash=initial)
        held = (result.nav[-2] - result.cash[-2]) / pre
        assert held == pytest.approx(round(held / 100.0) * 100.0)
        assert result.cash[-1] == pytest.approx(result.cash[-2] + held * cash_ps)
        assert result.nav[-1] == pytest.approx(held * (1.0 + ratio) * ex + result.cash[-1])


def _connect():
    if not _DB.is_file():
        pytest.skip("没有行情库")
    return sqlite3.connect(f"file:{_DB.resolve()}?mode=ro", uri=True)


def test_loader_matches_database_aggregates():
    connection = _connect()
    try:
        days = [
            row[0]
            for row in connection.execute(
                """
                SELECT ex_date FROM stock_dividend
                WHERE div_proc = '实施'
                GROUP BY ex_date
                ORDER BY ex_date
                LIMIT 8
                """
            )
        ]
        days.extend(["2019-04-23", "2018-09-19", "2024-06-19"])
        frames = []
        for day in days:
            part = pd.read_sql_query(
                """
                SELECT symbol, ex_date, div_proc, cash_div_tax, stk_bo_rate, stk_co_rate, stk_div
                FROM stock_dividend
                WHERE ex_date = ?
                """,
                connection,
                params=(day,),
            )
            frames.append(part)
    finally:
        connection.close()
    for part in frames:
        day = str(part["ex_date"].iloc[0])
        symbols = sorted(set(part["symbol"]))
        book = load_dividend_schedule(SimpleNamespace(db_path=_DB), [day], symbols)
        manual = events_to_schedule(part, [day], symbols)
        assert set(book) == set(manual)
        for key in book:
            assert book[key][0].tolist() == manual[key][0].tolist()
            assert book[key][1] == pytest.approx(manual[key][1])
            assert book[key][2] == pytest.approx(manual[key][2])
    moutai = next(part for part in frames if str(part["ex_date"].iloc[0]) == "2024-06-19")
    held = moutai[(moutai["symbol"] == "SH600519") & (moutai["div_proc"] == "实施")]
    assert not held.empty
    symbols = sorted(set(moutai["symbol"]))
    book = load_dividend_schedule(SimpleNamespace(db_path=_DB), ["2024-06-19"], symbols)
    loc = symbols.index("SH600519")
    idx = book["2024-06-19"][0].tolist()
    cash = float(book["2024-06-19"][1][idx.index(loc)])
    assert cash == pytest.approx(float(pd.to_numeric(held["cash_div_tax"]).fillna(0).sum()))
    assert cash != pytest.approx(0.0)


def test_engine_matches_real_opens_and_dividends():
    connection = _connect()
    try:
        events = pd.read_sql_query(
            """
            SELECT symbol, ex_date,
                   SUM(CASE WHEN cash_div_tax > 0 THEN cash_div_tax ELSE 0 END) AS cash_ps,
                   SUM(CASE
                         WHEN IFNULL(stk_bo_rate, 0) > 0 OR IFNULL(stk_co_rate, 0) > 0
                         THEN IFNULL(CASE WHEN stk_bo_rate > 0 THEN stk_bo_rate ELSE 0 END, 0)
                            + IFNULL(CASE WHEN stk_co_rate > 0 THEN stk_co_rate ELSE 0 END, 0)
                         WHEN stk_div > 0 THEN stk_div
                         ELSE 0
                       END) AS ratio
            FROM stock_dividend
            WHERE div_proc = '实施' AND ex_date BETWEEN '2020-01-01' AND '2024-12-31'
            GROUP BY symbol, ex_date
            HAVING cash_ps > 0 OR ratio > 0
            """,
            connection,
        )
        picked = []
        for kind, part in (
            ("cash", events[(events["ratio"] == 0) & (events["cash_ps"] > 0)]),
            ("share", events[(events["cash_ps"] == 0) & (events["ratio"] > 0)]),
            ("both", events[(events["cash_ps"] > 0) & (events["ratio"] > 0)]),
        ):
            part = part.sample(n=min(8, len(part)), random_state=7) if len(part) else part
            for _, row in part.iterrows():
                picked.append((kind, row))
        checked = 0
        for kind, row in picked:
            symbol = str(row["symbol"])
            ex_date = str(row["ex_date"])
            quotes = connection.execute(
                """
                SELECT trade_date, open FROM market_data
                WHERE symbol = ? AND trade_date <= ? AND open > 0
                ORDER BY trade_date DESC
                LIMIT 2
                """,
                (symbol, ex_date),
            ).fetchall()
            if len(quotes) < 2 or quotes[0][0] != ex_date:
                continue
            ex_open = float(quotes[0][1])
            prev_day, prev_open = str(quotes[1][0]), float(quotes[1][1])
            if prev_open < 5.0 or ex_open <= 0.0:
                continue
            cash_ps = float(row["cash_ps"])
            ratio = float(row["ratio"])
            shares = 1000.0
            initial = prev_open * shares
            dates = [prev_day, ex_date]
            # 前一日的前一交易日用来完成买入，价格等于前收开盘，使持仓刚好是 1000 股。
            anchor = connection.execute(
                """
                SELECT trade_date FROM market_data
                WHERE symbol = ? AND trade_date < ? AND open > 0
                ORDER BY trade_date DESC
                LIMIT 1
                """,
                (symbol, prev_day),
            ).fetchone()
            if anchor is None:
                continue
            start = str(anchor[0])
            book = events_to_schedule(
                pd.DataFrame(
                    [
                        _row(
                            symbol=symbol,
                            ex_date=ex_date,
                            cash_div_tax=cash_ps,
                            stk_bo_rate=ratio,
                        )
                    ]
                ),
                [start, prev_day, ex_date],
                [symbol],
            )
            result = _run(
                [start, prev_day, ex_date],
                [symbol],
                [[prev_open], [prev_open], [ex_open]],
                {start: {symbol: 1.0}, prev_day: {symbol: 1.0}, ex_date: {symbol: 1.0}},
                book,
                cash=initial,
            )
            held = (result.nav[-2] - result.cash[-2]) / prev_open
            if held < 100.0:
                continue
            assert result.cash[-1] == pytest.approx(result.cash[-2] + held * cash_ps), kind
            assert result.nav[-1] == pytest.approx(
                held * (1.0 + ratio) * ex_open + result.cash[-1]
            ), kind
            checked += 1
        assert checked >= 12
    finally:
        connection.close()
