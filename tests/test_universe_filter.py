#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""次新过滤按交易所日历，而不是回测窗口下标。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from FactorEvaluates.universe_filter import UniverseFilter
from StrategyEngine.backtest import Engine, PanelStore
from StrategyEngine.holdings import TargetHoldings


class _Storage:
    def __init__(self, basic: pd.DataFrame, calendar: list[str]):
        self._basic = basic
        self._calendar = list(calendar)

    def read_stock_basic(self) -> pd.DataFrame:
        return self._basic

    def list_trade_dates(self, start_date=None, end_date=None):
        out = list(self._calendar)
        if start_date:
            out = [d for d in out if d >= str(start_date)]
        if end_date:
            out = [d for d in out if d <= str(end_date)]
        return out

    def read_stock_namechange(self) -> pd.DataFrame:
        return pd.DataFrame()

    def read_index_constituents(self, index_code=None) -> pd.DataFrame:
        return pd.DataFrame()


def _opens(dates, symbols, price=10.0) -> pd.DataFrame:
    return pd.DataFrame(price, index=pd.Index(dates), columns=pd.Index(symbols))


class UniverseListingTests(unittest.TestCase):
    def test_veteran_is_tradable_on_short_window(self):
        calendar = pd.bdate_range("2019-01-02", periods=80).strftime("%Y-%m-%d").tolist()
        panel = calendar[-5:]
        basic = pd.DataFrame(
            {"symbol": ["AAA"], "name": ["老股"], "list_date": ["2019-01-02"]}
        )
        mask = UniverseFilter(_Storage(basic, calendar), min_list_days=60).build_mask(
            pd.Index(panel), pd.Index(["AAA"]), _opens(panel, ["AAA"])
        )
        self.assertTrue(bool(mask.iloc[0, 0]))
        self.assertTrue(bool(mask.iloc[-1, 0]))

    def test_recent_ipo_waits_out_min_days(self):
        calendar = pd.bdate_range("2020-01-02", periods=20).strftime("%Y-%m-%d").tolist()
        panel = calendar[-8:]
        basic = pd.DataFrame(
            {"symbol": ["BBB"], "name": ["次新"], "list_date": [calendar[0]]}
        )
        mask = UniverseFilter(_Storage(basic, calendar), min_list_days=15).build_mask(
            pd.Index(panel), pd.Index(["BBB"]), _opens(panel, ["BBB"])
        )
        self.assertFalse(bool(mask.iloc[0, 0]))

    def test_missing_basic_stays_tradable(self):
        calendar = pd.bdate_range("2020-01-02", periods=10).strftime("%Y-%m-%d").tolist()
        panel = calendar[-4:]
        basic = pd.DataFrame(columns=["symbol", "name", "list_date"])
        mask = UniverseFilter(_Storage(basic, calendar), min_list_days=60).build_mask(
            pd.Index(panel), pd.Index(["ZZZ"]), _opens(panel, ["ZZZ"])
        )
        self.assertTrue(bool(mask.iloc[0, 0]))


class UniverseDelistTests(unittest.TestCase):
    def test_cannot_buy_if_next_open_is_on_or_after_delist(self):
        calendar = pd.bdate_range("2020-01-02", periods=8).strftime("%Y-%m-%d").tolist()
        panel = calendar[:4]
        basic = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "name": ["退市股"],
                "list_date": ["2010-01-04"],
                "delist_date": [panel[2]],
            }
        )
        mask = UniverseFilter(_Storage(basic, calendar), min_list_days=0).build_mask(
            pd.Index(panel), pd.Index(["AAA"]), _opens(panel, ["AAA"])
        )
        self.assertTrue(bool(mask.iloc[0, 0]))
        self.assertFalse(bool(mask.iloc[1, 0]))
        self.assertFalse(bool(mask.iloc[2, 0]))

    def test_delist_on_aligned(self):
        basic = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "name": ["退", "活"],
                "delist_date": ["20181218", ""],
            }
        )
        got = UniverseFilter(_Storage(basic, ["2020-01-02"])).delist_on(
            ["BBB", "AAA", "ZZZ"]
        )
        self.assertEqual(list(got), ["", "2018-12-18", ""])


class _HoldOne:
    def score(self, ctx):
        w = np.zeros(len(ctx.symbols))
        w[0] = 1.0
        return TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, w)


class HaltMarkTests(unittest.TestCase):
    def test_halt_then_resume_keeps_last_price(self):
        open_px = np.array(
            [
                [10.0, 10.0],
                [10.0, 10.0],
                [np.nan, 10.0],
                [10.0, 10.0],
            ]
        )
        result = Engine(
            _px_store(open_px), initial_cash=10_000.0, commission=0.0, stamp=0.0
        ).run(_HoldOne())
        self.assertGreater(result.weights[1, 0], 0.99)
        self.assertGreater(result.weights[2, 0], 0.99)
        self.assertGreater(result.nav[2], 9_000.0)
        self.assertAlmostEqual(result.nav[2], result.nav[1], places=4)

    def test_halt_without_later_quote_still_holds(self):
        open_px = np.array(
            [
                [10.0, 10.0],
                [10.0, 10.0],
                [np.nan, 10.0],
                [np.nan, 10.0],
            ]
        )
        result = Engine(
            _px_store(open_px), initial_cash=10_000.0, commission=0.0, stamp=0.0
        ).run(_HoldOne())
        self.assertGreater(result.weights[1, 0], 0.99)
        self.assertGreater(result.weights[2, 0], 0.99)
        self.assertGreater(result.weights[3, 0], 0.99)
        self.assertGreater(result.nav[2], 9_000.0)
        self.assertAlmostEqual(result.nav[2], result.nav[1], places=4)
        self.assertAlmostEqual(result.nav[3], result.nav[2], places=4)

    def test_extending_future_quotes_does_not_rewrite_history(self):
        short_px = np.array(
            [
                [10.0, 10.0],
                [10.0, 10.0],
                [np.nan, 10.0],
                [np.nan, 10.0],
            ]
        )
        long_px = np.vstack([short_px, [[10.0, 10.0]]])
        short = Engine(
            _px_store(short_px), initial_cash=10_000.0, commission=0.0, stamp=0.0
        ).run(_HoldOne())
        long = Engine(
            _px_store(long_px), initial_cash=10_000.0, commission=0.0, stamp=0.0
        ).run(_HoldOne())
        n = len(short.dates)
        np.testing.assert_allclose(long.nav[:n], short.nav)
        np.testing.assert_allclose(long.cash[:n], short.cash)
        np.testing.assert_allclose(long.weights[:n], short.weights)

    def test_official_delist_cashes_out_even_if_quotes_remain(self):
        open_px = np.full((4, 2), 10.0)
        dates = pd.bdate_range("2020-01-02", periods=4).strftime("%Y-%m-%d").tolist()
        store = PanelStore.from_arrays(
            dates,
            ["s0", "s1"],
            open=open_px,
            close=open_px,
            mask=np.ones((4, 2), dtype=bool),
            delist_on=[dates[2], ""],
        )
        result = Engine(store, initial_cash=10_000.0, commission=0.0, stamp=0.0).run(
            _HoldOne()
        )
        self.assertGreater(result.weights[1, 0], 0.99)
        self.assertEqual(result.weights[2, 0], 0.0)
        self.assertGreater(result.cash[2], 9_000.0)
        self.assertAlmostEqual(result.nav[2], result.nav[1], places=4)


def _px_store(open_px: np.ndarray) -> PanelStore:
    n_dates, n_stocks = open_px.shape
    dates = pd.bdate_range("2020-01-02", periods=n_dates).strftime("%Y-%m-%d").tolist()
    symbols = [f"s{i}" for i in range(n_stocks)]
    return PanelStore.from_arrays(
        dates,
        symbols,
        open=open_px,
        close=np.nan_to_num(open_px, nan=10.0),
        mask=np.ones((n_dates, n_stocks), dtype=bool),
    )


if __name__ == "__main__":
    unittest.main()
