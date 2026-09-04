#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIT 股票池：历史 ST、当日开盘、指数 asof、catalog。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Universes.catalog import get, names, reset_cache
from Universes.rules import board_of_symbol


class _Storage:
    def __init__(self, basic=None, calendar=None, namechange=None, constituents=None):
        self._basic = basic if basic is not None else pd.DataFrame()
        self._calendar = list(calendar or [])
        self._namechange = namechange if namechange is not None else pd.DataFrame()
        self._constituents = constituents if constituents is not None else pd.DataFrame()

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
        return self._namechange

    def read_index_constituents(self, index_code=None) -> pd.DataFrame:
        frame = self._constituents
        if frame.empty or not index_code:
            return frame
        return frame[frame["index_code"] == index_code]


def _opens(dates, symbols, price=10.0) -> pd.DataFrame:
    return pd.DataFrame(price, index=pd.Index(dates), columns=pd.Index(symbols))


class CatalogTests(unittest.TestCase):
    def test_all_is_registered(self):
        reset_cache()
        self.assertIn("all", names())
        self.assertIn("hs300", names())
        self.assertIn("main", names())

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get("not_a_universe")
        self.assertIn("未知股票池", str(ctx.exception))


class PitStTests(unittest.TestCase):
    def test_current_st_does_not_wipe_history(self):
        dates = ["2020-01-02", "2020-01-03", "2020-06-01"]
        basic = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "name": ["*ST今"],
                "list_date": ["2010-01-04"],
            }
        )
        namechange = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "name": ["正常", "*ST今"],
                "start_date": ["2010-01-04", "2020-06-01"],
                "end_date": ["2020-05-31", ""],
            }
        )
        mask = get("all").build_mask(
            pd.Index(dates),
            pd.Index(["AAA"]),
            _opens(dates, ["AAA"]),
            _Storage(basic, dates, namechange),
            min_list_days=0,
        )
        self.assertTrue(bool(mask.loc["2020-01-02", "AAA"]))
        self.assertTrue(bool(mask.loc["2020-01-03", "AAA"]))
        self.assertFalse(bool(mask.loc["2020-06-01", "AAA"]))

    def test_historical_st_is_excluded_only_in_span(self):
        dates = ["2020-01-02", "2020-06-01"]
        basic = pd.DataFrame(
            {"symbol": ["BBB"], "name": ["正常"], "list_date": ["2010-01-04"]}
        )
        namechange = pd.DataFrame(
            {
                "symbol": ["BBB", "BBB"],
                "name": ["ST旧", "正常"],
                "start_date": ["2010-01-04", "2020-03-01"],
                "end_date": ["2020-02-28", ""],
            }
        )
        mask = get("all").build_mask(
            pd.Index(dates),
            pd.Index(["BBB"]),
            _opens(dates, ["BBB"]),
            _Storage(basic, dates, namechange),
            min_list_days=0,
        )
        self.assertFalse(bool(mask.loc["2020-01-02", "BBB"]))
        self.assertTrue(bool(mask.loc["2020-06-01", "BBB"]))


class OpenPitTests(unittest.TestCase):
    def test_halt_tomorrow_stays_in_universe_today(self):
        dates = ["2020-01-02", "2020-01-03"]
        basic = pd.DataFrame(
            {"symbol": ["AAA"], "name": ["活"], "list_date": ["2010-01-04"]}
        )
        open_px = pd.DataFrame(
            {"AAA": [10.0, np.nan]}, index=pd.Index(dates)
        )
        mask = get("all").build_mask(
            pd.Index(dates),
            pd.Index(["AAA"]),
            open_px,
            _Storage(basic, dates),
            min_list_days=0,
        )
        self.assertTrue(bool(mask.loc["2020-01-02", "AAA"]))
        self.assertFalse(bool(mask.loc["2020-01-03", "AAA"]))

    def test_require_next_open_still_drops_last_day(self):
        dates = ["2020-01-02", "2020-01-03"]
        basic = pd.DataFrame(
            {"symbol": ["AAA"], "name": ["活"], "list_date": ["2010-01-04"]}
        )
        mask = get("all").build_mask(
            pd.Index(dates),
            pd.Index(["AAA"]),
            _opens(dates, ["AAA"]),
            _Storage(basic, dates),
            min_list_days=0,
            require_next_open=True,
        )
        self.assertTrue(bool(mask.loc["2020-01-02", "AAA"]))
        self.assertFalse(bool(mask.loc["2020-01-03", "AAA"]))


class IndexAsOfTests(unittest.TestCase):
    def test_dropped_member_leaves_after_snapshot(self):
        dates = ["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"]
        basic = pd.DataFrame(
            {
                "symbol": ["SH600000", "SH600519"],
                "name": ["活", "活"],
                "list_date": ["2010-01-04", "2010-01-04"],
            }
        )
        cons = pd.DataFrame(
            {
                "index_code": ["000300.SH"] * 3,
                "symbol": ["SH600000", "SH600519", "SH600519"],
                "trade_date": ["2020-01-02", "2020-01-02", "2020-02-03"],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        mask = get("hs300").build_mask(
            pd.Index(dates),
            pd.Index(["SH600000", "SH600519"]),
            _opens(dates, ["SH600000", "SH600519"]),
            _Storage(basic, dates, constituents=cons),
            min_list_days=0,
        )
        self.assertTrue(bool(mask.loc["2020-01-02", "SH600000"]))
        self.assertTrue(bool(mask.loc["2020-01-03", "SH600000"]))
        self.assertFalse(bool(mask.loc["2020-02-03", "SH600000"]))
        self.assertTrue(bool(mask.loc["2020-02-03", "SH600519"]))

    def test_missing_constituents_raise(self):
        dates = ["2020-01-02"]
        basic = pd.DataFrame(
            {"symbol": ["SH600000"], "name": ["活"], "list_date": ["2010-01-04"]}
        )
        with self.assertRaises(ValueError) as ctx:
            get("hs300").build_mask(
                pd.Index(dates),
                pd.Index(["SH600000"]),
                _opens(dates, ["SH600000"]),
                _Storage(basic, dates),
                min_list_days=0,
            )
        self.assertIn("000300.SH", str(ctx.exception))


class BoardTests(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(board_of_symbol("SH600000"), "main")
        self.assertEqual(board_of_symbol("SZ000001"), "main")
        self.assertEqual(board_of_symbol("SZ300001"), "gem")
        self.assertEqual(board_of_symbol("SH688001"), "star")
        self.assertEqual(board_of_symbol("BJ430047"), "bse")

    def test_main_board_mask(self):
        dates = ["2020-01-02"]
        symbols = ["SH600000", "SZ300001", "SH688001"]
        basic = pd.DataFrame(
            {
                "symbol": symbols,
                "name": ["活", "活", "活"],
                "list_date": ["2010-01-04"] * 3,
            }
        )
        mask = get("main").build_mask(
            pd.Index(dates),
            pd.Index(symbols),
            _opens(dates, symbols),
            _Storage(basic, dates),
            min_list_days=0,
        )
        self.assertTrue(bool(mask.loc["2020-01-02", "SH600000"]))
        self.assertFalse(bool(mask.loc["2020-01-02", "SZ300001"]))
        self.assertFalse(bool(mask.loc["2020-01-02", "SH688001"]))


if __name__ == "__main__":
    unittest.main()
