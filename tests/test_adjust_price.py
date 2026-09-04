#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取层后复权：价格 × adj_factor，成交量 ÷ adj_factor，指数不复权。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from DailyUpdates.storage import SQLiteStorage


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000", "index_SH000300"],
            "date": ["2020-01-02", "2020-01-03", "2020-01-02"],
            "open": [10.0, 11.0, 4000.0],
            "close": [10.5, 11.5, 4010.0],
            "vol": [1000.0, 2000.0, 0.0],
            "amount": [10_500.0, 23_000.0, 0.0],
            "pct_chg": [1.0, 2.0, 0.5],
            "adj_factor": [2.0, 4.0, np.nan],
        }
    )


class AdjustPriceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = SQLiteStorage(str(Path(self._tmp.name) / "market.db"))
        self.storage.upsert_market_data(_market_frame())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _row(self, frame: pd.DataFrame, symbol: str, date: str) -> pd.Series:
        hit = frame.loc[
            (frame["ts_code"] == symbol) & (frame["trade_date"] == date)
        ]
        self.assertEqual(len(hit), 1)
        return hit.iloc[0]

    def test_raw_read_is_unchanged(self):
        frame = self.storage.read_market_data(
            fields=["open", "vol"], include_indexes=True
        )
        row = self._row(frame, "SH600000", "2020-01-02")
        self.assertAlmostEqual(row["open"], 10.0)
        self.assertAlmostEqual(row["vol"], 1000.0)

    def test_hfq_scales_price_up_and_volume_down(self):
        frame = self.storage.read_market_data(
            fields=["open", "close", "vol"], include_indexes=True, adjust="hfq"
        )
        row = self._row(frame, "SH600000", "2020-01-03")
        self.assertAlmostEqual(row["open"], 11.0 * 4.0)
        self.assertAlmostEqual(row["close"], 11.5 * 4.0)
        self.assertAlmostEqual(row["vol"], 2000.0 / 4.0)

    def test_hfq_keeps_vwap_consistent(self):
        """amount 不复权、vol 反向复权 => vwap 与价格同口径。"""
        frame = self.storage.read_market_data(
            fields=["close", "vol", "amount"], adjust="hfq"
        )
        row = self._row(frame, "SH600000", "2020-01-02")
        vwap = row["amount"] / row["vol"]
        raw_vwap = 10_500.0 / 1000.0
        self.assertAlmostEqual(vwap, raw_vwap * 2.0)

    def test_index_rows_are_not_adjusted(self):
        frame = self.storage.read_market_data(
            fields=["open"], include_indexes=True, adjust="hfq"
        )
        row = self._row(frame, "index_SH000300", "2020-01-02")
        self.assertAlmostEqual(row["open"], 4000.0)

    def test_non_price_fields_untouched(self):
        frame = self.storage.read_market_data(fields=["pct_chg"], adjust="hfq")
        row = self._row(frame, "SH600000", "2020-01-03")
        self.assertAlmostEqual(row["pct_chg"], 2.0)

    def test_returns_are_invariant_to_adjustment(self):
        """同一只股票的日间收益率不因复权而改变（除权日除外）。"""
        raw = self.storage.read_market_data(fields=["close"], adjust="none")
        raw_ratio = (
            self._row(raw, "SH600000", "2020-01-03")["close"]
            / self._row(raw, "SH600000", "2020-01-02")["close"]
        )
        hfq = self.storage.read_market_data(fields=["close"], adjust="hfq")
        hfq_ratio = (
            self._row(hfq, "SH600000", "2020-01-03")["close"]
            / self._row(hfq, "SH600000", "2020-01-02")["close"]
        )
        # 因子从 2 涨到 4，说明中间有除权：复权后的真实涨幅是原始涨幅的 2 倍
        self.assertAlmostEqual(hfq_ratio, raw_ratio * 2.0)

    def test_hfq_missing_factor_day_is_nan_not_raw(self):
        """缺 adj_factor 的交易日不得回退成未复权价，避免 20→11 伪收益。"""
        self.storage.upsert_market_data(
            pd.DataFrame(
                {
                    "symbol": ["SH600002", "SH600002"],
                    "date": ["2020-01-02", "2020-01-03"],
                    "open": [10.0, 11.0],
                    "close": [10.0, 11.0],
                    "adj_factor": [2.0, np.nan],
                }
            )
        )
        frame = self.storage.read_market_data(fields=["open"], adjust="hfq")
        d1 = self._row(frame, "SH600002", "2020-01-02")
        d2 = self._row(frame, "SH600002", "2020-01-03")
        self.assertAlmostEqual(d1["open"], 20.0)
        self.assertTrue(pd.isna(d2["open"]))
        self.storage.upsert_market_data(
            pd.DataFrame(
                {
                    "symbol": ["SH600001"],
                    "date": ["2020-01-02"],
                    "open": [10.0],
                    "close": [11.0],
                    "adj_factor": [np.nan],
                }
            )
        )
        frame = self.storage.read_market_data(
            fields=["open", "close"], adjust="hfq"
        )
        row = self._row(frame, "SH600001", "2020-01-02")
        self.assertTrue(pd.isna(row["open"]))
        self.assertTrue(pd.isna(row["close"]))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.storage.read_market_data(fields=["open"], adjust="qfq")

    def test_requested_columns_only(self):
        frame = self.storage.read_market_data(fields=["open"], adjust="hfq")
        self.assertEqual(list(frame.columns), ["ts_code", "trade_date", "open"])


if __name__ == "__main__":
    unittest.main()
