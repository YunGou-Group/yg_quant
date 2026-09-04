#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stock_basic 保存 Tushare delist_date。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from DailyUpdates.storage import SQLiteStorage


class StockBasicDelistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "yg_quant.db"
        self.storage = SQLiteStorage(str(self.db))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_roundtrip_delist_date(self):
        self.storage.replace_stock_basic(
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000018.SZ"],
                    "name": ["平安银行", "神州长城"],
                    "list_status": ["L", "D"],
                    "list_date": ["19910403", "19920616"],
                    "delist_date": [None, "2018-12-18"],
                }
            )
        )
        out = self.storage.read_stock_basic().set_index("symbol")
        self.assertEqual(out.loc["SZ000001", "delist_date"], "")
        self.assertEqual(out.loc["SZ000018", "delist_date"], "20181218")
        self.assertEqual(out.loc["SZ000018", "list_status"], "D")

    def test_upsert_updates_delist_date(self):
        self.storage.replace_stock_basic(
            pd.DataFrame({"ts_code": ["000018.SZ"], "name": ["神州长城"], "list_status": ["L"]})
        )
        self.storage.upsert_stock_basic(
            pd.DataFrame(
                {
                    "ts_code": ["000018.SZ"],
                    "name": ["神州长城"],
                    "list_status": ["D"],
                    "delist_date": ["20181218"],
                }
            )
        )
        out = self.storage.read_stock_basic()
        self.assertEqual(out.iloc[0]["delist_date"], "20181218")
        self.assertEqual(out.iloc[0]["list_status"], "D")

    def test_existing_db_gains_column(self):
        with self.storage._connect() as connection:
            cols = {row["name"] for row in connection.execute("PRAGMA table_info(stock_basic)")}
        self.assertIn("delist_date", cols)


if __name__ == "__main__":
    unittest.main()
