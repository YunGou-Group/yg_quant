#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行情宽表、sidecar 落库与面板合并：覆盖获取后的存储合同。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from DailyUpdates.data_fetcher.data_processor import DataProcessor
from DailyUpdates.data_fetcher.dataset_installer import DatasetInstaller
from DailyUpdates.storage import SQLiteStorage


class MarketStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = SQLiteStorage(str(Path(self.tmp.name) / "yg_quant.db"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_upsert_coalesce_keeps_existing_when_partial_null(self):
        self.storage.upsert_market_data(
            pd.DataFrame(
                {
                    "symbol": ["SH600000"],
                    "date": ["2020-01-02"],
                    "close": [10.5],
                    "adj_factor": [2.0],
                }
            )
        )
        self.storage.upsert_market_data(
            pd.DataFrame(
                {
                    "symbol": ["SH600000"],
                    "date": ["2020-01-02"],
                    "close": [np.nan],
                    "adj_factor": [2.1],
                }
            )
        )
        row = self.storage.read_market_data(
            fields=["close", "adj_factor"], adjust="none"
        ).iloc[0]
        self.assertAlmostEqual(row["close"], 10.5)
        self.assertAlmostEqual(row["adj_factor"], 2.1)

    def test_calendar_and_instruments_follow_upsert(self):
        self.storage.upsert_market_data(
            pd.DataFrame(
                {
                    "symbol": ["SH600000", "index_SH000300"],
                    "date": ["2020-01-02", "2020-01-02"],
                    "close": [10.0, 4000.0],
                }
            )
        )
        self.assertEqual(self.storage.list_trade_dates(), ["2020-01-02"])
        self.assertEqual(self.storage.get_instruments(), {"SH600000"})
        self.assertIn("index_SH000300", self.storage.get_instruments(include_indexes=True))

    def test_financial_keeps_update_flag_rows(self):
        frame = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "ann_date": ["20200331", "20200331"],
                "end_date": ["20191231", "20191231"],
                "update_flag": ["0", "1"],
                "roe": [10.0, 11.0],
            }
        )
        self.assertEqual(self.storage.replace_financial_indicator(frame), 2)
        out = self.storage.read_financial_indicator()
        self.assertEqual(len(out), 2)
        self.assertEqual(self.storage.get_financial_latest_end_date(), "2019-12-31")

    def test_industry_and_namechange_roundtrip(self):
        self.storage.replace_industry_classify(
            pd.DataFrame(
                {
                    "index_code": ["801010.SI"],
                    "industry_name": ["农林牧渔"],
                    "level": ["L1"],
                }
            ),
            src="SW2021",
        )
        self.storage.replace_industry_members(
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "l3_code": ["801016.SI"],
                    "l1_code": ["801010.SI"],
                    "in_date": ["20100101"],
                    "out_date": ["20200101"],
                    "is_new": ["N"],
                }
            ),
            src="SW2021",
        )
        members = self.storage.read_industry_members(src="SW2021", is_new=None)
        self.assertEqual(len(members), 1)
        self.assertEqual(members.iloc[0]["symbol"], "SZ000001")
        self.storage.replace_stock_namechange(
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["平安银行"],
                    "start_date": ["20100101"],
                    "end_date": [""],
                }
            )
        )
        self.assertEqual(self.storage.count_stock_namechange(), 1)
        self.storage.clear_stock_namechange()
        self.assertEqual(self.storage.count_stock_namechange(), 0)

    def test_index_constituents_upsert_does_not_wipe_older_month(self):
        first = pd.DataFrame(
            {
                "index_code": ["000300.SH"],
                "con_code": ["000001.SZ"],
                "trade_date": ["2020-01-31"],
                "weight": [1.0],
            }
        )
        self.storage.replace_index_constituents(first, index_codes=["000300.SH"])
        self.storage.upsert_index_constituents(
            pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "con_code": ["000002.SZ"],
                    "trade_date": ["2020-02-28"],
                    "weight": [2.0],
                }
            )
        )
        out = self.storage.read_index_constituents(index_code="000300.SH")
        self.assertEqual(len(out), 2)
        self.assertEqual(self.storage.get_index_constituent_latest_date(["000300.SH"]), "2020-02-28")


class ProcessorMergeTests(unittest.TestCase):
    def test_build_panel_merges_stock_and_index(self):
        processor = DataProcessor()
        frames = {
            "daily": pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20200102"],
                    "close": [10.0],
                    "vol": [100.0],
                }
            ),
            "index": pd.DataFrame(
                {
                    "ts_code": ["000300.SH"],
                    "trade_date": ["20200102"],
                    "close": [4000.0],
                    "vol": [1.0],
                }
            ),
        }
        config = {
            "daily": {
                "data_type": "daily",
                "fields": ["ts_code", "trade_date", "close", "vol"],
            },
            "index": {
                "data_type": "index",
                "fields": ["ts_code", "trade_date", "close", "vol"],
            },
        }
        out = processor.build_panel(
            frames, config, "20200102", set(), {"close", "vol"}
        )
        symbols = set(out["symbol"].astype(str))
        self.assertEqual(symbols, {"SZ000001", "index_SH000300"})

    def test_build_panel_keeps_stocks_when_index_comes_first(self):
        processor = DataProcessor()
        frames = {
            "index": pd.DataFrame(
                {
                    "ts_code": ["000300.SH"],
                    "trade_date": ["20200102"],
                    "close": [4000.0],
                    "vol": [1.0],
                }
            ),
            "daily": pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20200102"],
                    "close": [10.0],
                    "vol": [100.0],
                }
            ),
        }
        config = {
            "index": {
                "data_type": "index",
                "fields": ["ts_code", "trade_date", "close", "vol"],
            },
            "daily": {
                "data_type": "daily",
                "fields": ["ts_code", "trade_date", "close", "vol"],
            },
        }
        out = processor.build_panel(
            frames, config, "20200102", set(), {"close", "vol"}
        )
        symbols = list(out["symbol"].astype(str))
        self.assertEqual(symbols.count("SZ000001"), 1)
        self.assertEqual(symbols.count("index_SH000300"), 1)
        stock = out.loc[out["symbol"] == "SZ000001"].iloc[0]
        self.assertAlmostEqual(float(stock["close"]), 10.0)

    def test_range_panel_fills_missing_stock_on_every_date(self):
        processor = DataProcessor()
        frames = {
            "daily": pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "trade_date": ["20200102", "20200103"],
                    "close": [10.0, 10.5],
                }
            )
        }
        config = {
            "daily": {
                "data_type": "daily",
                "fields": ["ts_code", "trade_date", "close"],
            }
        }
        out = processor.build_panel(
            frames, config, "20200102", {"SZ000001", "SH600000"}, {"close"}
        )
        missing = out.loc[out["symbol"] == "SH600000"]
        self.assertEqual(len(missing), 2)
        self.assertTrue(missing["close"].isna().all())
        dates = set(pd.to_datetime(missing["date"]).dt.strftime("%Y-%m-%d"))
        self.assertEqual(dates, {"2020-01-02", "2020-01-03"})


class InstallerCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = SQLiteStorage(str(Path(self.tmp.name) / "yg_quant.db"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_adj_factor_completeness_ignores_placeholder_rows(self):
        self.storage.upsert_market_data(
            pd.DataFrame(
                {
                    "symbol": ["SH600000", "SH600001"],
                    "date": ["2020-01-02", "2020-01-02"],
                    "close": [10.0, np.nan],
                    "adj_factor": [2.0, np.nan],
                }
            )
        )
        installer = DatasetInstaller(self.storage, None, None)
        missing = installer._trade_dates_missing_fields(
            "2020-01-02", "2020-01-02", ["adj_factor"]
        )
        self.assertEqual(missing, [])


class FetchProcessStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = SQLiteStorage(str(Path(self.tmp.name) / "yg_quant.db"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_processor_then_upsert_roundtrip(self):
        processor = DataProcessor()
        frames = {
            "daily": pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "trade_date": ["20200102", "20200103"],
                    "open": [10.0, 10.2],
                    "close": [10.1, 10.3],
                    "vol": [100.0, 110.0],
                    "adj_factor": [1.0, 1.0],
                }
            ),
            "index": pd.DataFrame(
                {
                    "ts_code": ["000300.SH", "000300.SH"],
                    "trade_date": ["20200102", "20200103"],
                    "open": [4000.0, 4010.0],
                    "close": [4005.0, 4015.0],
                    "vol": [1.0, 1.0],
                }
            ),
        }
        config = {
            "daily": {
                "data_type": "daily",
                "fields": ["ts_code", "trade_date", "open", "close", "vol", "adj_factor"],
            },
            "index": {
                "data_type": "index",
                "fields": ["ts_code", "trade_date", "open", "close", "vol"],
            },
        }
        panel = processor.build_panel(
            frames, config, "20200102", {"SZ000001", "SH600000"}, {"open", "close", "vol", "adj_factor"}
        )
        written = self.storage.upsert_market_data(panel)
        self.assertGreater(written, 0)
        out = self.storage.read_market_data(
            fields=["open", "close", "adj_factor"],
            include_indexes=True,
            adjust="none",
        )
        symbols = set(out["ts_code"].astype(str))
        self.assertIn("SZ000001", symbols)
        self.assertIn("index_SH000300", symbols)
        self.assertIn("SH600000", symbols)
        self.assertEqual(self.storage.list_trade_dates(), ["2020-01-02", "2020-01-03"])
        suspended = out.loc[out["ts_code"] == "SH600000"]
        self.assertEqual(len(suspended), 2)
        self.assertTrue(suspended["close"].isna().all())
        hfq = self.storage.read_market_data(
            fields=["close"], symbols=["SZ000001"], adjust="hfq"
        )
        self.assertAlmostEqual(float(hfq.iloc[0]["close"]), 10.1)


if __name__ == "__main__":
    unittest.main()
