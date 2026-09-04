#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BinStorage 轴一致性：尾部追加、中间插入重映射、并发元数据写入。"""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from DailyUpdates.storage import BinStorage


def _frame(symbol: str, pairs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [symbol] * len(pairs),
            "trade_date": [d for d, _ in pairs],
            "factor_value": [v for _, v in pairs],
        }
    )


class BinStorageAxisTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = BinStorage(str(Path(self._tmp.name) / "factors"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _series(self, factor: str, symbol: str) -> pd.Series:
        panel = self.store.read_factor_panel(factor)
        return panel[symbol].dropna()

    def test_tail_append_keeps_alignment(self):
        self.store.upsert_factor_data(
            "f", _frame("SH600000", [("2020-01-02", 1.0), ("2020-01-03", 2.0)])
        )
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-06", 3.0)]))
        got = self._series("f", "SH600000")
        self.assertEqual(
            list(got.index), ["2020-01-02", "2020-01-03", "2020-01-06"]
        )
        np.testing.assert_allclose(got.to_numpy(), [1.0, 2.0, 3.0])

    def test_middle_insertion_remaps_existing_bins(self):
        """补一个漏掉的历史交易日，不能让之后的值整体错位。"""
        self.store.upsert_factor_data(
            "f",
            _frame(
                "SH600000",
                [("2020-01-02", 1.0), ("2020-01-06", 3.0), ("2020-01-07", 4.0)],
            ),
        )
        self.store.upsert_factor_data(
            "g", _frame("SZ000001", [("2020-01-02", 10.0), ("2020-01-07", 40.0)])
        )

        # 事后才发现 01-03 也是交易日
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-03", 2.0)]))

        self.assertEqual(
            self.store.read_calendar(),
            ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        )
        got = self._series("f", "SH600000")
        self.assertEqual(list(got.index), ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        np.testing.assert_allclose(got.to_numpy(), [1.0, 2.0, 3.0, 4.0])

        # 没有参与这次写入的因子也必须被一起重排
        other = self._series("g", "SZ000001")
        self.assertEqual(list(other.index), ["2020-01-02", "2020-01-07"])
        np.testing.assert_allclose(other.to_numpy(), [10.0, 40.0])

    def test_leading_insertion_remaps(self):
        self.store.upsert_factor_data(
            "f", _frame("SH600000", [("2020-02-03", 5.0), ("2020-02-04", 6.0)])
        )
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-02", 1.0)]))
        got = self._series("f", "SH600000")
        self.assertEqual(list(got.index), ["2020-01-02", "2020-02-03", "2020-02-04"])
        np.testing.assert_allclose(got.to_numpy(), [1.0, 5.0, 6.0])

    def test_concurrent_meta_writes_keep_every_factor(self):
        names = [f"f{i:02d}" for i in range(24)]

        def write(name: str) -> None:
            # 每个线程各建一个 BinStorage，模拟并发因子更新各持一份内存 meta
            store = BinStorage(str(self.store.root))
            store.upsert_factor_data(
                name, _frame("SH600000", [("2020-01-02", 1.0)])
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, names))

        fresh = BinStorage(str(self.store.root))
        for name in names:
            self.assertTrue(
                fresh.is_factor_registered(name), f"{name} 的元数据被并发写入覆盖了"
            )
            self.assertEqual(fresh.get_factor_latest_date(name), "2020-01-02")

    def test_delete_removes_meta_entry(self):
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-02", 1.0)]))
        self.store.upsert_factor_data("g", _frame("SH600000", [("2020-01-02", 2.0)]))
        self.store.delete_factor_data("f")
        fresh = BinStorage(str(self.store.root))
        self.assertFalse(fresh.is_factor_registered("f"))
        self.assertTrue(fresh.is_factor_registered("g"))

    def test_replace_overwrites_full_series(self):
        self.store.upsert_factor_data(
            "f", _frame("SH600000", [("2020-01-02", 1.0), ("2020-01-03", 2.0)])
        )
        self.store.upsert_factor_data(
            "f",
            _frame("SH600000", [("2020-01-02", 9.0), ("2020-01-03", 8.0)]),
            replace=True,
        )
        got = self._series("f", "SH600000")
        np.testing.assert_allclose(got.to_numpy(), [9.0, 8.0])

    def test_axis_files_are_never_partially_written(self):
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-02", 1.0)]))
        leftovers = list(self.store.root.rglob("*.tmp"))
        self.assertEqual(leftovers, [])
        self.assertFalse((self.store.root / "_axis.lock").exists())

    def test_mismatch_bin_length_is_rejected(self):
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-02", 1.0)]))
        path = next(self.store.features_dir.glob("*/f.day.bin"))
        extra = np.array([2.0], dtype="<f4")
        with path.open("ab") as handle:
            handle.write(extra.tobytes())
        with self.assertRaises(ValueError):
            self.store.read_factor_panel("f")

    def test_short_bin_reads_as_trailing_nan(self):
        """增量只写部分股票时，缺席文件读成尾部 NaN，不拒绝整面板。"""
        self.store.upsert_factor_data(
            "f",
            pd.concat(
                [
                    _frame("SH600000", [("2020-01-02", 1.0), ("2020-01-03", 2.0)]),
                    _frame("SZ000001", [("2020-01-02", 10.0), ("2020-01-03", 20.0)]),
                ],
                ignore_index=True,
            ),
        )
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-06", 3.0)]))
        panel = self.store.read_factor_panel("f")
        self.assertEqual(
            list(panel.index), ["2020-01-02", "2020-01-03", "2020-01-06"]
        )
        np.testing.assert_allclose(panel["SH600000"].to_numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(panel["SZ000001"].to_numpy(), [10.0, 20.0, np.nan])
        tail = self.store.read_factor_panel("f", start_date="2020-01-06")
        self.assertTrue(np.isnan(tail.loc["2020-01-06", "SZ000001"]))

    def test_middle_insert_remaps_short_bins(self):
        self.store.upsert_factor_data(
            "f",
            pd.concat(
                [
                    _frame("SH600000", [("2020-01-02", 1.0), ("2020-01-06", 3.0)]),
                    _frame("SZ000001", [("2020-01-02", 10.0), ("2020-01-06", 30.0)]),
                ],
                ignore_index=True,
            ),
        )
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-07", 4.0)]))
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-03", 2.0)]))
        panel = self.store.read_factor_panel("f")
        self.assertEqual(
            list(panel.index),
            ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        )
        np.testing.assert_allclose(panel["SH600000"].to_numpy(), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            panel["SZ000001"].to_numpy(), [10.0, np.nan, 30.0, np.nan]
        )

    def test_align_axis_false_writes_on_existing_calendar(self):
        self.store.ensure_calendar(["2020-01-02", "2020-01-03"])
        self.store.upsert_factor_data(
            "f",
            _frame("SH600000", [("2020-01-02", 1.0), ("2020-01-03", 2.0)]),
            align_axis=False,
        )
        got = self._series("f", "SH600000")
        np.testing.assert_allclose(got.to_numpy(), [1.0, 2.0])

    def test_incremental_patch_many_symbols(self):
        first = pd.concat(
            [
                _frame(f"SH{600000 + i}", [("2020-01-02", 1.0), ("2020-01-03", 2.0)])
                for i in range(48)
            ],
            ignore_index=True,
        )
        self.store.upsert_factor_data("f", first)
        extra = pd.concat(
            [_frame(f"SH{600000 + i}", [("2020-01-06", 3.0)]) for i in range(48)],
            ignore_index=True,
        )
        self.store.upsert_factor_data("f", extra)
        got = self._series("f", "SH600000")
        self.assertEqual(list(got.index), ["2020-01-02", "2020-01-03", "2020-01-06"])
        np.testing.assert_allclose(got.to_numpy(), [1.0, 2.0, 3.0])

    def test_stale_part_files_are_dropped_on_init(self):
        self.store.upsert_factor_data("f", _frame("SH600000", [("2020-01-02", 1.0)]))
        path = next(self.store.features_dir.glob("*/f.day.bin"))
        part = Path(str(path) + ".part")
        part.write_bytes(b"junk")
        self.assertTrue(part.exists())
        BinStorage(str(self.store.root))
        self.assertFalse(part.exists())

    def test_all_nan_does_not_advance_watermark(self):
        self.store.upsert_factor_data("f", _frame("SH600000", [("2026-09-01", 1.0)]))
        self.assertEqual(self.store.get_factor_latest_date("f"), "2026-09-01")
        rows = self.store.upsert_factor_data(
            "f", _frame("SH600000", [("2026-09-03", float("nan"))])
        )
        self.assertEqual(rows, 0)
        self.assertEqual(self.store.get_factor_latest_date("f"), "2026-09-01")


if __name__ == "__main__":
    unittest.main()
