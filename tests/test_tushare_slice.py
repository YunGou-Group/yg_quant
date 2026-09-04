#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tushare 宽区间按日切片；分页触达上限且下一页为空视为截断。"""

from __future__ import annotations

import pandas as pd
import pytest

from DailyUpdates.data_fetcher.data_sources.tushare_data_source import TushareDataSource


def test_daily_wide_range_slices_by_day(monkeypatch):
    src = TushareDataSource()
    seen = []

    def fake_call(getter, paras, fields=None):
        seen.append(paras.get("trade_date"))
        return pd.DataFrame(
            {"ts_code": ["000001.SZ"], "trade_date": [paras["trade_date"]], "close": [10.0]}
        )

    monkeypatch.setattr(src, "_call_api", fake_call)
    monkeypatch.setattr("DailyUpdates.data_fetcher.data_sources.tushare_data_source.time.sleep", lambda *_a, **_k: None)
    out = src._fetch_daily_by_day(
        lambda **_k: None, ["ts_code", "trade_date", "close"], "20200101", "20200103", "daily"
    )
    assert seen == ["20200101", "20200102", "20200103"]
    assert len(out) == 3


def test_daily_single_day_does_not_slice(monkeypatch):
    src = TushareDataSource()
    called = {}

    def fake_info(getter, paras, fields=None):
        called["paras"] = paras
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20200102"]})

    monkeypatch.setattr(src, "getTushareInfo", fake_info)
    src._fetch_daily_by_day(
        lambda **_k: None, ["ts_code", "trade_date"], "20200102", "20200102", "daily"
    )
    assert called["paras"] == {"trade_date": "20200102"}


def test_get_tushare_info_raises_on_page_cap_truncation():
    src = TushareDataSource()
    pages = [pd.DataFrame({"a": range(6000)}), pd.DataFrame()]

    def getter(**_kwargs):
        return pages.pop(0)

    with pytest.raises(RuntimeError, match="截断"):
        src.getTushareInfo(getter, {}, ["a"])


def test_get_tushare_info_short_last_page_is_complete():
    src = TushareDataSource()
    pages = [pd.DataFrame({"a": range(100)}), pd.DataFrame()]

    def getter(**_kwargs):
        return pages.pop(0)

    out = src.getTushareInfo(getter, {}, ["a"])
    assert len(out) == 100


def test_full_rebuild_updates_market_before_sidecar(monkeypatch, tmp_path):
    from DailyUpdates.data_fetcher.unified_scheduler import UnifiedScheduler

    order = []

    class FakeUpdater:
        def __init__(self, *args, dataset_config=None, **kwargs):
            self.cfg = dict(dataset_config or {})

        def update_all(self, end_date, start_date=None):
            order.append(("update", tuple(self.cfg.keys())))
            return True

        def install_new_dataset(self, name, config, start_date=None, end_date=None):
            order.append(("install", name, start_date, end_date))
            return True

    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.unified_scheduler.StockDataUpdater",
        FakeUpdater,
    )
    scheduler = UnifiedScheduler(db_path=str(tmp_path / "yg_quant.db"))
    config = {
        "daily": {
            "data_source": "Tushare",
            "data_type": "daily",
            "api_name": "daily",
            "fields": ["ts_code", "trade_date", "close"],
        },
        "stock_basic": {
            "data_source": "Tushare",
            "data_type": "stock_info",
            "api_name": "stock_basic",
            "fields": ["ts_code", "name"],
        },
    }
    result = scheduler.execute_schedule(config, end_date="20200110")
    assert result["success"] is True
    assert order[0][0] == "update"
    assert "daily" in order[0][1]
    assert "stock_basic" not in order[0][1]
    assert order[1][:2] == ("install", "stock_basic")
    assert order[1][2] == "20050101"
    assert order[1][3] == "20200110"


def test_index_weight_install_reuses_existing_rows():
    from DailyUpdates.data_fetcher.dataset_installer import DatasetInstaller

    class Store:
        def get_index_constituent_latest_date(self, codes=None):
            return "2026-08-31"

    mode, start = DatasetInstaller(Store(), None)._resolve_sidecar_mode(
        {
            "data_type": "index_constituent",
            "api_name": "index_weight",
            "index_list": ["000300.SH"],
        },
        "full",
    )
    assert mode == "incremental"
    assert start == "20260831"


def test_index_weight_install_full_when_table_empty():
    from DailyUpdates.data_fetcher.dataset_installer import DatasetInstaller

    class Store:
        def get_index_constituent_latest_date(self, codes=None):
            return None

    mode, start = DatasetInstaller(Store(), None)._resolve_sidecar_mode(
        {
            "data_type": "index_constituent",
            "api_name": "index_weight",
            "index_list": ["000300.SH"],
        },
        "full",
    )
    assert mode == "full"
    assert start is None


def _index_src(monkeypatch):
    src = TushareDataSource()
    src.pro = type("Pro", (), {"index_weight": object()})()
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.tushare_data_source.time.sleep",
        lambda *_a, **_k: None,
    )
    return src


def test_index_weight_probe_empty_fails_fast(monkeypatch):
    src = _index_src(monkeypatch)
    calls = []

    def fake_call(getter, paras, fields=None):
        calls.append(paras["index_code"])
        return pd.DataFrame()

    monkeypatch.setattr(src, "_call_api", fake_call)
    with pytest.raises(RuntimeError, match="index_weight"):
        src._fetch_index_weight(
            {"index_list": ["000300.SH", "000905.SH"], "pause_seconds": 0},
            ["index_code", "con_code", "trade_date", "weight"],
            "20200101",
            "20200229",
        )
    assert calls == ["000016.SH", "399300.SZ", "000905.SH"]


def test_index_weight_aliases_hs300_and_keeps_config_code(monkeypatch):
    src = _index_src(monkeypatch)

    def fake_call(getter, paras, fields=None):
        code = paras["index_code"]
        if code in {"000016.SH", "399300.SZ", "000905.SH"}:
            return pd.DataFrame(
                {
                    "index_code": [code],
                    "con_code": ["000001.SZ"],
                    "trade_date": [paras["start_date"]],
                    "weight": [1.0],
                }
            )
        return pd.DataFrame()

    monkeypatch.setattr(src, "_call_api", fake_call)
    out = src._fetch_index_weight(
        {"index_list": ["000300.SH"], "pause_seconds": 0},
        ["index_code", "con_code", "trade_date", "weight"],
        "20200101",
        "20200131",
    )
    assert not out.empty
    assert (out["index_code"] == "000300.SH").all()


def test_index_weight_probe_uses_previous_month(monkeypatch):
    src = _index_src(monkeypatch)
    windows = []

    def fake_call(getter, paras, fields=None):
        windows.append((paras["start_date"], paras["end_date"]))
        return pd.DataFrame()

    monkeypatch.setattr(src, "_call_api", fake_call)
    with pytest.raises(RuntimeError, match="index_weight"):
        src._assert_index_weight_available(
            ["index_code", "con_code", "trade_date", "weight"], "20260903"
        )
    assert windows == [("20260801", "20260831")] * 3


def test_index_weight_early_month_has_no_new_snapshot(monkeypatch):
    src = _index_src(monkeypatch)

    def fake_call(getter, paras, fields=None):
        start = paras["start_date"]
        if start.startswith("202608"):
            return pd.DataFrame(
                {
                    "index_code": [paras["index_code"]],
                    "con_code": ["000001.SZ"],
                    "trade_date": ["20260831"],
                    "weight": [1.0],
                }
            )
        return pd.DataFrame()

    monkeypatch.setattr(src, "_call_api", fake_call)
    out = src._fetch_index_weight(
        {"index_list": ["000016.SH"], "pause_seconds": 0},
        ["index_code", "con_code", "trade_date", "weight"],
        "20260901",
        "20260903",
    )
    assert out.empty
