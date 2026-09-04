#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Akshare 数据源把东财/申万接口映射成与 Tushare 相同的列。"""

from __future__ import annotations

import pandas as pd

from DailyUpdates.data_fetcher.data_sources.akshare_data_source import (
    AkshareDataSource,
    _limit_ratio,
    _sina_symbol,
    _to_ts_code,
)


def _hist_frame():
    return pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "股票代码": ["000001", "000001"],
            "开盘": [10.0, 10.2],
            "收盘": [10.1, 10.4],
            "最高": [10.2, 10.5],
            "最低": [9.9, 10.1],
            "成交量": [1000, 1100],
            "成交额": [1_010_000, 1_144_000],
            "振幅": [3.0, 4.0],
            "涨跌幅": [1.0, 3.0],
            "涨跌额": [0.1, 0.3],
            "换手率": [0.5, 0.6],
        }
    )


def test_to_ts_code_markets():
    assert _to_ts_code("000001") == "000001.SZ"
    assert _to_ts_code("600000.SH") == "600000.SH"
    assert _to_ts_code("688001") == "688001.SH"
    assert _to_ts_code("830001") == "830001.BJ"


def test_sina_symbol_keeps_explicit_market():
    assert _sina_symbol("000001") == "sz000001"
    assert _sina_symbol("600000.SH") == "sh600000"
    assert _sina_symbol("000300.SH") == "sh000300"
    assert _sina_symbol("399001.SZ") == "sz399001"


def test_limit_ratio_boards():
    assert _limit_ratio("000001.SZ", "平安银行") == 0.10
    assert _limit_ratio("300001.SZ", "特锐德") == 0.20
    assert _limit_ratio("688001.SH", "华兴源创") == 0.20
    assert _limit_ratio("000001.SZ", "*ST 假") == 0.05
    assert _limit_ratio("830001.BJ", "北交所") == 0.30


def test_daily_maps_unadjusted_bars(monkeypatch):
    src = AkshareDataSource()
    monkeypatch.setattr(src, "_codes", lambda _cfg: ["000001"])
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.stock_zh_a_hist",
        lambda **_k: _hist_frame(),
    )
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.time.sleep",
        lambda *_a, **_k: None,
    )
    out = src.fetch_data(
        {
            "data_source": "Akshare",
            "data_type": "daily",
            "api_name": "daily",
            "fields": ["ts_code", "trade_date", "open", "close", "pre_close", "vol", "amount"],
            "pause_seconds": 0,
        },
        "20240102",
        "20240103",
    )
    assert list(out["ts_code"].unique()) == ["000001.SZ"]
    assert list(out["trade_date"]) == ["20240102", "20240103"]
    assert out.loc[0, "pre_close"] == 10.0
    assert abs(out.loc[0, "amount"] - 1010.0) < 1e-6


def test_daily_falls_back_to_sina_when_eastmoney_fails(monkeypatch):
    src = AkshareDataSource()
    monkeypatch.setattr(src, "_codes", lambda _cfg: ["000001"])
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.time.sleep",
        lambda *_a, **_k: None,
    )

    def boom(**_k):
        raise ConnectionError("Remote end closed connection")

    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.stock_zh_a_hist",
        boom,
    )
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.stock_zh_a_daily",
        lambda **_k: pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02").date()],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "volume": [100000.0],
                "amount": [1_010_000.0],
                "outstanding_share": [1e9],
                "turnover": [0.005],
            }
        ),
    )
    out = src.fetch_data(
        {
            "data_type": "daily",
            "api_name": "daily",
            "fields": ["ts_code", "trade_date", "close", "vol", "amount"],
            "pause_seconds": 0,
        },
        "20240102",
        "20240102",
    )
    assert list(out["ts_code"]) == ["000001.SZ"]
    assert abs(out.loc[0, "vol"] - 1000.0) < 1e-6
    assert abs(out.loc[0, "amount"] - 1010.0) < 1e-6


def test_adj_factor_is_hfq_over_raw(monkeypatch):
    src = AkshareDataSource()
    monkeypatch.setattr(src, "_codes", lambda _cfg: ["000001"])
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.time.sleep",
        lambda *_a, **_k: None,
    )

    def fake_hist(symbol, period="daily", start_date="", end_date="", adjust=""):
        frame = _hist_frame()
        if adjust == "hfq":
            frame = frame.copy()
            frame["收盘"] = frame["收盘"] * 2
        return frame

    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.stock_zh_a_hist",
        fake_hist,
    )
    out = src.fetch_data(
        {
            "data_type": "daily",
            "api_name": "adj_factor",
            "fields": ["ts_code", "trade_date", "adj_factor"],
            "pause_seconds": 0,
        },
        "20240102",
        "20240103",
    )
    assert list(out["adj_factor"]) == [2.0, 2.0]


def test_stk_limit_from_pre_close(monkeypatch):
    src = AkshareDataSource()
    monkeypatch.setattr(
        src,
        "_fetch_daily",
        lambda *_a, **_k: pd.DataFrame(
            {"ts_code": ["000001.SZ", "300001.SZ"], "trade_date": ["20240102", "20240102"], "pre_close": [10.0, 10.0]}
        ),
    )
    out = src.fetch_data(
        {"data_type": "daily", "api_name": "stk_limit", "pause_seconds": 0},
        "20240102",
        "20240102",
    )
    by_code = out.set_index("ts_code")
    assert by_code.loc["000001.SZ", "up_limit"] == 11.0
    assert by_code.loc["000001.SZ", "down_limit"] == 9.0
    assert by_code.loc["300001.SZ", "up_limit"] == 12.0


def test_index_daily_keeps_ts_code(monkeypatch):
    src = AkshareDataSource()
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.index_zh_a_hist",
        lambda **_k: _hist_frame().drop(columns=["股票代码"]),
    )
    out = src.fetch_data(
        {
            "data_type": "index",
            "api_name": "index_daily",
            "index_list": ["000300.SH"],
            "pause_seconds": 0,
        },
        "20240102",
        "20240103",
    )
    assert (out["ts_code"] == "000300.SH").all()


def test_classify_maps_sw_levels(monkeypatch):
    src = AkshareDataSource()
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.sw_index_first_info",
        lambda: pd.DataFrame({"行业代码": ["801010"], "行业名称": ["农林牧渔"]}),
    )
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.sw_index_second_info",
        lambda: pd.DataFrame(
            {"行业代码": ["801011"], "行业名称": ["种植业"], "上级行业": ["农林牧渔"]}
        ),
    )
    monkeypatch.setattr(
        "DailyUpdates.data_fetcher.data_sources.akshare_data_source.ak.sw_index_third_info",
        lambda: pd.DataFrame(
            {"行业代码": ["801012"], "行业名称": ["种子"], "上级行业": ["种植业"]}
        ),
    )
    out = src.fetch_data(
        {
            "data_type": "industry",
            "api_name": "index_classify",
            "src": "SW2021",
            "levels": ["L1", "L2", "L3"],
            "fields": ["index_code", "industry_name", "parent_code", "level", "src"],
        },
        "",
        "",
    )
    l2 = out.loc[out["level"] == "L2"].iloc[0]
    assert l2["index_code"] == "801011.SI"
    assert l2["parent_code"] == "801010.SI"


def test_unknown_api_raises():
    src = AkshareDataSource()
    try:
        src.fetch_data({"data_type": "daily", "api_name": "not_a_real_api"}, "20240102", "20240102")
    except ValueError as exc:
        assert "不支持" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_updater_uses_range_window_from_source_class():
    from DailyUpdates.data_fetcher.stock_data_updater import StockDataUpdater
    from DailyUpdates.data_fetcher.data_sources.tushare_data_source import TushareDataSource

    class _Proc:
        data_source_classes = {
            "Akshare": AkshareDataSource,
            "Tushare": TushareDataSource,
        }

    updater = StockDataUpdater.__new__(StockDataUpdater)
    updater.data_processor = _Proc()
    updater.dataset_config = {
        "daily": {"data_source": "Akshare", "data_type": "daily", "api_name": "daily"}
    }
    assert updater._market_uses_range_window() is True
    updater.dataset_config = {
        "daily": {"data_source": "Tushare", "data_type": "daily", "api_name": "daily"}
    }
    assert updater._market_uses_range_window() is False
