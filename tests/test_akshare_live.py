#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实请求探测 Akshare 各接口。默认跳过，避免拖慢日常单测。

    python -m pytest tests/test_akshare_live.py -q -rs
或：

    $env:AKSHARE_LIVE=1; python -m pytest tests/test_akshare_live.py -q
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from DailyUpdates.data_fetcher.data_sources.akshare_data_source import AkshareDataSource

pytestmark = pytest.mark.skipif(
    os.getenv("AKSHARE_LIVE", "").strip().lower() not in {"1", "true", "yes"},
    reason="set AKSHARE_LIVE=1 to run live Akshare probes",
)

START = "20260824"
END = "20260828"
SYMBOLS = ["000001", "600000"]
PAUSE = {"pause_seconds": 0.2, "symbols": SYMBOLS}


def _src():
    return AkshareDataSource()


def test_live_trading_calendar():
    assert _src().is_trading_day("20260823") is False  # Sunday
    assert _src().is_trading_day("20260824") is True


def test_live_daily_two_stocks():
    out = _src().fetch_data(
        {
            **PAUSE,
            "data_type": "daily",
            "api_name": "daily",
            "fields": [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "vol",
                "amount",
            ],
        },
        START,
        END,
    )
    assert not out.empty, "daily 为空"
    assert {"000001.SZ", "600000.SH"} <= set(out["ts_code"].astype(str))
    assert out["close"].notna().any()
    print(out.head(4).to_string(index=False))


def test_live_daily_basic():
    out = _src().fetch_data(
        {
            **PAUSE,
            "data_type": "daily",
            "api_name": "daily_basic",
            "fields": ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate"],
        },
        START,
        END,
    )
    assert not out.empty, "daily_basic 为空"
    print(out.head(4).to_string(index=False))


def test_live_adj_factor():
    out = _src().fetch_data(
        {
            **PAUSE,
            "data_type": "daily",
            "api_name": "adj_factor",
            "fields": ["ts_code", "trade_date", "adj_factor"],
        },
        START,
        END,
    )
    assert not out.empty, "adj_factor 为空"
    factor = pd.to_numeric(out["adj_factor"], errors="coerce")
    assert factor.notna().any() and (factor > 0).any()
    print(out.head(4).to_string(index=False))


def test_live_stk_limit():
    out = _src().fetch_data(
        {
            **PAUSE,
            "data_type": "daily",
            "api_name": "stk_limit",
            "fields": ["ts_code", "trade_date", "up_limit", "down_limit"],
        },
        START,
        END,
    )
    assert not out.empty, "stk_limit 为空"
    assert (out["up_limit"] > out["down_limit"]).all()
    print(out.head(4).to_string(index=False))


def test_live_index_daily():
    out = _src().fetch_data(
        {
            "pause_seconds": 0.2,
            "data_type": "index",
            "api_name": "index_daily",
            "index_list": ["000300.SH"],
            "fields": ["ts_code", "trade_date", "open", "close", "vol"],
        },
        START,
        END,
    )
    assert not out.empty, "index_daily 为空"
    assert (out["ts_code"] == "000300.SH").all()
    print(out.to_string(index=False))


def test_live_index_classify():
    out = _src().fetch_data(
        {
            "data_type": "industry",
            "api_name": "index_classify",
            "src": "SW2021",
            "levels": ["L1"],
            "fields": ["index_code", "industry_name", "level", "src"],
        },
        "",
        "",
    )
    assert not out.empty, "index_classify 为空"
    assert set(out["level"]) == {"L1"}
    print(f"L1 industries: {len(out)}")


def test_live_stock_basic():
    out = _src().fetch_data(
        {
            "data_type": "stock_info",
            "api_name": "stock_basic",
            "fields": ["ts_code", "symbol", "name", "market", "list_status"],
        },
        "",
        "",
    )
    assert not out.empty, "stock_basic 为空"
    codes = set(out["ts_code"].astype(str))
    assert "000001.SZ" in codes
    assert "600000.SH" in codes
    print(f"stock_basic rows: {len(out)}")


def test_live_fina_indicator():
    out = _src().fetch_data(
        {
            "pause_seconds": 0.3,
            "symbols": ["000001"],
            "data_type": "financial",
            "api_name": "fina_indicator",
            "fields": ["ts_code", "end_date", "ann_date", "roe", "eps"],
        },
        "20240101",
        "20261231",
    )
    assert not out.empty, "fina_indicator 为空"
    print(out.head(4).to_string(index=False))


def test_live_index_weight():
    out = _src().fetch_data(
        {
            "data_type": "index_constituent",
            "api_name": "index_weight",
            "index_list": ["000300.SH"],
            "fields": ["index_code", "con_code", "trade_date", "weight"],
        },
        START,
        END,
    )
    assert not out.empty, "index_weight 为空"
    print(f"hs300 constituents: {len(out)}")
    print(out.head(3).to_string(index=False))
