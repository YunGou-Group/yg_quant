#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Tushare 官方 stk_limit 对照规则补齐。默认跳过。

    python -m pytest tests/test_tushare_limit_fill_live.py -q -rs
或：

    $env:TUSHARE_LIVE=1; python -m pytest tests/test_tushare_limit_fill_live.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from yg_quant_repo import default_db_path, load_repo_env

load_repo_env()

pytestmark = pytest.mark.skipif(
    os.getenv("TUSHARE_LIVE", "").strip().lower() not in {"1", "true", "yes"},
    reason="set TUSHARE_LIVE=1 to run live Tushare limit-fill probes",
)

# 库内空洞高峰 + 板块规则切换日。
HOLE_DAYS = ("2021-09-06", "2020-08-24", "2019-06-03")
# 排除已知规则边界后，官方有限价的行应几乎全部对上。
MIN_MATCH_RATE = 0.998
IPO_SENTINEL = 9999


def _token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        pytest.skip("没有 TUSHARE_TOKEN")
    return token


def _ymd8(day: str) -> str:
    return str(day).replace("-", "")[:8]


def _official(pro, day: str) -> pd.DataFrame:
    from DailyUpdates.storage.sqlite_storage import SQLiteStorage

    frame = pro.stk_limit(
        trade_date=_ymd8(day), fields="ts_code,trade_date,up_limit,down_limit"
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "off_up", "off_dn"])
    out = frame.copy()
    out["symbol"] = out["ts_code"].astype(str).map(SQLiteStorage._normalize_symbol)
    out["off_up"] = pd.to_numeric(out["up_limit"], errors="coerce")
    out["off_dn"] = pd.to_numeric(out["down_limit"], errors="coerce")
    return out[["symbol", "off_up", "off_dn"]]


def _context(storage):
    namechange = storage.read_stock_namechange()
    basic = storage.read_stock_basic()
    list_dates = None
    if basic is not None and not basic.empty and "list_date" in basic.columns:
        listed = pd.to_datetime(
            basic.drop_duplicates("symbol").set_index("symbol")["list_date"],
            errors="coerce",
        )
        list_dates = listed[listed.notna()]
    return namechange, list_dates


def _compare(storage, official: pd.DataFrame, day: str) -> pd.DataFrame:
    from DailyUpdates.data_fetcher.preprocessing.market_bars import sanitize_market_bars

    bars = storage.read_market_data(
        fields=["pre_close", "close", "up_limit", "down_limit"],
        start_date=day,
        end_date=day,
        include_indexes=False,
        ordered=False,
        adjust="none",
    )
    assert not bars.empty, f"{day} 库内无行情"
    bars = bars.rename(columns={"ts_code": "symbol"})
    masked = bars.copy()
    masked["up_limit"] = pd.NA
    masked["down_limit"] = pd.NA
    namechange, list_dates = _context(storage)
    try:
        calendar = storage.list_trade_dates()
    except Exception:
        calendar = None
    filled = sanitize_market_bars(
        masked,
        namechange=namechange,
        list_dates=list_dates,
        calendar=calendar,
    )
    merged = filled.merge(official, on="symbol", how="inner")
    merged["fill_up"] = pd.to_numeric(merged["up_limit"], errors="coerce")
    merged["fill_dn"] = pd.to_numeric(merged["down_limit"], errors="coerce")
    merged["pre"] = pd.to_numeric(merged["pre_close"], errors="coerce")
    return bars, merged


def _known_exception(row: pd.Series) -> bool:
    """无昨收、Tushare 上市初期哨兵，或官方在不设限窗口仍给了板块幅度。"""
    if pd.isna(row["pre"]) or row["pre"] <= 0:
        return True
    if row["off_up"] >= IPO_SENTINEL or (pd.notna(row["off_dn"]) and row["off_dn"] <= 0.02):
        return True
    if pd.isna(row["fill_up"]) and pd.notna(row["off_up"]):
        return True
    return False


@pytest.fixture(scope="module")
def _pro():
    import tushare as ts

    ts.set_token(_token())
    return ts.pro_api()


@pytest.fixture(scope="module")
def _storage():
    from DailyUpdates.storage.sqlite_storage import SQLiteStorage

    path = default_db_path()
    if not Path(path).is_file():
        pytest.skip(f"没有本地库 {path}")
    return SQLiteStorage(str(path))


@pytest.mark.parametrize("day", HOLE_DAYS)
def test_db_holes_are_also_missing_from_tushare(_pro, _storage, day):
    official = _official(_pro, day)
    assert not official.empty, f"{day} Tushare stk_limit 为空"
    bars, _ = _compare(_storage, official, day)
    missing = bars[
        bars["close"].notna()
        & (bars["up_limit"].isna() | bars["down_limit"].isna())
    ]
    api_has = missing[missing["symbol"].isin(set(official["symbol"]))]
    assert api_has.empty, (
        f"{day} 库内缺限价但 Tushare 有官方值: {api_has['symbol'].tolist()[:12]}"
    )


@pytest.mark.parametrize("day", HOLE_DAYS)
def test_rule_fill_matches_official_except_known_edges(_pro, _storage, day):
    official = _official(_pro, day)
    _, merged = _compare(_storage, official, day)
    comparable = merged[merged["off_up"].notna() & merged["off_dn"].notna()].copy()
    comparable["ok"] = (
        (comparable["fill_up"] - comparable["off_up"]).abs() <= 0.011
    ) & ((comparable["fill_dn"] - comparable["off_dn"]).abs() <= 0.011)
    remain = comparable.loc[~comparable["ok"]]
    remain = remain.loc[~remain.apply(_known_exception, axis=1)]
    rate = 1.0 - (len(remain) / len(comparable))
    print(
        f"{day} compared={len(comparable)} match={int(comparable['ok'].sum())} "
        f"unexplained={len(remain)} effective={rate:.4%}"
    )
    if len(remain):
        print(remain[["symbol", "pre", "fill_up", "off_up"]].head(12).to_string(index=False))
    assert rate >= MIN_MATCH_RATE, (
        f"{day} 排除已知边界后一致率 {rate:.4%} < {MIN_MATCH_RATE:.1%}"
    )
