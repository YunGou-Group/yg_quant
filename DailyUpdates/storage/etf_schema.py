#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""场内基金（ETF / LOF）表结构。与 A 股 market_data / stock_basic 隔离。

Tushare fund_daily：vol 为手，amount 为千元（与股票 daily 相同）。
"""

from __future__ import annotations

from typing import Tuple

ETF_BASIC_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "name",
    "management",
    "fund_type",
    "market",
    "status",
    "list_date",
    "delist_date",
)

ETF_TUSHARE_BASIC_FIELDS: Tuple[str, ...] = (
    "ts_code",
    "name",
    "management",
    "fund_type",
    "market",
    "status",
    "list_date",
    "delist_date",
)

ETF_BAR_FIELDS: Tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
)

ETF_TUSHARE_DAILY_FIELDS: Tuple[str, ...] = (
    "ts_code",
    "trade_date",
    *ETF_BAR_FIELDS,
)

# 固定池里的 LOF / 货币 ETF，market='E' 偶尔漏掉时补拉名单。
ETF_EXTRA_TS_CODES: Tuple[str, ...] = (
    "501018.SH",  # 南方原油 LOF
    "161226.SZ",  # 国投白银 LOF
    "511880.SH",  # 银华日利（防御底仓）
)
