#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""financial_indicator 字段：CNE5-lite / Barra10 描述子所需的 PIT 财务列。

对齐必须用 ann_date（公告日），不能用 end_date（报告期）。
"""

from __future__ import annotations

from typing import Tuple

# 主键列（与 Tushare fina_indicator 对齐）
FINANCIAL_KEY_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "ann_date",
    "end_date",
    "update_flag",
)

# 数值列：Barra10 风格描述子 + 现有小市值过滤
# Growth: netprofit_yoy / dt_netprofit_yoy / or_yoy / equity_yoy
# Leverage: debt_to_assets / debt_to_eqt / assets_to_eqt / longdeb_to_debt
# Earnings Yield 补充: eps / ocfps；BTOP 交叉校验: bps
FINANCIAL_VALUE_FIELDS: Tuple[str, ...] = (
    "roe",
    "roa",
    "eps",
    "ocfps",
    "bps",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "or_yoy",
    "equity_yoy",
    "debt_to_assets",
    "debt_to_eqt",
    "assets_to_eqt",
    "longdeb_to_debt",
)

# Tushare fina_indicator / fina_indicator_vip 请求字段
FINANCIAL_TUSHARE_FIELDS: Tuple[str, ...] = (
    "ts_code",
    "ann_date",
    "end_date",
    "update_flag",
    *FINANCIAL_VALUE_FIELDS,
)

# daily_basic 中 Barra10 还用得到、且 market_data 可自动加列的字段
DAILY_BASIC_BARRA_FIELDS: Tuple[str, ...] = (
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ttm",
    "turnover_rate",
    "turnover_rate_f",
    "total_mv",
    "circ_mv",
)
