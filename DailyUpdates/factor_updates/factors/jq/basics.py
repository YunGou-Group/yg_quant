#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽基础因子：市值、估值倒数、EBIT 等。"""

from __future__ import annotations

from ._fin import extract_fin
from ._wrappers import install

_DEPS = ("close", "total_mv", "circ_mv", "ps_ttm", "total_share")

SPECS = (
    ("basics_market_cap", "market_cap", "总市值", 40, _DEPS),
    ("basics_circulating_market_cap", "circ_cap", "流通市值", 40, _DEPS),
    ("basics_sales_to_price_ratio", "sales_to_price", "1 / PS_TTM", 40, _DEPS),
    (
        "basics_cash_flow_to_price_ratio",
        "cash_flow_to_price",
        "聚宽为 1/PCF_TTM；此处用每股经营现金流×总股本/总市值近似",
        40,
        _DEPS,
    ),
    (
        "basics_net_working_capital",
        "pit:working_capital",
        "流动资产-流动负债（聚宽 net_working_capital）",
        40,
        _DEPS,
    ),
    (
        "basics_working_capital",
        "pit:networking_capital",
        "Tushare 净营运资本（非聚宽同名科目）",
        40,
        _DEPS,
    ),
    ("basics_EBIT", "pit:ebit", "息税前利润", 40, _DEPS),
    ("basics_EBITDA", "pit:ebitda", "息税折旧摊销前利润", 40, _DEPS),
    ("basics_net_debt", "pit:netdebt", "净债务", 40, _DEPS),
    ("basics_retained_earnings", "pit:retained_earnings", "留存收益", 40, _DEPS),
    ("basics_fcff", "pit:fcff", "企业自由现金流", 40, _DEPS),
)

install(globals(), __name__, SPECS, extract_fin)
