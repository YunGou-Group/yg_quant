#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽成长因子：同比与 PEG。"""

from __future__ import annotations

from ._fin import extract_fin
from ._wrappers import install

_DEPS = ("close", "pe_ttm")

SPECS = (
    (
        "growth_operating_revenue_growth_rate",
        "pit:or_yoy",
        "营业收入同比",
        40,
        _DEPS,
    ),
    ("growth_net_profit_growth_rate", "pit:netprofit_yoy", "净利润同比", 40, _DEPS),
    ("growth_total_profit_growth_rate", "pit:ebt_yoy", "利润总额同比", 40, _DEPS),
    ("growth_total_asset_growth_rate", "pit:assets_yoy", "总资产同比", 40, _DEPS),
    (
        "growth_net_operate_cashflow_growth_rate",
        "pit:ocf_yoy",
        "经营现金流同比",
        40,
        _DEPS,
    ),
    (
        "growth_net_asset_growth_rate",
        "pit:equity_yoy",
        "Tushare 净资产同比（约四季）；聚宽为当季权益/三季前权益-1",
        40,
        _DEPS,
    ),
    ("growth_PEG", "peg", "PE_TTM / (净利同比×100)，PE 或增长为负缺失", 40, _DEPS),
)

install(globals(), __name__, SPECS, extract_fin)
