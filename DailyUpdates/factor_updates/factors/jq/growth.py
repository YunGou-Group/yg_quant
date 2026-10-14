#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽成长因子：同比与 PEG。"""

from __future__ import annotations

from ._fin import extract_fin
from ._wrappers import install

_DEPS = ("close", "pe_ttm")

SPECS = (
    ("growth_total_profit_growth_rate", "pit:ebt_yoy", "利润总额同比", 40, _DEPS),
    ("growth_total_asset_growth_rate", "pit:assets_yoy", "总资产同比", 40, _DEPS),
    (
        "growth_net_operate_cashflow_growth_rate",
        "pit:ocf_yoy",
        "经营现金流同比",
        40,
        _DEPS,
    ),
    ("growth_PEG", "peg", "PE_TTM / (净利同比×100)，PE 或增长为负缺失", 40, _DEPS),
    (
        "growth_adjusted_net_profit_yoy",
        "pit:dt_netprofit_yoy",
        "扣非净利润同比",
        40,
        _DEPS,
    ),
    (
        "growth_invest_cashflow_yoy",
        "ncf_inv_yoy",
        "投资活动现金流同比",
        40,
        _DEPS,
    ),
    (
        "growth_finance_cashflow_yoy",
        "ncf_fnc_yoy",
        "筹资活动现金流同比",
        40,
        _DEPS,
    ),
)

install(globals(), __name__, SPECS, extract_fin)
