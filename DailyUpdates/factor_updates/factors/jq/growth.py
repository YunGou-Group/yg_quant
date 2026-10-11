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
)

install(globals(), __name__, SPECS, extract_fin)
