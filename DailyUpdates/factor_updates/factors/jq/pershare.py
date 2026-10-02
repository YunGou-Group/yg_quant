#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽每股因子。"""

from __future__ import annotations

from ._fin import extract_fin
from ._wrappers import install

_DEPS = ("close",)

SPECS = (
    ("pershare_eps_ttm", "pit:eps", "当期每股收益（非聚宽 TTM 归母净利润/股本）", 40, _DEPS),
    ("pershare_net_asset_per_share", "pit:bps", "每股净资产", 40, _DEPS),
    ("pershare_operating_revenue_per_share", "pit:revenue_ps", "每股营业收入", 40, _DEPS),
    (
        "pershare_net_operate_cash_flow_per_share",
        "pit:ocfps",
        "每股经营现金流",
        40,
        _DEPS,
    ),
    ("pershare_cashflow_per_share_ttm", "pit:cfps", "每股现金流量净额", 40, _DEPS),
    (
        "pershare_capital_reserve_fund_per_share",
        "pit:capital_rese_ps",
        "每股资本公积",
        40,
        _DEPS,
    ),
    (
        "pershare_retained_profit_per_share",
        "pit:undist_profit_ps",
        "每股未分配利润",
        40,
        _DEPS,
    ),
    (
        "pershare_surplus_reserve_fund_per_share",
        "pit:surplus_rese_ps",
        "每股盈余公积",
        40,
        _DEPS,
    ),
)

install(globals(), __name__, SPECS, extract_fin)
