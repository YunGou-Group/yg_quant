#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽基础因子：市值、估值倒数、EBIT 等。"""

from __future__ import annotations

from ._fin import extract_fin
from ._wrappers import install

_DEPS = ("close", "total_mv", "circ_mv", "ps_ttm", "total_share")

SPECS = (
    ("basics_market_cap", "market_cap", "总市值", 40, _DEPS),
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
    ("basics_fcff", "pit:fcff", "企业自由现金流", 40, _DEPS),
    (
        "basics_sale_expense_ttm",
        "sale_expense_ttm",
        "销售费用率 × 营业总收入",
        40,
        _DEPS,
    ),
    (
        "basics_non_operating_net_profit_ttm",
        "pit:non_op_profit",
        "营业外收支净额（指标表，非严格 TTM 加总）",
        40,
        _DEPS,
    ),
    (
        "basics_financial_expense_ttm",
        "financial_expense_ttm",
        "财务费用率 × 营业总收入",
        40,
        _DEPS,
    ),
    (
        "basics_administration_expense_ttm",
        "administration_expense_ttm",
        "管理费用率 × 营业总收入",
        40,
        _DEPS,
    ),
    (
        "basics_value_change_profit_ttm",
        "pit:valuechange_income",
        "价值变动净收益（指标表）",
        40,
        _DEPS,
    ),
    (
        "basics_asset_impairment_loss_ttm",
        "asset_impairment_loss_ttm",
        "资产减值损失率 × 营业总收入",
        40,
        _DEPS,
    ),
    (
        "basics_non_recurring_gain_loss",
        "pit:extra_item",
        "非经常性损益",
        40,
        _DEPS,
    ),
)

install(globals(), __name__, SPECS, extract_fin)
