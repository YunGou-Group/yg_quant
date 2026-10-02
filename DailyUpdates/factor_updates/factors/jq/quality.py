#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽质量因子：财务指标按公告日前向填充。"""

from __future__ import annotations

from ._fin import extract_fin
from ._wrappers import install

_DEPS = ("close",)

SPECS = (
    ("quality_roe_ttm", "pit:roe", "权益回报率（指标表 PIT）", 40, _DEPS),
    ("quality_roa_ttm", "pit:roa", "资产回报率（指标表 PIT）", 40, _DEPS),
    ("quality_roic_ttm", "pit:roic", "投入资本回报率", 40, _DEPS),
    ("quality_gross_income_ratio", "pit:grossprofit_margin", "销售毛利率", 40, _DEPS),
    ("quality_net_profit_ratio", "pit:netprofit_margin", "销售净利率", 40, _DEPS),
    ("quality_operating_profit_ratio", "pit:op_of_gr", "营业利润率", 40, _DEPS),
    ("quality_current_ratio", "pit:current_ratio", "流动比率", 40, _DEPS),
    ("quality_quick_ratio", "pit:quick_ratio", "速动比率", 40, _DEPS),
    (
        "quality_cash_to_current_liability",
        "pit:cash_ratio",
        "Tushare 现金比率；聚宽为现金/流动负债12个月均值",
        40,
        _DEPS,
    ),
    ("quality_debt_to_asset_ratio", "pit:debt_to_assets", "资产负债率", 40, _DEPS),
    ("quality_debt_to_equity_ratio", "pit:debt_to_eqt", "产权比率", 40, _DEPS),
    ("quality_inventory_turnover_rate", "pit:inv_turn", "存货周转率", 40, _DEPS),
    ("quality_account_receivable_turnover_rate", "pit:ar_turn", "应收账款周转率", 40, _DEPS),
    ("quality_total_asset_turnover_rate", "pit:assets_turn", "总资产周转率", 40, _DEPS),
    ("quality_inventory_turnover_days", "pit:invturn_days", "存货周转天数", 40, _DEPS),
    (
        "quality_account_receivable_turnover_days",
        "pit:arturn_days",
        "应收账款周转天数",
        40,
        _DEPS,
    ),
    ("quality_OperatingCycle", "operating_cycle", "营业周期=存货+应收周转天数", 40, _DEPS),
    ("quality_admin_expense_rate", "pit:adminexp_of_gr", "管理费用率", 40, _DEPS),
    ("quality_financial_expense_rate", "pit:finaexp_of_gr", "财务费用率", 40, _DEPS),
    (
        "quality_sale_expense_to_operating_revenue",
        "pit:saleexp_to_gr",
        "销售费用率",
        40,
        _DEPS,
    ),
    ("quality_cash_rate_of_sales", "pit:ocf_to_or", "经营现金流 / 营业收入", 40, _DEPS),
    (
        "quality_goods_service_cash_to_operating_revenue_ttm",
        "pit:salescash_to_or",
        "销售收现 / 营业收入",
        40,
        _DEPS,
    ),
    ("quality_equity_to_asset_ratio", "equity_to_asset", "股东权益 / 总资产", 40, _DEPS),
    (
        "quality_dtprofit_to_profit",
        "pit:dtprofit_to_profit",
        "Tushare 扣非/净利润；聚宽 adjusted_profit_to_total_profit 为扣非/利润总额",
        40,
        _DEPS,
    ),
)

install(globals(), __name__, SPECS, extract_fin)
