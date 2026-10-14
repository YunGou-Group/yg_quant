#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽质量因子：财务指标按公告日前向填充。"""

from __future__ import annotations

from ._fin import extract_fin
from ._wrappers import install

_DEPS = ("close", "total_mv", "total_share")

SPECS = (
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
    (
        "quality_cfo_to_ev",
        "cfo_to_ev",
        "经营现金流 / (总市值+净债务)，近似聚宽 EV",
        40,
        _DEPS,
    ),
    (
        "quality_net_non_operating_income_to_total_profit",
        "pit:n_op_profit_of_ebt",
        "营业外收支净额 / 利润总额",
        40,
        _DEPS,
    ),
    (
        "quality_fixed_asset_ratio",
        "fixed_asset_ratio",
        "固定资产 / 总资产（总资产由账面权益还原）",
        40,
        _DEPS,
    ),
    ("quality_DEGM", "DEGM", "毛利率同比：当期毛利率/上年同期 - 1", 40, _DEPS),
    (
        "quality_net_operating_cash_flow_coverage",
        "net_operating_cash_flow_coverage",
        "每股经营现金流 / EPS，近似净利润现金含量",
        40,
        _DEPS,
    ),
    ("quality_MLEV", "MLEV", "长期负债 / (长期负债+总市值)", 40, _DEPS),
    (
        "quality_operating_profit_growth_rate",
        "pit:op_yoy",
        "营业利润同比",
        40,
        _DEPS,
    ),
    (
        "quality_debt_to_tangible_equity_ratio",
        "debt_to_tangible_equity_ratio",
        "负债 / 有形净值",
        40,
        _DEPS,
    ),
    (
        "quality_current_asset_turnover_rate",
        "pit:ca_turn",
        "流动资产周转率",
        40,
        _DEPS,
    ),
    (
        "quality_operating_profit_to_total_profit",
        "pit:opincome_of_ebt",
        "经营活动净收益 / 利润总额",
        40,
        _DEPS,
    ),
    (
        "quality_equity_to_fixed_asset_ratio",
        "equity_to_fixed_asset_ratio",
        "账面权益 / 固定资产",
        40,
        _DEPS,
    ),
    (
        "quality_fixed_assets_turnover_rate",
        "pit:fa_turn",
        "固定资产周转率",
        40,
        _DEPS,
    ),
    (
        "quality_invest_income_associates_to_total_profit",
        "pit:investincome_of_ebt",
        "联营合营投资收益 / 利润总额",
        40,
        _DEPS,
    ),
    ("quality_ACCA", "ACCA", "经营现金流/总资产 - ROA", 40, _DEPS),
    (
        "quality_total_profit_to_cost_ratio",
        "total_profit_to_cost_ratio",
        "EBIT/营业总收入 ÷ (成本+三费)/营收",
        40,
        _DEPS,
    ),
    (
        "quality_operating_cost_to_operating_revenue_ratio",
        "pit:cogs_of_sales",
        "销售成本率",
        40,
        _DEPS,
    ),
    ("quality_LVGI", "LVGI", "年报资产负债率本期/上期", 40, _DEPS),
    ("quality_SGI", "SGI", "年报每股营收本期/上期", 40, _DEPS),
    ("quality_GMI", "GMI", "年报毛利率上期/本期", 40, _DEPS),
    ("quality_DSRI", "DSRI", "年报应收周转天数本期/上期", 40, _DEPS),
    ("quality_SGAI", "SGAI", "年报销管费用率本期/上期", 40, _DEPS),
    (
        "quality_asset_turnover_ttm",
        "pit:assets_turn",
        "缺经营资产周转，用总资产周转率近似",
        40,
        _DEPS,
    ),
    ("quality_roe_ttm_8y", "roe_ttm_8y", "年报 ROE 最多 8 年几何平均", 40, _DEPS),
    ("quality_roa_ttm_8y", "roa_ttm_8y", "年报 ROA 最多 8 年几何平均", 40, _DEPS),
    ("quality_DEGM_8y", "DEGM_8y", "年报 DEGM 最多 8 年几何平均", 40, _DEPS),
    (
        "quality_margin_stability",
        "margin_stability",
        "过去最多 8 年年报毛利率均值/标准差",
        40,
        _DEPS,
    ),
    (
        "quality_maximum_margin",
        "maximum_margin",
        "max(margin_stability, DEGM_8y)",
        40,
        _DEPS,
    ),
    (
        "quality_cash_conversion_cycle",
        "cash_conversion_cycle",
        "现金循环周期：存货+应收天数−应付天数",
        40,
        _DEPS,
    ),
    (
        "quality_time_interest_earned",
        "time_interest_earned",
        "利息保障倍数：EBIT / 财务费用",
        40,
        _DEPS,
    ),
    (
        "quality_adjusted_roe",
        "adjusted_roe",
        "扣非净利润 / 账面净资产",
        40,
        _DEPS,
    ),
)

install(globals(), __name__, SPECS, extract_fin)
