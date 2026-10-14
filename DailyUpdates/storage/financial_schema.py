#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""financial_indicator 字段：PIT 财务列，对齐必须用 ann_date。

列名与 Tushare fina_indicator / fina_indicator_vip 一致。
质量、基础、成长、每股因子用指标表已算好的比率和同比。
另存资产负债表 / 现金流量表瘦身切片（仅因子缺口列），不灌整张三大报表。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

# 主键列（与 Tushare fina_indicator 对齐）
FINANCIAL_KEY_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "ann_date",
    "end_date",
    "update_flag",
)

# 数值列：风格描述子 + 质量 / 基础 / 成长 / 每股
# Tushare 比率类多为百分数（45 表示 45%），写入原值，换算留给因子。
FINANCIAL_VALUE_FIELDS: Tuple[str, ...] = (
    # 风格 / 过滤（已有）
    "roe",
    "roa",
    "eps",
    "ocfps",
    "bps",
    "revenue_ps",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "or_yoy",
    "equity_yoy",
    "debt_to_assets",
    "debt_to_eqt",
    "assets_to_eqt",
    "longdeb_to_debt",
    # 偿债 / 流动性
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "netdebt",
    "working_capital",
    "networking_capital",
    "ocf_to_debt",
    "ocf_to_netdebt",
    "ocf_to_shortdebt",
    "longdebt_to_workingcapital",
    "currentdebt_to_debt",
    # 周转
    "inv_turn",
    "ar_turn",
    "ca_turn",
    "fa_turn",
    "assets_turn",
    "invturn_days",
    "arturn_days",
    # 盈利
    "grossprofit_margin",
    "netprofit_margin",
    "cogs_of_sales",
    "op_of_gr",
    "roic",
    "npta",
    "ebit",
    "ebitda",
    "ebit_of_gr",
    # 费用
    "saleexp_to_gr",
    "adminexp_of_gr",
    "finaexp_of_gr",
    "impai_ttm",
    # 现金质量
    "ocf_to_or",
    "salescash_to_or",
    "ocf_to_opincome",
    "cfps",
    "fcff",
    "fcfe",
    # 结构
    "ca_to_assets",
    "nca_to_assets",
    "tbassets_to_totalassets",
    "tangible_asset",
    "retained_earnings",
    "invest_capital",
    "current_exint",
    "interestdebt",
    # 盈余质量
    "profit_dedt",
    "dtprofit_to_profit",
    "opincome_of_ebt",
    "n_op_profit_of_ebt",
    "investincome_of_ebt",
    "extra_item",
    "valuechange_income",
    # 每股
    "capital_rese_ps",
    "surplus_rese_ps",
    "undist_profit_ps",
    "retainedps",
    "ebit_ps",
    "total_revenue_ps",
    "fcff_ps",
    # 成长同比补齐
    "op_yoy",
    "ebt_yoy",
    "assets_yoy",
    "ocf_yoy",
    "basic_eps_yoy",
    "tr_yoy",
    "fixed_assets",
    "non_op_profit",
)

# Tushare fina_indicator / fina_indicator_vip 请求字段
FINANCIAL_TUSHARE_FIELDS: Tuple[str, ...] = (
    "ts_code",
    "ann_date",
    "end_date",
    "update_flag",
    *FINANCIAL_VALUE_FIELDS,
)

# 三大报表瘦身切片：官方字段名，见
# https://tushare.pro/document/2?doc_id=36
# https://tushare.pro/document/2?doc_id=44
# 筹资净额是 n_cash_flows_fnc_act，不是 n_cashflow_fnc_act。
STATEMENT_KEY_COLUMNS: Tuple[str, ...] = FINANCIAL_KEY_COLUMNS

BALANCESHEET_VALUE_FIELDS: Tuple[str, ...] = (
    "oth_eqt_tools",
    "acct_payable",
)
CASHFLOW_VALUE_FIELDS: Tuple[str, ...] = (
    "n_cashflow_inv_act",
    "n_cash_flows_fnc_act",
)
STATEMENT_FIELD_ALIASES: Dict[str, str] = {
    "n_cashflow_fnc_act": "n_cash_flows_fnc_act",
}

BALANCESHEET_TUSHARE_FIELDS: Tuple[str, ...] = (
    "ts_code",
    "ann_date",
    "end_date",
    "update_flag",
    *BALANCESHEET_VALUE_FIELDS,
)
CASHFLOW_TUSHARE_FIELDS: Tuple[str, ...] = (
    "ts_code",
    "ann_date",
    "end_date",
    "update_flag",
    *CASHFLOW_VALUE_FIELDS,
)

STATEMENT_TABLES: Tuple[str, ...] = ("balancesheet", "cashflow")
STATEMENT_API_TO_TABLE: Dict[str, str] = {
    "balancesheet": "balancesheet",
    "balancesheet_vip": "balancesheet",
    "cashflow": "cashflow",
    "cashflow_vip": "cashflow",
}
STATEMENT_VALUE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "balancesheet": BALANCESHEET_VALUE_FIELDS,
    "cashflow": CASHFLOW_VALUE_FIELDS,
}
STATEMENT_TUSHARE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "balancesheet": BALANCESHEET_TUSHARE_FIELDS,
    "cashflow": CASHFLOW_TUSHARE_FIELDS,
}


def statement_table_for_api(api_name: Optional[str]) -> Optional[str]:
    if not api_name:
        return None
    return STATEMENT_API_TO_TABLE.get(str(api_name).strip().lower())


def canonicalize_statement_fields(fields: Iterable[str]) -> Tuple[str, ...]:
    seen = []
    for name in fields:
        official = STATEMENT_FIELD_ALIASES.get(name, name)
        if official not in seen:
            seen.append(official)
    return tuple(seen)


# daily_basic 写入 market_data 的列（接口加列后按缺日期回填）
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
    "total_share",
    "float_share",
)
