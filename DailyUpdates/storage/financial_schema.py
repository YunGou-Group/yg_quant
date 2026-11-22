#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""financial_indicator 字段：PIT 财务列，对齐必须用 ann_date。

列名与 Tushare fina_indicator / fina_indicator_vip 一致。
质量、基础、成长、每股因子用指标表已算好的比率和同比。
利润表、资产负债表、现金流量表另存整张报表，不按因子用到的列裁剪。
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

# 三大报表整表。列名来自不指定 fields 时 Tushare 的返回顺序。
# https://tushare.pro/document/2?doc_id=33
# https://tushare.pro/document/2?doc_id=36
# https://tushare.pro/document/2?doc_id=44
# 筹资净额是 n_cash_flows_fnc_act，不是 n_cashflow_fnc_act。
# 主键含 report_type：同一公告里合并报表和母公司报表是不同的行。
STATEMENT_KEY_COLUMNS: Tuple[str, ...] = FINANCIAL_KEY_COLUMNS
STATEMENT_ROW_KEYS: Tuple[str, ...] = (
    "symbol",
    "ann_date",
    "end_date",
    "report_type",
    "update_flag",
)
STATEMENT_TEXT_FIELDS: Tuple[str, ...] = (
    "f_ann_date",
    "comp_type",
    "end_type",
)
STATEMENT_META_FIELDS = frozenset(
    {
        "ts_code",
        "symbol",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "update_flag",
    }
)
STATEMENT_FIELD_ALIASES: Dict[str, str] = {
    "n_cashflow_fnc_act": "n_cash_flows_fnc_act",
}

_INCOME_COLUMNS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,"
    "basic_eps,diluted_eps,total_revenue,revenue,int_income,prem_earned,"
    "comm_income,n_commis_income,n_oth_income,n_oth_b_income,prem_income,"
    "out_prem,une_prem_reser,reins_income,n_sec_tb_income,n_sec_uw_income,"
    "n_asset_mg_income,oth_b_income,fv_value_chg_gain,invest_income,"
    "ass_invest_income,forex_gain,total_cogs,oper_cost,int_exp,comm_exp,"
    "biz_tax_surchg,sell_exp,admin_exp,fin_exp,assets_impair_loss,prem_refund,"
    "compens_payout,reser_insur_liab,div_payt,reins_exp,oper_exp,"
    "compens_payout_refu,insur_reser_refu,reins_cost_refund,other_bus_cost,"
    "operate_profit,non_oper_income,non_oper_exp,nca_disploss,total_profit,"
    "income_tax,n_income,n_income_attr_p,minority_gain,oth_compr_income,"
    "t_compr_income,compr_inc_attr_p,compr_inc_attr_m_s,ebit,ebitda,"
    "insurance_exp,undist_profit,distable_profit,rd_exp,fin_exp_int_exp,"
    "fin_exp_int_inc,transfer_surplus_rese,transfer_housing_imprest,"
    "transfer_oth,adj_lossgain,withdra_legal_surplus,withdra_legal_pubfund,"
    "withdra_biz_devfund,withdra_rese_fund,withdra_oth_ersu,workers_welfare,"
    "distr_profit_shrhder,prfshare_payable_dvd,comshare_payable_dvd,"
    "capit_comstock_div,continued_net_profit,update_flag"
)
_BALANCESHEET_COLUMNS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,"
    "total_share,cap_rese,undistr_porfit,surplus_rese,special_rese,money_cap,"
    "trad_asset,notes_receiv,accounts_receiv,oth_receiv,prepayment,div_receiv,"
    "int_receiv,inventories,amor_exp,nca_within_1y,sett_rsrv,loanto_oth_bank_fi,"
    "premium_receiv,reinsur_receiv,reinsur_res_receiv,pur_resale_fa,"
    "oth_cur_assets,total_cur_assets,fa_avail_for_sale,htm_invest,"
    "lt_eqt_invest,invest_real_estate,time_deposits,oth_assets,lt_rec,"
    "fix_assets,cip,const_materials,fixed_assets_disp,produc_bio_assets,"
    "oil_and_gas_assets,intan_assets,r_and_d,goodwill,lt_amor_exp,"
    "defer_tax_assets,decr_in_disbur,oth_nca,total_nca,cash_reser_cb,"
    "depos_in_oth_bfi,prec_metals,deriv_assets,rr_reins_une_prem,"
    "rr_reins_outstd_cla,rr_reins_lins_liab,rr_reins_lthins_liab,"
    "refund_depos,ph_pledge_loans,refund_cap_depos,indep_acct_assets,"
    "client_depos,client_prov,transac_seat_fee,invest_as_receiv,total_assets,"
    "lt_borr,st_borr,cb_borr,depos_ib_deposits,loan_oth_bank,trading_fl,"
    "notes_payable,acct_payable,adv_receipts,sold_for_repur_fa,comm_payable,"
    "payroll_payable,taxes_payable,int_payable,div_payable,oth_payable,"
    "acc_exp,deferred_inc,st_bonds_payable,payable_to_reinsurer,"
    "rsrv_insur_cont,acting_trading_sec,acting_uw_sec,non_cur_liab_due_1y,"
    "oth_cur_liab,total_cur_liab,bond_payable,lt_payable,specific_payables,"
    "estimated_liab,defer_tax_liab,defer_inc_non_cur_liab,oth_ncl,total_ncl,"
    "depos_oth_bfi,deriv_liab,depos,agency_bus_liab,oth_liab,prem_receiv_adva,"
    "depos_received,ph_invest,reser_une_prem,reser_outstd_claims,"
    "reser_lins_liab,reser_lthins_liab,indept_acc_liab,pledge_borr,"
    "indem_payable,policy_div_payable,total_liab,treasury_share,"
    "ordin_risk_reser,forex_differ,invest_loss_unconf,minority_int,"
    "total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int,total_liab_hldr_eqy,"
    "lt_payroll_payable,oth_comp_income,oth_eqt_tools,oth_eqt_tools_p_shr,"
    "lending_funds,acc_receivable,st_fin_payable,payables,hfs_assets,hfs_sales,"
    "cost_fin_assets,fair_value_fin_assets,contract_assets,contract_liab,"
    "accounts_receiv_bill,accounts_pay,oth_rcv_total,fix_assets_total,"
    "cip_total,oth_pay_total,long_pay_total,debt_invest,oth_debt_invest,"
    "update_flag"
)
_CASHFLOW_COLUMNS = (
    "ts_code,ann_date,f_ann_date,end_date,comp_type,report_type,end_type,"
    "net_profit,finan_exp,c_fr_sale_sg,recp_tax_rends,n_depos_incr_fi,"
    "n_incr_loans_cb,n_inc_borr_oth_fi,prem_fr_orig_contr,n_incr_insured_dep,"
    "n_reinsur_prem,n_incr_disp_tfa,ifc_cash_incr,n_incr_disp_faas,"
    "n_incr_loans_oth_bank,n_cap_incr_repur,c_fr_oth_operate_a,"
    "c_inf_fr_operate_a,c_paid_goods_s,c_paid_to_for_empl,c_paid_for_taxes,"
    "n_incr_clt_loan_adv,n_incr_dep_cbob,c_pay_claims_orig_inco,"
    "pay_handling_chrg,pay_comm_insur_plcy,oth_cash_pay_oper_act,"
    "st_cash_out_act,n_cashflow_act,oth_recp_ral_inv_act,c_disp_withdrwl_invest,"
    "c_recp_return_invest,n_recp_disp_fiolta,n_recp_disp_sobu,"
    "stot_inflows_inv_act,c_pay_acq_const_fiolta,c_paid_invest,"
    "n_disp_subs_oth_biz,oth_pay_ral_inv_act,n_incr_pledge_loan,"
    "stot_out_inv_act,n_cashflow_inv_act,c_recp_borrow,proc_issue_bonds,"
    "oth_cash_recp_ral_fnc_act,stot_cash_in_fnc_act,free_cashflow,"
    "c_prepay_amt_borr,c_pay_dist_dpcp_int_exp,incl_dvd_profit_paid_sc_ms,"
    "oth_cashpay_ral_fnc_act,stot_cashout_fnc_act,n_cash_flows_fnc_act,"
    "eff_fx_flu_cash,n_incr_cash_cash_equ,c_cash_equ_beg_period,"
    "c_cash_equ_end_period,c_recp_cap_contrib,incl_cash_rec_saims,"
    "uncon_invest_loss,prov_depr_assets,depr_fa_coga_dpba,amort_intang_assets,"
    "lt_amort_deferred_exp,decr_deferred_exp,incr_acc_exp,loss_disp_fiolta,"
    "loss_scr_fa,loss_fv_chg,invest_loss,decr_def_inc_tax_assets,"
    "incr_def_inc_tax_liab,decr_inventories,decr_oper_payable,incr_oper_payable,"
    "others,im_net_cashflow_oper_act,conv_debt_into_cap,"
    "conv_copbonds_due_within_1y,fa_fnc_leases,im_n_incr_cash_equ,"
    "net_dism_capital_add,net_cash_rece_sec,credit_impa_loss,"
    "use_right_asset_dep,oth_loss_asset,end_bal_cash,beg_bal_cash,"
    "end_bal_cash_equ,beg_bal_cash_equ,update_flag"
)


def _statement_names(columns: str) -> Tuple[str, ...]:
    names = tuple(part.strip() for part in columns.split(",") if part.strip())
    if len(names) != len(set(names)):
        raise RuntimeError("报表字段名单有重复")
    return names


def _value_fields(columns: str) -> Tuple[str, ...]:
    return tuple(
        name for name in _statement_names(columns) if name not in STATEMENT_META_FIELDS
    )


INCOME_VALUE_FIELDS: Tuple[str, ...] = _value_fields(_INCOME_COLUMNS)
BALANCESHEET_VALUE_FIELDS: Tuple[str, ...] = _value_fields(_BALANCESHEET_COLUMNS)
CASHFLOW_VALUE_FIELDS: Tuple[str, ...] = _value_fields(_CASHFLOW_COLUMNS)
INCOME_TUSHARE_FIELDS: Tuple[str, ...] = _statement_names(_INCOME_COLUMNS)
BALANCESHEET_TUSHARE_FIELDS: Tuple[str, ...] = _statement_names(_BALANCESHEET_COLUMNS)
CASHFLOW_TUSHARE_FIELDS: Tuple[str, ...] = _statement_names(_CASHFLOW_COLUMNS)

STATEMENT_TABLES: Tuple[str, ...] = ("income", "balancesheet", "cashflow")
STATEMENT_API_TO_TABLE: Dict[str, str] = {
    "income": "income",
    "income_vip": "income",
    "balancesheet": "balancesheet",
    "balancesheet_vip": "balancesheet",
    "cashflow": "cashflow",
    "cashflow_vip": "cashflow",
}
STATEMENT_VALUE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "income": INCOME_VALUE_FIELDS,
    "balancesheet": BALANCESHEET_VALUE_FIELDS,
    "cashflow": CASHFLOW_VALUE_FIELDS,
}
STATEMENT_TUSHARE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "income": INCOME_TUSHARE_FIELDS,
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
