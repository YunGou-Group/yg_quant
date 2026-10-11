#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""短周期价量：隔夜/日内、反转、流动性、K 线结构。评估族名 short。"""

from __future__ import annotations

from ._price import extract_price
from ._wrappers import install

_DEPS = (
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "pre_close",
    "up_limit",
    "down_limit",
    "turnover_rate",
    "turnover_rate_f",
    "circ_mv",
)

SPECS = (
    ("short_overnight", "overnight", "开盘 / 昨收 - 1", 40, _DEPS),
    ("short_intraday", "intraday", "收盘 / 开盘 - 1", 40, _DEPS),
    ("short_ret1", "ret1", "1 日收益", 40, _DEPS),
    ("short_ret5", "ret5", "5 日收益", 40, _DEPS),
    ("short_ret10", "ret10", "10 日收益", 40, _DEPS),
    ("short_amihud5", "amihud5", "5 日 Amihud：均值 |收益|/成交额", 40, _DEPS),
    ("short_amihud20", "amihud20", "20 日 Amihud：均值 |收益|/成交额", 80, _DEPS),
    ("short_vol5", "vol5", "5 日平均换手率", 40, _DEPS),
    ("short_davol5", "davol5", "5 日均换手 / 20 日均换手", 80, _DEPS),
    ("short_range5", "range5", "5 日均振幅 (最高-最低)/收盘", 40, _DEPS),
    ("short_efficiency", "efficiency", "|收盘-开盘| / (最高-最低)", 40, _DEPS),
    ("short_upper_shadow", "upper_shadow", "上影线 / 收盘", 40, _DEPS),
    ("short_lower_shadow", "lower_shadow", "下影线 / 收盘", 40, _DEPS),
    ("short_close_loc", "close_loc", "(收盘-最低) / (最高-最低)", 40, _DEPS),
    ("short_limit_up_dist", "limit_up_dist", "(涨停价-收盘) / 昨收", 40, _DEPS),
    ("short_limit_dn_dist", "limit_dn_dist", "(收盘-跌停价) / 昨收", 40, _DEPS),
    ("short_maxret20", "maxret20", "过去 20 日最大单日收益", 80, _DEPS),
    ("short_minret20", "minret20", "过去 20 日最小单日收益", 80, _DEPS),
    ("short_corr_pv5", "corr_pv5", "5 日收益与成交量相关", 40, _DEPS),
    ("short_corr_pv20", "corr_pv20", "20 日收益与成交量相关", 80, _DEPS),
    ("short_vwap_dev", "vwap_dev", "收盘 / (成交额/成交量) - 1", 40, _DEPS),
    ("short_rsi6", "rsi6", "6 日 RSI（Wilder）", 40, _DEPS),
    ("short_rsi14", "rsi14", "14 日 RSI（Wilder）", 40, _DEPS),
    ("short_willr14", "willr14", "14 日威廉指标", 40, _DEPS),
    ("short_down_vol_ratio20", "down_vol_ratio20", "20 日下跌成交量占比", 80, _DEPS),
    ("short_amt_to_mv", "amt_to_mv", "成交额 / 流通市值", 40, _DEPS),
    ("short_atr5", "atr5", "5 日真实波幅均值", 40, _DEPS),
    ("short_jump5", "jump5", "5 日隔夜绝对收益均值", 40, _DEPS),
)

install(globals(), __name__, SPECS, extract_price)
