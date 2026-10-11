#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽动量因子（乖离、ROC、CCI、价格位置等，代表窗口）。"""

from __future__ import annotations

from ._price import extract_price
from ._wrappers import install

_DEPS = ("open", "high", "low", "close", "vol")

SPECS = (
    ("momentum_BIAS5", "bias5", "5日乖离率", 40, _DEPS),
    ("momentum_CCI20", "cci20", "20日顺势指标", 80, _DEPS),
    ("momentum_arron_up_25", "arron_up_25", "25日 Aroon 上轨", 80, _DEPS),
    ("momentum_arron_down_25", "arron_down_25", "25日 Aroon 下轨", 80, _DEPS),
    ("momentum_BIAS60", "bias60", "60日乖离率", 180, _DEPS),
    ("momentum_MASS", "mass", "梅斯线 SUM(EMA9(HL)/EMA9(EMA9(HL)),25)", 80, _DEPS),
    ("momentum_ROC120", "roc120", "120日价格变动速率", 400, _DEPS),
    ("momentum_single_day_VPT", "single_day_vpt", "单日价量趋势：日收益×成交量", 40, _DEPS),
    ("momentum_single_day_VPT_6", "single_day_vpt_6", "单日价量趋势 6 日均值", 40, _DEPS),
    ("momentum_single_day_VPT_12", "single_day_vpt_12", "单日价量趋势 12 日均值", 80, _DEPS),
)

install(globals(), __name__, SPECS, extract_price)
