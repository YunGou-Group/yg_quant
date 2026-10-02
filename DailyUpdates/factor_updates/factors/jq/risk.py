#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽风险因子（收益方差 / 偏度 / 峰度 / 夏普，代表窗口）。"""

from __future__ import annotations

from ._price import extract_price
from ._wrappers import install

_DEPS = ("close",)

SPECS = (
    ("risk_Variance20", "variance20", "20日年化收益方差", 80, _DEPS),
    ("risk_Variance60", "variance60", "60日年化收益方差", 180, _DEPS),
    ("risk_Skewness20", "skewness20", "20日收益偏度", 80, _DEPS),
    ("risk_Skewness60", "skewness60", "60日收益偏度", 180, _DEPS),
    ("risk_Kurtosis20", "kurtosis20", "20日收益峰度", 80, _DEPS),
    ("risk_sharpe_ratio_20", "sharpe20", "20日夏普（Rf=4%）", 80, _DEPS),
    ("risk_sharpe_ratio_60", "sharpe60", "60日夏普（Rf=4%）", 180, _DEPS),
)

install(globals(), __name__, SPECS, extract_price)
