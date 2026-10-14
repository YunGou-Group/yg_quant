#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽技术因子（均线比、布林、MACD、MFI，代表窗口）。"""

from __future__ import annotations

from ._price import extract_price
from ._wrappers import install

_DEPS = ("open", "high", "low", "close", "vol")
_DEPS_MCST = ("open", "high", "low", "close", "vol", "amount", "turnover_rate")

SPECS = (
    ("technical_MAC20", "mac20", "20日均线 / 收盘", 80, _DEPS),
    ("technical_MAC60", "mac60", "60日均线 / 收盘", 180, _DEPS),
    ("technical_boll_up", "boll_up", "布林上轨 / 收盘", 80, _DEPS),
    ("technical_boll_down", "boll_down", "布林下轨 / 收盘", 80, _DEPS),
    ("technical_EMA5", "ema5", "5日指数均线 / 收盘", 40, _DEPS),
    ("technical_KDJ_J", "kdj_j", "KDJ 的 J 值（N=9, M1=M2=3）", 40, _DEPS),
    ("technical_ADX", "adx", "平均趋向指数（M1=14, M2=6）", 80, _DEPS),
    ("technical_OBV", "obv", "能量潮", 450, _DEPS),
    ("technical_MDD60", "mdd60", "60日收盘价路径最大回撤", 180, _DEPS),
    (
        "technical_MCST",
        "mcst",
        "动态成本 DMA(成交额/量, 换手率)",
        450,
        _DEPS_MCST,
    ),
    ("technical_CR", "cr", "中间意愿指标 CR26", 80, _DEPS),
    ("technical_VOLT20", "volt20", "20日收盘价标准差", 80, _DEPS),
)

install(globals(), __name__, SPECS, extract_price)
