#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽技术因子（均线比、布林、MACD、MFI，代表窗口）。"""

from __future__ import annotations

from ._price import extract_price
from ._wrappers import install

_DEPS = ("open", "high", "low", "close", "vol")

SPECS = (
    ("technical_MAC20", "mac20", "20日均线 / 收盘", 80, _DEPS),
    ("technical_MAC60", "mac60", "60日均线 / 收盘", 180, _DEPS),
    ("technical_boll_up", "boll_up", "布林上轨 / 收盘", 80, _DEPS),
    ("technical_boll_down", "boll_down", "布林下轨 / 收盘", 80, _DEPS),
    ("technical_EMA5", "ema5", "5日指数均线 / 收盘", 40, _DEPS),
)

install(globals(), __name__, SPECS, extract_price)
