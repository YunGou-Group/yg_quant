#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽情绪因子（换手、量、AR/BR、ATR 等，代表窗口）。"""

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
    "turnover_rate",
    "turnover_rate_f",
)

SPECS = (
    ("emotion_VOL20", "vol20", "20日平均换手率", 80, _DEPS),
    ("emotion_VOL60", "vol60", "60日平均换手率", 180, _DEPS),
    ("emotion_DAVOL20", "davol20", "20日均换手 / 120日均换手", 400, _DEPS),
    ("emotion_VR", "vr", "26日成交量比率", 80, _DEPS),
    ("emotion_AR", "ar", "26日人气指标", 80, _DEPS),
    ("emotion_BR", "br", "26日意愿指标", 80, _DEPS),
    ("emotion_ARBR", "arbr", "AR 减 BR", 80, _DEPS),
    ("emotion_VROC12", "vroc12", "12日量变动速率", 80, _DEPS),
    ("emotion_VSTD20", "vstd20", "20日成交量标准差", 80, _DEPS),
    ("emotion_ATR14", "atr14", "14日真实波幅均值", 80, _DEPS),
    ("emotion_WVAD", "wvad", "6日威廉变异离散量", 80, _DEPS),
    ("emotion_PSY", "psy", "12日心理线", 80, _DEPS),
    ("emotion_turnover_volatility", "turnover_volatility", "20日换手率标准差", 80, _DEPS),
    ("emotion_money_flow_20", "money_flow_20", "20日资金流量之和", 80, _DEPS),
)

install(globals(), __name__, SPECS, extract_price)
