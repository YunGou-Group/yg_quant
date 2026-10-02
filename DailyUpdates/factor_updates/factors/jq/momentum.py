#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽动量因子（乖离、ROC、CCI、价格位置等，代表窗口）。"""

from __future__ import annotations

from ._price import extract_price
from ._wrappers import install

_DEPS = ("open", "high", "low", "close", "vol")

SPECS = (
    ("momentum_BIAS5", "bias5", "5日乖离率", 40, _DEPS),
    ("momentum_BIAS20", "bias20", "20日乖离率", 80, _DEPS),
    ("momentum_ROC20", "roc20", "20日价格变动速率", 80, _DEPS),
    ("momentum_ROC60", "roc60", "60日价格变动速率", 180, _DEPS),
    ("momentum_CCI20", "cci20", "20日顺势指标", 80, _DEPS),
    ("momentum_Price1M", "price1m", "收盘 / 21日均价 - 1", 80, _DEPS),
    ("momentum_Price1Y", "price1y", "收盘 / 250日均价 - 1", 450, _DEPS),
    ("momentum_PLRC12", "plrc12", "12日标准化收盘对时间的回归斜率", 80, _DEPS),
    ("momentum_TRIX10", "trix10", "10日三重指数平滑", 80, _DEPS),
    ("momentum_BBIC", "bbic", "BBI(3,6,12,24) / 收盘", 80, _DEPS),
    ("momentum_bull_power", "bull_power", "(最高 - EMA13) / 收盘", 80, _DEPS),
    ("momentum_bear_power", "bear_power", "(最低 - EMA13) / 收盘", 80, _DEPS),
    (
        "momentum_fifty_two_week_close_rank",
        "close_rank_1y",
        "收盘在过去250日窗口内的分位",
        450,
        _DEPS,
    ),
    ("momentum_Volume1M", "volume1m", "量相对20日均量 × 20日均收益", 80, _DEPS),
)

install(globals(), __name__, SPECS, extract_price)
