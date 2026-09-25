#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开源金工筹码收益：holding_ret 与市场符号调整 holding_ret_adj。"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

from ._chip_cyq import CHIP_WINDOW, holding_ret, holding_ret_adj

_CHIP_FIELDS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "turnover_rate_f",
    "turnover_rate",
]


class HoldingRetFactor(BaseFactor):
    name = "holding_ret"
    description = (
        "筹码收益：当日均价(O+H+L+C)/4 相对换手递推筹码加权成本的收益；"
        f"warmup={CHIP_WINDOW} 日"
    )
    dependencies: List[str] = list(_CHIP_FIELDS)
    role = "alpha"
    stage = "candidate"
    lookback_days = 450

    def calculate(
        self, data: pd.DataFrame, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        return holding_ret(data, start_date)


class HoldingRetAdjFactor(BaseFactor):
    name = "holding_ret_adj"
    description = (
        "筹码收益调整：holding_ret × sign(mkt_holding_ret)；"
        "mkt 为流通市值加权截面均值（缺市值则等权）；"
        "市场浮盈时做动量，否则做反转"
    )
    dependencies: List[str] = [*_CHIP_FIELDS, "circ_mv"]
    role = "alpha"
    stage = "candidate"
    lookback_days = 450

    def calculate(
        self, data: pd.DataFrame, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        return holding_ret_adj(data, start_date)
