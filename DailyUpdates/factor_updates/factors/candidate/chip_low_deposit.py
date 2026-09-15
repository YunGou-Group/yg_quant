#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""低位筹码沉积：近100日收盘中位数 / 平均筹码成本。"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

from ._chip_cyq import low_deposit


class ChipLowDepositFactor(BaseFactor):
    name = "chip_low_deposit"
    description = (
        "低位筹码沉积：c_t=(1-T_t)c_{t-1}+T_t(O+H+L+C)/4；"
        "因子=近100日收盘中位数/平均成本"
    )
    dependencies: List[str] = [
        "open",
        "high",
        "low",
        "close",
        "turnover_rate_f",
        "turnover_rate",
    ]
    role = "alpha"
    stage = "candidate"
    lookback_days = 400

    def calculate(
        self, data: pd.DataFrame, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        return low_deposit(data, start_date)
