#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""现价附近筹码浓度：换手递推后，成本落在现价 ±5% 内的质量。"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

from ._chip_cyq import near_price


class ChipNearPriceFactor(BaseFactor):
    name = "chip_near_price"
    description = (
        "现价附近筹码：d_t=(1-T_t)d_{t-1}+T_t f_t；"
        "因子=成本在现价±5%内的质量"
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
        return near_price(data, start_date)
