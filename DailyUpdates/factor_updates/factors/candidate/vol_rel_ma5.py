#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vol(i)/mean(vol(i-1)..vol(i-10)) * (-x)；x 为近 100 日收盘价高低分位 - 0.5。"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor


class VolRelMa5Factor(BaseFactor):
    name = "vol_rel_ma5"
    description = (
        "vol/前10日均量 * (-x)；"
        "x=(close-min100)/(max100-min100)-0.5，低位为负、高位为正"
    )
    dependencies: List[str] = ["vol", "close"]
    role = "alpha"
    stage = "candidate"
    lookback_days = 180

    def calculate(
        self, data: pd.DataFrame, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        df = data[["ts_code", "trade_date", "vol", "close"]].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(["ts_code", "trade_date"])

        g_vol = df.groupby("ts_code")["vol"]
        g_close = df.groupby("ts_code")["close"]
        ma10 = g_vol.transform(lambda s: s.shift(1).rolling(10).mean())
        lo = g_close.transform(lambda s: s.rolling(100).min())
        hi = g_close.transform(lambda s: s.rolling(100).max())
        span = hi - lo
        x = (df["close"] - lo) / span - 0.5
        x = x.where(span > 0)

        df["factor_value"] = (df["vol"] / ma10) * (-x)

        out = df[["ts_code", "trade_date", "factor_value"]].dropna()
        if start_date:
            start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            out = out[out["trade_date"] >= start]
        return out.reset_index(drop=True)
