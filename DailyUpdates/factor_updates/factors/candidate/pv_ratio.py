#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""今日 |(close-open)/vol| / 前10日 |价差和/带符号量|，符号取当日开收。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor


class PvRatioFactor(BaseFactor):
    name = "pv_ratio"
    description = (
        "sign(close-open) * |(close-open)/vol| / "
        "|sum(close-open)/sum(sign(close-open)*vol)|[i-10:i-1]"
    )
    dependencies: List[str] = ["open", "close", "vol"]
    role = "alpha"
    stage = "candidate"
    lookback_days = 40

    def calculate(
        self, data: pd.DataFrame, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        df = data[["ts_code", "trade_date", "open", "close", "vol"]].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(["ts_code", "trade_date"])

        move = df["close"] - df["open"]
        signed = np.sign(move) * df["vol"]
        today = move / df["vol"]
        past_move = move.groupby(df["ts_code"]).transform(
            lambda s: s.rolling(10).sum().shift(1)
        )
        past_sv = signed.groupby(df["ts_code"]).transform(
            lambda s: s.rolling(10).sum().shift(1)
        )
        past = past_move / past_sv
        df["factor_value"] = np.sign(move) * np.abs(today) / np.abs(past)
        bad = (
            (df["vol"] <= 0)
            | (past_sv == 0)
            | (past == 0)
            | ~np.isfinite(df["factor_value"])
        )
        df.loc[bad, "factor_value"] = np.nan

        out = df[["ts_code", "trade_date", "factor_value"]].dropna()
        if start_date:
            start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            out = out[out["trade_date"] >= start]
        return out.reset_index(drop=True)
