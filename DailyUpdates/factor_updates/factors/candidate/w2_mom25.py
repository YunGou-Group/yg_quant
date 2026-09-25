#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五福线性 W²：近 25 日加权对数价格回归，年化斜率 × R²。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

LOOKBACK_DAYS = 25
WINDOW = LOOKBACK_DAYS + 1


def w2_mom25_array(close: np.ndarray, lookback_days: int = LOOKBACK_DAYS) -> np.ndarray:
    """与 Strategies.wufu.momentum_score 相同：最近 lookback+1 个有效正价格。"""
    px = np.asarray(close, dtype=np.float64)
    out = np.full(px.size, np.nan, dtype=np.float64)
    need = int(lookback_days) + 1
    if px.size < need:
        return out
    ok = np.isfinite(px) & (px > 0.0)
    loc = np.flatnonzero(ok)
    if loc.size < need:
        return out
    logp = np.log(px[loc])
    x = np.arange(need, dtype=np.float64)
    weights = np.linspace(1.0, 2.0, need)
    w2 = weights ** 2
    w_sum = float(w2.sum())
    x_bar = float((w2 * x).sum() / w_sum)
    dx = x - x_bar
    var_x = float((w2 * dx ** 2).sum())
    if var_x == 0.0:
        out[loc[need - 1 :]] = 0.0
        return out
    windows = np.lib.stride_tricks.sliding_window_view(logp, need)
    y_bar = windows @ w2 / w_sum
    slope = (windows * (w2 * dx)).sum(axis=1) / var_x
    intercept = y_bar - slope * x_bar
    y_pred = slope[:, None] * x + intercept[:, None]
    ss_res = (weights * (windows - y_pred) ** 2).sum(axis=1)
    y_mean = windows.mean(axis=1)
    ss_tot = (weights * (windows - y_mean[:, None]) ** 2).sum(axis=1)
    r2 = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 0.0)
    annualized = np.exp(slope * 250.0) - 1.0
    out[loc[need - 1 :]] = annualized * r2
    return out


class W2Mom25Factor(BaseFactor):
    name = "w2_mom25"
    description = (
        "五福线性 W²：近25日对数收盘对时间做 w² 加权回归，"
        "因子=年化斜率×R²（权重 linspace 1→2）"
    )
    dependencies: List[str] = ["close"]
    role = "alpha"
    stage = "candidate"
    lookback_days = 60

    def calculate(
        self, data: pd.DataFrame, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        df = data[["ts_code", "trade_date", "close"]].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        close = df["close"].to_numpy(dtype=np.float64)
        values = np.full(len(df), np.nan, dtype=np.float64)
        for idx in df.groupby("ts_code", sort=False).indices.values():
            pos = np.asarray(idx)
            values[pos] = w2_mom25_array(close[pos])
        df["factor_value"] = values
        out = df.loc[
            np.isfinite(df["factor_value"]),
            ["ts_code", "trade_date", "factor_value"],
        ]
        if start_date:
            start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            out = out[out["trade_date"] >= start]
        return out.reset_index(drop=True)
