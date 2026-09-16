#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""换手递推筹码。低位=近100日收盘中位数/平均成本。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numba import njit

_MAX_TURN = 0.95
_MEDIAN_N = 100


def _turnover(data: pd.DataFrame) -> pd.Series:
    turn = None
    if "turnover_rate_f" in data.columns:
        turn = pd.to_numeric(data["turnover_rate_f"], errors="coerce")
        turn = turn.where(turn <= 1.0, turn / 100.0)
    if "turnover_rate" in data.columns:
        raw = pd.to_numeric(data["turnover_rate"], errors="coerce")
        raw = raw.where(raw <= 1.0, raw / 100.0)
        turn = raw if turn is None else turn.fillna(raw)
    if turn is None:
        raise ValueError("筹码因子需要 turnover_rate_f 或 turnover_rate")
    return turn


@njit
def avg_cost(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    turn: np.ndarray,
) -> np.ndarray:
    """c_t=(1-T_t)c_{t-1}+T_t (O+H+L+C)/4；首日 c=当日均价。"""
    n = close.shape[0]
    out = np.full(n, np.nan)
    cost = 0.0
    started = False
    for t in range(n):
        c = close[t]
        if not np.isfinite(c) or c <= 0.0:
            continue
        o = open_[t] if np.isfinite(open_[t]) else c
        h = high[t] if np.isfinite(high[t]) else c
        lo = low[t] if np.isfinite(low[t]) else c
        px = 0.25 * (o + h + lo + c)
        tau = turn[t]
        if not np.isfinite(tau) or tau < 0.0:
            tau = 0.0
        if tau > _MAX_TURN:
            tau = _MAX_TURN
        if not started:
            cost = px
            started = True
        else:
            cost = (1.0 - tau) * cost + tau * px
        out[t] = cost
    return out


def _frame(data: pd.DataFrame) -> pd.DataFrame:
    df = data[["ts_code", "trade_date", "open", "high", "low", "close"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df["turn"] = _turnover(data).to_numpy()
    return df.sort_values(["ts_code", "trade_date"])


def _stack(parts, start_date: Optional[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])
    out = pd.concat(parts, ignore_index=True).dropna(subset=["factor_value"])
    if start_date:
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        out = out[out["trade_date"] >= start]
    return out.reset_index(drop=True)


def low_deposit(data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
    df = _frame(data)
    parts = []
    for code, g in df.groupby("ts_code", sort=False):
        cost = avg_cost(
            g["open"].to_numpy(dtype=np.float64),
            g["high"].to_numpy(dtype=np.float64),
            g["low"].to_numpy(dtype=np.float64),
            g["close"].to_numpy(dtype=np.float64),
            g["turn"].to_numpy(dtype=np.float64),
        )
        mid = g["close"].rolling(_MEDIAN_N, min_periods=_MEDIAN_N).median().to_numpy()
        values = np.where(
            np.isfinite(mid) & np.isfinite(cost) & (mid > 0.0) & (cost > 0.0),
            mid / cost,
            np.nan,
        )
        parts.append(
            pd.DataFrame(
                {
                    "ts_code": code,
                    "trade_date": g["trade_date"].to_numpy(),
                    "factor_value": values,
                }
            )
        )
    return _stack(parts, start_date)
