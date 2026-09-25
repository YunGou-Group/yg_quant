#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""换手递推筹码。

低位沉积：近100日收盘中位数 / 平均成本。
筹码收益：当日均价相对平均成本；市场浮盈为正做动量，否则做反转。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numba import njit

_MAX_TURN = 0.95
_MEDIAN_N = 100
# 开源金工：忽略超过 250 个交易日的筹码；递推成本等价于无限历史，
# 输出前至少积累这么多有效成本，避免上市初期噪声。
CHIP_WINDOW = 250


def _turnover(data: pd.DataFrame) -> pd.Series:
    """Tushare turnover_rate / turnover_rate_f 是百分数（0.65 即 0.65%），不是小数。"""
    turn = None
    if "turnover_rate_f" in data.columns:
        turn = pd.to_numeric(data["turnover_rate_f"], errors="coerce")
    if "turnover_rate" in data.columns:
        raw = pd.to_numeric(data["turnover_rate"], errors="coerce")
        turn = raw if turn is None else turn.fillna(raw)
    if turn is None:
        raise ValueError("筹码因子需要 turnover_rate_f 或 turnover_rate")
    tau = turn.where(turn > 0.0) / 100.0
    return tau.clip(upper=_MAX_TURN).fillna(0.0)


@njit
def typical_px(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """成交均价 ≈ (O+H+L+C)/4；缺 OHLC 时回退到收盘。"""
    n = close.shape[0]
    out = np.full(n, np.nan)
    for t in range(n):
        c = close[t]
        if not np.isfinite(c) or c <= 0.0:
            continue
        o = open_[t] if np.isfinite(open_[t]) else c
        h = high[t] if np.isfinite(high[t]) else c
        lo = low[t] if np.isfinite(low[t]) else c
        out[t] = 0.25 * (o + h + lo + c)
    return out


@njit
def avg_cost_from_px(px: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """c_t=(1-T_t)c_{t-1}+T_t px_t；首日 c=当日均价。"""
    n = px.shape[0]
    out = np.full(n, np.nan)
    cost = 0.0
    started = False
    for t in range(n):
        p = px[t]
        if not np.isfinite(p) or p <= 0.0:
            continue
        tau = turn[t]
        if not np.isfinite(tau) or tau < 0.0:
            tau = 0.0
        if tau > _MAX_TURN:
            tau = _MAX_TURN
        if not started:
            cost = p
            started = True
        else:
            cost = (1.0 - tau) * cost + tau * p
        out[t] = cost
    return out


@njit
def avg_cost(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    turn: np.ndarray,
) -> np.ndarray:
    """c_t=(1-T_t)c_{t-1}+T_t (O+H+L+C)/4；首日 c=当日均价。"""
    return avg_cost_from_px(typical_px(open_, high, low, close), turn)


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


def _ohlc_arrays(g: pd.DataFrame):
    return (
        g["open"].to_numpy(dtype=np.float64),
        g["high"].to_numpy(dtype=np.float64),
        g["low"].to_numpy(dtype=np.float64),
        g["close"].to_numpy(dtype=np.float64),
        g["turn"].to_numpy(dtype=np.float64),
    )


def _mask_warmup(values: np.ndarray, cost: np.ndarray, min_bars: int) -> np.ndarray:
    if min_bars <= 1:
        return values
    count = np.cumsum(np.isfinite(cost).astype(np.int64))
    return np.where(count >= int(min_bars), values, np.nan)


def _market_holding_ret(work: pd.DataFrame) -> pd.Series:
    """截面加权均值：优先流通市值，否则等权。"""
    ret = work["holding_ret"]
    if "circ_mv" in work.columns:
        weight = pd.to_numeric(work["circ_mv"], errors="coerce")
        weight = weight.where(weight > 0.0)
        weighted = (ret * weight).groupby(work["trade_date"], sort=False).sum()
        denom = weight.groupby(work["trade_date"], sort=False).sum()
        mkt = weighted / denom.replace(0.0, np.nan)
    else:
        mkt = pd.Series(dtype=np.float64)
    equal = ret.groupby(work["trade_date"], sort=False).mean()
    if mkt.empty:
        return equal
    return mkt.fillna(equal)


def holding_ret(
    data: pd.DataFrame,
    start_date: Optional[str] = None,
    min_bars: int = CHIP_WINDOW,
) -> pd.DataFrame:
    """holding_ret = 当日均价 / 筹码加权成本 - 1。"""
    df = _frame(data)
    parts = []
    for code, g in df.groupby("ts_code", sort=False):
        open_, high, low, close, turn = _ohlc_arrays(g)
        px = typical_px(open_, high, low, close)
        cost = avg_cost_from_px(px, turn)
        values = np.where(
            np.isfinite(px) & np.isfinite(cost) & (cost > 0.0),
            px / cost - 1.0,
            np.nan,
        )
        values = _mask_warmup(values, cost, min_bars)
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


def holding_ret_adj(
    data: pd.DataFrame,
    start_date: Optional[str] = None,
    min_bars: int = CHIP_WINDOW,
) -> pd.DataFrame:
    """holding_ret_adj = holding_ret × sign(mkt_holding_ret)。"""
    raw = holding_ret(data, start_date=None, min_bars=min_bars)
    if raw.empty:
        return raw
    work = raw.rename(columns={"factor_value": "holding_ret"})
    if "circ_mv" in data.columns:
        mv = data[["ts_code"]].copy()
        mv["trade_date"] = pd.to_datetime(data["trade_date"]).dt.strftime("%Y-%m-%d")
        mv["circ_mv"] = pd.to_numeric(data["circ_mv"], errors="coerce")
        work = work.merge(mv, on=["ts_code", "trade_date"], how="left")
    mkt = _market_holding_ret(work)
    work["factor_value"] = work["holding_ret"] * np.sign(
        work["trade_date"].map(mkt)
    )
    return _stack([work[["ts_code", "trade_date", "factor_value"]]], start_date)


def low_deposit(data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
    df = _frame(data)
    parts = []
    for code, g in df.groupby("ts_code", sort=False):
        open_, high, low, close, turn = _ohlc_arrays(g)
        cost = avg_cost(open_, high, low, close, turn)
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
