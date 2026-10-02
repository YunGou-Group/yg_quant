#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽价量 / 财务因子共用的长宽表工具。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def stock_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data
    if "ts_code" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "ts_code"})
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "trade_date"})
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame = frame.loc[~frame["ts_code"].str.startswith("index_")]
    return frame.sort_values(["ts_code", "trade_date"])


def to_wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in frame.columns:
        raise ValueError(f"缺少行情字段: {field}")
    wide = frame.pivot(index="trade_date", columns="ts_code", values=field)
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide.sort_index()


def wide_to_long(wide: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
    if wide is None or wide.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])
    work = wide
    if start_date:
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        work = work.loc[work.index >= start]
    stacked = work.stack(future_stack=True).rename("factor_value").reset_index()
    stacked.columns = ["trade_date", "ts_code", "factor_value"]
    stacked["factor_value"] = stacked["factor_value"].replace([np.inf, -np.inf], np.nan)
    stacked = stacked.dropna(subset=["factor_value"])
    return stacked.reset_index(drop=True)
