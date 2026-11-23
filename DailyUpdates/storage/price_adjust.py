"""Shared stock price-adjustment rules for SQLite and market projections."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

ADJ_FACTOR_COLUMN = "adj_factor"
ADJ_PRICE_FIELDS = frozenset(
    {"open", "high", "low", "close", "pre_close", "up_limit", "down_limit"}
)
ADJ_VOLUME_FIELDS = frozenset({"vol"})
ADJUSTABLE_FIELDS = ADJ_PRICE_FIELDS | ADJ_VOLUME_FIELDS
ADJUST_MODES = ("none", "hfq")


def adjusted_panel(
    values: pd.DataFrame, factor: pd.DataFrame, field: str
) -> pd.DataFrame:
    """Apply the repository HFQ contract to one aligned wide panel."""
    usable = factor.where(factor > 0)
    if field in ADJ_PRICE_FIELDS:
        return values * usable
    if field in ADJ_VOLUME_FIELDS:
        return values / usable
    return values


def adjust_long_frame(
    frame: pd.DataFrame, requested: Sequence[str]
) -> pd.DataFrame:
    """Apply HFQ in place to a stock/index long frame and return requested columns."""
    if frame.empty:
        return frame.loc[:, ["ts_code", "trade_date", *requested]]
    factor = pd.to_numeric(frame[ADJ_FACTOR_COLUMN], errors="coerce")
    usable = factor > 0
    stock_rows = ~frame["ts_code"].astype(str).str.startswith("index_")
    scaled = factor.copy()
    scaled.loc[~stock_rows] = 1.0
    scaled.loc[stock_rows & ~usable] = np.nan
    factor_arr = scaled.to_numpy(dtype="float64")
    for field in requested:
        if field in ADJ_PRICE_FIELDS:
            frame[field] = pd.to_numeric(frame[field], errors="coerce") * factor_arr
        elif field in ADJ_VOLUME_FIELDS:
            frame[field] = pd.to_numeric(frame[field], errors="coerce") / factor_arr
    return frame.loc[:, ["ts_code", "trade_date", *requested]]
