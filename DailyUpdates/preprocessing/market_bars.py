#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""个股行情清洗：占位价、复权因子跳变。指数行原样通过。"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger("MarketPreprocess")

MIN_RAW_PRICE = 0.05
MAX_ADJ_STEP = 8.0
_PRICE_COLS = ("open", "high", "low", "close", "pre_close")
_CODE_COLS = ("ts_code", "symbol")
_DATE_COLS = ("trade_date", "date")


def _first_col(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def sanitize_market_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """把占位价和单日 adj_factor 暴跳置空，供复权/收益计算使用。

    不改库。北交所 0.01 开盘、因子一天从 1 跳到上百，都会造成后复权日收益
    几十到上千倍，分层净值 / PnL / 市值分层会被单点拉爆。
    """
    if frame is None or frame.empty:
        return frame
    code_col = _first_col(frame, _CODE_COLS)
    date_col = _first_col(frame, _DATE_COLS)
    if code_col is None or date_col is None:
        return frame

    out = frame.copy()
    codes = out[code_col].astype(str)
    stock = ~codes.str.startswith("index_")
    n_tiny = 0
    n_jump = 0

    price_cols = [c for c in _PRICE_COLS if c in out.columns]
    if price_cols:
        tiny = pd.Series(False, index=out.index)
        for col in price_cols:
            px = pd.to_numeric(out[col], errors="coerce")
            tiny = tiny | (px.notna() & (px <= MIN_RAW_PRICE))
        tiny &= stock
        n_tiny = int(tiny.sum())
        if n_tiny:
            for col in price_cols:
                out.loc[tiny, col] = pd.NA

    if "adj_factor" in out.columns:
        order = out.index.to_numpy()
        ranked = out.sort_values([code_col, date_col], kind="mergesort")
        adj = pd.to_numeric(ranked["adj_factor"], errors="coerce")
        prev = adj.groupby(ranked[code_col].astype(str), sort=False).shift(1)
        ratio = adj / prev
        jump = (
            ~ranked[code_col].astype(str).str.startswith("index_")
            & prev.notna()
            & (prev > 0)
            & adj.notna()
            & (adj > 0)
            & ((ratio >= MAX_ADJ_STEP) | (ratio <= 1.0 / MAX_ADJ_STEP))
        )
        n_jump = int(jump.sum())
        if n_jump:
            ranked.loc[jump, "adj_factor"] = pd.NA
        ranked["_orig_order"] = order
        # sort_values 打乱了位置，用原 index 对齐
        out["adj_factor"] = ranked["adj_factor"].reindex(out.index)

    if n_tiny or n_jump:
        logger.warning(
            "行情预处理：占位价 %s 行，adj_factor 跳变 %s 行（阈值价≤%s 或因子变幅≥×%s）",
            n_tiny,
            n_jump,
            MIN_RAW_PRICE,
            int(MAX_ADJ_STEP),
        )
    return out
