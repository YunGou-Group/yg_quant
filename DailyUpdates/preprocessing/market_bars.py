#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""个股行情入库清洗：占位价、OHLC、成交量、复权因子跳变。指数行原样通过。"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MarketPreprocess")

MIN_RAW_PRICE = 0.05
MAX_ADJ_STEP = 8.0
_PRICE_COLS = ("open", "high", "low", "close", "pre_close")
_OHLC_COLS = ("open", "high", "low", "close")
_CODE_COLS = ("ts_code", "symbol")
_DATE_COLS = ("trade_date", "date")


def _first_col(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _stock_mask(codes: pd.Series) -> pd.Series:
    return ~codes.astype(str).str.startswith("index_")


def sanitize_market_bars(
    frame: pd.DataFrame,
    *,
    prev_adj: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """把脏点置空后再写入。指数行不改价格/因子。

    新交易日 INSERT 会落下 NaN。同一天重拉时 storage upsert 用 COALESCE，
    NaN 清不掉库里旧的 0.01，旧脏点需要全量重建。
    """
    if frame is None or frame.empty:
        return frame
    code_col = _first_col(frame, _CODE_COLS)
    date_col = _first_col(frame, _DATE_COLS)
    if code_col is None or date_col is None:
        return frame

    out = frame.copy()
    stock = _stock_mask(out[code_col])
    n_tiny = n_ohlc = n_vol = n_adj_sign = n_jump = 0

    numeric_cols = [
        col
        for col in out.columns
        if col not in {code_col, date_col} and pd.api.types.is_numeric_dtype(out[col])
    ]
    if numeric_cols and stock.any():
        work = out.loc[stock, numeric_cols]
        coerced = work.apply(lambda col: pd.to_numeric(col, errors="coerce"))
        bad = ~np.isfinite(coerced.to_numpy(dtype="float64")) & work.notna().to_numpy()
        if bad.any():
            cleaned = coerced.mask(~np.isfinite(coerced))
            out.loc[stock, numeric_cols] = cleaned

    out = out.drop_duplicates([code_col, date_col], keep="last")
    stock = _stock_mask(out[code_col])

    price_cols = [c for c in _PRICE_COLS if c in out.columns]
    if price_cols and stock.any():
        tiny = pd.Series(False, index=out.index)
        for col in price_cols:
            px = pd.to_numeric(out[col], errors="coerce")
            tiny = tiny | (px.notna() & (px <= MIN_RAW_PRICE))
        tiny &= stock
        n_tiny = int(tiny.sum())
        if n_tiny:
            for col in price_cols:
                out.loc[tiny, col] = pd.NA

    ohlc = [c for c in _OHLC_COLS if c in out.columns]
    if {"high", "low"}.issubset(ohlc) and ("open" in ohlc or "close" in ohlc) and stock.any():
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        open_ = pd.to_numeric(out["open"], errors="coerce") if "open" in ohlc else None
        close = pd.to_numeric(out["close"], errors="coerce") if "close" in ohlc else None
        if open_ is not None and close is not None:
            body_hi = np.maximum(open_, close)
            body_lo = np.minimum(open_, close)
        elif open_ is not None:
            body_hi = body_lo = open_
        else:
            body_hi = body_lo = close
        broken = stock & (
            (high.notna() & body_hi.notna() & (high < body_hi))
            | (low.notna() & body_lo.notna() & (low > body_lo))
        )
        n_ohlc = int(broken.sum())
        if n_ohlc:
            for col in ohlc:
                out.loc[broken, col] = pd.NA

    for col in ("vol", "amount"):
        if col not in out.columns:
            continue
        val = pd.to_numeric(out[col], errors="coerce")
        bad = stock & val.notna() & (val <= 0)
        n_vol += int(bad.sum())
        if bad.any():
            out.loc[bad, col] = pd.NA

    if "adj_factor" in out.columns:
        adj = pd.to_numeric(out["adj_factor"], errors="coerce")
        nonpos = stock & adj.notna() & (adj <= 0)
        n_adj_sign = int(nonpos.sum())
        if n_adj_sign:
            out.loc[nonpos, "adj_factor"] = pd.NA
        ranked = out.sort_values([code_col, date_col], kind="mergesort")
        adj = pd.to_numeric(ranked["adj_factor"], errors="coerce")
        codes = ranked[code_col].astype(str)
        prev = adj.groupby(codes, sort=False).shift(1)
        if prev_adj is not None and len(prev_adj):
            mapped = codes.map(pd.to_numeric(prev_adj, errors="coerce"))
            prev = prev.where(prev.notna(), mapped)
        ratio = adj / prev
        jump = (
            ~codes.str.startswith("index_")
            & prev.notna()
            & (prev > 0)
            & adj.notna()
            & (adj > 0)
            & ((ratio >= MAX_ADJ_STEP) | (ratio <= 1.0 / MAX_ADJ_STEP))
        )
        n_jump = int(jump.sum())
        if n_jump:
            ranked.loc[jump, "adj_factor"] = pd.NA
        out["adj_factor"] = ranked["adj_factor"].reindex(out.index)

    if n_tiny or n_ohlc or n_vol or n_adj_sign or n_jump:
        logger.warning(
            "行情预处理：占位价 %s 行，OHLC %s 行，成交 %s 行，"
            "adj≤0 %s 行，adj 跳变 %s 行（价≤%s 或因子变幅≥×%s）",
            n_tiny,
            n_ohlc,
            n_vol,
            n_adj_sign,
            n_jump,
            MIN_RAW_PRICE,
            int(MAX_ADJ_STEP),
        )
    return out


def load_prev_adj(storage: Any, before_date: str) -> Optional[pd.Series]:
    """库内 ``before_date`` 之前最近一个交易日的 adj_factor，index 为代码。"""
    if storage is None or not before_date:
        return None
    try:
        available = set(storage.get_market_fields())
        if "adj_factor" not in available:
            return None
        day = str(before_date).replace("-", "")
        if len(day) >= 8:
            iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        else:
            iso = str(before_date)
        calendar = storage.list_trade_dates(end_date=iso)
        prior = [d for d in calendar if str(d) < iso]
        if not prior:
            return None
        prev_day = max(prior)
        frame = storage.read_market_data(
            fields=["adj_factor"],
            start_date=prev_day,
            end_date=prev_day,
            adjust="none",
        )
    except Exception:
        logger.debug("无法加载昨日 adj_factor", exc_info=True)
        return None
    if frame is None or frame.empty or "adj_factor" not in frame.columns:
        return None
    code_col = _first_col(frame, _CODE_COLS)
    if code_col is None:
        return None
    values = pd.to_numeric(frame["adj_factor"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame[code_col].astype(str), name="adj_factor")
    series = series[~series.index.str.startswith("index_")]
    series = series[series.notna() & (series > 0)]
    return series if not series.empty else None
