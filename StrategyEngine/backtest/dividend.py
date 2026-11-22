#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""除权日分红送股。只读 stock_dividend，不改成交价。

现金用每股税前 cash_div_tax，在除权日加到隔夜持股上。
股数乘 1+送股+转增。分项都空时才用送转合计 stk_div，避免和分项再加一次。
同一股票同一除权日的多条「实施」加总。预案、股东大会通过、预披露不入账。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DividendBook = Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]

_FIELDS = ("symbol", "ex_date", "div_proc", "cash_div_tax", "stk_bo_rate", "stk_co_rate", "stk_div")


def load_dividend_schedule(
    storage,
    dates: Sequence[str],
    symbols: Sequence[str],
) -> DividendBook:
    """按面板日期和股票，从库里读出实施分红。表不存在则返回空账本。"""
    path = _db_path(storage)
    if path is None or not path.is_file():
        return {}
    if not dates or not symbols:
        return {}
    start = _iso(dates[0])
    end = _iso(dates[-1])
    frame = _read_implemented(path, start, end)
    book = events_to_schedule(frame, dates, symbols)
    return book


def events_to_schedule(
    frame: Optional[pd.DataFrame],
    dates: Sequence[str],
    symbols: Sequence[str],
) -> DividendBook:
    """把分红明细收成除权日 → (股票下标, 每股现金, 股数乘数)。"""
    if frame is None or frame.empty:
        return {}
    work = frame.copy()
    for name in _FIELDS:
        if name not in work.columns:
            work[name] = np.nan
    if "div_proc" in frame.columns:
        proc = work["div_proc"].fillna("").astype(str).str.strip()
        work = work.loc[proc.eq("实施")].copy()
    if work.empty:
        return {}
    work["symbol"] = work["symbol"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    work["ex_date"] = work["ex_date"].map(_iso)
    cash = _nonneg(work["cash_div_tax"])
    bonus = _nonneg(work["stk_bo_rate"])
    transfer = _nonneg(work["stk_co_rate"])
    total = _nonneg(work["stk_div"])
    ratio = np.where((bonus == 0.0) & (transfer == 0.0), total, bonus + transfer)
    keep = np.flatnonzero((cash > 0.0) | (ratio > 0.0))
    if keep.size == 0:
        return {}
    work = work.iloc[keep].copy()
    work["cash"] = cash[keep]
    work["ratio"] = ratio[keep]
    grouped = work.groupby(["ex_date", "symbol"], sort=False)[["cash", "ratio"]].sum()
    allowed_dates = {_iso(day) for day in dates}
    index = {str(symbol): i for i, symbol in enumerate(symbols)}
    buckets: Dict[str, list] = {}
    for (day, symbol), row in grouped.iterrows():
        day_s = str(day)
        if day_s not in allowed_dates:
            continue
        loc = index.get(str(symbol))
        if loc is None:
            continue
        buckets.setdefault(day_s, []).append(
            (loc, float(row["cash"]), 1.0 + float(row["ratio"]))
        )
    book: DividendBook = {}
    for day, rows in buckets.items():
        rows.sort(key=lambda item: item[0])
        idx = np.asarray([item[0] for item in rows], dtype=np.int64)
        cash_ps = np.asarray([item[1] for item in rows], dtype=np.float64)
        mult = np.asarray([item[2] for item in rows], dtype=np.float64)
        book[day] = (idx, cash_ps, mult)
    return book


def _read_implemented(path: Path, start: str, end: str) -> pd.DataFrame:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        found = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'stock_dividend'"
        ).fetchone()
        if not found:
            return pd.DataFrame(columns=list(_FIELDS))
        return pd.read_sql_query(
            """
            SELECT symbol, ex_date, div_proc, cash_div_tax, stk_bo_rate, stk_co_rate, stk_div
            FROM stock_dividend
            WHERE div_proc = '实施' AND ex_date >= ? AND ex_date <= ?
            """,
            connection,
            params=(start, end),
        )
    finally:
        connection.close()


def _db_path(storage) -> Optional[Path]:
    raw = getattr(storage, "db_path", None)
    if raw is None:
        return None
    return Path(raw).expanduser()


def _iso(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nat", "none", "nan"}:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _nonneg(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
