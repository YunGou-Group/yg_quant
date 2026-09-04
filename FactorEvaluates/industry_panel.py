#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""申万行业：按 in_date/out_date 铺成 date × symbol 代码表。不进 Bin。"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("FactorEvaluates")

INDUSTRY_SRC = "SW2021"


def load_l1_code_panel(
    storage,
    dates: pd.Index,
    symbols: pd.Index,
    src: str = INDUSTRY_SRC,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """返回申万一级行业代码宽表，以及 code → 中文名。"""
    return load_code_panel(storage, dates, symbols, src=src, level="l1")


def load_code_panel(
    storage,
    dates: pd.Index,
    symbols: pd.Index,
    src: str = INDUSTRY_SRC,
    level: str = "l1",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """返回行业代码宽表（level=l1/l2/l3），以及 code → 中文名。"""
    dates = pd.Index(pd.to_datetime(dates).strftime("%Y-%m-%d"))
    symbols = pd.Index([str(s) for s in symbols])
    members = _read_members(storage, src, level)
    if members.empty:
        logger.warning("industry_member 为空，level=%s 无行业代码", level)
        empty = pd.DataFrame(np.nan, index=dates, columns=symbols)
        return empty, {}
    labels = (
        members.dropna(subset=["industry_code"])
        .drop_duplicates("industry_code", keep="last")
        .set_index("industry_code")["industry_name"]
        .astype(str)
        .to_dict()
    )
    panel = _fill_code_panel(members, dates, symbols)
    return panel, {str(k): str(v) for k, v in labels.items() if str(k)}


def _read_members(storage, src: str, level: str = "l1") -> pd.DataFrame:
    frame = storage.read_industry_members(src=src, level=level, is_new=None)
    if frame.empty:
        frame = storage.read_industry_members(src=None, level=level, is_new=None)
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["industry_code"] = out["industry_code"].astype(str)
    if "industry_name" not in out.columns:
        out["industry_name"] = out["industry_code"]
    return out


def _fill_code_panel(
    members: pd.DataFrame, dates: pd.Index, symbols: pd.Index
) -> pd.DataFrame:
    date_ts = pd.to_datetime(dates)
    pos = {str(sym): i for i, sym in enumerate(symbols)}
    grid = np.full((len(dates), len(symbols)), None, dtype=object)
    work = members.dropna(subset=["symbol", "industry_code"]).copy()
    work["in_ts"] = pd.to_datetime(work.get("in_date"), errors="coerce")
    work["out_ts"] = pd.to_datetime(work.get("out_date"), errors="coerce")
    work = work.sort_values(["in_ts"], na_position="first")
    start0 = date_ts.min()
    open_end = date_ts.max() + pd.Timedelta(days=1)
    for rec in work.itertuples(index=False):
        j = pos.get(str(rec.symbol))
        if j is None:
            continue
        start = rec.in_ts if pd.notna(rec.in_ts) else start0
        stop = rec.out_ts if pd.notna(rec.out_ts) else open_end
        sl = (date_ts >= start) & (date_ts < stop)
        grid[np.asarray(sl), j] = str(rec.industry_code)
    return pd.DataFrame(grid, index=dates, columns=symbols)
