#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引擎共用的长宽表工具。

只缓存「同一份行情 DataFrame」上的规范化长表和字段宽表。
用弱引用核对对象身份，避免 ``id()`` 回收复用把旧宽表发给新行情。
不改变 BaseFactor 契约：自定义因子仍接收原始 data，可不走这里。
"""

from __future__ import annotations

import threading
import weakref
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_LOCK = threading.RLock()
_FRAME_CACHE: Dict[int, Tuple[weakref.ReferenceType, pd.DataFrame]] = {}
_WIDE_CACHE: Dict[int, Tuple[weakref.ReferenceType, Dict[str, pd.DataFrame]]] = {}


def _alive(cache: dict, obj: pd.DataFrame):
    entry = cache.get(id(obj))
    if entry is None:
        return None
    ref, payload = entry
    if ref() is obj:
        return payload
    cache.pop(id(obj), None)
    return None


def stock_frame(data: pd.DataFrame) -> pd.DataFrame:
    """规范化长表。同一行情对象复用，供价量 / 财务 / 风格引擎共享。"""
    with _LOCK:
        hit = _alive(_FRAME_CACHE, data)
        if hit is not None:
            return hit

    frame = data
    if "ts_code" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "ts_code"})
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "trade_date"})
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame = frame.loc[~frame["ts_code"].str.startswith("index_")]
    frame = frame.sort_values(["ts_code", "trade_date"])

    with _LOCK:
        hit = _alive(_FRAME_CACHE, data)
        if hit is not None:
            return hit
        _FRAME_CACHE.clear()
        _WIDE_CACHE.clear()
        _FRAME_CACHE[id(data)] = (weakref.ref(data), frame)
        return frame


def to_wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in frame.columns:
        raise ValueError(f"缺少行情字段: {field}")
    with _LOCK:
        bucket = _alive(_WIDE_CACHE, frame)
        if bucket is not None and field in bucket:
            return bucket[field]

    wide = frame.pivot(index="trade_date", columns="ts_code", values=field)
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide = wide.sort_index()

    with _LOCK:
        bucket = _alive(_WIDE_CACHE, frame)
        if bucket is None:
            bucket = {}
            _WIDE_CACHE[id(frame)] = (weakref.ref(frame), bucket)
        if field in bucket:
            return bucket[field]
        bucket[field] = wide
        return wide


def events_to_wide(
    events: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    """公告事件按 ann_date 前向填充到交易日日历。"""
    empty = pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float64)
    if events is None or events.empty:
        return empty
    piece = events.dropna(subset=["value"]).copy()
    if piece.empty:
        return empty
    piece["ann_date"] = pd.to_datetime(piece["ann_date"]).dt.strftime("%Y-%m-%d")
    piece = piece.drop_duplicates(["ts_code", "ann_date"], keep="last")
    announced = piece.pivot(index="ann_date", columns="ts_code", values="value")
    calendar = pd.Index(index.astype(str))
    combined = calendar.union(announced.index).sort_values()
    wide = announced.reindex(combined).ffill().reindex(calendar)
    return wide.reindex(columns=columns).astype(np.float64)


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


def clear_panel_cache() -> None:
    with _LOCK:
        _FRAME_CACHE.clear()
        _WIDE_CACHE.clear()
