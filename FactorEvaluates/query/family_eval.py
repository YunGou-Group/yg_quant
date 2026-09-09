#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""家族三类聚合：方向无关 mean、带符号 abs_mean、两腿按 rank_ic 择腿。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..batch.batch_result_writer import read_table
from ..batch.family_key import family_of
from .datasets import run_dir
from .factors import _summary_frame


_UNSIGNED_PREFIXES = (
    "coverage_rate",
    "missing_rate",
    "missing_samples",
    "total_samples",
    "unique_rate",
    "extreme_value_ratio",
    "factor_mean",
    "factor_std",
    "factor_min",
    "factor_max",
    "factor_autocorr",
    "iqr",
    "quantile_turnover",
)
_SIGNED_PREFIXES = (
    "rank_ic",
    "ic",
    "nonlinear_ic",
    "mi_ic",
    "pure_ic",
    "weighted_ic",
    "factor_returns",
    "size_ls",
    "size_rank_ls",
    "quantile_spread",
)
_LONG_LEGS = ("long_only_return", "weighted_long_pnl", "size_long", "size_rank_long")
_SHORT_LEGS = {
    "long_only_return": "short_only_return",
    "weighted_long_pnl": "weighted_short_pnl",
    "size_long": "size_short",
    "size_rank_long": "size_rank_short",
}
_WINDOW_METRICS = (
    "rank_ic",
    "ic",
    "coverage_rate",
    "quantile_spread",
    "long_short_term_consistency",
    "rolling_ic",
    "weighted_pnl",
    "long_only_return",
    "factor_returns",
    "size_ls",
)


def _col(frame: pd.DataFrame, metric: str) -> Optional[str]:
    for key in (f"{metric}_mean", metric, f"{metric}_ir"):
        if key in frame.columns:
            return key
    return None


def family_eval(
    run_id: Optional[str] = None,
    date_range=None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(date_range, dict):
        start = start or date_range.get("start")
        end = end or date_range.get("end")
    if start or end:
        frame = _window_summary(run_id, start, end)
    else:
        frame = _summary_frame(run_id)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, name in enumerate(frame["factor_name"].astype(str)):
        groups[family_of(name)].append(i)
    items = []
    for fam, idxs in sorted(groups.items()):
        sub = frame.iloc[idxs]
        row: Dict[str, Any] = {"family": fam, "n": int(len(idxs))}
        rank_col = _col(sub, "rank_ic")
        direction = None
        if rank_col:
            vals = pd.to_numeric(sub[rank_col], errors="coerce")
            direction = np.sign(vals.to_numpy(dtype=np.float64))
            direction[direction == 0] = 1.0
            best = vals.abs().idxmax() if vals.notna().any() else None
            if best is not None:
                row["best_member"] = str(sub.loc[best, "factor_name"])
        for metric in _UNSIGNED_PREFIXES:
            col = _col(sub, metric)
            if col:
                row[f"{metric}_mean"] = _mean(sub[col])
        for metric in _SIGNED_PREFIXES:
            col = _col(sub, metric)
            if col:
                vals = pd.to_numeric(sub[col], errors="coerce")
                row[f"{metric}_mean"] = _mean(vals)
                row[f"{metric}_abs_mean"] = _mean(vals.abs())
        if direction is not None:
            for long_name in _LONG_LEGS:
                long_col = _col(sub, long_name)
                short_col = _col(sub, _SHORT_LEGS.get(long_name, ""))
                if not long_col:
                    continue
                long_v = pd.to_numeric(sub[long_col], errors="coerce").to_numpy(dtype=np.float64)
                if short_col:
                    short_v = pd.to_numeric(sub[short_col], errors="coerce").to_numpy(dtype=np.float64)
                    chosen = np.where(direction >= 0, long_v, short_v)
                else:
                    chosen = long_v * direction
                row[f"{long_name}@dir_mean"] = _mean(pd.Series(chosen))
        items.append(row)
    return {"items": items, "start": start, "end": end, "run_id": run_id}


def _window_summary(run_id: Optional[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """自定义时段从 daily 重算均值；缺列则退回 summary 概览。"""
    base = _summary_frame(run_id)
    names = base["factor_name"].astype(str)
    daily_dir = run_dir(run_id) / "daily"
    extra: Dict[str, List[Any]] = {"factor_name": names.tolist()}
    loaded = 0
    for metric in _WINDOW_METRICS:
        path = daily_dir / f"{metric}.parquet"
        if not path.is_file():
            continue
        try:
            frame = read_table(path)
        except Exception:
            continue
        idx = pd.to_datetime(frame.index, errors="coerce")
        mask = pd.Series(True, index=frame.index)
        if start:
            mask &= idx >= pd.Timestamp(start)
        if end:
            mask &= idx <= pd.Timestamp(end)
        sliced = frame.loc[mask]
        if sliced.empty:
            continue
        means = sliced.mean(axis=0, numeric_only=True)
        extra[f"{metric}_mean"] = [None if _nan(means.get(n)) else float(means.get(n)) for n in names]
        loaded += 1
    if loaded < 2:
        return base
    return pd.DataFrame(extra)


def _mean(series) -> Optional[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _nan(value: Any) -> bool:
    try:
        return value != value or value is None
    except Exception:
        return False
