#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""净值指标、成交明细、可选基准超额、回测快照 JSON。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .engine import EngineResult

_PathLike = Union[str, Path]
_VOL_EPS = 1e-12


def summarize(
    result: EngineResult,
    *,
    benchmark: Optional[pd.Series] = None,
    n_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    nav = np.asarray(result.nav, dtype=np.float64)
    n = int(nav.size)
    out: Dict[str, Any] = {
        "n_days": n,
        "start": result.dates[0] if result.dates else None,
        "end": result.dates[-1] if result.dates else None,
        "nav_start": float(nav[0]) if n else None,
        "nav_end": float(nav[-1]) if n else None,
        "total_return": None,
        "annualized": None,
        "volatility": None,
        "sharpe": None,
        "max_drawdown": None,
        "calmar": None,
        "turnover_mean": None,
        "fee_total": None,
        "gross_return": None,
        "excess_annualized": None,
        "risk_free_rate": float(risk_free_rate),
    }
    if n < 2 or nav[0] <= 0:
        return out
    out["total_return"] = float(nav[-1] / nav[0] - 1.0)
    span = max(n - 1, 1)
    out["annualized"] = float((nav[-1] / nav[0]) ** (n_per_year / span) - 1.0)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.where(peak > 0, peak, np.nan) - 1.0
    out["max_drawdown"] = float(np.nanmin(dd))
    prev = np.where(nav[:-1] > 0, nav[:-1], np.nan)
    rets = nav[1:] / prev - 1.0
    rets = rets[np.isfinite(rets)]
    if rets.size >= 2:
        out["volatility"] = float(np.std(rets, ddof=1) * np.sqrt(n_per_year))
    elif rets.size == 1:
        out["volatility"] = 0.0
    vol = out["volatility"]
    if vol is not None and vol > _VOL_EPS:
        out["sharpe"] = float((out["annualized"] - float(risk_free_rate)) / vol)
    dd_val = out["max_drawdown"]
    if dd_val is not None and dd_val < 0:
        out["calmar"] = float(out["annualized"] / abs(dd_val))
    turn = np.asarray(result.turnover, dtype=np.float64)
    out["turnover_mean"] = float(np.mean(turn)) if turn.size else 0.0
    fee_total = 0.0
    has_fee = False
    for item in list(result.fills) + list(getattr(result, "forced_fills", ())):
        if item is None:
            continue
        fee = getattr(item, "fee", None)
        if fee is None:
            continue
        has_fee = True
        fee_total += float(fee)
    if has_fee:
        out["fee_total"] = fee_total
        if n >= 2 and nav[0] > 0:
            out["gross_return"] = float((nav[-1] + fee_total) / nav[0] - 1.0)
    if benchmark is not None and not benchmark.empty:
        bench = _aligned_nav(benchmark, result.dates)
        if bench is not None and bench[0] > 0:
            b_ann = float((bench[-1] / bench[0]) ** (n_per_year / span) - 1.0)
            out["excess_annualized"] = out["annualized"] - b_ann
    return out


def write_run_snapshot(path: _PathLike, payload: Dict[str, Any]) -> Path:
    """回测关键结果快照，UTF-8 JSON。"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest


def trades_frame(result: EngineResult) -> pd.DataFrame:
    rows = []
    for item in list(result.fills) + list(getattr(result, "forced_fills", ())):
        if item is None:
            continue
        rows.extend(item.to_rows())
    if not rows:
        return pd.DataFrame(
            columns=["date", "symbol", "side", "intended", "filled", "price", "reason", "fee"]
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def _aligned_nav(close: pd.Series, dates) -> Optional[np.ndarray]:
    series = pd.to_numeric(close, errors="coerce").copy()
    series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
    series = series[~series.index.duplicated(keep="last")].reindex(list(dates))
    values = series.to_numpy(dtype=np.float64)
    if not np.isfinite(values).any():
        return None
    first = next((v for v in values if np.isfinite(v) and v > 0), None)
    if first is None:
        return None
    filled = pd.Series(values).ffill().to_numpy(dtype=np.float64)
    return filled / first


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return str(value)
