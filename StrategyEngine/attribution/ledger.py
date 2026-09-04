#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从成交明细还原每日股数，再拼 Brison 所需的持仓 / 交易 / 日收益表。"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CASH_BUCKET = "现金"
UNCLASSIFIED = "未分类"
_QTY_EPS = 1e-9


def normalize_day(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.replace("/", "-")[:10]


def load_trades(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "symbol", "side", "filled", "price", "fee"]
        )
    out = pd.DataFrame(
        {
            "date": frame["date"].map(normalize_day),
            "symbol": frame["symbol"].astype(str),
            "side": frame["side"].astype(str).str.lower() if "side" in frame else "buy",
            "filled": pd.to_numeric(frame.get("filled", 0.0), errors="coerce").fillna(0.0),
            "price": pd.to_numeric(frame.get("price"), errors="coerce"),
            "fee": pd.to_numeric(frame.get("fee", 0.0), errors="coerce").fillna(0.0)
            if "fee" in frame.columns
            else 0.0,
        }
    )
    return out[out["date"] != ""].copy()


def load_equity(path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        return pd.DataFrame(columns=["date", "nav", "cash"])
    dates = [normalize_day(idx) for idx in frame.index]
    out = pd.DataFrame({"date": dates})
    out["nav"] = pd.to_numeric(frame["nav"], errors="coerce").to_numpy() if "nav" in frame else np.nan
    if "cash" in frame.columns:
        out["cash"] = pd.to_numeric(frame["cash"], errors="coerce").to_numpy()
    else:
        out["cash"] = np.nan
    return out[out["date"] != ""].copy()


def signed_fill(side: str, filled: float) -> float:
    qty = float(filled)
    key = str(side or "").strip().lower()
    if key in {"sell", "s", "short"}:
        return -abs(qty)
    if key in {"buy", "b", "long", ""}:
        return abs(qty) if qty >= 0 else qty
    return qty


def positions_by_date(
    dates: Sequence[str],
    trades: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """每个交易日开盘成交之后的股数。"""
    by_day: Dict[str, List[Tuple[str, float]]] = {}
    if trades is not None and not trades.empty:
        for rec in trades.itertuples(index=False):
            day = normalize_day(getattr(rec, "date", ""))
            symbol = str(getattr(rec, "symbol", "") or "")
            if not day or not symbol:
                continue
            qty = signed_fill(getattr(rec, "side", ""), getattr(rec, "filled", 0.0))
            by_day.setdefault(day, []).append((symbol, qty))
    held: Dict[str, float] = {}
    out: Dict[str, Dict[str, float]] = {}
    for day in dates:
        for symbol, qty in by_day.get(day, ()):
            held[symbol] = held.get(symbol, 0.0) + qty
            if abs(held[symbol]) < _QTY_EPS:
                held.pop(symbol, None)
        out[day] = {key: float(val) for key, val in held.items() if abs(val) >= _QTY_EPS}
    return out


def previous_positions(
    dates: Sequence[str],
    positions: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    prev: Dict[str, Dict[str, float]] = {}
    last: Dict[str, float] = {}
    for day in dates:
        prev[day] = dict(last)
        last = dict(positions.get(day) or {})
    return prev


def trade_rows(
    trades: pd.DataFrame,
    mark_close: Mapping[str, Mapping[str, float]],
) -> List[dict]:
    rows: List[dict] = []
    if trades is None or trades.empty:
        return rows
    for rec in trades.itertuples(index=False):
        day = normalize_day(getattr(rec, "date", ""))
        symbol = str(getattr(rec, "symbol", "") or "")
        filled = float(getattr(rec, "filled", 0.0) or 0.0)
        price = getattr(rec, "price", np.nan)
        if not day or not symbol or abs(filled) < _QTY_EPS:
            continue
        mark = _price(mark_close, day, symbol)
        exec_px = float(price) if price is not None and np.isfinite(price) else mark
        if exec_px is None or not np.isfinite(exec_px):
            continue
        rows.append(
            {
                "period": day,
                "asset_id": symbol,
                "asset_class": "stock",
                "side": str(getattr(rec, "side", "buy") or "buy"),
                "quantity": abs(filled),
                "execution_price": float(exec_px),
                "mark_price": float(mark if mark is not None and np.isfinite(mark) else exec_px),
                "commission": float(getattr(rec, "fee", 0.0) or 0.0),
                "tax": 0.0,
            }
        )
    return rows


def holding_rows(
    dates: Sequence[str],
    start_positions: Mapping[str, Mapping[str, float]],
    close: Mapping[str, Mapping[str, float]],
) -> List[dict]:
    rows: List[dict] = []
    prev_day = None
    for day in dates:
        start_px_day = prev_day
        held = start_positions.get(day) or {}
        for symbol, qty in held.items():
            if abs(qty) < _QTY_EPS:
                continue
            start_px = _price(close, start_px_day, symbol) if start_px_day else None
            end_px = _price(close, day, symbol)
            if start_px is None or end_px is None:
                continue
            rows.append(
                {
                    "period": day,
                    "asset_id": symbol,
                    "asset_class": "stock",
                    "start_quantity": float(qty),
                    "start_price": float(start_px),
                    "end_price": float(end_px),
                }
            )
        prev_day = day
    return rows


def close_wealth(
    day: str,
    cash: float,
    qty: Mapping[str, float],
    close: Mapping[str, Mapping[str, float]],
) -> Optional[float]:
    marked = 0.0
    for symbol, shares in qty.items():
        px = _price(close, day, symbol)
        if px is None:
            return None
        marked += float(shares) * float(px)
    if not np.isfinite(cash):
        return None
    return float(cash) + marked


def period_returns(
    dates: Sequence[str],
    cash_by_date: Mapping[str, float],
    positions: Mapping[str, Mapping[str, float]],
    close: Mapping[str, Mapping[str, float]],
    *,
    initial_cash: float,
    benchmark_close: Optional[Mapping[str, float]] = None,
) -> List[dict]:
    rows: List[dict] = []
    prev_wealth = float(initial_cash)
    prev_bench = None
    for day in dates:
        cash = cash_by_date.get(day)
        if cash is None or not np.isfinite(cash):
            cash = prev_wealth
        wealth = close_wealth(day, float(cash), positions.get(day) or {}, close)
        if wealth is None or prev_wealth <= 0:
            prev_wealth = wealth if wealth is not None else prev_wealth
            continue
        port_r = wealth / prev_wealth - 1.0
        bench_r = 0.0
        if benchmark_close:
            px = benchmark_close.get(day)
            if px is not None and prev_bench is not None and prev_bench > 0 and px > 0:
                bench_r = float(px) / float(prev_bench) - 1.0
            if px is not None and px > 0:
                prev_bench = float(px)
        rows.append(
            {
                "period": day,
                "portfolio_return": float(port_r),
                "benchmark_return": float(bench_r),
                "nav_begin": float(prev_wealth),
            }
        )
        prev_wealth = wealth
    return rows


def beginning_weights(
    dates: Sequence[str],
    start_positions: Mapping[str, Mapping[str, float]],
    close: Mapping[str, Mapping[str, float]],
    cash_by_date: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    """每个归因日用期初市值权重（昨收 × 昨持股 / 期初 NAV），与行业 BF 同一套账。"""
    out: Dict[str, Dict[str, float]] = {}
    prev_day = None
    prev_cash = None
    for day in dates:
        if prev_day is None:
            prev_day = day
            prev_cash = cash_by_date.get(day)
            continue
        qty = start_positions.get(day) or {}
        cash_v = float(prev_cash or 0.0)
        weights = _rescale_to_mv(qty, close, prev_day, cash_v)
        if weights:
            out[str(day)] = weights
        prev_day = day
        prev_cash = cash_by_date.get(day, prev_cash)
    return out


def industry_sides(
    dates: Sequence[str],
    start_positions: Mapping[str, Mapping[str, float]],
    close: Mapping[str, Mapping[str, float]],
    cash_by_date: Mapping[str, float],
    industry_name: Mapping[str, Mapping[str, str]],
    *,
    benchmark_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, List[dict]]:
    portfolio: List[dict] = []
    benchmark: List[dict] = []
    prev_day = None
    prev_cash = None
    for day in dates:
        if prev_day is None:
            prev_day = day
            prev_cash = cash_by_date.get(day)
            continue
        labels = industry_name.get(prev_day) or industry_name.get(day) or {}
        qty = start_positions.get(day) or {}
        cash_v = float(prev_cash or 0.0)
        port_w = _rescale_to_mv(qty, close, prev_day, cash_v)
        port_r = {
            symbol: _stock_return(close, prev_day, day, symbol) for symbol in list(port_w)
        }
        port_r = {key: val for key, val in port_r.items() if val is not None}
        port_w = {key: port_w[key] for key in port_r}
        cash_w = max(0.0, 1.0 - sum(port_w.values()))
        portfolio.extend(_bucket_rows(day, port_w, port_r, labels, cash_w))
        if benchmark_weights:
            bw = dict(benchmark_weights.get(prev_day) or benchmark_weights.get(day) or {})
            br = {
                sym: _stock_return(close, prev_day, day, sym)
                for sym in bw
            }
            br = {k: v for k, v in br.items() if v is not None}
            bench_stock_w = {k: float(bw[k]) for k in br if float(bw.get(k, 0) or 0) > 0}
            bench_cash = max(0.0, 1.0 - sum(bench_stock_w.values()))
            benchmark.extend(_bucket_rows(day, bench_stock_w, br, labels, bench_cash))
        prev_day = day
        prev_cash = cash_by_date.get(day, prev_cash)
    return {"portfolio": portfolio, "benchmark": benchmark}


def _rescale_to_mv(
    qty: Mapping[str, float],
    close: Mapping[str, Mapping[str, float]],
    day: str,
    cash: float,
) -> Dict[str, float]:
    parts: Dict[str, float] = {}
    total = float(cash) if np.isfinite(cash) else 0.0
    for symbol, shares in qty.items():
        px = _price(close, day, symbol)
        if px is None or abs(shares) < _QTY_EPS:
            continue
        mv = float(shares) * float(px)
        parts[symbol] = mv
        total += mv
    if total <= 0:
        return {}
    return {sym: mv / total for sym, mv in parts.items()}


def _stock_return(
    close: Mapping[str, Mapping[str, float]],
    start_day: str,
    end_day: str,
    symbol: str,
) -> Optional[float]:
    a = _price(close, start_day, symbol)
    b = _price(close, end_day, symbol)
    if a is None or b is None or a <= 0:
        return None
    return float(b) / float(a) - 1.0


def _bucket_rows(
    day: str,
    weights: Mapping[str, float],
    returns: Mapping[str, float],
    labels: Mapping[str, str],
    cash_weight: float,
) -> List[dict]:
    buckets: Dict[str, List[float]] = {}
    for symbol, weight in weights.items():
        w = float(weight)
        if w <= 0:
            continue
        name = str(labels.get(symbol) or UNCLASSIFIED)
        r = float(returns.get(symbol) or 0.0)
        acc = buckets.setdefault(name, [0.0, 0.0])
        acc[0] += w
        acc[1] += w * r
    rows = []
    for name, (w, wr) in sorted(buckets.items()):
        rows.append(
            {
                "period": day,
                "industry": name,
                "weight": float(w),
                "return": float(wr / w) if w else 0.0,
            }
        )
    cash_w = float(cash_weight)
    if cash_w > 1e-12:
        rows.append(
            {"period": day, "industry": CASH_BUCKET, "weight": cash_w, "return": 0.0}
        )
    return rows


def _price(panel: Mapping[str, Mapping[str, float]], day: Optional[str], symbol: str) -> Optional[float]:
    if not day:
        return None
    row = panel.get(day)
    if not row:
        return None
    value = row.get(symbol)
    if value is None or not np.isfinite(value) or float(value) <= 0:
        return None
    return float(value)


def pivot_prices(frame: pd.DataFrame, field: str = "close") -> Dict[str, Dict[str, float]]:
    if frame is None or frame.empty or field not in frame.columns:
        return {}
    work = frame.copy()
    day_col = "trade_date" if "trade_date" in work.columns else "date"
    sym_col = "ts_code" if "ts_code" in work.columns else "symbol"
    work["day"] = work[day_col].map(normalize_day)
    out: Dict[str, Dict[str, float]] = {}
    values = pd.to_numeric(work[field], errors="coerce")
    symbols = work[sym_col].astype(str)
    for day, symbol, px in zip(work["day"], symbols, values):
        if not day or not symbol or not np.isfinite(px) or float(px) <= 0:
            continue
        out.setdefault(str(day), {})[str(symbol)] = float(px)
    return out


def series_to_map(series: pd.Series) -> Dict[str, float]:
    if series is None or getattr(series, "empty", True):
        return {}
    out: Dict[str, float] = {}
    for idx, value in series.items():
        day = normalize_day(idx)
        px = pd.to_numeric(value, errors="coerce")
        if day and np.isfinite(px) and float(px) > 0:
            out[day] = float(px)
    return out
