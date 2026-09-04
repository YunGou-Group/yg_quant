#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归因数值例程共用的表读取与收尾。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .models import normalize_asset_class, warning


_EPS = 1e-14
_PERIOD_COLUMNS = ("period", "date", "trading_date", "trade_date", "timestamp")
_ID_COLUMNS = ("asset_id", "order_book_id", "instrument", "symbol", "security_id")


def _as_frame(data: Any, name: str = "data") -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame().T
    try:
        return pd.DataFrame(data).copy()
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise TypeError(f"{name} must be a DataFrame or JSON-like records") from exc


def _pick(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    available = set(columns)
    return next((column for column in aliases if column in available), None)


def _numbers(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    default: float = 0.0,
    required: bool = False,
    label: Optional[str] = None,
) -> pd.Series:
    column = _pick(frame.columns, aliases)
    if column is None:
        if required:
            raise ValueError(f"missing required column: {label or aliases[0]}")
        return pd.Series(default, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(default).astype(float)


def _period(frame: pd.DataFrame) -> pd.Series:
    column = _pick(frame.columns, _PERIOD_COLUMNS)
    if column is None:
        return pd.Series("TOTAL", index=frame.index, dtype=object)
    return frame[column].where(frame[column].notna(), "UNKNOWN")


def _asset_id(frame: pd.DataFrame) -> pd.Series:
    column = _pick(frame.columns, _ID_COLUMNS)
    if column is None:
        return pd.Series([f"row_{index}" for index in range(len(frame))], index=frame.index)
    return frame[column].astype(str)


def _asset_classes(frame: pd.DataFrame, warnings: List[Dict[str, Any]]) -> pd.Series:
    column = _pick(frame.columns, ("asset_class", "asset_type", "type"))
    if column is None:
        return pd.Series(None, index=frame.index, dtype=object)
    normalized = frame[column].map(lambda value: normalize_asset_class(value, strict=False))
    invalid = normalized.isna() & frame[column].notna()
    if invalid.any():
        values = sorted({str(value) for value in frame.loc[invalid, column].tolist()})
        warnings.append(
            warning(
                "unknown_asset_class",
                "Some rows contain an unsupported asset class.",
                scope="asset_class",
                values=values,
                row_count=int(invalid.sum()),
            )
        )
    return normalized


def _finish(
    frame: pd.DataFrame,
    warnings: Optional[List[Dict[str, Any]]] = None,
    reconciliations: Optional[List[Dict[str, Any]]] = None,
    **attrs: Any,
) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    frame.attrs["quality_warnings"] = list(warnings or [])
    frame.attrs["reconciliation"] = list(reconciliations or [])
    frame.attrs.update(attrs)
    return frame


def _fees(frame: pd.DataFrame) -> pd.Series:
    total_column = _pick(frame.columns, ("fees", "fee", "total_fee", "transaction_cost"))
    if total_column is not None:
        return pd.to_numeric(frame[total_column], errors="coerce").fillna(0.0).abs().astype(float)
    parts = pd.Series(0.0, index=frame.index)
    for column in ("commission", "tax", "slippage", "exchange_fee", "borrow_fee"):
        if column in frame:
            parts = parts + pd.to_numeric(frame[column], errors="coerce").fillna(0.0).abs()
    return parts.astype(float)
