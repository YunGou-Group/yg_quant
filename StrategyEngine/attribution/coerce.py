#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 payload 收成表，并拼 JSON 报告段。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import normalize_asset_class


RETURN_UNIT = "decimal_return (0.0322 = 3.22%)"
AMOUNT_UNIT = "portfolio_currency_amount"


def _as_frame(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame().T
    return pd.DataFrame(data).copy()


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _period_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("period", "date", "trading_date", "trade_date", "timestamp"):
        if column in frame:
            return frame[column].where(frame[column].notna(), "UNKNOWN")
    return pd.Series("TOTAL", index=frame.index)


def _numeric(frame: pd.DataFrame, names: Sequence[str]) -> Optional[pd.Series]:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).astype(float)
    return None


def _prepare_period_returns(data: Any) -> pd.DataFrame:
    frame = _as_frame(data)
    if frame.empty:
        return pd.DataFrame(columns=["period", "portfolio_return", "benchmark_return", "active_return"])
    portfolio = _numeric(frame, ("portfolio_return", "strategy_return", "return"))
    benchmark = _numeric(frame, ("benchmark_return", "index_return"))
    active = _numeric(frame, ("active_return", "excess_return"))
    if portfolio is None and benchmark is not None and active is not None:
        portfolio = benchmark + active
    if benchmark is None and portfolio is not None and active is not None:
        benchmark = portfolio - active
    if portfolio is None or benchmark is None:
        raise ValueError("returns requires portfolio_return and benchmark_return (or active_return to derive one)")
    result = pd.DataFrame(
        {
            "period": _period_series(frame),
            "portfolio_return": portfolio,
            "benchmark_return": benchmark,
        }
    )
    result["active_return"] = result["portfolio_return"] - result["benchmark_return"]
    optional_return_aliases = {
        "trade_return": ("trade_return", "trading_return", "transaction_return"),
        "holding_return": ("holding_return", "position_return"),
        "leverage_return": ("leverage_return", "financing_return"),
        "benchmark_holding_return": (
            "benchmark_holding_return",
            "benchmark_position_return",
            "index_holding_return",
        ),
    }
    for canonical, aliases in optional_return_aliases.items():
        source = next((column for column in aliases if column in frame), None)
        if source is not None:
            result[canonical] = pd.to_numeric(frame[source], errors="coerce")
    nav_column = next(
        (
            column
            for column in ("nav_begin", "beginning_nav", "begin_nav", "opening_nav", "portfolio_begin_value")
            if column in frame
        ),
        None,
    )
    if nav_column is not None:
        result["nav_begin"] = pd.to_numeric(frame[nav_column], errors="coerce")
    # One record per period is expected.  If duplicates are supplied, they are
    # geometrically compounded rather than silently taking an arbitrary row.
    grouped_rows: List[Dict[str, Any]] = []
    for period_value, rows in result.groupby("period", sort=False, dropna=False):
        p = float(np.prod(1.0 + rows["portfolio_return"].to_numpy(dtype=float)) - 1.0)
        b = float(np.prod(1.0 + rows["benchmark_return"].to_numpy(dtype=float)) - 1.0)
        grouped_rows.append(
            {
                "period": period_value,
                "portfolio_return": p,
                "benchmark_return": b,
                "active_return": p - b,
                **{
                    column: (
                        float(rows[column].dropna().sum())
                        if rows[column].notna().any()
                        else np.nan
                    )
                    for column in optional_return_aliases
                    if column in rows
                },
                **(
                    {
                        "nav_begin": (
                            float(rows["nav_begin"].dropna().iloc[0])
                            if rows["nav_begin"].notna().any()
                            else np.nan
                        )
                    }
                    if "nav_begin" in rows
                    else {}
                ),
            }
        )
    return pd.DataFrame(grouped_rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if value is pd.NA:
        return None
    return value


def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    return _json_value(clean.to_dict("records"))


def _units_for_columns(columns: Iterable[str]) -> Dict[str, str]:
    units: Dict[str, str] = {}
    for column in columns:
        lower = column.lower()
        if lower.endswith("_return") or "_return_" in lower or lower in {
            "allocation",
            "selection",
            "interaction",
            "total_effect",
            "active_contribution",
            "contribution",
            "linked_effect",
            "residual",
        } or lower.endswith("_linked") or lower.endswith("_contribution"):
            units[column] = RETURN_UNIT
        elif "mtm" in lower or lower in {"fees", "absolute_notional", "opening_market_value"}:
            units[column] = AMOUNT_UNIT
        elif "weight" in lower or "exposure" in lower or lower in {"carino_k", "carino_k", "linking_factor"}:
            units[column] = "ratio"
        elif "price" in lower:
            units[column] = "price_in_portfolio_currency"
        elif "quantity" in lower:
            units[column] = "units_or_contracts"
        else:
            units[column] = "text_or_dimensionless"
    return units


def _section(
    key: str,
    title: str,
    method: str,
    frame: pd.DataFrame,
    *,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    columns = list(frame.columns)
    result: Dict[str, Any] = {
        "key": key,
        "title": title,
        "method": method,
        "columns": columns,
        "units": _units_for_columns(columns),
        "rows": _records(frame),
        "totals": _json_value(frame.attrs.get("totals", [])),
    }
    if note:
        result["note"] = note
    return result


def _collect_frame_metadata(
    frame: pd.DataFrame,
    warnings: List[Dict[str, Any]],
    reconciliations: List[Dict[str, Any]],
) -> None:
    warnings.extend(_json_value(frame.attrs.get("quality_warnings", [])))
    reconciliations.extend(_json_value(frame.attrs.get("reconciliation", [])))


def _infer_asset_pool(payload: Mapping[str, Any]) -> List[str]:
    inferred: List[str] = []
    for key, canonical in (("bond", "bond"), ("bonds", "bond"), ("future", "future"), ("futures", "future"), ("option", "option"), ("options", "option")):
        if key in payload and payload[key] is not None and canonical not in inferred:
            inferred.append(canonical)
    if "stock" in payload or "industry_attribution" in payload or "factor_attribution" in payload:
        inferred.append("stock")
    for data_key in ("trades", "holdings"):
        frame = _as_frame(payload.get(data_key))
        if frame.empty:
            continue
        class_column = next((column for column in ("asset_class", "asset_type", "type") if column in frame), None)
        if class_column:
            for value in frame[class_column].dropna().tolist():
                normalized = normalize_asset_class(value, strict=False)
                if normalized is not None and normalized not in inferred:
                    inferred.append(normalized)
    return inferred


def _extract_sided(data: Any) -> Tuple[Any, Any, Any]:
    """Return portfolio, benchmark and combined representations."""

    if isinstance(data, Mapping):
        portfolio = _first(data, ("portfolio", "strategy", "portfolio_rows"))
        benchmark = _first(data, ("benchmark", "index", "benchmark_rows"))
        combined = _first(data, ("rows", "data", "combined"))
        if portfolio is None and benchmark is None and combined is None:
            # A dict-of-lists is itself a valid pandas/JSON table.
            combined = data
        return portfolio, benchmark, combined
    return None, None, data


def _attach_factor_weights(data: Any, portfolio_weights: Any, benchmark_weights: Any) -> Any:
    """Join separately supplied security weights to RQData-style factor rows."""

    frame = _as_frame(data)
    if frame.empty:
        return data
    period_aliases = ("period", "date", "trading_date", "trade_date", "timestamp")
    asset_aliases = ("asset_id", "order_book_id", "instrument_id", "instrument", "symbol", "security_id")
    data_period = next((column for column in period_aliases if column in frame), None)
    data_asset = next((column for column in asset_aliases if column in frame), None)
    if data_asset is None:
        return frame
    frame["__asset_key"] = frame[data_asset].astype(str)
    if data_period is not None:
        frame["__period_key"] = frame[data_period].astype(str)
    for side, weights_data in (("portfolio", portfolio_weights), ("benchmark", benchmark_weights)):
        target = f"{side}_weight"
        if target in frame or weights_data is None:
            continue
        weights = _as_frame(weights_data)
        if weights.empty:
            continue
        weight_asset = next((column for column in asset_aliases if column in weights), None)
        weight_period = next((column for column in period_aliases if column in weights), None)
        weight_column = next(
            (column for column in (target, "weight", "average_weight", "avg_weight") if column in weights),
            None,
        )
        if weight_asset is None or weight_column is None:
            continue
        weights = weights.copy()
        weights["__asset_key"] = weights[weight_asset].astype(str)
        join_keys = ["__asset_key"]
        if data_period is not None and weight_period is not None:
            weights["__period_key"] = weights[weight_period].astype(str)
            join_keys.insert(0, "__period_key")
        weights[target] = pd.to_numeric(weights[weight_column], errors="coerce")
        weights = weights.groupby(join_keys, sort=False, dropna=False)[target].sum().reset_index()
        frame = frame.merge(weights, how="left", on=join_keys, sort=False)
    return frame.drop(columns=[column for column in ("__period_key", "__asset_key") if column in frame])


def _table_from_frame(key: str, title: str, method: str, frame: pd.DataFrame, note: Optional[str] = None) -> Dict[str, Any]:
    return _section(key, title, method, frame, note=note)
