#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成交与持仓盯市（含费用）。"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ._common import (
    _EPS,
    _PERIOD_COLUMNS,
    _as_frame,
    _asset_classes,
    _asset_id,
    _fees,
    _finish,
    _numbers,
    _period,
    _pick,
)
from .models import warning


def calculate_signed_trade_mtm(trades: Any) -> pd.DataFrame:
    """Mark trades to period end using signed quantities and subtract fees.

    A buy/cover has positive signed quantity and a sell/short has negative
    signed quantity, so ``q_signed * (mark - execution) * multiplier`` has the
    correct P&L sign for both directions.  Missing marks are conservatively set
    to the execution price and reported as a quality warning.
    """

    frame = _as_frame(trades, "trades")
    warnings: List[Dict[str, Any]] = []
    if frame.empty:
        return _finish(
            pd.DataFrame(
                columns=[
                    "period",
                    "asset_id",
                    "asset_class",
                    "signed_quantity",
                    "execution_price",
                    "mark_price",
                    "multiplier",
                    "gross_trade_mtm",
                    "fees",
                    "trade_mtm",
                ]
            )
        )

    signed_column = _pick(frame.columns, ("signed_quantity", "signed_qty", "net_quantity"))
    quantity = _numbers(
        frame,
        (signed_column,) if signed_column else ("quantity", "qty", "volume"),
        required=True,
        label="signed_quantity or quantity",
    )
    if signed_column is None:
        side_column = _pick(frame.columns, ("side", "direction", "action", "trade_side"))
        if side_column is not None:
            side = frame[side_column].astype(str).str.strip().str.lower()
            buy_values = {
                "buy",
                "b",
                "long",
                "open_long",
                "close_short",
                "cover",
                "买",
                "买入",
                "多",
                "平空",
            }
            sell_values = {
                "sell",
                "s",
                "short",
                "open_short",
                "close_long",
                "卖",
                "卖出",
                "空",
                "平多",
            }
            signs = side.map(lambda value: 1.0 if value in buy_values else (-1.0 if value in sell_values else np.nan))
            unknown = signs.isna()
            if unknown.any():
                warnings.append(
                    warning(
                        "unknown_trade_side",
                        "Unknown trade sides kept the sign supplied in quantity.",
                        scope="trades",
                        values=sorted(set(side[unknown].tolist())),
                        row_count=int(unknown.sum()),
                    )
                )
                signs = signs.where(~unknown, np.sign(quantity).replace(0.0, 1.0))
            quantity = quantity.abs() * signs

    execution = _numbers(
        frame,
        ("execution_price", "trade_price", "price", "fill_price"),
        required=True,
        label="execution_price",
    )
    mark_column = _pick(frame.columns, ("mark_price", "end_price", "close_price", "period_end_price"))
    if mark_column is None:
        mark = execution.copy()
        warnings.append(
            warning(
                "missing_trade_mark",
                "Trade mark price is missing; execution price was used, producing zero gross trade MTM.",
                scope="trades",
                row_count=len(frame),
            )
        )
    else:
        raw_mark = pd.to_numeric(frame[mark_column], errors="coerce")
        missing_mark = raw_mark.isna()
        mark = raw_mark.where(~missing_mark, execution).astype(float)
        if missing_mark.any():
            warnings.append(
                warning(
                    "missing_trade_mark",
                    "Missing trade marks were replaced with execution prices.",
                    scope="trades",
                    row_count=int(missing_mark.sum()),
                )
            )
    multiplier = _numbers(frame, ("multiplier", "contract_multiplier"), default=1.0)
    multiplier = multiplier.where(multiplier != 0.0, 1.0)
    fees = _fees(frame)
    gross = quantity * (mark - execution) * multiplier

    result = pd.DataFrame(
        {
            "period": _period(frame),
            "asset_id": _asset_id(frame),
            "asset_class": _asset_classes(frame, warnings),
            "signed_quantity": quantity,
            "execution_price": execution,
            "mark_price": mark,
            "multiplier": multiplier,
            "gross_trade_mtm": gross,
            "fees": fees,
            "trade_mtm": gross - fees,
            "absolute_notional": quantity.abs() * execution.abs() * multiplier.abs(),
        }
    )
    return _finish(
        result,
        warnings,
        totals={
            "gross_trade_mtm": float(gross.sum()),
            "fees": float(fees.sum()),
            "trade_mtm": float((gross - fees).sum()),
        },
    )


def calculate_holding_mtm(holdings: Any) -> pd.DataFrame:
    """Calculate period MTM for opening holdings and subtract holding costs."""

    frame = _as_frame(holdings, "holdings")
    warnings: List[Dict[str, Any]] = []
    if frame.empty:
        return _finish(
            pd.DataFrame(
                columns=[
                    "period",
                    "asset_id",
                    "asset_class",
                    "quantity",
                    "start_price",
                    "end_price",
                    "multiplier",
                    "gross_holding_mtm",
                    "fees",
                    "holding_mtm",
                ]
            )
        )

    quantity = _numbers(
        frame,
        ("start_quantity", "opening_quantity", "quantity", "position", "qty"),
        required=True,
        label="start_quantity",
    )
    start_price = _numbers(
        frame,
        ("start_price", "begin_price", "opening_price", "cost_price", "previous_close"),
        required=True,
        label="start_price",
    )
    end_column = _pick(frame.columns, ("end_price", "mark_price", "close_price", "period_end_price"))
    if end_column is None:
        end_price = start_price.copy()
        warnings.append(
            warning(
                "missing_holding_mark",
                "Holding end price is missing; start price was used, producing zero gross holding MTM.",
                scope="holdings",
                row_count=len(frame),
            )
        )
    else:
        raw_end = pd.to_numeric(frame[end_column], errors="coerce")
        missing = raw_end.isna()
        end_price = raw_end.where(~missing, start_price).astype(float)
        if missing.any():
            warnings.append(
                warning(
                    "missing_holding_mark",
                    "Missing holding end prices were replaced with start prices.",
                    scope="holdings",
                    row_count=int(missing.sum()),
                )
            )
    multiplier = _numbers(frame, ("multiplier", "contract_multiplier"), default=1.0)
    multiplier = multiplier.where(multiplier != 0.0, 1.0)
    fees = _fees(frame)
    gross = quantity * (end_price - start_price) * multiplier
    result = pd.DataFrame(
        {
            "period": _period(frame),
            "asset_id": _asset_id(frame),
            "asset_class": _asset_classes(frame, warnings),
            "quantity": quantity,
            "start_price": start_price,
            "end_price": end_price,
            "multiplier": multiplier,
            "gross_holding_mtm": gross,
            "fees": fees,
            "holding_mtm": gross - fees,
            "opening_market_value": quantity * start_price * multiplier,
        }
    )
    return _finish(
        result,
        warnings,
        totals={
            "gross_holding_mtm": float(gross.sum()),
            "fees": float(fees.sum()),
            "holding_mtm": float((gross - fees).sum()),
        },
    )


calculate_trade_mtm = calculate_signed_trade_mtm
