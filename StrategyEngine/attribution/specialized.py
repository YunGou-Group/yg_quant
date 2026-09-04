#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""债券 / 期货 / 期权成分分解。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

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
from .models import reconciliation, warning


def _decimal_change(frame: pd.DataFrame, decimal_aliases: Sequence[str], bps_aliases: Sequence[str]) -> pd.Series:
    decimal_column = _pick(frame.columns, decimal_aliases)
    if decimal_column is not None:
        return pd.to_numeric(frame[decimal_column], errors="coerce").fillna(0.0).astype(float)
    bps_column = _pick(frame.columns, bps_aliases)
    if bps_column is not None:
        return pd.to_numeric(frame[bps_column], errors="coerce").fillna(0.0).astype(float) / 10000.0
    return pd.Series(0.0, index=frame.index)


def _direct_or(frame: pd.DataFrame, aliases: Sequence[str], calculated: pd.Series) -> pd.Series:
    column = _pick(frame.columns, aliases)
    if column is None:
        return calculated.astype(float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(calculated).astype(float)


def _actual_return(
    frame: pd.DataFrame,
    component_sum: pd.Series,
    warnings: List[Dict[str, Any]],
    scope: str,
) -> Tuple[pd.Series, bool]:
    column = _pick(frame.columns, ("actual_return", "total_return", "return", "portfolio_return"))
    if column is not None:
        return pd.to_numeric(frame[column], errors="coerce").fillna(component_sum).astype(float), True
    pnl_column = _pick(frame.columns, ("pnl", "profit_loss", "actual_pnl"))
    value_column = _pick(frame.columns, ("start_value", "opening_value", "capital", "market_value"))
    if pnl_column is not None and value_column is not None:
        pnl = pd.to_numeric(frame[pnl_column], errors="coerce").fillna(0.0)
        value = pd.to_numeric(frame[value_column], errors="coerce").replace(0.0, np.nan)
        calculated = (pnl / value).replace([np.inf, -np.inf], np.nan).fillna(component_sum)
        return calculated.astype(float), True
    warnings.append(
        warning(
            f"missing_{scope}_actual_return",
            f"{scope.title()} actual return is missing; component sum was used and residual is zero.",
            severity="info",
            scope=scope,
            row_count=len(frame),
        )
    )
    return component_sum.astype(float), False


def _component_output(
    frame: pd.DataFrame,
    components: Mapping[str, pd.Series],
    actual: pd.Series,
    warnings: List[Dict[str, Any]],
    *,
    scope: str,
    tolerance: float,
    identity_name: str,
) -> pd.DataFrame:
    weight = _numbers(frame, ("weight", "portfolio_weight", "average_weight"), default=1.0)
    result = pd.DataFrame(
        {
            "period": _period(frame),
            "asset_id": _asset_id(frame),
            "weight": weight,
            "actual_return": actual,
        }
    )
    component_sum = pd.Series(0.0, index=frame.index)
    for name, values in components.items():
        result[f"{name}_return"] = values.astype(float)
        component_sum = component_sum + values.astype(float)
    residual = actual - component_sum
    result["residual_return"] = residual
    result["actual_contribution"] = weight * actual
    for name in components:
        result[f"{name}_contribution"] = weight * result[f"{name}_return"]
    result["residual_contribution"] = weight * residual
    contribution_columns = [f"{name}_contribution" for name in components] + ["residual_contribution"]
    result["explained_contribution"] = result[contribution_columns].sum(axis=1)

    reconciliations: List[Dict[str, Any]] = []
    for period_value, rows in result.groupby("period", sort=False, dropna=False):
        actual_total = float(rows["actual_contribution"].sum())
        explained_total = float(rows["explained_contribution"].sum())
        reconciliations.append(
            reconciliation(
                f"{scope}:{period_value}",
                actual_total,
                explained_total,
                tolerance=tolerance,
                scope=scope,
            )
        )
        pre_residual = float((rows["weight"] * rows[[f"{name}_return" for name in components]].sum(axis=1)).sum())
        period_residual = actual_total - pre_residual
        if abs(period_residual) > max(tolerance, 1e-6):
            warnings.append(
                warning(
                    f"large_{scope}_residual",
                    f"{scope.title()} model leaves a material residual; it is shown explicitly so totals reconcile.",
                    scope=scope,
                    period=period_value,
                    residual=period_residual,
                )
            )
    totals = []
    for period_value, rows in result.groupby("period", sort=False, dropna=False):
        total: Dict[str, Any] = {
            "period": period_value,
            "actual_contribution": float(rows["actual_contribution"].sum()),
        }
        for column in contribution_columns:
            total[column] = float(rows[column].sum())
        total["explained_contribution"] = float(rows["explained_contribution"].sum())
        totals.append(total)
    return _finish(result, warnings, reconciliations, totals=totals, method=identity_name)


def bond_attribution(data: Any, *, tolerance: float = 1e-10) -> pd.DataFrame:
    """Decompose bond return into carry, curve, spread, convexity and residual."""

    frame = _as_frame(data, "bond attribution")
    if frame.empty:
        return _finish(pd.DataFrame())
    warnings: List[Dict[str, Any]] = []
    start_value = _numbers(frame, ("start_value", "opening_value", "market_value"), default=1.0).replace(0.0, np.nan)
    carry_calculated = (
        _numbers(frame, ("coupon_income",), default=0.0)
        + _numbers(frame, ("accrual_income", "accrued_interest_change"), default=0.0)
    ) / start_value
    if not any(column in frame for column in ("coupon_income", "accrual_income", "accrued_interest_change")):
        annual_yield = _numbers(frame, ("yield_to_maturity", "annual_yield", "coupon_rate"), default=0.0)
        days = _numbers(frame, ("day_count", "days", "holding_days"), default=0.0)
        carry_calculated = annual_yield * days / 365.0
    carry = _direct_or(frame, ("carry_return", "carry_effect", "carry"), carry_calculated.fillna(0.0))
    curve_change = _decimal_change(
        frame,
        ("risk_free_yield_change", "curve_change", "government_yield_change"),
        ("risk_free_yield_change_bps", "curve_change_bps", "government_yield_change_bps"),
    )
    spread_change = _decimal_change(
        frame,
        ("spread_change", "credit_spread_change"),
        ("spread_change_bps", "credit_spread_change_bps"),
    )
    duration = _numbers(frame, ("modified_duration", "duration"), default=0.0)
    spread_duration = _numbers(frame, ("spread_duration",), default=0.0)
    convexity_value = _numbers(frame, ("convexity",), default=0.0)
    curve = _direct_or(frame, ("curve_return", "curve_effect"), -duration * curve_change)
    spread = _direct_or(frame, ("spread_return", "spread_effect"), -spread_duration * spread_change)
    total_yield_change = curve_change + spread_change
    convexity = _direct_or(
        frame,
        ("convexity_return", "convexity_effect"),
        0.5 * convexity_value * total_yield_change.pow(2),
    )
    components = {"carry": carry, "curve": curve, "spread": spread, "convexity": convexity}
    component_sum = sum(components.values(), pd.Series(0.0, index=frame.index))
    actual, _ = _actual_return(frame, component_sum, warnings, "bond")
    return _component_output(
        frame,
        components,
        actual,
        warnings,
        scope="bond",
        tolerance=tolerance,
        identity_name="bond carry/curve/spread/convexity",
    )


def futures_attribution(data: Any, *, tolerance: float = 1e-10) -> pd.DataFrame:
    """Decompose futures return into underlying, basis, roll, collateral and leverage."""

    frame = _as_frame(data, "futures attribution")
    if frame.empty:
        return _finish(pd.DataFrame())
    warnings: List[Dict[str, Any]] = []
    initial_price = _numbers(frame, ("initial_price", "start_price", "underlying_start_price"), default=1.0).replace(0.0, np.nan)
    price_change = _numbers(frame, ("price_change", "underlying_price_change"), default=0.0)
    underlying = _direct_or(
        frame,
        ("underlying_return", "spot_return", "price_return", "underlying_effect"),
        (price_change / initial_price).fillna(0.0),
    )
    if "end_basis" in frame and "start_basis" in frame:
        basis_calculated = (
            pd.to_numeric(frame["end_basis"], errors="coerce").fillna(0.0)
            - pd.to_numeric(frame["start_basis"], errors="coerce").fillna(0.0)
        ) / initial_price
    elif "basis_change" in frame:
        basis_raw = pd.to_numeric(frame["basis_change"], errors="coerce").fillna(0.0)
        basis_calculated = basis_raw / initial_price if any(c in frame for c in ("initial_price", "start_price")) else basis_raw
    else:
        basis_calculated = pd.Series(0.0, index=frame.index)
    basis = _direct_or(frame, ("basis_return", "basis_effect"), basis_calculated.fillna(0.0))
    roll = _direct_or(
        frame,
        ("roll_return", "roll_effect", "roll_yield"),
        pd.Series(0.0, index=frame.index),
    )
    collateral = _direct_or(
        frame,
        ("collateral_return", "collateral_effect", "cash_return"),
        pd.Series(0.0, index=frame.index),
    )
    leverage_ratio = _numbers(frame, ("leverage", "leverage_ratio"), default=1.0)
    leverage = _direct_or(
        frame,
        ("leverage_return", "leverage_effect"),
        (leverage_ratio - 1.0) * (underlying + basis + roll),
    )
    components = {
        "underlying": underlying,
        "basis": basis,
        "roll": roll,
        "collateral": collateral,
        "leverage": leverage,
    }
    component_sum = sum(components.values(), pd.Series(0.0, index=frame.index))
    actual, _ = _actual_return(frame, component_sum, warnings, "future")
    return _component_output(
        frame,
        components,
        actual,
        warnings,
        scope="future",
        tolerance=tolerance,
        identity_name="futures underlying/basis/roll/collateral/leverage",
    )


def option_attribution(data: Any, *, tolerance: float = 1e-10) -> pd.DataFrame:
    """Greek P&L attribution: delta, gamma, vega, theta, rho and residual.

    When precomputed ``*_return``/``*_effect`` fields are absent, Greek P&L is
    divided by ``start_value``.  Theta is assumed to use the same time unit as
    ``elapsed_days`` (default one).
    """

    frame = _as_frame(data, "option attribution")
    if frame.empty:
        return _finish(pd.DataFrame())
    warnings: List[Dict[str, Any]] = []
    start_value = _numbers(
        frame, ("start_value", "opening_value", "option_value", "premium"), default=1.0
    ).replace(0.0, np.nan)
    underlying_change = _numbers(
        frame, ("underlying_price_change", "underlying_change", "spot_change", "delta_s"), default=0.0
    )
    volatility_change = _decimal_change(
        frame,
        ("volatility_change", "implied_volatility_change", "iv_change"),
        ("volatility_change_bps", "iv_change_bps"),
    )
    rate_change = _decimal_change(
        frame,
        ("rate_change", "interest_rate_change"),
        ("rate_change_bps", "interest_rate_change_bps"),
    )
    elapsed = _numbers(frame, ("elapsed_days", "day_count", "days"), default=1.0)
    delta = _direct_or(
        frame,
        ("delta_return", "delta_effect"),
        (_numbers(frame, ("delta",), default=0.0) * underlying_change / start_value).fillna(0.0),
    )
    gamma = _direct_or(
        frame,
        ("gamma_return", "gamma_effect"),
        (
            0.5
            * _numbers(frame, ("gamma",), default=0.0)
            * underlying_change.pow(2)
            / start_value
        ).fillna(0.0),
    )
    vega = _direct_or(
        frame,
        ("vega_return", "vega_effect"),
        (_numbers(frame, ("vega",), default=0.0) * volatility_change / start_value).fillna(0.0),
    )
    theta = _direct_or(
        frame,
        ("theta_return", "theta_effect"),
        (_numbers(frame, ("theta",), default=0.0) * elapsed / start_value).fillna(0.0),
    )
    rho = _direct_or(
        frame,
        ("rho_return", "rho_effect"),
        (_numbers(frame, ("rho",), default=0.0) * rate_change / start_value).fillna(0.0),
    )
    components = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
    component_sum = sum(components.values(), pd.Series(0.0, index=frame.index))
    actual, _ = _actual_return(frame, component_sum, warnings, "option")
    return _component_output(
        frame,
        components,
        actual,
        warnings,
        scope="option",
        tolerance=tolerance,
        identity_name="option delta/gamma/vega/theta/rho",
    )


def reconcile_components(
    actual: float,
    components: Mapping[str, float],
    *,
    name: str = "attribution",
    tolerance: float = 1e-10,
    scope: str = "portfolio",
) -> Dict[str, Any]:
    """Public reconciliation helper used by adapters and tests."""

    return reconciliation(name, actual, sum(float(value) for value in components.values()), tolerance=tolerance, scope=scope)
