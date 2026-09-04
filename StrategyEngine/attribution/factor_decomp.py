#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主动暴露 × 因子收益 + 特异 / 残差。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

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


def _factor_returns_long(data: Any) -> pd.DataFrame:
    frame = _as_frame(data, "factor_returns")
    if frame.empty:
        return pd.DataFrame(columns=["period", "factor", "factor_return"])
    factor_column = _pick(frame.columns, ("factor", "factor_name", "style_factor"))
    return_column = _pick(frame.columns, ("factor_return", "return", "factor_premium"))
    if factor_column is not None and return_column is not None:
        return pd.DataFrame(
            {
                "period": _period(frame),
                "factor": frame[factor_column].astype(str),
                "factor_return": pd.to_numeric(frame[return_column], errors="coerce"),
            }
        )
    id_vars = [_pick(frame.columns, _PERIOD_COLUMNS)]
    id_vars = [value for value in id_vars if value is not None]
    value_vars = [
        column
        for column in frame.columns
        if column not in id_vars and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not value_vars:
        raise ValueError("factor_returns must be long-form or contain numeric factor columns")
    melted = frame.melt(id_vars=id_vars, value_vars=value_vars, var_name="factor", value_name="factor_return")
    melted["period"] = _period(melted)
    return melted[["period", "factor", "factor_return"]]


def _exposures_long(data: Any, factor_names: Sequence[str]) -> pd.DataFrame:
    frame = _as_frame(data, "exposures")
    if frame.empty:
        return pd.DataFrame(
            columns=["period", "factor", "portfolio_exposure", "benchmark_exposure", "active_exposure"]
        )
    factor_column = _pick(frame.columns, ("factor", "factor_name", "style_factor"))
    if factor_column is None:
        value_vars = [column for column in factor_names if column in frame.columns]
        if not value_vars:
            raise ValueError("exposures must contain a factor column or wide factor exposure columns")
        id_vars = [column for column in frame.columns if column not in value_vars]
        frame = frame.melt(id_vars=id_vars, value_vars=value_vars, var_name="factor", value_name="exposure")
        factor_column = "factor"
    period_values = _period(frame)
    factor_values = frame[factor_column].astype(str)
    active_column = _pick(frame.columns, ("active_exposure", "relative_exposure"))
    portfolio_column = _pick(frame.columns, ("portfolio_exposure", "strategy_exposure"))
    benchmark_column = _pick(frame.columns, ("benchmark_exposure", "index_exposure"))
    exposure_column = _pick(frame.columns, ("exposure", "factor_exposure", "value"))
    p_weight_column = _pick(frame.columns, ("portfolio_weight", "strategy_weight", "weight"))
    b_weight_column = _pick(frame.columns, ("benchmark_weight", "index_weight"))

    if active_column is not None:
        active = pd.to_numeric(frame[active_column], errors="coerce").fillna(0.0)
        p_exposure = (
            pd.to_numeric(frame[portfolio_column], errors="coerce").fillna(0.0)
            if portfolio_column is not None
            else active
        )
        b_exposure = (
            pd.to_numeric(frame[benchmark_column], errors="coerce").fillna(0.0)
            if benchmark_column is not None
            else p_exposure - active
        )
    elif portfolio_column is not None or benchmark_column is not None:
        p_exposure = (
            pd.to_numeric(frame[portfolio_column], errors="coerce").fillna(0.0)
            if portfolio_column is not None
            else pd.Series(0.0, index=frame.index)
        )
        b_exposure = (
            pd.to_numeric(frame[benchmark_column], errors="coerce").fillna(0.0)
            if benchmark_column is not None
            else pd.Series(0.0, index=frame.index)
        )
        active = p_exposure - b_exposure
    elif exposure_column is not None and (p_weight_column is not None or b_weight_column is not None):
        exposure = pd.to_numeric(frame[exposure_column], errors="coerce").fillna(0.0)
        p_weight = (
            pd.to_numeric(frame[p_weight_column], errors="coerce").fillna(0.0)
            if p_weight_column is not None
            else pd.Series(0.0, index=frame.index)
        )
        b_weight = (
            pd.to_numeric(frame[b_weight_column], errors="coerce").fillna(0.0)
            if b_weight_column is not None
            else pd.Series(0.0, index=frame.index)
        )
        p_exposure = exposure * p_weight
        b_exposure = exposure * b_weight
        active = p_exposure - b_exposure
    else:
        raise ValueError(
            "exposures require active_exposure, portfolio/benchmark_exposure, or exposure with weights"
        )

    result = pd.DataFrame(
        {
            "period": period_values,
            "factor": factor_values,
            "portfolio_exposure": p_exposure,
            "benchmark_exposure": b_exposure,
            "active_exposure": active,
        }
    )
    return result.groupby(["period", "factor"], sort=False, dropna=False).sum(numeric_only=True).reset_index()


def _specific_effects(data: Any) -> pd.DataFrame:
    frame = _as_frame(data, "specific_returns")
    if frame.empty:
        return pd.DataFrame(columns=["period", "contribution"])
    period_values = _period(frame)
    direct = _pick(frame.columns, ("specific_contribution", "active_specific_contribution"))
    if direct is not None:
        contribution = pd.to_numeric(frame[direct], errors="coerce").fillna(0.0)
    elif "portfolio_specific_return" in frame.columns or "benchmark_specific_return" in frame.columns:
        p = _numbers(frame, ("portfolio_specific_return",), default=0.0)
        b = _numbers(frame, ("benchmark_specific_return",), default=0.0)
        contribution = p - b
    else:
        specific_column = _pick(frame.columns, ("specific_return", "idiosyncratic_return", "residual_return"))
        if specific_column is None:
            raise ValueError("specific_returns is missing specific_return or specific_contribution")
        specific = pd.to_numeric(frame[specific_column], errors="coerce").fillna(0.0)
        if "active_weight" in frame:
            contribution = specific * pd.to_numeric(frame["active_weight"], errors="coerce").fillna(0.0)
        else:
            p_weight = _numbers(frame, ("portfolio_weight", "strategy_weight", "weight"), default=0.0)
            b_weight = _numbers(frame, ("benchmark_weight", "index_weight"), default=0.0)
            contribution = specific * (p_weight - b_weight)
    return pd.DataFrame({"period": period_values, "contribution": contribution}).groupby(
        "period", sort=False, dropna=False
    ).sum().reset_index()


def factor_attribution(
    exposures: Any,
    factor_returns: Any,
    specific_returns: Any = None,
    *,
    actual_active_returns: Any = None,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Attribute stock active return to supplied factor and specific returns.

    The function intentionally does not fetch a risk model: the data adapter can
    supply Barra-style exposures and factor returns from rqdatac (or any other
    licensed source) without coupling credentials to the numerical engine.
    """

    warnings: List[Dict[str, Any]] = []
    factor_return_frame = _factor_returns_long(factor_returns)
    factor_names = factor_return_frame["factor"].drop_duplicates().tolist()
    exposure_frame = _exposures_long(exposures, factor_names)
    merged = exposure_frame.merge(
        factor_return_frame,
        how="outer",
        on=["period", "factor"],
        sort=False,
        indicator=True,
    )
    missing_exposure = merged["_merge"].eq("right_only")
    missing_return = merged["_merge"].eq("left_only")
    if missing_exposure.any():
        warnings.append(
            warning(
                "missing_factor_exposure",
                "Factor returns without matching active exposures were assigned zero exposure.",
                scope="factor",
                row_count=int(missing_exposure.sum()),
            )
        )
    if missing_return.any():
        warnings.append(
            warning(
                "missing_factor_return",
                "Active exposures without matching factor returns were assigned zero return.",
                scope="factor",
                row_count=int(missing_return.sum()),
            )
        )
    for column in ("portfolio_exposure", "benchmark_exposure", "active_exposure", "factor_return"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged["contribution"] = merged["active_exposure"] * merged["factor_return"]
    merged["component_type"] = "factor"
    result = merged[
        [
            "period",
            "factor",
            "component_type",
            "portfolio_exposure",
            "benchmark_exposure",
            "active_exposure",
            "factor_return",
            "contribution",
        ]
    ].copy()

    specific = _specific_effects(specific_returns)
    if specific.empty:
        warnings.append(
            warning(
                "missing_specific_returns",
                "Specific returns were not supplied; factor attribution covers systematic effects only.",
                severity="info",
                scope="factor",
            )
        )
    else:
        specific_rows = pd.DataFrame(
            {
                "period": specific["period"],
                "factor": "Specific",
                "component_type": "specific",
                "portfolio_exposure": np.nan,
                "benchmark_exposure": np.nan,
                "active_exposure": np.nan,
                "factor_return": np.nan,
                "contribution": specific["contribution"],
            }
        )
        result = pd.concat([result, specific_rows], ignore_index=True)

    reconciliations: List[Dict[str, Any]] = []
    if actual_active_returns is not None:
        actual_frame = _as_frame(actual_active_returns, "actual_active_returns")
        actual_col = _pick(actual_frame.columns, ("active_return", "actual_active_return", "return"))
        if actual_col is None:
            raise ValueError("actual_active_returns requires active_return")
        actual = pd.DataFrame(
            {
                "period": _period(actual_frame),
                "actual": pd.to_numeric(actual_frame[actual_col], errors="coerce").fillna(0.0),
            }
        ).groupby("period", sort=False).first().reset_index()
        explained = result.groupby("period", sort=False)["contribution"].sum().rename("explained").reset_index()
        checks = actual.merge(explained, how="outer", on="period").fillna(0.0)
        residual_rows: List[Dict[str, Any]] = []
        for row in checks.itertuples(index=False):
            check = reconciliation(
                f"factor:{row.period}", row.actual, row.explained, tolerance=tolerance, scope="factor"
            )
            residual = float(check["difference"])
            reconciliations.append(
                reconciliation(
                    f"factor_with_residual:{row.period}",
                    row.actual,
                    row.explained + residual,
                    tolerance=tolerance,
                    scope="factor",
                )
            )
            residual_rows.append(
                {
                    "period": row.period,
                    "factor": "Residual",
                    "component_type": "residual",
                    "portfolio_exposure": np.nan,
                    "benchmark_exposure": np.nan,
                    "active_exposure": np.nan,
                    "factor_return": np.nan,
                    "contribution": residual,
                }
            )
            if abs(residual) > max(tolerance, 1e-6):
                warnings.append(
                    warning(
                        "large_factor_residual",
                        "Factor and specific effects leave a material unexplained active return.",
                        scope="factor",
                        period=row.period,
                        residual=residual,
                    )
                )
        result = pd.concat([result, pd.DataFrame(residual_rows)], ignore_index=True)

    totals = (
        result.groupby(["period", "component_type"], sort=False)["contribution"]
        .sum()
        .reset_index()
        .to_dict("records")
    )
    return _finish(result, warnings, reconciliations, totals=totals, method="linear factor model")
