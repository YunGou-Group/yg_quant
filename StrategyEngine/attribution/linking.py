#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多期 Carino 链接。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

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


def _carino_coefficient(portfolio_return: np.ndarray, benchmark_return: np.ndarray) -> np.ndarray:
    active = portfolio_return - benchmark_return
    numerator = np.log1p(portfolio_return) - np.log1p(benchmark_return)
    limit = 1.0 / (1.0 + benchmark_return)
    return np.where(np.abs(active) > _EPS, numerator / active, limit)


def carino_link(
    period_returns: Any,
    effects: Any = None,
    *,
    effect_columns: Optional[Sequence[str]] = None,
    tolerance: float = 1e-10,
    enforce_reconciliation: bool = True,
) -> pd.DataFrame:
    """Carino-link single-period attribution effects across multiple periods.

    Linked effect is ``effect_t * k_t / K``, where ``k_t`` and ``K`` are the
    period and full-horizon logarithmic return coefficients.  If supplied
    effects do not sum to period active return, an explicit ``linking_residual``
    is added (to the first row of each period) so the linked result reconciles.
    """

    returns_frame = _as_frame(period_returns, "period_returns")
    if returns_frame.empty:
        return _finish(pd.DataFrame())
    p_col = _pick(returns_frame.columns, ("portfolio_return", "strategy_return", "return"))
    b_col = _pick(returns_frame.columns, ("benchmark_return", "index_return"))
    if p_col is None or b_col is None:
        raise ValueError("period_returns requires portfolio_return and benchmark_return")
    returns = pd.DataFrame(
        {
            "period": _period(returns_frame),
            "portfolio_return": pd.to_numeric(returns_frame[p_col], errors="coerce").fillna(0.0),
            "benchmark_return": pd.to_numeric(returns_frame[b_col], errors="coerce").fillna(0.0),
        }
    ).groupby("period", sort=False, dropna=False).agg(
        portfolio_return=("portfolio_return", "first"), benchmark_return=("benchmark_return", "first")
    ).reset_index()
    if (returns[["portfolio_return", "benchmark_return"]] <= -1.0).any().any():
        raise ValueError("Carino linking requires every period return to be greater than -100%")

    if effects is None:
        effect_frame = returns_frame.copy()
    else:
        effect_frame = _as_frame(effects, "effects")
    effect_frame = effect_frame.copy()
    effect_frame["period"] = _period(effect_frame)
    if effect_columns is None:
        excluded = set(_PERIOD_COLUMNS) | {
            "portfolio_return",
            "strategy_return",
            "return",
            "benchmark_return",
            "index_return",
            "portfolio_weight",
            "benchmark_weight",
            "carino_k",
            "carino_K",
            "linking_factor",
        }
        effect_columns = [
            column
            for column in effect_frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(effect_frame[column])
        ]
    effect_columns = list(effect_columns)
    if not effect_columns:
        raise ValueError("no attribution effect columns were supplied for Carino linking")
    for column in effect_columns:
        if column not in effect_frame:
            raise ValueError(f"effect column {column!r} is missing")
        effect_frame[column] = pd.to_numeric(effect_frame[column], errors="coerce").fillna(0.0)
    result = effect_frame.merge(returns, how="left", on="period", suffixes=("", "_period"), sort=False)
    missing_returns = result["portfolio_return"].isna() | result["benchmark_return"].isna()
    if missing_returns.any():
        raise ValueError("effects contain periods that are absent from period_returns")

    warnings: List[Dict[str, Any]] = []
    if enforce_reconciliation:
        result["linking_residual"] = 0.0
        period_effects = result.groupby("period", sort=False)[effect_columns].sum().sum(axis=1)
        active_map = returns.set_index("period").eval("portfolio_return - benchmark_return")
        residual_map = active_map.subtract(period_effects, fill_value=0.0)
        for period_value, difference in residual_map.items():
            if abs(float(difference)) > tolerance:
                indices = result.index[result["period"] == period_value]
                if len(indices):
                    result.loc[indices[0], "linking_residual"] = float(difference)
                    warnings.append(
                        warning(
                            "carino_period_residual_added",
                            "A period residual was added before linking because supplied effects did not equal active return.",
                            scope="linking",
                            period=period_value,
                            difference=float(difference),
                        )
                    )
        if result["linking_residual"].abs().max() > tolerance:
            effect_columns.append("linking_residual")
        else:
            result = result.drop(columns=["linking_residual"])

    rp = result["portfolio_return"].to_numpy(dtype=float)
    rb = result["benchmark_return"].to_numpy(dtype=float)
    result["carino_k"] = _carino_coefficient(rp, rb)
    total_portfolio = float(np.prod(1.0 + returns["portfolio_return"].to_numpy(dtype=float)) - 1.0)
    total_benchmark = float(np.prod(1.0 + returns["benchmark_return"].to_numpy(dtype=float)) - 1.0)
    if total_portfolio <= -1.0 or total_benchmark <= -1.0:
        raise ValueError("invalid cumulative return for Carino linking")
    total_active = total_portfolio - total_benchmark
    if abs(total_active) > _EPS:
        total_k = (np.log1p(total_portfolio) - np.log1p(total_benchmark)) / total_active
    else:
        total_k = 1.0 / (1.0 + total_benchmark)
    result["carino_K"] = float(total_k)
    result["linking_factor"] = result["carino_k"] / float(total_k)
    linked_columns: List[str] = []
    for column in effect_columns:
        linked = f"{column}_linked"
        result[linked] = result[column] * result["linking_factor"]
        linked_columns.append(linked)
    if len(effect_columns) == 1:
        result["linked_effect"] = result[linked_columns[0]]

    linked_total = float(result[linked_columns].sum().sum())
    check = reconciliation(
        "carino_total_active_return",
        total_active,
        linked_total,
        tolerance=tolerance,
        scope="linking",
    )
    if not check["passed"]:
        warnings.append(
            warning(
                "carino_reconciliation_failed",
                "Linked attribution does not reconcile to cumulative active return.",
                scope="linking",
                difference=check["difference"],
            )
        )
    return _finish(
        result,
        warnings,
        [check],
        totals={
            "portfolio_return": total_portfolio,
            "benchmark_return": total_benchmark,
            "active_return": total_active,
            "linked_effect": linked_total,
            "carino_K": float(total_k),
        },
        effect_columns=effect_columns,
        linked_columns=linked_columns,
    )
