#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多资产 BHB 与股票行业 Brinson-Fachler。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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
from .models import normalize_asset_class, reconciliation, warning


def _allocation_side(
    data: Any,
    side: str,
    group_aliases: Sequence[str],
    *,
    normalize_classes: bool = False,
    warnings: List[Dict[str, Any]],
) -> pd.DataFrame:
    frame = _as_frame(data, side)
    output_columns = ["period", "group", f"{side}_weight", f"{side}_return", f"{side}_contribution", f"{side}_present"]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    group_column = _pick(frame.columns, group_aliases)
    if group_column is None:
        raise ValueError(f"{side} data is missing {group_aliases[0]}")
    groups = frame[group_column].astype(object)
    if normalize_classes:
        groups = groups.map(lambda value: normalize_asset_class(value, strict=False))
        invalid = groups.isna()
        if invalid.any():
            warnings.append(
                warning(
                    "unknown_asset_class",
                    f"Unsupported {side} asset-class rows were excluded.",
                    scope="asset_class",
                    values=sorted({str(value) for value in frame.loc[invalid, group_column]}),
                    row_count=int(invalid.sum()),
                )
            )
            frame = frame.loc[~invalid].copy()
            groups = groups.loc[~invalid]
    weight_column = _pick(frame.columns, (f"{side}_weight", "weight", "average_weight", "avg_weight"))
    return_column = _pick(frame.columns, (f"{side}_return", "return", "period_return", "asset_return"))
    contribution_column = _pick(frame.columns, (f"{side}_contribution", "contribution"))
    if weight_column is None:
        raise ValueError(f"{side} data is missing weight")
    if return_column is None and contribution_column is None:
        raise ValueError(f"{side} data is missing return or contribution")
    work = pd.DataFrame(
        {
            "period": _period(frame),
            "group": groups,
            "weight": pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0),
        }
    )
    if return_column is not None:
        work["return"] = pd.to_numeric(frame[return_column], errors="coerce").fillna(0.0)
        work["contribution"] = work["weight"] * work["return"]
    else:
        work["contribution"] = pd.to_numeric(frame[contribution_column], errors="coerce").fillna(0.0)
        work["return"] = np.where(work["weight"].abs() > _EPS, work["contribution"] / work["weight"], 0.0)
    grouped = work.groupby(["period", "group"], sort=False, dropna=False).agg(
        weight=("weight", "sum"), contribution=("contribution", "sum"), fallback_return=("return", "mean")
    )
    grouped = grouped.reset_index()
    grouped["return"] = np.where(
        grouped["weight"].abs() > _EPS,
        grouped["contribution"] / grouped["weight"],
        grouped["fallback_return"],
    )
    grouped[f"{side}_present"] = True
    return grouped.rename(
        columns={
            "weight": f"{side}_weight",
            "return": f"{side}_return",
            "contribution": f"{side}_contribution",
        }
    )[output_columns]


def _split_combined_attribution(frame: pd.DataFrame, group_aliases: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_column = _pick(frame.columns, group_aliases)
    period_column = _pick(frame.columns, _PERIOD_COLUMNS)
    if group_column is None:
        raise ValueError(f"combined attribution data is missing {group_aliases[0]}")
    shared = {group_column: frame[group_column]}
    if period_column is not None:
        shared[period_column] = frame[period_column]
    portfolio = pd.DataFrame(shared)
    benchmark = pd.DataFrame(shared)
    for side, destination in (("portfolio", portfolio), ("benchmark", benchmark)):
        for metric in ("weight", "return", "contribution"):
            column = _pick(frame.columns, (f"{side}_{metric}", f"{side[:1]}_{metric}"))
            if column is not None:
                destination[f"{side}_{metric}"] = frame[column]
    return portfolio, benchmark


def multi_asset_bhb(portfolio: Any, benchmark: Any = None, *, tolerance: float = 1e-10) -> pd.DataFrame:
    """Multi-asset BHB attribution with interaction folded into selection.

    For groups held on both sides, the no-interaction formulas are::

        allocation = (w_p - w_b) * r_b
        selection  = w_p * (r_p - r_b)

    This exactly reconciles to ``w_p*r_p - w_b*r_b``.  A portfolio-only or
    benchmark-only asset class cannot support a selection comparison, so its
    complete active contribution is assigned to allocation as required.
    """

    warnings: List[Dict[str, Any]] = []
    if benchmark is None:
        combined = _as_frame(portfolio, "asset-class attribution")
        portfolio, benchmark = _split_combined_attribution(
            combined, ("asset_class", "asset_type", "type")
        )
    p = _allocation_side(
        portfolio,
        "portfolio",
        ("asset_class", "asset_type", "type"),
        normalize_classes=True,
        warnings=warnings,
    )
    b = _allocation_side(
        benchmark,
        "benchmark",
        ("asset_class", "asset_type", "type"),
        normalize_classes=True,
        warnings=warnings,
    )
    merged = p.merge(b, how="outer", on=["period", "group"], sort=False)
    if merged.empty:
        return _finish(pd.DataFrame(), warnings)
    for column in (
        "portfolio_weight",
        "portfolio_return",
        "portfolio_contribution",
        "benchmark_weight",
        "benchmark_return",
        "benchmark_contribution",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    p_side = merged["portfolio_present"].fillna(False) & (merged["portfolio_weight"].abs() > _EPS)
    b_side = merged["benchmark_present"].fillna(False) & (merged["benchmark_weight"].abs() > _EPS)
    both = p_side & b_side
    active = merged["portfolio_contribution"] - merged["benchmark_contribution"]
    regular_allocation = (merged["portfolio_weight"] - merged["benchmark_weight"]) * merged["benchmark_return"]
    regular_selection = merged["portfolio_weight"] * (
        merged["portfolio_return"] - merged["benchmark_return"]
    )
    merged["allocation"] = np.where(both, regular_allocation, active)
    merged["selection"] = np.where(both, regular_selection, 0.0)
    merged["interaction"] = 0.0
    merged["total_effect"] = merged["allocation"] + merged["selection"]
    merged["active_contribution"] = active
    merged["one_sided"] = np.where(both, "both", np.where(p_side, "portfolio_only", "benchmark_only"))
    merged = merged.rename(columns={"group": "asset_class"})

    reconciliations: List[Dict[str, Any]] = []
    totals: List[Dict[str, Any]] = []
    for period_value, rows in merged.groupby("period", sort=False, dropna=False):
        p_weight = float(rows["portfolio_weight"].sum())
        b_weight = float(rows["benchmark_weight"].sum())
        if abs(p_weight - 1.0) > 1e-6:
            warnings.append(
                warning(
                    "portfolio_weights_not_one",
                    "Portfolio asset-class weights do not sum to one.",
                    scope="asset_class",
                    period=period_value,
                    weight_sum=p_weight,
                )
            )
        if abs(b_weight - 1.0) > 1e-6:
            warnings.append(
                warning(
                    "benchmark_weights_not_one",
                    "Benchmark asset-class weights do not sum to one.",
                    scope="asset_class",
                    period=period_value,
                    weight_sum=b_weight,
                )
            )
        actual = float(rows["active_contribution"].sum())
        explained = float(rows["total_effect"].sum())
        check = reconciliation(
            f"asset_class:{period_value}", actual, explained, tolerance=tolerance, scope="asset_class"
        )
        reconciliations.append(check)
        totals.append(
            {
                "period": period_value,
                "portfolio_return": float(rows["portfolio_contribution"].sum()),
                "benchmark_return": float(rows["benchmark_contribution"].sum()),
                "active_return": actual,
                "allocation": float(rows["allocation"].sum()),
                "selection": float(rows["selection"].sum()),
                "total_effect": explained,
            }
        )
    columns = [
        "period",
        "asset_class",
        "portfolio_weight",
        "benchmark_weight",
        "portfolio_return",
        "benchmark_return",
        "portfolio_contribution",
        "benchmark_contribution",
        "active_contribution",
        "allocation",
        "selection",
        "interaction",
        "total_effect",
        "one_sided",
    ]
    return _finish(merged[columns], warnings, reconciliations, totals=totals, method="BHB no-interaction")


def stock_industry_brinson_fachler(
    portfolio: Any,
    benchmark: Any = None,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Stock-industry Brinson-Fachler attribution without interaction.

    Allocation is measured relative to the total benchmark return and
    interaction is folded into selection.  As in the multi-asset routine,
    one-sided industries are assigned wholly to allocation.
    """

    warnings: List[Dict[str, Any]] = []
    aliases = ("industry", "industry_name", "industry_code", "sector")
    if benchmark is None:
        portfolio, benchmark = _split_combined_attribution(_as_frame(portfolio), aliases)
    p = _allocation_side(portfolio, "portfolio", aliases, warnings=warnings)
    b = _allocation_side(benchmark, "benchmark", aliases, warnings=warnings)
    merged = p.merge(b, how="outer", on=["period", "group"], sort=False)
    if merged.empty:
        return _finish(pd.DataFrame(), warnings)
    for column in (
        "portfolio_weight",
        "portfolio_return",
        "portfolio_contribution",
        "benchmark_weight",
        "benchmark_return",
        "benchmark_contribution",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    # An explicit map avoids version-specific groupby.apply behavior.
    benchmark_map: Dict[Any, float] = {}
    for period_value, rows in merged.groupby("period", sort=False, dropna=False):
        weight_sum = float(rows["benchmark_weight"].sum())
        benchmark_map[period_value] = (
            float(rows["benchmark_contribution"].sum()) / weight_sum if abs(weight_sum) > _EPS else 0.0
        )
    merged["benchmark_total_return"] = merged["period"].map(benchmark_map).astype(float)
    p_side = merged["portfolio_present"].fillna(False) & (merged["portfolio_weight"].abs() > _EPS)
    b_side = merged["benchmark_present"].fillna(False) & (merged["benchmark_weight"].abs() > _EPS)
    both = p_side & b_side
    active = merged["portfolio_contribution"] - merged["benchmark_contribution"]
    regular_allocation = (merged["portfolio_weight"] - merged["benchmark_weight"]) * (
        merged["benchmark_return"] - merged["benchmark_total_return"]
    )
    regular_selection = merged["portfolio_weight"] * (
        merged["portfolio_return"] - merged["benchmark_return"]
    )
    # With no comparator, the full relative-to-benchmark contribution belongs
    # to allocation.  This preserves BF's benchmark-total baseline.
    one_sided_allocation = np.where(
        p_side,
        merged["portfolio_contribution"] - merged["portfolio_weight"] * merged["benchmark_total_return"],
        -merged["benchmark_contribution"] + merged["benchmark_weight"] * merged["benchmark_total_return"],
    )
    merged["allocation"] = np.where(both, regular_allocation, one_sided_allocation)
    merged["selection"] = np.where(both, regular_selection, 0.0)
    merged["interaction"] = 0.0
    merged["total_effect"] = merged["allocation"] + merged["selection"]
    merged["active_contribution"] = active
    merged["one_sided"] = np.where(both, "both", np.where(p_side, "portfolio_only", "benchmark_only"))
    merged = merged.rename(columns={"group": "industry"})

    reconciliations: List[Dict[str, Any]] = []
    totals: List[Dict[str, Any]] = []
    for period_value, rows in merged.groupby("period", sort=False, dropna=False):
        p_weight = float(rows["portfolio_weight"].sum())
        b_weight = float(rows["benchmark_weight"].sum())
        if abs(p_weight - b_weight) > 1e-6:
            warnings.append(
                warning(
                    "industry_weight_scope_mismatch",
                    "Portfolio and benchmark industry weights cover different totals; BF cannot fully reconcile without a cash/other bucket.",
                    scope="industry",
                    period=period_value,
                    portfolio_weight_sum=p_weight,
                    benchmark_weight_sum=b_weight,
                )
            )
        if abs(p_weight - 1.0) > 1e-6 or abs(b_weight - 1.0) > 1e-6:
            warnings.append(
                warning(
                    "industry_weights_not_one",
                    "Industry weights are expected to be normalized within the stock sleeve.",
                    severity="info",
                    scope="industry",
                    period=period_value,
                    portfolio_weight_sum=p_weight,
                    benchmark_weight_sum=b_weight,
                )
            )
        actual = float(rows["active_contribution"].sum())
        explained = float(rows["total_effect"].sum())
        check = reconciliation(f"industry:{period_value}", actual, explained, tolerance=tolerance, scope="industry")
        reconciliations.append(check)
        if not check["passed"]:
            warnings.append(
                warning(
                    "industry_reconciliation_failed",
                    "Industry attribution does not reconcile; check sleeve weight coverage.",
                    scope="industry",
                    period=period_value,
                    difference=check["difference"],
                )
            )
        totals.append(
            {
                "period": period_value,
                "portfolio_return": float(rows["portfolio_contribution"].sum()),
                "benchmark_return": float(rows["benchmark_contribution"].sum()),
                "active_return": actual,
                "allocation": float(rows["allocation"].sum()),
                "selection": float(rows["selection"].sum()),
                "total_effect": explained,
            }
        )
    columns = [
        "period",
        "industry",
        "portfolio_weight",
        "benchmark_weight",
        "portfolio_return",
        "benchmark_return",
        "benchmark_total_return",
        "active_contribution",
        "allocation",
        "selection",
        "interaction",
        "total_effect",
        "one_sided",
    ]
    return _finish(
        merged[columns], warnings, reconciliations, totals=totals, method="Brinson-Fachler no-interaction"
    )


calculate_bhb_attribution = multi_asset_bhb
industry_attribution = stock_industry_brinson_fachler
brinson_fachler = stock_industry_brinson_fachler
