#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按资产池跑盯市、Brinson、因子与专业分解，输出 JSON 报告。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .brinson import multi_asset_bhb, stock_industry_brinson_fachler
from .coerce import (
    AMOUNT_UNIT,
    RETURN_UNIT,
    _attach_factor_weights,
    _collect_frame_metadata,
    _extract_sided,
    _first,
    _infer_asset_pool,
    _json_value,
    _mapping,
    _prepare_period_returns,
    _records,
    _section,
    _units_for_columns,
)
from .factor_decomp import factor_attribution
from .linking import link_brinson_effects, subsequent_growth
from .models import (
    EXPECTED_INPUT_FIELDS,
    AttributionOutput,
    normalize_asset_classes,
    reconciliation,
    warning,
)
from .mtm import calculate_holding_mtm, calculate_signed_trade_mtm
from .specialized import bond_attribution, futures_attribution, option_attribution


class AttributionEngine:
    """State-free façade; instantiate when dependency injection prefers objects."""

    def __init__(self, *, tolerance: float = 1e-10):
        self.tolerance = float(tolerance)

    def run(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return run_attribution(payload, tolerance=self.tolerance)


def run_attribution(payload: Mapping[str, Any], *, tolerance: float = 1e-10) -> Dict[str, Any]:
    """Run every attribution model relevant to the declared asset pool.

    Invalid or incomplete optional sections produce structured quality warnings
    while valid sections still render.  A non-mapping payload is a programming
    error and raises ``TypeError``.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    quality_warnings: List[Dict[str, Any]] = []
    reconciliations: List[Dict[str, Any]] = []
    sections: Dict[str, Any] = {}
    tables: List[Dict[str, Any]] = []

    raw_pool = _first(payload, ("asset_pool", "asset_classes", "asset_class"))
    try:
        asset_pool = normalize_asset_classes(raw_pool, strict=True) if raw_pool is not None else []
    except ValueError as exc:
        asset_pool = normalize_asset_classes(raw_pool, strict=False)
        quality_warnings.append(
            warning("unknown_asset_pool", str(exc), scope="asset_pool", supplied=_json_value(raw_pool))
        )
    if not asset_pool:
        asset_pool = _infer_asset_pool(payload)
        quality_warnings.append(
            warning(
                "asset_pool_inferred" if asset_pool else "missing_asset_pool",
                "Asset pool was inferred from supplied sections."
                if asset_pool
                else "Asset pool is missing and could not be inferred.",
                severity="info" if asset_pool else "warning",
                scope="asset_pool",
                inferred=asset_pool,
            )
        )

    # Returns -----------------------------------------------------------------
    returns_data = _first(
        payload,
        ("returns", "portfolio_daily", "daily_returns", "period_returns", "performance"),
    )
    try:
        periods_frame = _prepare_period_returns(returns_data)
    except (ValueError, TypeError) as exc:
        periods_frame = pd.DataFrame(columns=["period", "portfolio_return", "benchmark_return", "active_return"])
        quality_warnings.append(warning("invalid_returns", str(exc), scope="returns"))
    if periods_frame.empty:
        quality_warnings.append(
            warning(
                "missing_returns",
                "Portfolio and benchmark period returns are required for total-return and Carino summaries.",
                scope="returns",
            )
        )

    # Signed trade and holding MTM --------------------------------------------
    try:
        trade_frame = calculate_signed_trade_mtm(payload.get("trades"))
        _collect_frame_metadata(trade_frame, quality_warnings, reconciliations)
    except (ValueError, TypeError, KeyError) as exc:
        trade_frame = pd.DataFrame()
        quality_warnings.append(warning("invalid_trades", str(exc), scope="trades"))
    try:
        holding_frame = calculate_holding_mtm(payload.get("holdings"))
        _collect_frame_metadata(holding_frame, quality_warnings, reconciliations)
    except (ValueError, TypeError, KeyError) as exc:
        holding_frame = pd.DataFrame()
        quality_warnings.append(warning("invalid_holdings", str(exc), scope="holdings"))

    pnl_periods: List[pd.DataFrame] = []
    if not trade_frame.empty:
        pnl_periods.append(
            trade_frame.groupby("period", sort=False).agg(
                trade_mtm=("trade_mtm", "sum"), trade_fees=("fees", "sum")
            ).reset_index()
        )
    if not holding_frame.empty:
        pnl_periods.append(
            holding_frame.groupby("period", sort=False).agg(
                holding_mtm=("holding_mtm", "sum"), holding_fees=("fees", "sum")
            ).reset_index()
        )
    if pnl_periods:
        pnl_by_period = pnl_periods[0]
        for next_frame in pnl_periods[1:]:
            pnl_by_period = pnl_by_period.merge(next_frame, how="outer", on="period", sort=False)
        for column in ("trade_mtm", "trade_fees", "holding_mtm", "holding_fees"):
            if column not in pnl_by_period:
                pnl_by_period[column] = 0.0
            pnl_by_period[column] = pd.to_numeric(pnl_by_period[column], errors="coerce").fillna(0.0)
        pnl_by_period["total_mtm"] = pnl_by_period["trade_mtm"] + pnl_by_period["holding_mtm"]
        pnl_by_period["total_fees"] = pnl_by_period["trade_fees"] + pnl_by_period["holding_fees"]
    else:
        pnl_by_period = pd.DataFrame(
            columns=["period", "trade_mtm", "holding_mtm", "total_mtm", "trade_fees", "holding_fees", "total_fees"]
        )

    # Convert MTM amounts to return contributions when beginning NAV is
    # available.  The amount tables remain present regardless, which makes the
    # behavior explicit instead of fabricating a denominator.
    pnl_rows_supplied = not trade_frame.empty or not holding_frame.empty
    for column in ("trade_return", "holding_return", "total_mtm_return"):
        if column not in periods_frame:
            periods_frame[column] = np.nan
    if "leverage_return" not in periods_frame:
        periods_frame["leverage_return"] = 0.0
    else:
        periods_frame["leverage_return"] = pd.to_numeric(
            periods_frame["leverage_return"], errors="coerce"
        ).fillna(0.0)
    if not periods_frame.empty and "nav_begin" in periods_frame and pnl_rows_supplied:
        pnl_lookup = pnl_by_period.set_index("period") if not pnl_by_period.empty else pd.DataFrame()
        trade_amounts: List[float] = []
        holding_amounts: List[float] = []
        for period_value in periods_frame["period"]:
            if not pnl_by_period.empty and period_value in pnl_lookup.index:
                pnl_row = pnl_lookup.loc[period_value]
                # Duplicate labels should not occur after grouping, but sum is
                # a safe fallback if an exotic period object compares equal.
                if isinstance(pnl_row, pd.DataFrame):
                    trade_amounts.append(float(pnl_row["trade_mtm"].sum()))
                    holding_amounts.append(float(pnl_row["holding_mtm"].sum()))
                else:
                    trade_amounts.append(float(pnl_row["trade_mtm"]))
                    holding_amounts.append(float(pnl_row["holding_mtm"]))
            else:
                trade_amounts.append(0.0)
                holding_amounts.append(0.0)
        nav_values = pd.to_numeric(periods_frame["nav_begin"], errors="coerce")
        valid_nav = nav_values > 0.0
        if not trade_frame.empty:
            periods_frame.loc[valid_nav, "trade_return"] = (
                np.asarray(trade_amounts, dtype=float)[valid_nav.to_numpy()] / nav_values[valid_nav]
            )
        else:
            periods_frame.loc[valid_nav, "trade_return"] = periods_frame.loc[
                valid_nav, "trade_return"
            ].fillna(0.0)
        if not holding_frame.empty:
            periods_frame.loc[valid_nav, "holding_return"] = (
                np.asarray(holding_amounts, dtype=float)[valid_nav.to_numpy()] / nav_values[valid_nav]
            )
        else:
            periods_frame.loc[valid_nav, "holding_return"] = periods_frame.loc[
                valid_nav, "holding_return"
            ].fillna(
                periods_frame.loc[valid_nav, "portfolio_return"]
                - periods_frame.loc[valid_nav, "trade_return"]
                - periods_frame.loc[valid_nav, "leverage_return"]
            )
        periods_frame.loc[valid_nav, "total_mtm_return"] = (
            periods_frame.loc[valid_nav, "trade_return"]
            + periods_frame.loc[valid_nav, "holding_return"]
        )
        invalid_nav = ~valid_nav
        if invalid_nav.any() and pnl_rows_supplied:
            quality_warnings.append(
                warning(
                    "invalid_nav_begin_for_pnl_returns",
                    "Beginning NAV is missing or non-positive for some periods; their MTM remains amount-only.",
                    severity="info",
                    scope="returns",
                    periods=_json_value(periods_frame.loc[invalid_nav, "period"].tolist()),
                )
            )
    elif pnl_rows_supplied:
        quality_warnings.append(
            warning(
                "missing_nav_begin_for_pnl_returns",
                "Trade and holding MTM are shown as amounts because returns/portfolio_daily has no nav_begin (or beginning_nav).",
                severity="info",
                scope="returns",
            )
        )

    # With no transaction/holding ledgers, the least-assumptive Brinson basis
    # is that the entire portfolio return is holding return.  Explicit component
    # returns in returns/portfolio_daily take precedence.
    if not pnl_rows_supplied and not periods_frame.empty:
        periods_frame["trade_return"] = periods_frame["trade_return"].fillna(0.0)
        periods_frame["holding_return"] = periods_frame["holding_return"].fillna(
            periods_frame["portfolio_return"] - periods_frame["leverage_return"]
        )
        periods_frame["total_mtm_return"] = (
            periods_frame["trade_return"] + periods_frame["holding_return"]
        )

    if "benchmark_holding_return" not in periods_frame:
        periods_frame["benchmark_holding_return"] = periods_frame["benchmark_return"]
    else:
        periods_frame["benchmark_holding_return"] = pd.to_numeric(
            periods_frame["benchmark_holding_return"], errors="coerce"
        ).fillna(periods_frame["benchmark_return"])
    periods_frame["holding_active_return"] = (
        periods_frame["holding_return"] - periods_frame["benchmark_holding_return"]
    )

    periods_frame["pnl_return_residual"] = np.where(
        periods_frame["total_mtm_return"].notna(),
        periods_frame["portfolio_return"]
        - periods_frame["total_mtm_return"]
        - periods_frame["leverage_return"],
        np.nan,
    )
    valid_pnl_return_rows = periods_frame["total_mtm_return"].notna()
    for row in periods_frame.loc[valid_pnl_return_rows].itertuples(index=False):
        explained_return = float(
            row.trade_return + row.holding_return + row.leverage_return + row.pnl_return_residual
        )
        reconciliations.append(
            reconciliation(
                f"pnl_return:{row.period}",
                float(row.portfolio_return),
                explained_return,
                tolerance=tolerance,
                scope="pnl_return",
            )
        )
        if abs(float(row.pnl_return_residual)) > max(tolerance, 1e-6):
            quality_warnings.append(
                warning(
                    "pnl_return_residual_present",
                    "Portfolio return contains effects outside signed trade, leverage and opening-holding MTM; the difference is shown as residual.",
                    severity="info",
                    scope="pnl_return",
                    period=_json_value(row.period),
                    residual=float(row.pnl_return_residual),
                )
            )

    if pnl_rows_supplied and not pnl_by_period.empty and not periods_frame.empty:
        known_periods = set(periods_frame["period"].tolist())
        unmatched_periods = [
            value for value in pnl_by_period["period"].tolist() if value not in known_periods
        ]
        if unmatched_periods:
            quality_warnings.append(
                warning(
                    "pnl_period_missing_from_returns",
                    "Some MTM periods are absent from returns/portfolio_daily and cannot be converted to return contributions.",
                    severity="info",
                    scope="returns",
                    periods=_json_value(unmatched_periods),
                )
            )

    if not pnl_by_period.empty:
        period_return_fields = periods_frame[
            ["period", "nav_begin", "trade_return", "holding_return", "leverage_return", "total_mtm_return", "benchmark_holding_return", "holding_active_return", "pnl_return_residual"]
            if "nav_begin" in periods_frame
            else ["period", "trade_return", "holding_return", "leverage_return", "total_mtm_return", "benchmark_holding_return", "holding_active_return", "pnl_return_residual"]
        ]
        pnl_by_period = pnl_by_period.merge(period_return_fields, how="left", on="period", sort=False)
    pnl = {
        "trade_columns": list(trade_frame.columns),
        "trade_units": _units_for_columns(trade_frame.columns),
        "trades": _records(trade_frame),
        "holding_columns": list(holding_frame.columns),
        "holding_units": _units_for_columns(holding_frame.columns),
        "holdings": _records(holding_frame),
        "by_period_columns": list(pnl_by_period.columns),
        "by_period_units": _units_for_columns(pnl_by_period.columns),
        "by_period": _records(pnl_by_period),
        "totals": {
            "trade_mtm": float(trade_frame["trade_mtm"].sum()) if "trade_mtm" in trade_frame else 0.0,
            "holding_mtm": float(holding_frame["holding_mtm"].sum()) if "holding_mtm" in holding_frame else 0.0,
            "fees": (
                float(trade_frame["fees"].sum()) if "fees" in trade_frame else 0.0
            )
            + (float(holding_frame["fees"].sum()) if "fees" in holding_frame else 0.0),
        },
    }
    pnl["totals"]["total_mtm"] = pnl["totals"]["trade_mtm"] + pnl["totals"]["holding_mtm"]

    # Multi-asset BHB ---------------------------------------------------------
    asset_data = _first(
        payload,
        ("asset_class_attribution", "asset_attribution", "asset_class_rows", "allocation_data"),
    )
    if asset_data is None:
        portfolio_assets = _first(payload, ("portfolio_asset_classes", "portfolio_allocation"))
        benchmark_assets = _first(payload, ("benchmark_asset_classes", "benchmark_allocation"))
    else:
        portfolio_assets, benchmark_assets, combined_assets = _extract_sided(asset_data)
        if portfolio_assets is None and benchmark_assets is None:
            portfolio_assets = combined_assets
            benchmark_assets = None
    asset_frame = pd.DataFrame()
    if asset_data is not None or (portfolio_assets is not None and benchmark_assets is not None):
        try:
            asset_frame = multi_asset_bhb(portfolio_assets, benchmark_assets, tolerance=tolerance)
            _collect_frame_metadata(asset_frame, quality_warnings, reconciliations)
            section = _section(
                "asset_class",
                "资产类别归因",
                "BHB no-interaction; one-sided classes assigned to allocation",
                asset_frame,
            )
            sections["asset_class"] = section
            tables.append(section)
        except (ValueError, TypeError, KeyError) as exc:
            quality_warnings.append(warning("invalid_asset_class_attribution", str(exc), scope="asset_class"))
    elif len(asset_pool) > 1:
        quality_warnings.append(
            warning(
                "missing_asset_class_attribution",
                "A multi-asset pool needs portfolio and benchmark asset-class weights and returns.",
                scope="asset_class",
            )
        )

    # Stock industry BF and factor model -------------------------------------
    stock_config = _mapping(payload.get("stock"))
    industry_data = _first(payload, ("industry_attribution", "stock_industry"))
    if industry_data is None:
        industry_data = _first(stock_config, ("industry", "industries", "industry_attribution"))
    industry_frame = pd.DataFrame()
    if industry_data is not None:
        p_industry, b_industry, combined_industry = _extract_sided(industry_data)
        if p_industry is None and b_industry is None:
            p_industry, b_industry = combined_industry, None
        try:
            industry_frame = stock_industry_brinson_fachler(
                p_industry, b_industry, tolerance=tolerance
            )
            _collect_frame_metadata(industry_frame, quality_warnings, reconciliations)
            section = _section(
                "industry",
                "股票行业归因",
                "Brinson-Fachler no-interaction",
                industry_frame,
            )
            sections["industry"] = section
            tables.append(section)
        except (ValueError, TypeError, KeyError) as exc:
            quality_warnings.append(warning("invalid_industry_attribution", str(exc), scope="industry"))
    elif "stock" in asset_pool:
        quality_warnings.append(
            warning(
                "missing_industry_data",
                "Stock attribution needs portfolio and benchmark industry weights and returns.",
                severity="info",
                scope="industry",
            )
        )

    factor_config = _first(payload, ("factor_attribution", "factor"))
    if factor_config is None:
        factor_config = _first(stock_config, ("factor", "factors", "factor_attribution"))
    factor_frame = pd.DataFrame()
    if factor_config is not None:
        factor_map = _mapping(factor_config)
        exposures = _first(factor_map, ("exposures", "factor_exposures"))
        factor_returns_data = _first(factor_map, ("factor_returns", "returns"))
        specific = _first(factor_map, ("specific_returns", "specific"))
        portfolio_factor_weights = _first(
            factor_map, ("portfolio_weights", "strategy_weights", "portfolio_holdings")
        )
        benchmark_factor_weights = _first(
            factor_map, ("benchmark_weights", "index_weights", "benchmark_holdings")
        )
        if portfolio_factor_weights is None:
            portfolio_factor_weights = _first(
                payload, ("portfolio_factor_weights", "portfolio_security_weights")
            )
        if benchmark_factor_weights is None:
            benchmark_factor_weights = _first(
                payload, ("benchmark_factor_weights", "benchmark_security_weights")
            )
        exposures = _attach_factor_weights(
            exposures, portfolio_factor_weights, benchmark_factor_weights
        )
        specific = _attach_factor_weights(
            specific, portfolio_factor_weights, benchmark_factor_weights
        )
        actual_factor = _first(factor_map, ("actual_active_returns", "active_returns"))
        if actual_factor is None and asset_pool == ["stock"] and not periods_frame.empty:
            if periods_frame["holding_active_return"].notna().all():
                actual_factor = periods_frame[["period", "holding_active_return"]].rename(
                    columns={"holding_active_return": "active_return"}
                )
            else:
                actual_factor = periods_frame[["period", "active_return"]]
        try:
            factor_frame = factor_attribution(
                exposures,
                factor_returns_data,
                specific,
                actual_active_returns=actual_factor,
                tolerance=tolerance,
            )
            _collect_frame_metadata(factor_frame, quality_warnings, reconciliations)
            section = _section(
                "factor",
                "股票因子归因",
                "linear active-exposure × factor-return model plus specific return",
                factor_frame,
                note="Factor data is supplied by the data adapter (for example rqdatac/Barra-style fields).",
            )
            sections["factor"] = section
            tables.append(section)
        except (ValueError, TypeError, KeyError) as exc:
            quality_warnings.append(warning("invalid_factor_attribution", str(exc), scope="factor"))
    elif "stock" in asset_pool:
        quality_warnings.append(
            warning(
                "missing_factor_data",
                "Stock factor attribution needs exposures, factor returns and preferably specific returns.",
                severity="info",
                scope="factor",
            )
        )

    # Asset-specific decompositions ------------------------------------------
    decomposition_specs = [
        (
            "bond",
            ("bond_attribution", "bond", "bonds"),
            bond_attribution,
            "债券收益归因",
            "carry / curve / spread / convexity / residual",
        ),
        (
            "future",
            ("futures_attribution", "future_attribution", "future", "futures"),
            futures_attribution,
            "期货收益归因",
            "underlying / basis / roll / collateral / leverage / residual",
        ),
        (
            "option",
            ("option_attribution", "option", "options"),
            option_attribution,
            "期权希腊字母归因",
            "delta / gamma / vega / theta / rho / residual",
        ),
    ]
    for asset_class, aliases, calculator, title, method in decomposition_specs:
        raw = _first(payload, aliases)
        if isinstance(raw, Mapping):
            nested_rows = _first(raw, ("rows", "data", "components", "attribution"))
            if nested_rows is not None:
                raw = nested_rows
        if raw is not None:
            try:
                frame = calculator(raw, tolerance=tolerance)
                _collect_frame_metadata(frame, quality_warnings, reconciliations)
                key = "futures" if asset_class == "future" else ("options" if asset_class == "option" else "bond")
                section = _section(key, title, method, frame)
                sections[key] = section
                tables.append(section)
            except (ValueError, TypeError, KeyError) as exc:
                quality_warnings.append(
                    warning(f"invalid_{asset_class}_attribution", str(exc), scope=asset_class)
                )
        elif asset_class in asset_pool:
            quality_warnings.append(
                warning(
                    f"missing_{asset_class}_data",
                    f"Declared {asset_class} assets have no component-attribution input.",
                    severity="info",
                    scope=asset_class,
                )
            )

    # Carino linking: asset-class BHB, else stock-industry BF -----------------
    linked_frame = pd.DataFrame()
    brinson_return_basis = "not_available"
    brinson_effect_source = "none"
    effect_source = asset_frame if not asset_frame.empty else industry_frame
    if not effect_source.empty:
        brinson_effect_source = "asset_class" if not asset_frame.empty else "industry"
        totals_basis = (
            "asset_class_input_fallback"
            if brinson_effect_source == "asset_class"
            else "industry_input_fallback"
        )
        try:
            linked_frame, brinson_return_basis, link_warnings = link_brinson_effects(
                effect_source,
                periods_frame,
                tolerance=tolerance,
                totals=effect_source.attrs.get("totals", []),
                totals_basis=totals_basis,
            )
            quality_warnings.extend(link_warnings)
            if not linked_frame.empty:
                _collect_frame_metadata(linked_frame, quality_warnings, reconciliations)
                section = _section(
                    "linked",
                    "多期链接归因",
                    "Carino logarithmic linking",
                    linked_frame,
                    note=f"Return basis: {brinson_return_basis}; effects: {brinson_effect_source}",
                )
                sections["linked"] = section
                tables.append(section)
        except (ValueError, TypeError, KeyError) as exc:
            quality_warnings.append(warning("invalid_carino_linking", str(exc), scope="linking"))

    # Summary -----------------------------------------------------------------
    if not periods_frame.empty:
        portfolio_total = float(np.prod(1.0 + periods_frame["portfolio_return"].to_numpy(dtype=float)) - 1.0)
        benchmark_total = float(np.prod(1.0 + periods_frame["benchmark_return"].to_numpy(dtype=float)) - 1.0)
        active_total = portfolio_total - benchmark_total
    elif not asset_frame.empty and asset_frame.attrs.get("totals"):
        totals_frame = pd.DataFrame(asset_frame.attrs["totals"])
        portfolio_total = float(np.prod(1.0 + totals_frame["portfolio_return"].to_numpy(dtype=float)) - 1.0)
        benchmark_total = float(np.prod(1.0 + totals_frame["benchmark_return"].to_numpy(dtype=float)) - 1.0)
        active_total = portfolio_total - benchmark_total
    else:
        portfolio_total = benchmark_total = active_total = 0.0

    if not periods_frame.empty and periods_frame["holding_return"].notna().all():
        holding_cumulative_return = float(
            np.prod(1.0 + periods_frame["holding_return"].to_numpy(dtype=float)) - 1.0
        )
        benchmark_holding_cumulative_return = float(
            np.prod(1.0 + periods_frame["benchmark_holding_return"].to_numpy(dtype=float)) - 1.0
        )
        holding_active_total = holding_cumulative_return - benchmark_holding_cumulative_return
    else:
        # The fallback is explicitly labelled in meta/methodology and warnings.
        holding_cumulative_return = portfolio_total
        benchmark_holding_cumulative_return = benchmark_total
        holding_active_total = active_total

    if not linked_frame.empty:
        linked_columns = linked_frame.attrs.get("linked_columns", [])
        attribution_total = float(linked_frame[linked_columns].sum().sum()) if linked_columns else 0.0
        allocation_total = float(linked_frame["allocation_linked"].sum()) if "allocation_linked" in linked_frame else 0.0
        selection_total = float(linked_frame["selection_linked"].sum()) if "selection_linked" in linked_frame else 0.0
    elif not asset_frame.empty:
        attribution_total = float(asset_frame["total_effect"].sum())
        allocation_total = float(asset_frame["allocation"].sum())
        selection_total = float(asset_frame["selection"].sum())
    elif not industry_frame.empty:
        attribution_total = float(industry_frame["total_effect"].sum())
        allocation_total = float(industry_frame["allocation"].sum())
        selection_total = float(industry_frame["selection"].sum())
    else:
        attribution_total = 0.0
        allocation_total = 0.0
        selection_total = 0.0
    attribution_gap = (
        holding_active_total - attribution_total
        if (not asset_frame.empty or not industry_frame.empty or not linked_frame.empty)
        else None
    )

    # Link P&L return contributions through the portfolio wealth chain.  For a
    # single period this is exactly MTM / beginning NAV.  For multiple periods,
    # each effect is grown by all subsequent portfolio returns; trade, holding
    # and the explicit residual therefore sum to compounded portfolio return.
    if (
        not periods_frame.empty
        and periods_frame[["trade_return", "holding_return", "leverage_return"]].notna().all().all()
        and (
            pnl_by_period.empty
            or set(pnl_by_period["period"].tolist()).issubset(set(periods_frame["period"].tolist()))
        )
    ):
        portfolio_path = periods_frame["portfolio_return"].to_numpy(dtype=float)
        trade_path = periods_frame["trade_return"].to_numpy(dtype=float)
        holding_path = periods_frame["holding_return"].to_numpy(dtype=float)
        leverage_path = periods_frame["leverage_return"].to_numpy(dtype=float)
        growth_after = subsequent_growth(portfolio_path)
        linked_trade_return = float(np.sum(trade_path * growth_after))
        linked_holding_return = float(np.sum(holding_path * growth_after))
        linked_leverage_return: Optional[float] = float(np.sum(leverage_path * growth_after))
        residual_path = portfolio_path - trade_path - holding_path - leverage_path
        linked_pnl_residual = float(np.sum(residual_path * growth_after))
    else:
        linked_trade_return = None
        linked_holding_return = None
        linked_leverage_return = None
        linked_pnl_residual = None
    if (
        linked_trade_return is not None
        and linked_holding_return is not None
        and linked_leverage_return is not None
        and linked_pnl_residual is not None
    ):
        reconciliations.append(
            reconciliation(
                "pnl_return:linked_total",
                portfolio_total,
                linked_trade_return
                + linked_holding_return
                + linked_leverage_return
                + linked_pnl_residual,
                tolerance=tolerance,
                scope="pnl_return",
            )
        )
    if attribution_gap is not None:
        overall_check = reconciliation(
            "overall_active_return",
            holding_active_total,
            attribution_total,
            tolerance=tolerance,
            scope="portfolio",
        )
        reconciliations.append(overall_check)
    else:
        overall_check = None
    summary = {
        "portfolio_return": portfolio_total,
        "benchmark_return": benchmark_total,
        "active_return": active_total,
        "holding_cumulative_return": holding_cumulative_return,
        "benchmark_holding_return": benchmark_holding_cumulative_return,
        "holding_active_return": holding_active_total,
        "allocation_effect": allocation_total,
        "selection_effect": selection_total,
        "attribution_total": attribution_total,
        "attribution_gap": attribution_gap,
        "trade_mtm": pnl["totals"]["trade_mtm"],
        "holding_mtm": pnl["totals"]["holding_mtm"],
        "total_mtm": pnl["totals"]["total_mtm"],
        "fees": pnl["totals"]["fees"],
        "trade_return": linked_trade_return,
        "holding_return": linked_holding_return,
        "leverage_return": linked_leverage_return,
        "pnl_return_residual": linked_pnl_residual,
        "reconciled": bool(all(record.get("passed", False) for record in reconciliations))
        if reconciliations
        else None,
    }
    if attribution_gap is not None and abs(attribution_gap) > max(tolerance, 1e-8):
        quality_warnings.append(
            warning(
                "overall_reconciliation_gap",
                "Attributed effects do not equal cumulative holding active return.",
                scope="portfolio",
                difference=attribution_gap,
            )
        )

    warning_severities = {item.get("severity", "warning") for item in quality_warnings}
    status = "warning" if warning_severities.intersection({"warning", "error"}) else "ok"
    period_columns = list(periods_frame.columns)
    schemas = {
        "periods": period_columns,
        "quality_warnings": ["code", "message", "severity", "scope", "details"],
        "reconciliation": ["name", "actual", "explained", "difference", "tolerance", "passed", "scope"],
        "pnl.trades": list(trade_frame.columns),
        "pnl.holdings": list(holding_frame.columns),
        "pnl.by_period": list(pnl_by_period.columns),
        "sections": {key: value["columns"] for key, value in sections.items()},
    }
    output = AttributionOutput(
        status=status,
        meta={
            "asset_pool": asset_pool,
            "methodology": {
                "asset_class": "BHB no-interaction",
                "stock_industry": "Brinson-Fachler no-interaction",
                "multi_period": "Carino",
                "return_basis": brinson_return_basis,
                "allocation_selection": brinson_effect_source,
            },
            "units": {
                "return_and_contribution": RETURN_UNIT,
                "pnl_and_fees": AMOUNT_UNIT,
                "weight_and_exposure": "ratio",
            },
            "summary_units": {
                "portfolio_return": RETURN_UNIT,
                "benchmark_return": RETURN_UNIT,
                "active_return": RETURN_UNIT,
                "holding_cumulative_return": RETURN_UNIT,
                "benchmark_holding_return": RETURN_UNIT,
                "holding_active_return": RETURN_UNIT,
                "allocation_effect": RETURN_UNIT,
                "selection_effect": RETURN_UNIT,
                "attribution_total": RETURN_UNIT,
                "attribution_gap": RETURN_UNIT,
                "trade_mtm": AMOUNT_UNIT,
                "holding_mtm": AMOUNT_UNIT,
                "total_mtm": AMOUNT_UNIT,
                "fees": AMOUNT_UNIT,
                "trade_return": RETURN_UNIT,
                "holding_return": RETURN_UNIT,
                "leverage_return": RETURN_UNIT,
                "pnl_return_residual": RETURN_UNIT,
                "reconciled": "boolean",
            },
            "schemas": schemas,
            "expected_input_fields": EXPECTED_INPUT_FIELDS,
        },
        summary=_json_value(summary),
        periods=_records(periods_frame),
        pnl=_json_value(pnl),
        sections=_json_value(sections),
        tables=_json_value(tables),
        reconciliation=_json_value(reconciliations),
        quality_warnings=_json_value(quality_warnings),
    )
    return output.to_dict()


build_attribution_report = run_attribution


__all__ = [
    "AMOUNT_UNIT",
    "AttributionEngine",
    "RETURN_UNIT",
    "build_attribution_report",
    "run_attribution",
]
