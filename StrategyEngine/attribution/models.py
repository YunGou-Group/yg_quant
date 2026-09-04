#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归因报告用的警告、对账和输出结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ASSET_CLASSES = ("stock", "etf", "bond", "future", "option", "cash")


# Canonical ingestion contract.  Alternative aliases are accepted by the
# engine, but exposing one authoritative set of names lets the website build
# upload templates and field-help text without duplicating backend knowledge.
EXPECTED_INPUT_FIELDS: Dict[str, Dict[str, Sequence[str]]] = {
    "returns": {
        "required": ("period", "portfolio_return", "benchmark_return"),
        "optional": ("nav_begin", "portfolio_value", "benchmark_value", "currency"),
    },
    "trades": {
        "required": (
            "period",
            "asset_id",
            "asset_class",
            "quantity or signed_quantity",
            "execution_price",
            "mark_price",
        ),
        "optional": ("side", "multiplier", "fees", "commission", "tax", "slippage"),
    },
    "holdings": {
        "required": ("period", "asset_id", "asset_class", "quantity", "start_price", "end_price"),
        "optional": ("multiplier", "fees", "weight", "market_value"),
    },
    "asset_class_portfolio_and_benchmark": {
        "required": ("period", "asset_class", "weight", "return"),
        "optional": ("contribution",),
    },
    "stock_industry_portfolio_and_benchmark": {
        "required": ("period", "industry", "weight", "return"),
        "optional": ("industry_code",),
    },
    "stock_factor_exposures": {
        "required": ("period", "factor", "active_exposure"),
        "optional": (
            "asset_id",
            "exposure",
            "portfolio_weight",
            "benchmark_weight",
            "portfolio_exposure",
            "benchmark_exposure",
        ),
    },
    "stock_factor_returns": {
        "required": ("period", "factor", "factor_return"),
        "optional": (),
    },
    "stock_specific_returns": {
        "required": ("period", "specific_contribution or specific_return with weights"),
        "optional": ("asset_id", "portfolio_weight", "benchmark_weight", "active_weight"),
    },
    "bond": {
        "required": ("period", "asset_id", "weight", "actual_return"),
        "optional": (
            "carry_return",
            "modified_duration",
            "risk_free_yield_change",
            "spread_duration",
            "spread_change",
            "convexity",
            "curve_return",
            "spread_return",
            "convexity_return",
        ),
    },
    "future": {
        "required": ("period", "asset_id", "weight", "actual_return"),
        "optional": (
            "underlying_return",
            "basis_return",
            "roll_return",
            "collateral_return",
            "leverage",
            "leverage_return",
        ),
    },
    "option": {
        "required": ("period", "asset_id", "weight", "actual_return"),
        "optional": (
            "start_value",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "underlying_price_change",
            "volatility_change",
            "elapsed_days",
            "rate_change",
            "delta_return",
            "gamma_return",
            "vega_return",
            "theta_return",
            "rho_return",
        ),
    },
}


_ASSET_CLASS_ALIASES = {
    # Equity
    "stock": "stock",
    "stocks": "stock",
    "equity": "stock",
    "equities": "stock",
    "a_share": "stock",
    "ashare": "stock",
    "股票": "stock",
    "权益": "stock",
    # ETF
    "etf": "etf",
    "etfs": "etf",
    "exchange_traded_fund": "etf",
    "fund": "etf",
    "基金": "etf",
    "交易所交易基金": "etf",
    # Fixed income
    "bond": "bond",
    "bonds": "bond",
    "fixed_income": "bond",
    "fixedincome": "bond",
    "debt": "bond",
    "债券": "bond",
    "固收": "bond",
    # Futures
    "future": "future",
    "futures": "future",
    "fut": "future",
    "期货": "future",
    # Options
    "option": "option",
    "options": "option",
    "opt": "option",
    "期权": "option",
    # Cash
    "cash": "cash",
    "money_market": "cash",
    "money": "cash",
    "现金": "cash",
    "货币": "cash",
}


def normalize_asset_class(value: Any, *, strict: bool = True) -> Optional[str]:
    """Return one of the six canonical asset-class identifiers.

    Parameters
    ----------
    value:
        English or Chinese asset-class name.  Whitespace and hyphens are
        normalized before alias matching.
    strict:
        If true (the default), invalid values raise ``ValueError``.  If false,
        ``None`` is returned so ingestion code can emit a quality warning and
        continue processing the valid rows.
    """

    if value is None:
        if strict:
            raise ValueError("asset class is required")
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _ASSET_CLASS_ALIASES.get(key)
    if normalized is None and strict:
        allowed = ", ".join(ASSET_CLASSES)
        raise ValueError(f"unknown asset class {value!r}; expected one of {allowed}")
    return normalized


def normalize_asset_classes(values: Any, *, strict: bool = True) -> List[str]:
    """Normalize a string or iterable into a de-duplicated ordered list."""

    if values is None:
        return []
    if isinstance(values, str):
        # ``future+option`` and comma-separated values are common UI inputs.
        raw_values: Iterable[Any] = values.replace(",", "+").split("+")
    else:
        raw_values = values
    result: List[str] = []
    for value in raw_values:
        normalized = normalize_asset_class(value, strict=strict)
        if normalized is not None and normalized not in result:
            result.append(normalized)
    return result


@dataclass(frozen=True)
class QualityWarning:
    """A machine-readable, user-presentable data-quality observation."""

    code: str
    message: str
    severity: str = "warning"
    scope: str = "input"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Reconciliation:
    """Comparison between a reported total and attributed components."""

    name: str
    actual: float
    explained: float
    difference: float
    tolerance: float
    passed: bool
    scope: str = "portfolio"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributionTable:
    """A display-ready report table plus its calculation metadata."""

    key: str
    title: str
    method: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)
    columns: List[Dict[str, str]] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.note is None:
            value.pop("note", None)
        return value


@dataclass
class AttributionOutput:
    """Top-level response consumed by the website and downloadable JSON."""

    status: str
    meta: Dict[str, Any]
    summary: Dict[str, Any]
    periods: List[Dict[str, Any]]
    pnl: Dict[str, Any]
    sections: Dict[str, Any]
    tables: List[Dict[str, Any]]
    reconciliation: List[Dict[str, Any]]
    quality_warnings: List[Dict[str, Any]]
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def warning(
    code: str,
    message: str,
    *,
    severity: str = "warning",
    scope: str = "input",
    **details: Any,
) -> Dict[str, Any]:
    """Convenience constructor used by numerical routines."""

    return QualityWarning(
        code=code,
        message=message,
        severity=severity,
        scope=scope,
        details=details,
    ).to_dict()


def reconciliation(
    name: str,
    actual: float,
    explained: float,
    *,
    tolerance: float = 1e-10,
    scope: str = "portfolio",
) -> Dict[str, Any]:
    """Build a reconciliation record using a scale-aware tolerance."""

    actual_value = float(actual)
    explained_value = float(explained)
    difference = actual_value - explained_value
    scaled_tolerance = float(tolerance) * max(1.0, abs(actual_value), abs(explained_value))
    return Reconciliation(
        name=name,
        actual=actual_value,
        explained=explained_value,
        difference=float(difference),
        tolerance=scaled_tolerance,
        passed=bool(abs(difference) <= scaled_tolerance),
        scope=scope,
    ).to_dict()


__all__ = [
    "ASSET_CLASSES",
    "EXPECTED_INPUT_FIELDS",
    "AttributionOutput",
    "AttributionTable",
    "QualityWarning",
    "Reconciliation",
    "normalize_asset_class",
    "normalize_asset_classes",
    "reconciliation",
    "warning",
]
