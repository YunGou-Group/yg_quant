#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本地 CNE5-lite 风格接到 run_attribution 的 stock.factor 输入。

逻辑对齐原业绩归因器 methodology §5：
  active_exposure = portfolio_exposure - benchmark_exposure
  factor_contribution = active_exposure × factor_return
十个风格名与 RiceQuant Barra v1 相同；原料是 bin 里的 style_* + 现算 NLSIZE，
行业哑变量与截距（市场联动）来自同一张 X_T。不是 RQData；缺 bin 直接失败。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from FactorEvaluates.exposure_engine import (
    INDUSTRY_PREFIX,
    NLSIZE_NAME,
    RAW_STYLE_NAMES,
    SIZE_NAME,
    ExposureEngine,
    ExposureMatrix,
    is_industry_col,
)
from FactorEvaluates.factor_panel_loader import FactorPanelLoader
from FactorEvaluates.industry_panel import load_l1_code_panel
from FactorEvaluates.market_panel_loader import MarketPanelLoader
from FactorEvaluates.style_return_engine import StyleReturnEngine
from Universes.catalog import build_mask

from . import ledger

_LOG = logging.getLogger("StrategyEngine")

# RiceQuant model="v1" 的固定十风格，顺序与 rqdata_adapter 一致。
BARRA_V1_STYLE_FACTORS: Tuple[str, ...] = (
    "beta",
    "momentum",
    "size",
    "earnings_yield",
    "residual_volatility",
    "growth",
    "book_to_price",
    "leverage",
    "liquidity",
    "non_linear_size",
)

LOCAL_TO_BARRA: Dict[str, str] = {
    "style_beta": "beta",
    "style_momentum": "momentum",
    "style_size": "size",
    "style_ey": "earnings_yield",
    "style_resvol": "residual_volatility",
    "style_growth": "growth",
    "style_leverage": "leverage",
    "style_btop": "book_to_price",
    "style_liquidity": "liquidity",
    NLSIZE_NAME: "non_linear_size",
}

FACTOR_LABELS: Dict[str, str] = {
    "beta": "贝塔",
    "momentum": "动量",
    "size": "规模",
    "earnings_yield": "盈利收益",
    "residual_volatility": "残余波动",
    "growth": "成长",
    "leverage": "杠杆",
    "liquidity": "流动性",
    "book_to_price": "账面市值比",
    "non_linear_size": "非线性规模",
    "comovement": "市场联动",
    "Specific": "特异收益",
    "Residual": "未解释残差",
}

GROUP_STYLE = "风格偏好"
GROUP_INDUSTRY = "行业偏好"
GROUP_MARKET = "市场联动"
GROUP_SPECIFIC = "特异收益"
GROUP_ORDER = (GROUP_STYLE, GROUP_INDUSTRY, GROUP_MARKET, GROUP_SPECIFIC)

_INDUSTRY_PREFIX = "industry:"


def build_factor_payload(
    storage,
    dates: Sequence[str],
    start_positions: Mapping[str, Mapping[str, float]],
    close_map: Mapping[str, Mapping[str, float]],
    cash_by_date: Mapping[str, float],
    *,
    benchmark_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    db_path: Optional[str] = None,
) -> Tuple[dict, str]:
    """组合/基准期初权重 × 期初 X，乘当日截面因子收益。"""
    days = [str(d) for d in dates if str(d)]
    if len(days) < 2:
        raise ValueError("交易日不足，无法做风格因子归因。")
    try:
        payload = _build(
            storage,
            days,
            start_positions,
            close_map,
            cash_by_date,
            benchmark_weights=benchmark_weights,
            db_path=db_path,
        )
    except FileNotFoundError as exc:
        raise ValueError("风格 Bin 未就绪，无法做风格因子归因。") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"风格因子面板组装失败: {exc}") from exc
    if not payload:
        raise ValueError("风格暴露或因子收益为空，无法做风格因子归因。")
    return payload, (
        "风格因子用本地 CNE5-lite（style_* + 现算 NLSIZE + 申万一级），"
        "对应 Barra v1 十风格名；不是 RQData。因子收益是全市场截面 WLS；"
        "行业哑变量丢掉出现最多的一个做参照并进市场联动，与评估 X_T 相同。"
    )


def summarize_factor_rows(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    """把引擎逐日因子行收成一张表：贡献加总，暴露取日均。"""
    buckets: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("factor") or "").strip()
        if not key:
            continue
        item = buckets.setdefault(
            key,
            {
                "key": canonical_key(key),
                "name": display_name(key),
                "group": group_of(key),
                "portfolio_exposure": 0.0,
                "benchmark_exposure": 0.0,
                "active_exposure": 0.0,
                "factor_return": 0.0,
                "contribution": 0.0,
            },
        )
        contrib = _float(row.get("contribution"))
        item["contribution"] += contrib
        item["portfolio_exposure"] += _float(row.get("portfolio_exposure"))
        item["benchmark_exposure"] += _float(row.get("benchmark_exposure"))
        item["active_exposure"] += _float(row.get("active_exposure"))
        item["factor_return"] += _float(row.get("factor_return"))
        counts[key] = counts.get(key, 0) + 1
    out: List[Dict[str, Any]] = []
    for key, item in buckets.items():
        n = max(int(counts.get(key, 1)), 1)
        item["portfolio_exposure"] /= n
        item["benchmark_exposure"] /= n
        item["active_exposure"] /= n
        item["factor_return"] /= n
        out.append(item)
    return _sort_factor_rows(out)


def factor_group_totals(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    totals = {name: 0.0 for name in GROUP_ORDER}
    for row in rows:
        group = str(row.get("group") or "")
        if group in totals:
            totals[group] += _float(row.get("contribution"))
    return [{"group": name, "contribution": totals[name]} for name in GROUP_ORDER]


def group_of(factor: str) -> str:
    key = canonical_key(factor)
    if key in BARRA_V1_STYLE_FACTORS:
        return GROUP_STYLE
    if key == "comovement":
        return GROUP_MARKET
    if key in {"Specific", "Residual", "specific", "residual"}:
        return GROUP_SPECIFIC
    if key.startswith(_INDUSTRY_PREFIX) or is_industry_col(factor) or factor.startswith(_INDUSTRY_PREFIX):
        return GROUP_INDUSTRY
    component = str(factor)
    if component.lower() in {"specific", "residual"}:
        return GROUP_SPECIFIC
    return GROUP_STYLE


def canonical_key(factor: str) -> str:
    text = str(factor or "").strip()
    if text in LOCAL_TO_BARRA:
        return LOCAL_TO_BARRA[text]
    if text in {"intercept", "Intercept"}:
        return "comovement"
    if text.lower() == "specific":
        return "Specific"
    if text.lower() == "residual":
        return "Residual"
    return text


def display_name(factor: str) -> str:
    key = canonical_key(factor)
    if key in FACTOR_LABELS:
        return FACTOR_LABELS[key]
    if key.startswith(_INDUSTRY_PREFIX):
        return key[len(_INDUSTRY_PREFIX) :]
    return key


def _build(
    storage,
    dates: Sequence[str],
    start_positions: Mapping[str, Mapping[str, float]],
    close_map: Mapping[str, Mapping[str, float]],
    cash_by_date: Mapping[str, float],
    *,
    benchmark_weights: Optional[Mapping[str, Mapping[str, float]]],
    db_path: Optional[str],
) -> Optional[dict]:
    start, end = dates[0], dates[-1]
    loader = FactorPanelLoader()
    raw = loader.load_many(RAW_STYLE_NAMES, start_date=start, end_date=end)
    if SIZE_NAME not in raw or raw[SIZE_NAME] is None or raw[SIZE_NAME].empty:
        return None
    market = MarketPanelLoader(db_path)
    close = market.load_field("close", start_date=start, end_date=end, adjust="none")
    if close is None or close.empty:
        return None
    close = _normalize_panel(close)
    raw_styles = {
        name: _normalize_panel(panel).reindex(index=close.index, columns=close.columns)
        for name, panel in raw.items()
        if panel is not None and not panel.empty
    }
    if SIZE_NAME not in raw_styles:
        return None
    mask = close.notna()
    try:
        universe = build_mask("all", close.index, close.columns, close, storage)
        if universe is not None and not universe.empty:
            aligned = universe.copy()
            aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
            aligned.columns = [str(c) for c in aligned.columns]
            mask = mask & aligned.reindex(index=close.index, columns=close.columns).fillna(False)
    except Exception:
        _LOG.warning("因子归因股票池 mask 失败，改用有收盘价的股票", exc_info=True)

    codes = None
    labels: Dict[str, str] = {}
    try:
        codes, labels = load_l1_code_panel(storage, close.index, close.columns)
    except Exception:
        _LOG.warning("因子归因加载行业失败，风格仍继续", exc_info=True)
        codes = None

    exposures = ExposureEngine().build(
        raw_styles,
        mask,
        industry_codes=codes,
        industry_labels=labels,
    )
    period_dates = list(dates[1:])
    lagged = _lag_exposures(exposures, dates)
    if lagged is None:
        return None
    close_on_dates = close.reindex(index=list(dates))
    returns = close_on_dates.pct_change().reindex(index=period_dates, columns=close.columns)
    lagged_mask = _lag_frame(mask.astype(float), dates).fillna(0) > 0
    lagged_mask = lagged_mask.reindex(index=period_dates, columns=close.columns).fillna(False)
    lagged_mask = lagged_mask & returns.notna()
    # 期初 X 对应当日 close-to-close，与持仓盯市同一段收益。
    factor_ret = StyleReturnEngine().estimate(
        returns, lagged, mask=lagged_mask
    )
    if factor_ret.empty:
        return None

    port_w = ledger.beginning_weights(dates, start_positions, close_map, cash_by_date)
    bench_w: Dict[str, Dict[str, float]] = {}
    prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    if benchmark_weights:
        for day in period_dates:
            src = prev.get(day)
            bench_w[day] = dict(
                benchmark_weights.get(src) or benchmark_weights.get(day) or {}
            )

    names = _factor_names(lagged, factor_ret)
    wp = _weight_matrix(port_w, period_dates, close.columns)
    wb = _weight_matrix(bench_w, period_dates, close.columns)
    x_stack, x_names = _design(lagged, period_dates, close.columns, names)

    exposures_rows: List[dict] = []
    factor_return_rows: List[dict] = []
    f_mat = factor_ret.reindex(index=period_dates, columns=x_names).to_numpy(dtype=np.float64)
    r_mat = returns.reindex(index=period_dates, columns=close.columns).to_numpy(dtype=np.float64)
    wp_n = wp.to_numpy(dtype=np.float64)
    wb_n = wb.to_numpy(dtype=np.float64)

    for k, name in enumerate(x_names):
        xk = x_stack[:, :, k]
        finite = np.isfinite(xk)
        x_filled = np.where(finite, xk, 0.0)
        port_exp = (wp_n * x_filled).sum(axis=1)
        bench_exp = (wb_n * x_filled).sum(axis=1)
        f_k = f_mat[:, k]
        published = _publish_name(name, lagged)
        for i, day in enumerate(period_dates):
            fr = f_k[i]
            if not np.isfinite(fr):
                fr = 0.0
            exposures_rows.append(
                {
                    "period": day,
                    "factor": published,
                    "portfolio_exposure": float(port_exp[i]),
                    "benchmark_exposure": float(bench_exp[i]),
                    "active_exposure": float(port_exp[i] - bench_exp[i]),
                }
            )
            factor_return_rows.append(
                {
                    "period": day,
                    "factor": published,
                    "factor_return": float(fr),
                }
            )

    fitted = np.zeros(r_mat.shape, dtype=np.float64)
    for k in range(len(x_names)):
        xk = x_stack[:, :, k]
        fk = f_mat[:, k][:, None]
        fitted += np.where(np.isfinite(xk) & np.isfinite(fk), xk * fk, 0.0)
    valid_x = np.isfinite(x_stack).all(axis=2) if x_stack.size else np.zeros(r_mat.shape, dtype=bool)
    valid_f = np.isfinite(f_mat).all(axis=1)[:, None] if f_mat.size else False
    resid = np.where(np.isfinite(r_mat) & valid_x & valid_f, r_mat - fitted, np.nan)
    specific_rows = []
    active_w = wp_n - wb_n
    spec = np.nansum(np.where(np.isfinite(resid), active_w * resid, 0.0), axis=1)
    for i, day in enumerate(period_dates):
        specific_rows.append(
            {
                "period": day,
                "specific_contribution": float(spec[i]) if np.isfinite(spec[i]) else 0.0,
            }
        )

    if not exposures_rows or not factor_return_rows:
        return None
    return {
        "exposures": exposures_rows,
        "factor_returns": factor_return_rows,
        "specific_returns": specific_rows,
    }


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out.columns = [str(c) for c in out.columns]
    return out


def _lag_frame(frame: pd.DataFrame, dates: Sequence[str]) -> pd.DataFrame:
    """把 T-1 行对齐到归因日 T。"""
    period_dates = list(dates[1:])
    prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    work = _normalize_panel(frame)
    rows = []
    for day in period_dates:
        src = prev[day]
        if src in work.index:
            rows.append(work.loc[src])
        else:
            rows.append(pd.Series(np.nan, index=work.columns))
    return pd.DataFrame(rows, index=period_dates, columns=work.columns)


def _lag_exposures(exposures: ExposureMatrix, dates: Sequence[str]) -> Optional[ExposureMatrix]:
    """把 T-1 的 X 对齐到归因日 T：因子解释的是昨收到今收。"""
    if len(dates) < 2:
        return None
    panels = {name: _lag_frame(panel, dates) for name, panel in exposures.panels.items()}
    weight_frame = _lag_frame(exposures.weights, dates) if exposures.weights is not None else None
    return ExposureMatrix(
        panels=panels,
        names=exposures.names,
        weighted=exposures.weighted,
        winsor_p=exposures.winsor_p,
        weights=weight_frame,
        industry_names=exposures.industry_names,
        industry_labels=dict(exposures.industry_labels),
        industry_ref=exposures.industry_ref,
    )


def _factor_names(exposures: ExposureMatrix, factor_ret: pd.DataFrame) -> List[str]:
    names: List[str] = []
    if "intercept" in factor_ret.columns:
        names.append("intercept")
    for name in exposures.names:
        if name in exposures.panels:
            names.append(name)
    return names


def _design(
    exposures: ExposureMatrix,
    dates: Sequence[str],
    symbols: pd.Index,
    names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    cols = pd.Index([str(c) for c in symbols])
    idx = pd.Index(list(dates))
    stacked = []
    used: List[str] = []
    for name in names:
        if name == "intercept":
            stacked.append(np.ones((len(idx), len(cols)), dtype=np.float64))
            used.append("intercept")
            continue
        panel = exposures.panels.get(name)
        if panel is None:
            continue
        arr = panel.reindex(index=idx, columns=cols).to_numpy(dtype=np.float64)
        stacked.append(arr)
        used.append(name)
    if not stacked:
        return np.zeros((len(idx), len(cols), 0)), []
    return np.stack(stacked, axis=2), used


def _weight_matrix(
    weights: Mapping[str, Mapping[str, float]],
    dates: Sequence[str],
    symbols: pd.Index,
) -> pd.DataFrame:
    cols = [str(c) for c in symbols]
    rows = []
    for day in dates:
        series = pd.Series(weights.get(day) or {}, dtype=float)
        rows.append(series)
    frame = pd.DataFrame(rows, index=list(dates))
    return frame.reindex(columns=cols).fillna(0.0)


def _publish_name(name: str, exposures: ExposureMatrix) -> str:
    if name == "intercept":
        return "comovement"
    if name in LOCAL_TO_BARRA:
        return LOCAL_TO_BARRA[name]
    if is_industry_col(name):
        code = name[len(INDUSTRY_PREFIX) :] if name.startswith(INDUSTRY_PREFIX) else name
        label = (
            exposures.industry_labels.get(name)
            or exposures.industry_labels.get(code)
            or code
        )
        return f"{_INDUSTRY_PREFIX}{label}"
    return name


def _sort_factor_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rank = {name: i for i, name in enumerate(BARRA_V1_STYLE_FACTORS)}
    group_rank = {name: i for i, name in enumerate(GROUP_ORDER)}

    def key(row: Mapping[str, Any]):
        group = str(row.get("group") or "")
        ident = str(row.get("key") or "")
        if group == GROUP_STYLE:
            return (group_rank.get(group, 9), rank.get(ident, 99), ident)
        if ident == "comovement":
            return (group_rank.get(GROUP_MARKET, 9), 0, ident)
        if ident == "Specific":
            return (group_rank.get(GROUP_SPECIFIC, 9), 0, ident)
        if ident == "Residual":
            return (group_rank.get(GROUP_SPECIFIC, 9), 1, ident)
        return (group_rank.get(group, 9), 0, str(row.get("name") or ident))

    return [dict(row) for row in sorted(rows, key=key)]


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return number
