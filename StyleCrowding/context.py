#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享 Context：面板、分组、风格收益、特异收益、分化度。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from DailyUpdates.storage import SQLiteStorage
from FactorEvaluates.exposure_engine import ExposureEngine, NLSIZE_NAME, RAW_STYLE_NAMES, SIZE_NAME
from FactorEvaluates.factor_panel_loader import FactorPanelLoader
from FactorEvaluates.market_panel_loader import MarketPanelLoader
from FactorEvaluates.style_return_engine import StyleReturnEngine
from Universes.catalog import build_mask as build_universe_mask
from yg_quant_repo import default_db_path, default_factor_dir

from .config import StyleCrowdingSettings
from .universe import (
    GroupSnapshot,
    build_groups_for_style,
    reverse_group_snapshots,
)

logger = logging.getLogger("StyleCrowding")

BTOP_NAME = "style_btop"


@dataclass
class StyleCrowdingContext:
    settings: StyleCrowdingSettings
    dates: Tuple[str, ...]
    output_dates: Tuple[str, ...]
    factor_ids: Tuple[str, ...]
    symbols: Tuple[str, ...]
    close: pd.DataFrame
    pre_close: pd.DataFrame
    pb: pd.DataFrame
    pe: pd.DataFrame
    ps: pd.DataFrame
    circ_mv: pd.DataFrame
    mask: pd.DataFrame
    style_panels: Dict[str, pd.DataFrame]
    exposure_btop: pd.DataFrame
    close_returns: pd.DataFrame
    groups: Dict[int, Dict[str, Dict[str, GroupSnapshot]]] = field(default_factory=dict)
    dispersion: Dict[int, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    factor_returns: Dict[int, pd.DataFrame] = field(default_factory=dict)
    canonical_factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk_factor_returns: Dict[str, pd.DataFrame] = field(default_factory=dict)
    specific_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    symbol_index: Dict[str, int] = field(default_factory=dict)
    _reverse_dispersion: Dict[Tuple[int, str], pd.DataFrame] = field(
        default_factory=dict, repr=False
    )
    pairwise_cache: Dict[Tuple[str, int], pd.Series] = field(
        default_factory=dict, repr=False
    )

    def group(self, group_count: int, factor_id: str, date: str) -> Optional[GroupSnapshot]:
        return self.groups.get(group_count, {}).get(factor_id, {}).get(date)

    def dispersion_for(
        self, group_count: int, factor_id: str, orientation: str
    ) -> pd.DataFrame:
        """按 orientation 取分散度。

        reverse 的「高组」是正向的低组，直接复用正向 dispersion 会把
        07/08 之类的指标算到反了的那一侧上。
        """
        positive = self.dispersion.get(group_count, {}).get(factor_id)
        if orientation != "reverse":
            return positive if positive is not None else pd.DataFrame()
        key = (int(group_count), str(factor_id))
        cached = self._reverse_dispersion.get(key)
        if cached is not None:
            return cached
        snapshots = self.groups.get(group_count, {}).get(factor_id)
        if not snapshots:
            return positive if positive is not None else pd.DataFrame()
        flipped = reverse_group_snapshots({factor_id: snapshots})[factor_id]
        frame = _build_dispersion_for_groups(
            self.dates, self.close_returns, flipped, self.settings
        )
        self._reverse_dispersion[key] = frame
        return frame


def _iso_dates(index: pd.Index) -> Tuple[str, ...]:
    return tuple(pd.to_datetime(index).strftime("%Y-%m-%d"))


def _align_panel(frame: pd.DataFrame, dates: pd.Index, symbols: pd.Index) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out.columns = [str(c) for c in out.columns]
    return out.reindex(index=dates, columns=symbols)


def _group_mean(values: pd.Series, members: Tuple[str, ...]) -> float:
    if not members:
        return float("nan")
    sample = pd.to_numeric(values.reindex(list(members)), errors="coerce")
    sample = sample[np.isfinite(sample.to_numpy(dtype="float64"))]
    return float(sample.mean()) if sample.size else float("nan")


def _group_std(values: pd.Series, members: Tuple[str, ...], min_obs: int = 2) -> float:
    if len(members) < min_obs:
        return float("nan")
    sample = pd.to_numeric(values.reindex(list(members)), errors="coerce")
    sample = sample[np.isfinite(sample.to_numpy(dtype="float64"))]
    if sample.size < min_obs:
        return float("nan")
    return float(sample.std(ddof=1))


def _build_close_returns(close: pd.DataFrame, pre_close: pd.DataFrame) -> pd.DataFrame:
    pre = pre_close.reindex(index=close.index, columns=close.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close / pre - 1.0
    return ret.where(np.isfinite(pre) & (pre != 0))


def _build_dispersion_for_groups(
    dates: Tuple[str, ...],
    close_returns: pd.DataFrame,
    snapshots: Dict[str, GroupSnapshot],
    settings: StyleCrowdingSettings,
) -> pd.DataFrame:
    rows = []
    top_counts = settings.dispersion_top_counts
    for date in dates:
        snap = snapshots.get(date)
        ret = close_returns.loc[date] if date in close_returns.index else pd.Series(dtype="float64")
        market = ret[np.isfinite(ret.to_numpy(dtype="float64"))]
        market_std = float(market.std(ddof=1)) if market.size >= 2 else float("nan")
        row = {"trade_date": date, "market_std": market_std, "d_top": float("nan"), "d_rel": float("nan")}
        for n in top_counts:
            row[f"d_top_{n}"] = float("nan")
        row["d_bottom_50"] = float("nan")
        if snap is not None and snap.valid:
            row["d_top"] = _group_std(ret, snap.high)
            if np.isfinite(market_std) and market_std > 0 and np.isfinite(row["d_top"]):
                row["d_rel"] = row["d_top"] / market_std
            for n in top_counts:
                if len(snap.high) >= n:
                    row[f"d_top_{n}"] = _group_std(ret, snap.high[-n:])
            if len(snap.low) >= 50:
                row["d_bottom_50"] = _group_std(ret, snap.low[:50])
        rows.append(row)
    return pd.DataFrame(rows).set_index("trade_date")


def _build_hl_returns_from_groups(
    dates: Tuple[str, ...],
    close_returns: pd.DataFrame,
    snapshots: Dict[str, GroupSnapshot],
) -> pd.Series:
    values = []
    for date in dates:
        snap = snapshots.get(date)
        if snap is None or not snap.valid:
            values.append(float("nan"))
            continue
        ret = close_returns.loc[date]
        hi = _group_mean(ret, snap.high)
        lo = _group_mean(ret, snap.low)
        values.append(hi - lo if np.isfinite(hi) and np.isfinite(lo) else float("nan"))
    return pd.Series(values, index=dates, dtype="float64")


def _compute_specific_returns(
    returns: pd.DataFrame,
    exposure_engine: ExposureEngine,
    raw_styles: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    circ_mv: pd.DataFrame,
    min_obs: int,
) -> np.ndarray:
    dates = returns.index
    symbols = returns.columns
    weights = np.sqrt(circ_mv.reindex(index=dates, columns=symbols).clip(lower=0).to_numpy(dtype="float64"))
    matrix = exposure_engine.build(raw_styles, mask, weights=pd.DataFrame(weights, index=dates, columns=symbols))
    aligned = matrix.align(dates, symbols)
    y = returns.to_numpy(dtype="float64")
    m = mask.fillna(False).to_numpy(dtype=bool) & np.isfinite(y)
    style_names = [name for name in aligned.names if name in aligned.panels]
    n_dates, n_symbols = y.shape
    residuals = np.full((n_dates, n_symbols), np.nan, dtype=np.float32)
    engine = StyleReturnEngine(min_obs=min_obs)
    for t in range(n_dates):
        valid = m[t]
        if int(valid.sum()) < min_obs:
            continue
        cols = [aligned.panels[name].to_numpy(dtype="float64")[t] for name in style_names]
        x = np.column_stack([np.ones(n_symbols)] + cols)
        x = np.where(valid[:, None], x, np.nan)
        beta = engine.estimate_day(x, y[t], min_obs=min_obs)
        if not np.isfinite(beta).any():
            continue
        fitted = np.nansum(x * beta, axis=1)
        resid = y[t] - fitted
        residuals[t, valid] = resid[valid].astype(np.float32)
    return residuals


def _market_cap_weighted_return(close_returns: pd.DataFrame, circ_mv: pd.DataFrame) -> pd.Series:
    aligned_mv = circ_mv.reindex(index=close_returns.index, columns=close_returns.columns)
    ret = close_returns.to_numpy(dtype="float64")
    w = aligned_mv.to_numpy(dtype="float64")
    valid = np.isfinite(ret) & np.isfinite(w) & (w > 0)
    out = np.full(ret.shape[0], np.nan, dtype="float64")
    for t in range(ret.shape[0]):
        m = valid[t]
        if int(m.sum()) < 10:
            continue
        out[t] = np.average(ret[t, m], weights=w[t, m])
    return pd.Series(out, index=close_returns.index, dtype="float64")


def build_context(settings: StyleCrowdingSettings) -> StyleCrowdingContext:
    loader = MarketPanelLoader(str(default_db_path()))
    factor_loader = FactorPanelLoader(str(default_factor_dir()))
    storage = SQLiteStorage(str(default_db_path()))

    start = settings.start_date or None
    end = settings.end_date or None
    close = loader.load_field("close", start_date=start, end_date=end)
    pre_close = loader.load_field("pre_close", start_date=start, end_date=end)
    pb = loader.load_field("pb", start_date=start, end_date=end)
    pe = loader.load_field("pe", start_date=start, end_date=end)
    ps = loader.load_field("ps", start_date=start, end_date=end)
    circ_mv = loader.load_field("circ_mv", start_date=start, end_date=end)
    open_panel = loader.load_open(start_date=start, end_date=end)

    style_names = list(RAW_STYLE_NAMES)
    raw_styles = factor_loader.load_many(style_names, start_date=start, end_date=end)
    dates = _iso_dates(close.index)
    symbols = tuple(str(c) for c in close.columns)
    date_index = pd.Index(dates)
    symbol_index = pd.Index(symbols)
    close = _align_panel(close, date_index, symbol_index)
    pre_close = _align_panel(pre_close, date_index, symbol_index)
    pb = _align_panel(pb, date_index, symbol_index)
    pe = _align_panel(pe, date_index, symbol_index)
    ps = _align_panel(ps, date_index, symbol_index)
    circ_mv = _align_panel(circ_mv, date_index, symbol_index)
    for name in style_names:
        raw_styles[name] = _align_panel(raw_styles[name], date_index, symbol_index)

    mask = build_universe_mask(
        settings.universe, date_index, symbol_index, open_panel, storage
    )
    close_returns = _build_close_returns(close, pre_close)

    exposure_engine = ExposureEngine()
    exposure_matrix = exposure_engine.build(raw_styles, mask, weights=circ_mv)
    exposure_btop = exposure_matrix.panels.get(BTOP_NAME, raw_styles.get(BTOP_NAME, pd.DataFrame()))

    style_engine = StyleReturnEngine(min_obs=settings.min_obs_style_wls)
    style_returns = style_engine.estimate(close_returns, exposure_matrix, mask=mask)

    targets = settings.style_targets()
    groups: Dict[int, Dict[str, Dict[str, GroupSnapshot]]] = {}
    dispersion: Dict[int, Dict[str, pd.DataFrame]] = {}
    factor_returns: Dict[int, pd.DataFrame] = {}

    for group_count in settings.group_counts:
        groups[group_count] = {}
        dispersion[group_count] = {}
        hl_cols = {}
        for factor_id in targets:
            if factor_id == NLSIZE_NAME:
                panel = exposure_matrix.panels.get(NLSIZE_NAME)
            else:
                panel = raw_styles.get(factor_id)
            if panel is None or panel.empty:
                continue
            snapshots = build_groups_for_style(
                panel, factor_id, settings, group_count=group_count, mask=mask
            )
            groups[group_count][factor_id] = snapshots
            dispersion[group_count][factor_id] = _build_dispersion_for_groups(
                dates, close_returns, snapshots, settings
            )
            hl_cols[factor_id] = _build_hl_returns_from_groups(dates, close_returns, snapshots)
        factor_returns[group_count] = pd.DataFrame(hl_cols, index=dates)

    market_ret = _market_cap_weighted_return(close_returns, circ_mv)
    risk_returns: Dict[str, pd.DataFrame] = {}
    for factor_id in targets:
        if factor_id not in style_returns.columns:
            continue
        risk_returns[factor_id] = pd.DataFrame(
            {"target_return": style_returns[factor_id], "market_return": market_ret},
            index=dates,
        )

    canonical = pd.DataFrame(
        {fid: style_returns[fid] for fid in targets if fid in style_returns.columns},
        index=dates,
    )

    specific = _compute_specific_returns(
        close_returns,
        exposure_engine,
        raw_styles,
        mask,
        circ_mv,
        settings.min_obs_style_wls,
    )

    active = tuple(sorted(set(groups.get(settings.group_counts[0], {}).keys())))
    ctx = StyleCrowdingContext(
        settings=settings,
        dates=dates,
        output_dates=dates,
        factor_ids=active,
        symbols=symbols,
        close=close,
        pre_close=pre_close,
        pb=pb,
        pe=pe,
        ps=ps,
        circ_mv=circ_mv,
        mask=mask,
        style_panels=raw_styles,
        exposure_btop=exposure_btop,
        close_returns=close_returns,
        groups=groups,
        dispersion=dispersion,
        factor_returns=factor_returns,
        canonical_factor_returns=canonical,
        risk_factor_returns=risk_returns,
        specific_returns=specific,
        symbol_index={sym: i for i, sym in enumerate(symbols)},
    )
    logger.info(
        "Context 就绪：%d 日 × %d 股，%d 目标",
        len(dates),
        len(symbols),
        len(active),
    )
    return ctx
