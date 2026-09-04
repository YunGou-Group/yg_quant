#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拥挤指标计算。"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from ..config import StyleCrowdingSettings
from ..context import StyleCrowdingContext
from ..normalization import prior_rolling_zscore, rolling_mean
from ..universe import GroupSnapshot, reverse_group_snapshots, snapshot_usable


def _compound(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.any(finite <= -1.0):
        return float("nan")
    return float(np.expm1(np.log1p(finite).sum()))


def _group_stat(values: pd.Series, members: Tuple[str, ...], agg: str, min_obs: int) -> float:
    sample = pd.to_numeric(values.reindex(list(members)), errors="coerce")
    sample = sample[np.isfinite(sample.to_numpy(dtype="float64"))]
    if sample.size < min_obs:
        return float("nan")
    return float(sample.mean() if agg == "mean" else sample.median())


def _valuation_series(
    ctx: StyleCrowdingContext,
    groups: Mapping[str, GroupSnapshot],
    field: str,
    *,
    log_ratio: bool,
    diff: bool,
    settings: StyleCrowdingSettings,
) -> pd.Series:
    raw = []
    inv = {"pb": ctx.pb, "pe": ctx.pe, "ps": ctx.ps}[field]
    for date in ctx.dates:
        snap = groups.get(date)
        if snap is None or not snap.valid:
            raw.append(float("nan"))
            continue
        if field == "pb":
            h = _group_stat(1.0 / inv.loc[date].replace(0, np.nan), snap.high, settings.valuation_aggregation, settings.min_group_size)
            l = _group_stat(1.0 / inv.loc[date].replace(0, np.nan), snap.low, settings.valuation_aggregation, settings.min_group_size)
            if log_ratio:
                raw.append(float(np.log(l / h)) if np.isfinite(h) and np.isfinite(l) and h > 0 and l > 0 else float("nan"))
            else:
                raw.append(float("nan"))
        elif field == "ps":
            h = _group_stat(1.0 / inv.loc[date].replace(0, np.nan), snap.high, settings.valuation_aggregation, settings.min_group_size)
            l = _group_stat(1.0 / inv.loc[date].replace(0, np.nan), snap.low, settings.valuation_aggregation, settings.min_group_size)
            raw.append(float(np.log(l / h)) if np.isfinite(h) and np.isfinite(l) and h > 0 and l > 0 else float("nan"))
        else:
            h = _group_stat(1.0 / inv.loc[date], snap.high, settings.valuation_aggregation, settings.min_group_size)
            l = _group_stat(1.0 / inv.loc[date], snap.low, settings.valuation_aggregation, settings.min_group_size)
            raw.append(l - h if diff and np.isfinite(h) and np.isfinite(l) else float("nan"))
    series = pd.Series(raw, index=ctx.dates, dtype="float64")
    return prior_rolling_zscore(
        series,
        window=settings.prior_z_window,
        min_history=settings.valuation_z_min_prior,
    )


def _dispersion_spike(
    dispersion: pd.DataFrame,
    base_col: str,
    settings: StyleCrowdingSettings,
    *,
    relative_top: bool = False,
) -> pd.Series:
    source = pd.to_numeric(dispersion.get(base_col, pd.Series(np.nan, index=dispersion.index)), errors="coerce")
    if relative_top and "market_std" in dispersion.columns:
        market = pd.to_numeric(dispersion["market_std"], errors="coerce")
        source = (source / market).where(market > 0)
    d20 = rolling_mean(source, settings.article_window, min_observations=16)
    delta = d20 - d20.shift(20)
    z1 = prior_rolling_zscore(d20, window=settings.prior_z_window, min_history=settings.prior_z_min_history)
    z2 = prior_rolling_zscore(delta, window=settings.prior_z_window, min_history=settings.prior_z_min_history)
    return 0.5 * z1 + 0.5 * z2


def _pairwise_corr_day(
    residuals: np.ndarray,
    t: int,
    snap: GroupSnapshot,
    *,
    window: int,
    min_pair: int,
    min_ratio: float,
    symbol_index: Mapping[str, int],
) -> float:
    start = max(0, t - window + 1)
    window_mat = residuals[start : t + 1]
    if window_mat.shape[0] < window:
        return float("nan")

    def leg_corr(members: Tuple[str, ...]) -> float:
        locs = [symbol_index.get(s, -1) for s in members]
        locs = [x for x in locs if x >= 0]
        if len(locs) < 2:
            return float("nan")
        mat = window_mat[:, locs]
        finite = np.isfinite(mat)
        filled = np.where(finite, mat, 0.0)
        col_sum = filled.sum(axis=1, keepdims=True)
        n_finite = finite.sum(axis=1, keepdims=True).astype(np.float64)
        n_others = n_finite - finite.astype(np.float64)
        mean_others = np.where(
            n_others >= 1.0,
            (col_sum - filled) / np.maximum(n_others, 1.0),
            np.nan,
        )
        valid = finite & np.isfinite(mean_others)
        n_obs = valid.sum(axis=0)
        a0 = np.where(valid, mat, 0.0)
        b0 = np.where(valid, mean_others, 0.0)
        n_safe = np.maximum(n_obs, 1)
        mean_a = a0.sum(axis=0) / n_safe
        mean_b = b0.sum(axis=0) / n_safe
        da = np.where(valid, mat - mean_a, 0.0)
        db = np.where(valid, mean_others - mean_b, 0.0)
        cov = (da * db).sum(axis=0)
        var_a = (da * da).sum(axis=0)
        var_b = (db * db).sum(axis=0)
        n_m1 = np.maximum(n_obs - 1, 1)
        std_a = np.sqrt(var_a / n_m1)
        std_b = np.sqrt(var_b / n_m1)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.sqrt(var_a * var_b)
        ok = (n_obs >= min_pair) & (std_a > 1e-15) & (std_b > 1e-15) & np.isfinite(corr)
        if int(ok.sum()) < min_ratio * mat.shape[1]:
            return float("nan")
        return float(np.mean(corr[ok])) if np.any(ok) else float("nan")

    hi = leg_corr(snap.high)
    lo = leg_corr(snap.low)
    if np.isfinite(hi) and np.isfinite(lo):
        return 0.5 * (hi + lo)
    if np.isfinite(hi):
        return hi
    if np.isfinite(lo):
        return lo
    return float("nan")


def _ewma_ratio(window: pd.DataFrame, half_life: int) -> float:
    values = window[["target_return", "market_return"]].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        return float("nan")
    length = len(values)
    decay = float(0.5 ** (1.0 / half_life))
    weights = decay ** np.arange(length - 1, -1, -1, dtype="float64")
    weights /= weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    variances = np.sum(centered * centered * weights[:, None], axis=0)
    if variances[1] <= 1e-18 or variances[0] < 0:
        return float("nan")
    return float(np.sqrt(variances[0]) / np.sqrt(variances[1]))


def compute_indicators(
    ctx: StyleCrowdingContext,
    factor_id: str,
    group_count: int,
    orientation: str,
) -> Dict[str, pd.Series]:
    settings = ctx.settings
    skip = set(settings.skip_indicators)
    out: Dict[str, pd.Series] = {}
    base_groups = ctx.groups[group_count][factor_id]
    groups = (
        reverse_group_snapshots({factor_id: base_groups})[factor_id]
        if orientation == "reverse"
        else base_groups
    )
    dispersion = ctx.dispersion_for(group_count, factor_id, orientation)
    hl = ctx.factor_returns[group_count][factor_id]
    if orientation == "reverse":
        hl = -hl
    canonical = ctx.canonical_factor_returns.get(factor_id, pd.Series(np.nan, index=ctx.dates))

    def put(key: str, series: pd.Series) -> None:
        if key in skip:
            return
        out[key] = series.reindex(ctx.output_dates)

    # Valuation
    put("01_valuation_bp_z", _valuation_series(ctx, groups, "pb", log_ratio=True, diff=False, settings=settings))
    put("01_valuation_ps_z", _valuation_series(ctx, groups, "ps", log_ratio=True, diff=False, settings=settings))
    put("01_valuation_pe_z", _valuation_series(ctx, groups, "pe", log_ratio=False, diff=True, settings=settings))

    # Article B/P exposure spread
    art_raw = []
    for date in ctx.dates:
        snap = groups.get(date)
        # article 口径的覆盖度门槛比主口径松（论文原样），单独判定
        if not snapshot_usable(snap, min_coverage=settings.article_min_group_coverage):
            art_raw.append(float("nan"))
            continue
        exp = ctx.exposure_btop.loc[date] if date in ctx.exposure_btop.index else pd.Series(dtype="float64")
        hi = _group_stat(exp, snap.high, "mean", settings.min_group_size)
        lo = _group_stat(exp, snap.low, "mean", settings.min_group_size)
        art_raw.append(hi - lo if np.isfinite(hi) and np.isfinite(lo) else float("nan"))
    put("01_article_valuation_spread_bp_exposure_20d", rolling_mean(pd.Series(art_raw, index=ctx.dates), settings.article_window, 16))

    # Price stretch
    stretch = hl.rolling(settings.price_stretch_window, min_periods=settings.price_stretch_min_obs).apply(_compound, raw=True)
    stretch.iloc[: max(settings.price_stretch_window - 1, 0)] = np.nan
    put("02_price_stretch_60d", stretch)

    # Article dispersion
    for key, col in (
        ("03_article_within_group_dispersion_close_20d", "d_top"),
        ("03_article_within_group_dispersion_close_top50_20d", "d_top_50"),
        ("03_article_within_group_dispersion_close_top100_20d", "d_top_100"),
        ("03_article_within_group_dispersion_close_top200_20d", "d_top_200"),
    ):
        src = pd.to_numeric(dispersion[col], errors="coerce")
        put(key, rolling_mean(src, settings.article_window, 16))

    # Low dispersion
    for key, col, rel_top in (
        ("03_low_dispersion_relative_20d", "d_rel", False),
        ("03_low_dispersion_relative_top50_20d", "d_top_50", True),
        ("03_low_dispersion_relative_top100_20d", "d_top_100", True),
        ("03_low_dispersion_relative_top200_20d", "d_top_200", True),
    ):
        src = pd.to_numeric(dispersion[col], errors="coerce")
        if rel_top:
            market = pd.to_numeric(dispersion["market_std"], errors="coerce")
            src = (src / market).where(market > 0)
        put(key, -rolling_mean(src, settings.article_window, 16))

    # Dispersion spike
    spike_map = {
        "04_dispersion_spike_raw": "d_top",
        "04_dispersion_spike_raw_top50": "d_top_50",
        "04_dispersion_spike_raw_top100": "d_top_100",
        "04_dispersion_spike_raw_top200": "d_top_200",
        "04_dispersion_spike_relative": "d_rel",
        "04_dispersion_spike_relative_top50": "d_top_50",
        "04_dispersion_spike_relative_top100": "d_top_100",
        "04_dispersion_spike_relative_top200": "d_top_200",
    }
    for key, col in spike_map.items():
        rel = key.startswith("04_dispersion_spike_relative_top")
        spike = _dispersion_spike(dispersion, col, settings, relative_top=rel)
        put(key, spike)
        if key in {"04_dispersion_spike_raw", "04_dispersion_spike_relative"}:
            mom_key = key + "_momentum_20d"
            put(mom_key, spike - spike.shift(20))

    # Factor volatility (group invariant — uses canonical WLS style return)
    vol = canonical.rolling(settings.factor_vol_window, min_periods=settings.factor_vol_min_obs).std(ddof=1) * np.sqrt(252.0)
    vol.iloc[: max(settings.factor_vol_window - 1, 0)] = np.nan
    put("05_factor_volatility_60d", vol)
    prior = vol.shift(20)
    put("05_factor_volatility_momentum_20d", (vol / prior - 1.0).where(prior > 0))
    put(
        "05_factor_volatility_of_volatility_20d",
        vol.rolling(20, min_periods=16).std(ddof=1),
    )

    mom21 = canonical.rolling(settings.factor_momentum_window, min_periods=settings.factor_momentum_window).apply(
        lambda x: _compound(x) if np.isfinite(x).all() else float("nan"), raw=True
    )
    put("06_factor_cumulative_return_momentum_21d", mom21)

    # Pairwise correlation：同一 (style, group_count) 正反向共用（两腿平均与方向无关）
    if "pairwise" not in skip and "07_pairwise_correlation_63d" not in skip:
        cache_key = (str(factor_id), int(group_count))
        cached = ctx.pairwise_cache.get(cache_key)
        if cached is not None:
            put("07_pairwise_correlation_63d", cached)
        else:
            raw_pair = []
            dates_list = list(ctx.dates)
            for t, date in enumerate(dates_list):
                snap = groups.get(date)
                if snap is None or not snap.valid or ctx.specific_returns.size == 0:
                    raw_pair.append(float("nan"))
                    continue
                raw_pair.append(
                    _pairwise_corr_day(
                        ctx.specific_returns,
                        t,
                        snap,
                        window=settings.pairwise_window,
                        min_pair=settings.pairwise_min_pair_obs,
                        min_ratio=settings.pairwise_min_member_ratio,
                        symbol_index=ctx.symbol_index,
                    )
                )
            pair_series = pd.Series(raw_pair, index=ctx.dates, dtype="float64")
            scored = prior_rolling_zscore(
                pair_series, window=settings.prior_z_window, min_history=settings.prior_z_min_history
            )
            ctx.pairwise_cache[cache_key] = scored
            put("07_pairwise_correlation_63d", scored)

    # Relative factor volatility
    risk = ctx.risk_factor_returns.get(factor_id)
    if risk is not None and not risk.empty:
        ratio = pd.Series(np.nan, index=ctx.dates, dtype="float64")
        w = settings.risk_covariance_window
        for t in range(w - 1, len(ctx.dates)):
            window = risk.iloc[t - w + 1 : t + 1]
            ratio.iloc[t] = _ewma_ratio(window, settings.risk_covariance_half_life)
        put(
            "08_relative_factor_volatility",
            prior_rolling_zscore(ratio, window=settings.prior_z_window, min_history=settings.relative_vol_z_min_prior),
        )

    return out
