#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截面矩阵核：单因子 compute 与批量 compute_matrix 共用。"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .context import BatchEvalContext, EvalContext
from .metric_result import MetricResult


def rank_cols(values: np.ndarray) -> np.ndarray:
    """列内秩，平均处理并列。NaN 保持 NaN。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True
    else:
        squeeze = False
    n, p = arr.shape
    out = np.full((n, p), np.nan, dtype=np.float64)
    for j in range(p):
        col = arr[:, j]
        valid = np.isfinite(col)
        m = int(valid.sum())
        if m < 2:
            continue
        # pandas C 实现，比纯 Python argsort 循环快一个数量级
        out[valid, j] = pd.Series(col[valid]).rank(method="average").to_numpy(dtype=np.float64)
    return out[:, 0] if squeeze else out


def pearson_cols(left: np.ndarray, right: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """两张 (n, p) 矩阵按列 Pearson：left[:, j] 对 right[:, j]。"""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
        b = b[:, None]
    valid = np.isfinite(a) & np.isfinite(b)
    count = valid.sum(axis=0)
    az = np.where(valid, a, 0.0)
    bz = np.where(valid, b, 0.0)
    cnt = np.maximum(count, 1)
    am = az.sum(axis=0) / cnt
    bm = bz.sum(axis=0) / cnt
    ac = np.where(valid, a - am, 0.0)
    bc = np.where(valid, b - bm, 0.0)
    cov = (ac * bc).sum(axis=0)
    va = (ac * ac).sum(axis=0)
    vb = (bc * bc).sum(axis=0)
    den = np.sqrt(va * vb)
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    ok = (count >= min_obs) & (den > 1e-15)
    np.divide(cov, den, out=out, where=ok)
    return out


def pearson_pairwise(factors: np.ndarray, returns: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """factors (n, p) 与 returns (n,) 的列 Pearson。"""
    f = np.asarray(factors, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if f.ndim == 1:
        f = f[:, None]
    valid = np.isfinite(f) & np.isfinite(r)[:, None]
    count = valid.sum(axis=0)
    fz = np.where(valid, f, 0.0)
    rz = np.where(valid, r[:, None], 0.0)
    cnt = np.maximum(count, 1)
    f_mean = fz.sum(axis=0) / cnt
    r_mean = rz.sum(axis=0) / cnt
    fc = np.where(valid, f - f_mean, 0.0)
    rc = np.where(valid, r[:, None] - r_mean, 0.0)
    cov = (fc * rc).sum(axis=0)
    var_f = (fc * fc).sum(axis=0)
    var_r = (rc * rc).sum(axis=0)
    den = np.sqrt(var_f * var_r)
    out = np.full(f.shape[1], np.nan, dtype=np.float64)
    ok = (count >= min_obs) & (den > 1e-15)
    np.divide(cov, den, out=out, where=ok)
    return out


def spearman_pairwise(factors: np.ndarray, returns: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """factors (n, p) 与 returns (n,) 的列 Spearman（RankIC）。

    排名只在「因子与收益都有效」的交集上做。各自在全集上排名再取交集算相关，
    等价于把缺失样本挤占的名次留在序列里，会系统性压低 |IC|，且不同覆盖度的
    因子之间不可比。
    """
    f = np.asarray(factors, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    valid = np.isfinite(f) & np.isfinite(r)[:, None]
    masked_f = np.where(valid, f, np.nan)
    masked_r = np.where(valid, r[:, None], np.nan)
    return pearson_cols(rank_cols(masked_f), rank_cols(masked_r), min_obs=min_obs)


def assign_quantiles(factors: np.ndarray, n_q: int) -> np.ndarray:
    """列内等频分位标签 1..n_q，无效为 0。"""
    f = np.asarray(factors, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
        squeeze = True
    else:
        squeeze = False
    ranks = rank_cols(f)
    valid = np.isfinite(ranks)
    count = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (ranks - 1.0) * n_q / np.maximum(count, 1)
        labels = np.minimum(scaled, n_q - 1).astype(np.int32) + 1
    out = np.where(valid & (count >= n_q), labels, 0).astype(np.int32)
    return out[:, 0] if squeeze else out


def assign_quantiles_on_returns(
    factors: np.ndarray, returns: Optional[np.ndarray], n_q: int
) -> np.ndarray:
    """按 T 日因子分桶。returns 不参与分组，避免未来缺失改写分位边界。"""
    del returns
    return assign_quantiles(factors, n_q)


def quality_stats(factors: np.ndarray, extreme_threshold: float = 3.0) -> Dict[str, np.ndarray]:
    f = np.asarray(factors, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    n, p = f.shape
    nan_mask = ~np.isfinite(f)
    n_valid = (~nan_mask).sum(axis=0).astype(np.float64)
    n_missing = nan_mask.sum(axis=0).astype(np.float64)
    n_total = float(n)
    coverage = n_valid / n_total if n_total else np.full(p, np.nan)
    unique_count = np.empty(p, dtype=np.float64)
    for j in range(p):
        col = f[~nan_mask[:, j], j]
        unique_count[j] = float(len(np.unique(col))) if col.size else np.nan
    unique_rate = np.where(n_valid > 0, unique_count / n_valid, np.nan)
    mu = np.nanmean(f, axis=0)
    std = np.nanstd(f, axis=0, ddof=1)
    extreme = np.full(p, np.nan)
    for j in range(p):
        if n_valid[j] <= 0:
            continue
        if not np.isfinite(std[j]) or std[j] <= 1e-15:
            extreme[j] = 0.0
            continue
        col = f[~nan_mask[:, j], j]
        extreme[j] = float(np.mean(np.abs(col - mu[j]) > extreme_threshold * std[j]))
    q75 = np.nanpercentile(f, 75, axis=0)
    q25 = np.nanpercentile(f, 25, axis=0)
    return {
        "coverage_rate": coverage,
        "missing_rate": n_missing / n_total if n_total else np.full(p, np.nan),
        "missing_samples": n_missing,
        "valid_samples": n_valid,
        "total_samples": np.full(p, n_total),
        "unique_count": unique_count,
        "unique_rate": unique_rate,
        "extreme_value_ratio": extreme,
        "iqr": q75 - q25,
    }


def nan_skew(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = np.sum(np.isfinite(x), axis=0).astype(np.float64)
    mu = np.nanmean(x, axis=0)
    xc = x - mu
    m2 = np.nanmean(xc ** 2, axis=0)
    m3 = np.nanmean(xc ** 3, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = m3 / np.power(m2, 1.5)
        adj = np.sqrt(n * (n - 1)) / np.maximum(n - 2, 1.0)
        out = adj * g1
    out[n < 3] = np.nan
    return out


def nan_kurtosis(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = np.sum(np.isfinite(x), axis=0).astype(np.float64)
    mu = np.nanmean(x, axis=0)
    xc = x - mu
    m2 = np.nanmean(xc ** 2, axis=0)
    m4 = np.nanmean(xc ** 4, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g2 = m4 / np.power(m2, 2) - 3.0
    out = g2
    out[n < 4] = np.nan
    return out


def quantile_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    bins = np.empty(n, dtype=np.int32)
    bins[order] = np.minimum(np.arange(n) * n_bins // n, n_bins - 1)
    return bins


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """一维距离相关。调用方应先降采样。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 10:
        return float("nan")
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    b = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov = float(np.mean(a * b))
    dvar_x = float(np.mean(a * a))
    dvar_y = float(np.mean(b * b))
    den = np.sqrt(dvar_x * dvar_y)
    if den <= 0:
        return float("nan")
    return float(np.sqrt(max(dcov / den, 0.0)))


def spearman_corr_matrix(factors: np.ndarray, min_obs: int = 20) -> np.ndarray:
    """factors (n_stocks, n_factors) 的截面 Spearman 相关。"""
    ranked = rank_cols(factors)
    p = ranked.shape[1]
    out = np.full((p, p), np.nan, dtype=np.float64)
    valid_n = np.isfinite(ranked).sum(axis=0)
    centered = ranked - np.nanmean(ranked, axis=0)
    centered = np.where(np.isfinite(ranked), centered, 0.0)
    gram = centered.T @ centered
    norms = np.sqrt(np.diag(gram))
    den = norms[:, None] * norms[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(den > 0, gram / den, np.nan)
    enough = (valid_n[:, None] >= min_obs) & (valid_n[None, :] >= min_obs)
    out[enough] = corr[enough]
    np.fill_diagonal(out, 1.0)
    return out


def spearman_cross_corr(
    left: np.ndarray, right: np.ndarray, min_obs: int = 20
) -> np.ndarray:
    """left (n_stocks, p) 与 right (n_stocks, q) 的截面 Spearman 交叉块。

    排名口径与 ``spearman_corr_matrix`` 一致（各列在自身有效集上排名），
    这样批内对角块与跨批非对角块可以直接拼成同一张相关矩阵。
    """
    ra = rank_cols(np.asarray(left, dtype=np.float64))
    rb = rank_cols(np.asarray(right, dtype=np.float64))
    ok_a = np.isfinite(ra)
    ok_b = np.isfinite(rb)
    na = ok_a.sum(axis=0)
    nb = ok_b.sum(axis=0)
    mean_a = np.where(na > 0, np.where(ok_a, ra, 0.0).sum(axis=0) / np.maximum(na, 1), 0.0)
    mean_b = np.where(nb > 0, np.where(ok_b, rb, 0.0).sum(axis=0) / np.maximum(nb, 1), 0.0)
    ca = np.where(ok_a, ra - mean_a, 0.0)
    cb = np.where(ok_b, rb - mean_b, 0.0)
    gram = ca.T @ cb
    den = np.sqrt((ca * ca).sum(axis=0))[:, None] * np.sqrt((cb * cb).sum(axis=0))[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(den > 0, gram / den, np.nan)
    enough = (na[:, None] >= min_obs) & (nb[None, :] >= min_obs)
    return np.where(enough, corr, np.nan)


def nan_cumsum(values: np.ndarray) -> np.ndarray:
    """skipna 累计：NaN 当 0 累加，位置仍写出当前和（对齐 pandas skipna=True）。"""
    filled = np.where(np.isfinite(values), values, 0.0)
    return np.cumsum(filled, axis=0)


def prior_rolling_zscore(
    series: pd.Series,
    *,
    window: int = 252,
    min_history: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """严格 t 以前 rolling 窗口的 Z 分数；当前值不参与阈值。"""
    window = int(window)
    min_history = int(min_history)
    if window < 2:
        raise ValueError("window 必须至少为 2")
    if min_history < 1 or min_history > window:
        raise ValueError("min_history 无效")
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    prior = values.shift(1)
    rolling = prior.rolling(window=window, min_periods=1)
    count = rolling.count()
    mean = rolling.mean()
    std = rolling.std(ddof=ddof)
    result = (values - mean) / std
    valid = values.notna() & (count >= min_history) & np.isfinite(std) & (std > 0.0)
    return result.where(valid).astype("float64")


def causal_percentile(series: pd.Series, *, min_prior: int = 252) -> pd.Series:
    """当前值相对严格历史经验分位；相同值按 <= 计数。"""
    from bisect import bisect_right, insort

    min_prior = int(min_prior)
    history: list[float] = []
    output: list[float] = []
    for raw in pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64"):
        if not np.isfinite(raw):
            output.append(float("nan"))
            continue
        value = float(raw)
        output.append(
            bisect_right(history, value) / len(history) if len(history) >= min_prior else float("nan")
        )
        insort(history, value)
    return pd.Series(output, index=series.index, dtype="float64")


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """列向滚动均值；窗口内有效点不足 ``window`` 则为 NaN。"""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 1:
        return values.astype(np.float64, copy=True)
    n = values.shape[0]
    filled = np.where(np.isfinite(values), values, 0.0)
    csum = np.nancumsum(filled, axis=0)
    ccnt = np.cumsum(np.isfinite(values).astype(np.float64), axis=0)
    for i in range(window - 1, n):
        lo = i - window
        s = csum[i] - (csum[lo] if lo >= 0 else 0)
        c = ccnt[i] - (ccnt[lo] if lo >= 0 else 0)
        out[i] = np.where(c >= window, s / np.maximum(c, 1.0), np.nan)
    return out


def icir_row(daily: np.ndarray) -> np.ndarray:
    """``mean/std``，形状 (1, n_factors)。"""
    mean = np.nanmean(daily, axis=0)
    std = np.nanstd(daily, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(std > 0, mean / std, np.nan)[None, :]


def expanding_icir(daily: np.ndarray) -> np.ndarray:
    """逐日 expanding IR：``mean(IC_1..t) / std(IC_1..t, ddof=1)``。

    ``daily`` 形状 ``(n_dates, n_factors)``，输出同形。
    """
    x = np.asarray(daily, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_dates, n_factors = x.shape
    out = np.full((n_dates, n_factors), np.nan, dtype=np.float64)
    count = np.zeros(n_factors, dtype=np.float64)
    cumsum = np.zeros(n_factors, dtype=np.float64)
    cumsum_sq = np.zeros(n_factors, dtype=np.float64)
    for t in range(n_dates):
        row = x[t]
        valid = np.isfinite(row)
        count[valid] += 1.0
        cumsum[valid] += row[valid]
        cumsum_sq[valid] += row[valid] ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = cumsum / np.maximum(count, 1.0)
            var = (cumsum_sq - cumsum * mean) / np.maximum(count - 1.0, 1.0)
            ir = np.where((count > 1) & (var > 1e-30), mean / np.sqrt(var), np.nan)
        out[t] = ir
    return out


def summarize_daily(series: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(clean))
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "icir": float("nan"), "positive_ratio": float("nan"), "n_days": 0}
    values = clean.to_numpy(dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    icir = float(mean / std) if n > 1 and std > 0 and np.isfinite(std) else float("nan")
    return {
        "mean": mean,
        "std": std,
        "icir": icir,
        "positive_ratio": float(np.mean(values > 0)),
        "n_days": n,
    }


def barra_row(ctx: EvalContext, date_i: int, style_names: Tuple[str, ...]) -> Optional[np.ndarray]:
    raw = ctx.intermediates.get("style_raw")
    if not isinstance(raw, Mapping):
        return None
    cols = []
    n_stocks = ctx.factor.shape[1]
    for name in style_names:
        panel = raw.get(name)
        if panel is None:
            cols.append(np.full(n_stocks, np.nan))
            continue
        row = panel.reindex(index=ctx.factor.index, columns=ctx.factor.columns)
        cols.append(row.to_numpy(dtype=np.float64)[date_i])
    return np.column_stack(cols)


def apply_daily_matrix(metric, ctx: EvalContext, params: Mapping) -> Dict[str, pd.Series]:
    """把 compute_matrix 逐日套到单因子宽表上。"""
    factor = ctx.masked_factor()
    fwd = ctx.masked_fwd_ret()
    values = factor.to_numpy(dtype=np.float64, copy=False)
    rets = fwd.to_numpy(dtype=np.float64, copy=False)
    mask_arr = None
    if ctx.universe_mask is not None:
        mask_arr = ctx.universe_mask.to_numpy(dtype=bool, copy=False)
    n_dates = values.shape[0]
    buckets: Dict[str, list] = {}
    style_raw = ctx.intermediates.get("style_raw")
    from .exposure_engine import RAW_STYLE_NAMES

    size_panel = style_raw.get("style_size") if isinstance(style_raw, Mapping) else None
    for i in range(n_dates):
        mask = mask_arr[i] if mask_arr is not None else np.ones(values.shape[1], dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        barra = None
        size = None
        if isinstance(style_raw, Mapping):
            barra = barra_row(ctx, i, RAW_STYLE_NAMES)
        if size_panel is not None:
            size = size_panel.reindex(index=factor.index, columns=factor.columns).to_numpy(dtype=np.float64)[i]
        batch = BatchEvalContext(
            factors=values[i, :, None],
            returns=rets[i],
            mask=mask,
            barra=barra,
            size=size,
            date=str(factor.index[i]),
            n_quantiles=int(params.get("n_quantiles", 5)),
            min_obs=int(params.get("min_obs", 20)),
        )
        day = metric.compute_matrix(batch, params)
        for key, arr in day.items():
            vec = np.asarray(arr, dtype=np.float64).reshape(-1)
            buckets.setdefault(key, []).append(float(vec[0]) if vec.size else float("nan"))
    index = factor.index
    return {
        key: pd.Series(vals, index=index, name=key, dtype="float64")
        for key, vals in buckets.items()
    }


def result_from_daily(
    daily: Mapping[str, pd.Series],
    *,
    extra_scalars: Optional[Mapping[str, float]] = None,
    primary: Optional[str] = None,
    extra_series: Optional[Mapping[str, pd.Series]] = None,
) -> MetricResult:
    series = {key: value.astype("float64") for key, value in daily.items()}
    if extra_series:
        series.update(extra_series)
    key = primary or (next(iter(daily)) if daily else None)
    scalars: Dict[str, float] = {}
    if key is not None:
        scalars.update(summarize_daily(daily[key]))
    if extra_scalars:
        scalars.update(dict(extra_scalars))
    return MetricResult(scalars=scalars, series=series)
