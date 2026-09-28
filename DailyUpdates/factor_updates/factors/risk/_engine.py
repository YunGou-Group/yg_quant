#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽 CNE5 风格因子。

时间序列描述子在这里计算。多描述子因子先按聚宽规则对每个描述子做
2.5 倍标准差去极值、市值加权均值 / 等权标准差标准化，再按权重合成
（缺失描述子的权重在剩余项上重归一），然后做定义里要求的正交。

``style_size`` 入库值仍是总市值的自然对数，暴露矩阵用 ``exp(style_size)``
做权重。最终的风格因子标准化放在 ``ExposureEngine``，按当天股票池再做一次。

和聚宽原文的差别（库内没有对应数据）：
- 盈利、成长里的分析师预期项视为缺失，权重重归一。
- 优先股账面按 0。长期债务用「非流动负债/负债合计 × 产权比率」近似。
- 银行、保险的营收增长记缺失（保险不用已赚保费替代）。
- 行业缺失填充用申万一级当前成分，不是聚宽一级的历史成分。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numba import njit

from yg_quant_repo import default_db_path

logger = logging.getLogger("Style.engine")

BETA_WINDOW = 252
BETA_HALFLIFE = 63
BETA_MIN_OBS = 22
MOMENTUM_WINDOW = 504
MOMENTUM_HALFLIFE = 126
MOMENTUM_LAG = 21
MOMENTUM_MIN_OBS = 126
DSTD_WINDOW = 252
DSTD_HALFLIFE = 42
HIST_SIGMA_MIN_OBS = 22
MONTH_DAYS = 21
RANGE_MONTHS = 12
TURNOVER_MONTH = 21
TURNOVER_QUARTER = 63
TURNOVER_YEAR = 252

RESVOL_WEIGHTS = (0.74, 0.16, 0.10)
LIQUIDITY_WEIGHTS = (0.35, 0.35, 0.30)
EY_WEIGHTS = (0.68, 0.21, 0.11)
GROWTH_WEIGHTS = (0.18, 0.11, 0.24, 0.47)
LEVERAGE_WEIGHTS = (0.38, 0.35, 0.27)
WINSOR_SIGMA = 2.5

_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE: Dict[int, "StyleEngine"] = {}

_RATIO_FIELDS = ("debt_to_assets", "debt_to_eqt", "longdeb_to_debt")


def _resolve_db_path() -> Optional[str]:
    path = default_db_path()
    return str(path) if path.is_file() else None


def _normalize_market(data: pd.DataFrame) -> pd.DataFrame:
    frame = data
    if "ts_code" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "ts_code"})
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "trade_date"})
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["ts_code"] = frame["ts_code"].astype(str)
    mask = ~frame["ts_code"].str.startswith("index_")
    frame = frame.loc[mask]
    return frame.sort_values(["ts_code", "trade_date"])


def _to_wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in frame.columns:
        raise ValueError(f"风格描述子缺少行情字段: {field}")
    wide = frame.pivot(index="trade_date", columns="ts_code", values=field)
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide.sort_index()


def _wide_to_long(wide: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
    if start_date:
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        wide = wide.loc[wide.index >= start]
    stacked = wide.stack(future_stack=True).rename("factor_value").reset_index()
    stacked.columns = ["trade_date", "ts_code", "factor_value"]
    stacked["factor_value"] = stacked["factor_value"].replace([np.inf, -np.inf], np.nan)
    stacked = stacked.dropna(subset=["factor_value"])
    return stacked.reset_index(drop=True)


def _slope_ratio(values: np.ndarray) -> float:
    """五年（或四年）水平对 0..n-1 的斜率，除以绝对值均值。"""
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 4:
        return np.nan
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    var_x = float(np.dot(x, x))
    if var_x <= 0:
        return np.nan
    slope = float(np.dot(x, y - y.mean()) / var_x)
    denom = float(np.mean(np.abs(y)))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return slope / denom


def scale_percent_ratio(series: pd.Series) -> pd.Series:
    """Tushare 比率常以百分数存储。中位数大于 2 时视为百分数。"""
    values = pd.to_numeric(series, errors="coerce")
    sample = values.to_numpy(dtype=np.float64, copy=False)
    med = np.nanmedian(np.abs(sample)) if sample.size else np.nan
    if np.isfinite(med) and med > 2.0:
        return values / 100.0
    return values


def annual_regression_growth(fin: pd.DataFrame, field: str) -> pd.DataFrame:
    """年报序列的聚宽增长斜率。返回 ts_code, ann_date, value。"""
    empty = pd.DataFrame(columns=["ts_code", "ann_date", "value"])
    if fin is None or fin.empty or field not in fin.columns:
        return empty
    work = fin[["ts_code", "ann_date", "end_date", field]].copy()
    work["ann_date"] = pd.to_datetime(work["ann_date"], errors="coerce")
    work["end_date"] = pd.to_datetime(work["end_date"], errors="coerce")
    work[field] = pd.to_numeric(work[field], errors="coerce")
    work = work.dropna(subset=["ts_code", "ann_date", "end_date", field])
    work = work.loc[work["end_date"].dt.month == 12]
    if work.empty:
        return empty
    work = work.sort_values(["ts_code", "end_date", "ann_date"])
    work = work.drop_duplicates(["ts_code", "end_date"], keep="last")
    rows: List[Tuple[str, pd.Timestamp, float]] = []
    for code, grp in work.groupby("ts_code", sort=False):
        end = grp["end_date"].to_numpy()
        ann = grp["ann_date"].to_numpy()
        val = grp[field].to_numpy(dtype=np.float64)
        for i in range(val.size):
            known = (ann <= ann[i]) & (end <= end[i])
            idx = np.flatnonzero(known)
            if idx.size < 4:
                continue
            growth = _slope_ratio(val[idx[-5:]])
            if np.isfinite(growth):
                rows.append((str(code), pd.Timestamp(ann[i]), float(growth)))
    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=["ts_code", "ann_date", "value"])
    return out.drop_duplicates(["ts_code", "ann_date"], keep="last")


def ttm_from_ytd(fin: pd.DataFrame, field: str) -> pd.DataFrame:
    """累计流量转 TTM：年报即当年，季报 = 本期累计 + 上年年报 - 上年同期累计。"""
    empty = pd.DataFrame(columns=["ts_code", "ann_date", "value"])
    if fin is None or fin.empty or field not in fin.columns:
        return empty
    work = fin[["ts_code", "ann_date", "end_date", field]].copy()
    work["ann_date"] = pd.to_datetime(work["ann_date"], errors="coerce")
    work["end_date"] = pd.to_datetime(work["end_date"], errors="coerce")
    work[field] = pd.to_numeric(work[field], errors="coerce")
    work = work.dropna(subset=["ts_code", "ann_date", "end_date", field])
    if work.empty:
        return empty
    work = work.sort_values(["ts_code", "end_date", "ann_date"])
    work = work.drop_duplicates(["ts_code", "end_date"], keep="last")
    rows: List[Tuple[str, pd.Timestamp, float]] = []
    for code, grp in work.groupby("ts_code", sort=False):
        by_period = {}
        for end, ann, value in zip(
            grp["end_date"].to_numpy(),
            grp["ann_date"].to_numpy(),
            grp[field].to_numpy(dtype=np.float64),
        ):
            stamp = pd.Timestamp(end)
            by_period[(int(stamp.year), int(stamp.month))] = (
                pd.Timestamp(ann),
                float(value),
            )
        for (year, month), (ann, ytd) in by_period.items():
            if not np.isfinite(ytd):
                continue
            if month == 12:
                rows.append((str(code), ann, ytd))
                continue
            annual = by_period.get((year - 1, 12))
            prev = by_period.get((year - 1, month))
            if annual is None or prev is None:
                continue
            if annual[0] > ann or prev[0] > ann:
                continue
            rows.append((str(code), ann, ytd + annual[1] - prev[1]))
    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=["ts_code", "ann_date", "value"])
    return out.drop_duplicates(["ts_code", "ann_date"], keep="last")


def _events_to_wide(
    events: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    empty = pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float64)
    if events is None or events.empty:
        return empty
    piece = events.dropna(subset=["value"]).copy()
    if piece.empty:
        return empty
    piece["ann_date"] = pd.to_datetime(piece["ann_date"]).dt.strftime("%Y-%m-%d")
    piece = piece.drop_duplicates(["ts_code", "ann_date"], keep="last")
    announced = piece.pivot(index="ann_date", columns="ts_code", values="value")
    calendar = pd.Index(index.astype(str))
    combined = calendar.union(announced.index).sort_values()
    wide = announced.reindex(combined).ffill().reindex(calendar)
    return wide.reindex(columns=columns).astype(np.float64)


def _zero_after_listing(ret: np.ndarray, vol: Optional[np.ndarray]) -> np.ndarray:
    """上市前保持缺失；上市后停牌或缺失收益记 0。"""
    out = np.array(ret, dtype=np.float64, copy=True)
    finite = np.isfinite(out)
    seen = np.maximum.accumulate(finite.astype(np.int8), axis=0).astype(bool)
    out = np.where(seen & ~finite, 0.0, out)
    if vol is not None:
        halted = seen & np.isfinite(vol) & (vol <= 0)
        out = np.where(halted, 0.0, out)
    return out


def _winsor_sigma(arr: np.ndarray, sigma: float = WINSOR_SIGMA, min_obs: int = 10) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True)
    finite = np.isfinite(out)
    count = finite.sum(axis=1)
    ok = count >= int(min_obs)
    if not ok.any():
        return out
    masked = np.where(finite, out, np.nan)
    mu = np.full(out.shape[0], np.nan)
    sd = np.full(out.shape[0], np.nan)
    mu[ok] = np.nanmean(masked[ok], axis=1)
    sd[ok] = np.nanstd(masked[ok], axis=1)
    use = ok & np.isfinite(sd) & (sd > 1e-12)
    if not use.any():
        return out
    lo = mu - float(sigma) * sd
    hi = mu + float(sigma) * sd
    clipped = np.clip(out, lo[:, None], hi[:, None])
    out[use] = clipped[use]
    return out


def _standardize(arr: np.ndarray, cap: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """市值加权均值、等权标准差。"""
    if not np.isfinite(arr).any():
        return np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    w = np.where(finite & np.isfinite(cap) & (cap > 0), cap, 0.0)
    sw = w.sum(axis=1)
    mu = np.sum(np.where(w > 0, arr, 0.0) * w, axis=1) / np.where(sw > 0, sw, np.nan)
    count = finite.sum(axis=1)
    plain = np.where(finite, arr, np.nan)
    eq_mu = np.full(arr.shape[0], np.nan)
    var = np.full(arr.shape[0], np.nan)
    has = count > 0
    if has.any():
        eq_mu[has] = np.nanmean(plain[has], axis=1)
        var[has] = np.nanmean((plain[has] - eq_mu[has, None]) ** 2, axis=1)
    sd = np.sqrt(var)
    sd = np.where((count >= int(min_obs)) & np.isfinite(sd) & (sd > 1e-12), sd, np.nan)
    return (arr - mu[:, None]) / sd[:, None]


def _combine(parts: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """缺失描述子不参与，剩余权重重新归一。"""
    acc = np.zeros(parts[0].shape, dtype=np.float64)
    wsum = np.zeros(parts[0].shape, dtype=np.float64)
    for part, weight in zip(parts, weights):
        mask = np.isfinite(part)
        acc = np.where(mask, acc + float(weight) * part, acc)
        wsum = np.where(mask, wsum + float(weight), wsum)
    return np.divide(acc, wsum, out=np.full(acc.shape, np.nan), where=wsum > 0)


def _residualize(y: np.ndarray, regressors: Sequence[np.ndarray], cap: np.ndarray) -> np.ndarray:
    """逐日市值加权回归取残差。样本太小则保留原值。"""
    out = np.array(y, dtype=np.float64, copy=True)
    n_reg = len(regressors)
    min_obs = n_reg + 5
    for t in range(y.shape[0]):
        mask = np.isfinite(y[t]) & np.isfinite(cap[t]) & (cap[t] > 0)
        for reg in regressors:
            mask &= np.isfinite(reg[t])
        idx = np.flatnonzero(mask)
        if idx.size < min_obs:
            continue
        yy = y[t, idx]
        weights = cap[t, idx]
        weights = weights / weights.sum()
        design = [np.ones(idx.size)]
        design.extend(reg[t, idx] for reg in regressors)
        x_mat = np.column_stack(design)
        scale = np.sqrt(weights)
        coef, _, _, _ = np.linalg.lstsq(x_mat * scale[:, None], yy * scale, rcond=None)
        resid = np.full(y.shape[1], np.nan)
        resid[idx] = yy - x_mat @ coef
        known = np.isfinite(y[t])
        out[t, known] = np.where(np.isfinite(resid[known]), resid[known], y[t, known])
    return out


def _industry_fill(
    values: np.ndarray, log_size: np.ndarray, groups: Dict[str, np.ndarray]
) -> np.ndarray:
    """行业内对对数市值回归，填充缺失的风格值。"""
    if not groups:
        return values
    out = np.array(values, dtype=np.float64, copy=True)
    for t in range(out.shape[0]):
        y = out[t]
        x = log_size[t]
        if not np.isfinite(y).any():
            continue
        for idx in groups.values():
            member_y = y[idx]
            member_x = x[idx]
            obs = np.isfinite(member_y) & np.isfinite(member_x)
            miss = ~np.isfinite(member_y) & np.isfinite(member_x)
            if int(obs.sum()) < 5 or not miss.any():
                continue
            xx = member_x[obs]
            yy = member_y[obs]
            xc = xx - xx.mean()
            var_x = float(np.dot(xc, xc))
            if var_x <= 1e-18:
                pred = np.full(int(miss.sum()), float(yy.mean()))
            else:
                slope = float(np.dot(xc, yy - yy.mean()) / var_x)
                intercept = float(yy.mean() - slope * xx.mean())
                pred = intercept + slope * member_x[miss]
            filled = member_y.copy()
            filled[miss] = pred
            y[idx] = filled
        out[t] = y
    return out


@njit(cache=True)
def _ewm_beta_alpha_1d(y, x, window, decay, min_obs):
    n = y.shape[0]
    beta = np.empty(n)
    alpha = np.empty(n)
    beta[:] = np.nan
    alpha[:] = np.nan
    acc_w = 0.0
    acc_x = 0.0
    acc_y = 0.0
    acc_xy = 0.0
    acc_x2 = 0.0
    drop_w = decay ** window
    for t in range(n):
        acc_w *= decay
        acc_x *= decay
        acc_y *= decay
        acc_xy *= decay
        acc_x2 *= decay
        if t >= window:
            old_x = x[t - window]
            old_y = y[t - window]
            acc_w -= drop_w
            acc_x -= drop_w * old_x
            acc_y -= drop_w * old_y
            acc_xy -= drop_w * old_x * old_y
            acc_x2 -= drop_w * old_x * old_x
        acc_w += 1.0
        xt = x[t]
        yt = y[t]
        acc_x += xt
        acc_y += yt
        acc_xy += xt * yt
        acc_x2 += xt * xt
        nobs = window if t + 1 >= window else t + 1
        if nobs < min_obs or acc_w <= 1e-12:
            continue
        mean_x = acc_x / acc_w
        mean_y = acc_y / acc_w
        var_x = acc_x2 / acc_w - mean_x * mean_x
        if var_x <= 1e-18:
            continue
        cov = acc_xy / acc_w - mean_x * mean_y
        b = cov / var_x
        beta[t] = b
        alpha[t] = mean_y - b * mean_x
    return beta, alpha


@njit(cache=True)
def _resid_std_1d(y, x, alpha, beta, window, min_obs):
    n = y.shape[0]
    out = np.empty(n)
    out[:] = np.nan
    cy = np.zeros(n + 1)
    cx = np.zeros(n + 1)
    cy2 = np.zeros(n + 1)
    cx2 = np.zeros(n + 1)
    cxy = np.zeros(n + 1)
    for t in range(n):
        cy[t + 1] = cy[t] + y[t]
        cx[t + 1] = cx[t] + x[t]
        cy2[t + 1] = cy2[t] + y[t] * y[t]
        cx2[t + 1] = cx2[t] + x[t] * x[t]
        cxy[t + 1] = cxy[t] + x[t] * y[t]
    for t in range(n):
        a = alpha[t]
        b = beta[t]
        if np.isnan(a) or np.isnan(b):
            continue
        left = 0 if t + 1 < window else t + 1 - window
        nobs = t + 1 - left
        if nobs < min_obs:
            continue
        inv = 1.0 / nobs
        mean_y = (cy[t + 1] - cy[left]) * inv
        mean_x = (cx[t + 1] - cx[left]) * inv
        mean_y2 = (cy2[t + 1] - cy2[left]) * inv
        mean_x2 = (cx2[t + 1] - cx2[left]) * inv
        mean_xy = (cxy[t + 1] - cxy[left]) * inv
        ee = (
            mean_y2
            + a * a
            + b * b * mean_x2
            - 2.0 * a * mean_y
            - 2.0 * b * mean_xy
            + 2.0 * a * b * mean_x
        )
        me = mean_y - a - b * mean_x
        var = ee - me * me
        if var < 0.0:
            var = 0.0
        out[t] = np.sqrt(var)
    return out


@njit(cache=True)
def _ewm_std_1d(z, window, decay, min_obs):
    n = z.shape[0]
    out = np.empty(n)
    out[:] = np.nan
    acc_w = 0.0
    acc_z = 0.0
    acc_z2 = 0.0
    drop_w = decay ** window
    for t in range(n):
        acc_w *= decay
        acc_z *= decay
        acc_z2 *= decay
        if t >= window:
            old = z[t - window]
            acc_w -= drop_w
            acc_z -= drop_w * old
            acc_z2 -= drop_w * old * old
        acc_w += 1.0
        zt = z[t]
        acc_z += zt
        acc_z2 += zt * zt
        nobs = window if t + 1 >= window else t + 1
        if nobs < min_obs or acc_w <= 1e-12:
            continue
        mean = acc_z / acc_w
        var = acc_z2 / acc_w - mean * mean
        if var < 0.0:
            var = 0.0
        out[t] = np.sqrt(var)
    return out


@njit(cache=True)
def _ewm_mean_1d(z, window, decay, min_obs):
    n = z.shape[0]
    out = np.empty(n)
    out[:] = np.nan
    acc_w = 0.0
    acc_z = 0.0
    drop_w = decay ** window
    for t in range(n):
        acc_w *= decay
        acc_z *= decay
        if t >= window:
            old = z[t - window]
            acc_w -= drop_w
            acc_z -= drop_w * old
        acc_w += 1.0
        acc_z += z[t]
        nobs = window if t + 1 >= window else t + 1
        if nobs < min_obs or acc_w <= 1e-12:
            continue
        out[t] = acc_z / acc_w
    return out


def _first_valid(column: np.ndarray) -> int:
    finite = np.isfinite(column)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return -1
    return int(idx[0])


def _beta_panel(stock: np.ndarray, market: np.ndarray):
    rows, cols = stock.shape
    beta = np.full((rows, cols), np.nan)
    alpha = np.full((rows, cols), np.nan)
    decay = 0.5 ** (1.0 / BETA_HALFLIFE)
    for j in range(cols):
        start = _first_valid(stock[:, j])
        if start < 0:
            continue
        b, a = _ewm_beta_alpha_1d(
            stock[start:, j], market[start:], BETA_WINDOW, decay, BETA_MIN_OBS
        )
        beta[start:, j] = b
        alpha[start:, j] = a
    return beta, alpha


def _hist_sigma_panel(stock, market, alpha, beta):
    rows, cols = stock.shape
    out = np.full((rows, cols), np.nan)
    for j in range(cols):
        start = _first_valid(stock[:, j])
        if start < 0:
            continue
        out[start:, j] = _resid_std_1d(
            stock[start:, j],
            market[start:],
            alpha[start:, j],
            beta[start:, j],
            BETA_WINDOW,
            HIST_SIGMA_MIN_OBS,
        )
    return out


def _ewm_std_panel(values: np.ndarray, window: int, halflife: int, min_obs: int):
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan)
    decay = 0.5 ** (1.0 / halflife)
    for j in range(cols):
        start = _first_valid(values[:, j])
        if start < 0:
            continue
        out[start:, j] = _ewm_std_1d(values[start:, j], window, decay, min_obs)
    return out


def _ewm_mean_panel(values: np.ndarray, window: int, halflife: int, min_obs: int):
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan)
    decay = 0.5 ** (1.0 / halflife)
    for j in range(cols):
        start = _first_valid(values[:, j])
        if start < 0:
            continue
        out[start:, j] = _ewm_mean_1d(values[start:, j], window, decay, min_obs)
    return out


class StyleEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        financial: Optional[pd.DataFrame] = None,
        load_reference: bool = True,
        list_dates: Optional[pd.Series] = None,
        industries: Optional[pd.Series] = None,
        sector_names: Optional[pd.Series] = None,
    ):
        self.frame = _normalize_market(data)
        self._wides: Dict[str, pd.DataFrame] = {}
        self._financial_override = financial
        self._load_reference = load_reference
        self._list_dates = list_dates
        self._industries = industries
        self._sector_names = sector_names
        self._cache: Dict[str, pd.DataFrame] = {}
        self._fin: Optional[pd.DataFrame] = None
        self._fin_loaded = False
        self._ref_loaded = False
        self._returns: Optional[np.ndarray] = None
        self._market: Optional[np.ndarray] = None
        self._beta: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def wide(self, field: str) -> pd.DataFrame:
        cached = self._wides.get(field)
        if cached is not None:
            return cached
        panel = _to_wide(self.frame, field)
        self._wides[field] = panel
        return panel

    def _template(self) -> pd.DataFrame:
        for field in ("total_mv", "circ_mv", "close", "pct_chg", "pb", "pe_ttm"):
            if field in self.frame.columns:
                return self.wide(field)
        raise ValueError("风格引擎需要至少一列行情字段作为日历模板")

    def _array(self, field: str) -> np.ndarray:
        template = self._template()
        panel = self.wide(field).reindex(index=template.index, columns=template.columns)
        return panel.to_numpy(dtype=np.float64)

    def _optional_array(self, field: str) -> Optional[np.ndarray]:
        if field not in self.frame.columns:
            return None
        return self._array(field)

    def _frame(self, values: np.ndarray) -> pd.DataFrame:
        template = self._template()
        return pd.DataFrame(values, index=template.index, columns=template.columns)

    def _positive(self, field: str) -> np.ndarray:
        values = self._array(field)
        return np.where(np.isfinite(values) & (values > 0), values, np.nan)

    def compute(self, key: str) -> pd.DataFrame:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        dispatch = {
            "size": self._size,
            "beta": self._beta_frame,
            "momentum": self._momentum,
            "resvol": self._resvol,
            "btop": self._btop,
            "liquidity": self._liquidity,
            "ey": self._earnings_yield,
            "growth": self._growth,
            "leverage": self._leverage,
        }
        if key not in dispatch:
            raise KeyError(f"未知风格描述子: {key}")
        frame = self._frame(dispatch[key]())
        self._cache[key] = frame
        return frame

    def _size(self) -> np.ndarray:
        cap = self._positive("total_mv")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(cap)

    def _returns_and_market(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._returns is not None and self._market is not None:
            return self._returns, self._market
        ret = self._array("pct_chg") / 100.0
        vol = self._optional_array("vol")
        stock = _zero_after_listing(ret, vol)
        stock = np.where(np.isfinite(stock) & (stock <= -0.999), -0.999, stock)
        cap = self._positive("circ_mv")
        valid = np.isfinite(stock) & np.isfinite(cap)
        num = np.sum(np.where(valid, stock * cap, 0.0), axis=1)
        den = np.sum(np.where(valid, cap, 0.0), axis=1)
        market = np.divide(num, den, out=np.zeros(num.shape), where=den > 0)
        self._returns = stock
        self._market = market
        return stock, market

    def _beta_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._beta is not None and self._alpha is not None:
            return self._beta, self._alpha
        stock, market = self._returns_and_market()
        beta, alpha = _beta_panel(stock, market)
        self._beta = beta
        self._alpha = alpha
        return beta, alpha

    def _beta_frame(self) -> np.ndarray:
        beta, _ = self._beta_arrays()
        return beta

    def _momentum(self) -> np.ndarray:
        stock, market = self._returns_and_market()
        mkt = np.clip(market, -0.999, None)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = np.log1p(stock) - np.log1p(mkt)
        excess = np.where(np.isfinite(stock), excess, np.nan)
        means = _ewm_mean_panel(
            excess, MOMENTUM_WINDOW, MOMENTUM_HALFLIFE, MOMENTUM_MIN_OBS
        )
        out = np.full(means.shape, np.nan)
        lag = MOMENTUM_LAG
        if means.shape[0] > lag:
            out[lag:] = means[:-lag]
        return out

    def _daily_std(self) -> np.ndarray:
        stock, market = self._returns_and_market()
        excess = np.where(np.isfinite(stock), stock - market[:, None], np.nan)
        return _ewm_std_panel(excess, DSTD_WINDOW, DSTD_HALFLIFE, DSTD_WINDOW)

    def _cumulative_range(self) -> np.ndarray:
        close = self._positive("close")
        frame = self._frame(close)
        pieces = []
        for k in range(RANGE_MONTHS):
            current = frame.shift(MONTH_DAYS * k)
            previous = frame.shift(MONTH_DAYS * (k + 1))
            pieces.append(current / previous.where(previous > 0) - 1.0)
        stacked = np.stack([piece.to_numpy(dtype=np.float64) for piece in pieces], axis=0)
        finite = np.isfinite(stacked).all(axis=0)
        high = np.max(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0)
        low = np.min(np.where(np.isfinite(stacked), stacked, np.inf), axis=0)
        out = np.where(finite, high - low, np.nan)
        return self._mask_young_listings(out, months=6)

    def _hist_sigma(self) -> np.ndarray:
        stock, market = self._returns_and_market()
        beta, alpha = self._beta_arrays()
        return _hist_sigma_panel(stock, market, alpha, beta)

    def _resvol(self) -> np.ndarray:
        cap = self._positive("total_mv")
        parts = [
            _standardize(_winsor_sigma(self._daily_std()), cap),
            _standardize(_winsor_sigma(self._cumulative_range()), cap),
            _standardize(_winsor_sigma(self._hist_sigma()), cap),
        ]
        combined = _combine(parts, RESVOL_WEIGHTS)
        beta, _ = self._beta_arrays()
        residual = _residualize(
            combined,
            [_winsor_sigma(beta), _winsor_sigma(self._size())],
            cap,
        )
        return self._fill(residual)

    def _btop(self) -> np.ndarray:
        pb = self._array("pb")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(pb) & (pb > 0), 1.0 / pb, np.nan)

    def _liquidity(self) -> np.ndarray:
        primary = self._optional_array("turnover_rate")
        fallback = self._optional_array("turnover_rate_f")
        if primary is None and fallback is None:
            raise ValueError("style_liquidity 需要 turnover_rate 或 turnover_rate_f")
        if primary is None:
            turn = fallback
        elif fallback is None:
            turn = primary
        else:
            turn = np.where(np.isfinite(primary), primary, fallback)
        frame = self._frame(turn)

        def _log_roll(window: int, how: str) -> np.ndarray:
            rolled = frame.rolling(window, min_periods=window)
            values = rolled.sum() if how == "sum" else rolled.mean()
            raw = values.to_numpy(dtype=np.float64)
            return np.where(np.isfinite(raw) & (raw > 0), np.log(raw), np.nan)

        cap = self._positive("total_mv")
        parts = [
            _standardize(_winsor_sigma(_log_roll(TURNOVER_MONTH, "sum")), cap),
            _standardize(_winsor_sigma(_log_roll(TURNOVER_QUARTER, "mean")), cap),
            _standardize(_winsor_sigma(_log_roll(TURNOVER_YEAR, "mean")), cap),
        ]
        combined = _combine(parts, LIQUIDITY_WEIGHTS)
        residual = _residualize(combined, [_winsor_sigma(self._size())], cap)
        return self._fill(residual)

    def _earnings_yield(self) -> np.ndarray:
        pe = self._array("pe_ttm") if "pe_ttm" in self.frame.columns else None
        if pe is None and "pe" in self.frame.columns:
            pe = self._array("pe")
        elif pe is not None and "pe" in self.frame.columns:
            fallback = self._array("pe")
            pe = np.where(np.isfinite(pe), pe, fallback)
        if pe is None:
            raise ValueError("style_ey 需要 pe_ttm 或 pe")
        with np.errstate(divide="ignore", invalid="ignore"):
            trailing = np.where(np.isfinite(pe) & (pe != 0), 1.0 / pe, np.nan)
        cash = self._cash_earnings_yield()
        predicted = np.full(trailing.shape, np.nan)
        cap = self._positive("total_mv")
        parts = [
            _standardize(_winsor_sigma(predicted), cap),
            _standardize(_winsor_sigma(cash), cap),
            _standardize(_winsor_sigma(trailing), cap),
        ]
        return self._fill(_combine(parts, EY_WEIGHTS))

    def _cash_earnings_yield(self) -> np.ndarray:
        template = self._template()
        fin = self._financial_frame()
        ocf = _events_to_wide(ttm_from_ytd(fin, "ocfps"), template.index, template.columns)
        bps = self._pit_level("bps")
        pb = self._array("pb") if "pb" in self.frame.columns else np.full(ocf.shape, np.nan)
        book = bps.to_numpy(dtype=np.float64)
        ocf_arr = ocf.to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            price = np.where((book > 0) & np.isfinite(pb) & (pb > 0), book * pb, np.nan)
            return np.where(np.isfinite(price) & (price != 0), ocf_arr / price, np.nan)

    def _growth(self) -> np.ndarray:
        template = self._template()
        fin = self._financial_frame()
        earnings = _events_to_wide(
            annual_regression_growth(fin, "eps"), template.index, template.columns
        ).to_numpy(dtype=np.float64)
        sales = _events_to_wide(
            annual_regression_growth(fin, "revenue_ps"), template.index, template.columns
        ).to_numpy(dtype=np.float64)
        blocked = self._sales_blocked()
        if blocked is not None:
            sales = sales.copy()
            sales[:, blocked] = np.nan
        cap = self._positive("total_mv")
        empty = np.full(earnings.shape, np.nan)
        parts = [
            _standardize(_winsor_sigma(empty), cap),
            _standardize(_winsor_sigma(empty), cap),
            _standardize(_winsor_sigma(earnings), cap),
            _standardize(_winsor_sigma(sales), cap),
        ]
        return self._fill(_combine(parts, GROWTH_WEIGHTS))

    def _leverage(self) -> np.ndarray:
        ratios = {
            name: self._pit_ratio(name).to_numpy(dtype=np.float64) for name in _RATIO_FIELDS
        }
        debt_to_assets = ratios["debt_to_assets"]
        longdeb = ratios["longdeb_to_debt"]
        debt_to_eqt = ratios["debt_to_eqt"]
        btop = self._btop()
        ld_over_be = longdeb * debt_to_eqt
        valid_ld = np.isfinite(ld_over_be) & (ld_over_be >= 0)
        book = np.where(valid_ld, 1.0 + ld_over_be, np.nan)
        book = np.where(np.isfinite(book) & (book > 0) & (book < 100), book, np.nan)
        market = np.where(
            valid_ld & np.isfinite(btop) & (btop > 0),
            1.0 + ld_over_be * btop,
            np.nan,
        )
        cap = self._positive("total_mv")
        parts = [
            _standardize(_winsor_sigma(market), cap),
            _standardize(_winsor_sigma(debt_to_assets), cap),
            _standardize(_winsor_sigma(book), cap),
        ]
        return self._fill(_combine(parts, LEVERAGE_WEIGHTS))

    def _mask_young_listings(self, values: np.ndarray, months: int) -> np.ndarray:
        listed = self._list_date_array()
        if listed is None:
            return values
        template = self._template()
        trade = pd.to_datetime(template.index).to_numpy()
        cutoff = (listed + pd.DateOffset(months=int(months))).to_numpy()
        known = pd.notna(listed).to_numpy()
        too_new = known[None, :] & (trade[:, None] < cutoff[None, :])
        return np.where(too_new, np.nan, values)

    def _fill(self, values: np.ndarray) -> np.ndarray:
        groups = self._industry_groups()
        if not groups:
            return values
        return _industry_fill(values, self._size(), groups)

    def _financial_frame(self) -> pd.DataFrame:
        if self._fin_loaded:
            return self._fin if self._fin is not None else pd.DataFrame()
        self._fin_loaded = True
        if self._financial_override is not None:
            frame = self._financial_override.copy()
        else:
            db_path = _resolve_db_path()
            if not db_path:
                self._fin = pd.DataFrame()
                return self._fin
            from DailyUpdates.storage.sqlite_storage import SQLiteStorage

            frame = SQLiteStorage(db_path).read_financial_indicator()
        if frame is None or frame.empty:
            self._fin = pd.DataFrame()
            return self._fin
        if "ts_code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "ts_code"})
        frame["ts_code"] = frame["ts_code"].astype(str)
        for name in _RATIO_FIELDS:
            if name in frame.columns:
                frame[name] = scale_percent_ratio(frame[name])
        self._fin = frame
        return frame

    def _pit_level(self, field: str) -> pd.DataFrame:
        template = self._template()
        fin = self._financial_frame()
        if fin.empty or field not in fin.columns:
            return pd.DataFrame(np.nan, index=template.index, columns=template.columns)
        events = fin[["ts_code", "ann_date", field]].rename(columns={field: "value"})
        events = events.dropna(subset=["value"])
        return _events_to_wide(events, template.index, template.columns)

    def _pit_ratio(self, field: str) -> pd.DataFrame:
        return self._pit_level(field)

    def _sales_blocked(self) -> Optional[np.ndarray]:
        names = self._sector_name_array()
        if names is None:
            return None
        blocked = np.array(
            [("银行" in text) or ("保险" in text) for text in names], dtype=bool
        )
        if not blocked.any():
            return None
        return blocked

    def _ensure_reference(self) -> None:
        if self._ref_loaded:
            return
        self._ref_loaded = True
        if self._list_dates is not None or self._industries is not None or not self._load_reference:
            return
        db_path = _resolve_db_path()
        if not db_path:
            return
        from DailyUpdates.storage.sqlite_storage import SQLiteStorage

        storage = SQLiteStorage(db_path)
        basic = storage.read_stock_basic()
        if basic is not None and not basic.empty and "symbol" in basic.columns:
            self._list_dates = basic.set_index("symbol")["list_date"]
            if "industry" in basic.columns:
                self._sector_names = basic.set_index("symbol")["industry"]
        try:
            members = storage.read_industry_members(level="l1", is_new="Y")
        except Exception:
            logger.warning("读取申万一级行业失败，跳过风格缺失填充", exc_info=True)
            members = pd.DataFrame()
        if members is not None and not members.empty and "symbol" in members.columns:
            self._industries = members.drop_duplicates("symbol", keep="last").set_index("symbol")[
                "industry_code"
            ]

    def _list_date_array(self) -> Optional[pd.Series]:
        self._ensure_reference()
        if self._list_dates is None:
            return None
        template = self._template()
        aligned = self._list_dates.reindex(template.columns)
        parsed = pd.to_datetime(aligned, errors="coerce")
        if parsed.isna().all():
            return None
        return parsed

    def _sector_name_array(self) -> Optional[List[str]]:
        self._ensure_reference()
        if self._sector_names is None:
            return None
        template = self._template()
        aligned = self._sector_names.reindex(template.columns)
        return [("" if pd.isna(value) else str(value)) for value in aligned.tolist()]

    def _industry_groups(self) -> Dict[str, np.ndarray]:
        self._ensure_reference()
        if self._industries is None:
            return {}
        template = self._template()
        aligned = self._industries.reindex(template.columns)
        groups: Dict[str, List[int]] = {}
        for pos, value in enumerate(aligned.tolist()):
            if pd.isna(value) or str(value).strip() == "":
                continue
            groups.setdefault(str(value), []).append(pos)
        return {code: np.asarray(idx, dtype=np.int64) for code, idx in groups.items()}


def get_prepared_engine(
    data: pd.DataFrame,
    *,
    financial: Optional[pd.DataFrame] = None,
    load_reference: bool = True,
) -> StyleEngine:
    key = id(data)
    with _CACHE_LOCK:
        cached = _ENGINE_CACHE.get(key)
        if cached is not None and financial is None and load_reference:
            return cached
        engine = StyleEngine(data, financial=financial, load_reference=load_reference)
        if financial is None and load_reference:
            _ENGINE_CACHE.clear()
            _ENGINE_CACHE[key] = engine
        return engine


def clear_prepared_engine() -> None:
    with _CACHE_LOCK:
        _ENGINE_CACHE.clear()


def extract_style(
    data: pd.DataFrame,
    key: str,
    start_date: Optional[str] = None,
    *,
    financial: Optional[pd.DataFrame] = None,
    load_reference: bool = True,
) -> pd.DataFrame:
    """计算一个风格因子，返回 ``ts_code, trade_date, factor_value``。"""
    engine = get_prepared_engine(
        data, financial=financial, load_reference=load_reference
    )
    started = time.perf_counter()
    wide = engine.compute(key)
    frame = _wide_to_long(wide, start_date=start_date)
    logger.info(
        "风格 %s 完成 rows=%s (%.2fs)",
        key,
        len(frame),
        time.perf_counter() - started,
    )
    return frame


STYLE_KEYS: List[str] = [
    "size",
    "beta",
    "momentum",
    "resvol",
    "btop",
    "liquidity",
    "ey",
    "growth",
    "leverage",
]
