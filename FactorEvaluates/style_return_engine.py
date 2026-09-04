#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格收益：同一张 X_T 上对契约 B 远期收益做截面 WLS。

r = a + X f + e。f 是风格当天的因子收益，不是因子值。
不做 open shift；收益只来自传入的远期收益面板。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .exposure_engine import ExposureMatrix, _align, _zscore_sample


def _wls(x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return beta
    sw = np.sqrt(np.where(np.isfinite(weights) & (weights > 0), weights, 0.0))
    if float(sw.sum()) <= 0:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return beta
    return np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]


class StyleReturnEngine:
    def __init__(self, min_obs: int = 20):
        self.min_obs = int(min_obs)

    def estimate(
        self,
        returns: pd.DataFrame,
        exposures: ExposureMatrix,
        mask: Optional[pd.DataFrame] = None,
        min_obs: Optional[int] = None,
    ) -> pd.DataFrame:
        """逐日 WLS：r = a + Xf + e。返回 date × (intercept + 风格) 的 f_t。"""
        ret_n = returns.copy()
        ret_n.index = pd.to_datetime(ret_n.index).strftime("%Y-%m-%d")
        ret_n.columns = [str(c) for c in ret_n.columns]
        dates = ret_n.index
        symbols = ret_n.columns
        aligned = exposures.align(dates, symbols)
        y = ret_n.to_numpy(dtype=np.float64)
        if mask is None:
            m = np.isfinite(y)
        else:
            m = _align(mask.astype(float), dates, symbols).fillna(0).to_numpy() > 0
            m = m & np.isfinite(y)
        style_names = [name for name in aligned.names if name in aligned.panels]
        cols = [aligned.panels[name].to_numpy(dtype=np.float64) for name in style_names]
        weight = None
        if aligned.weights is not None:
            weight = aligned.weights.to_numpy(dtype=np.float64)
        names = ["intercept", *style_names]
        out = np.full((y.shape[0], len(names)), np.nan, dtype=np.float64)
        if not cols:
            return pd.DataFrame(out, index=dates, columns=names)
        k = len(cols)
        floor = int(min_obs if min_obs is not None else max(self.min_obs, k + 5))
        for t in range(y.shape[0]):
            valid = m[t]
            for col in cols:
                valid = valid & np.isfinite(col[t])
            n = int(valid.sum())
            if n < floor:
                continue
            x = np.column_stack([np.ones(n)] + [col[t, valid] for col in cols])
            w = None if weight is None else weight[t, valid]
            out[t] = _wls(x, y[t, valid], w)
        return pd.DataFrame(out, index=dates, columns=names)

    def factor_return(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        mask: Optional[pd.DataFrame] = None,
        min_obs: Optional[int] = None,
    ) -> pd.Series:
        """因子组合日收益 λ_t：当日样本上 r = a + λ z(F) + e 的斜率。"""
        factor_n = factor.copy()
        factor_n.index = pd.to_datetime(factor_n.index).strftime("%Y-%m-%d")
        factor_n.columns = [str(c) for c in factor_n.columns]
        ret_n = _align(returns, factor_n.index, factor_n.columns)
        y = factor_n.to_numpy(dtype=np.float64)
        r = ret_n.to_numpy(dtype=np.float64)
        if mask is None:
            m = np.isfinite(y) & np.isfinite(r)
        else:
            m = _align(mask.astype(float), factor_n.index, factor_n.columns).fillna(0).to_numpy() > 0
            m = m & np.isfinite(y) & np.isfinite(r)
        floor = int(min_obs if min_obs is not None else max(self.min_obs, 8))
        lam = np.full(y.shape[0], np.nan, dtype=np.float64)
        for t in range(y.shape[0]):
            valid = m[t]
            if int(valid.sum()) < floor:
                continue
            zt = _zscore_sample(y[t, valid])
            if zt is None:
                continue
            x = np.column_stack([np.ones(zt.size), zt])
            beta = _wls(x, r[t, valid], None)
            lam[t] = beta[1]
        return pd.Series(lam, index=factor_n.index, name="factor_return")

    def estimate_day(
        self,
        x: np.ndarray,
        returns: np.ndarray,
        min_obs: Optional[int] = None,
    ) -> np.ndarray:
        """一天：r = X f，X 已含 intercept 列。返回 (k,) 含 intercept。"""
        design = np.asarray(x, dtype=np.float64)
        r = np.asarray(returns, dtype=np.float64).reshape(-1)
        k = design.shape[1]
        out = np.full(k, np.nan, dtype=np.float64)
        if design.ndim != 2 or r.size != design.shape[0]:
            return out
        valid = np.isfinite(design).all(axis=1) & np.isfinite(r)
        n_style = max(k - 1, 0)
        floor = int(min_obs if min_obs is not None else max(self.min_obs, n_style + 5))
        if int(valid.sum()) < floor:
            return out
        out[:] = _wls(design[valid], r[valid], None)
        return out

    def factor_return_day(
        self,
        factors: np.ndarray,
        returns: np.ndarray,
        min_obs: Optional[int] = None,
    ) -> np.ndarray:
        """一天多因子：每列 r = a + λ z(F_j) 的斜率，形状 (p,)。"""
        f = np.asarray(factors, dtype=np.float64)
        if f.ndim == 1:
            f = f[:, None]
        r = np.asarray(returns, dtype=np.float64).reshape(-1)
        p = f.shape[1]
        out = np.full(p, np.nan, dtype=np.float64)
        floor = int(min_obs if min_obs is not None else max(self.min_obs, 8))
        for j in range(p):
            valid = np.isfinite(f[:, j]) & np.isfinite(r)
            if int(valid.sum()) < floor:
                continue
            zt = _zscore_sample(f[valid, j])
            if zt is None:
                continue
            x = np.column_stack([np.ones(zt.size), zt])
            beta = _wls(x, r[valid], None)
            out[j] = beta[1]
        return out


def estimate_style_returns(
    returns: pd.DataFrame,
    exposures: ExposureMatrix,
    mask: Optional[pd.DataFrame] = None,
    min_obs: int = 20,
) -> pd.DataFrame:
    return StyleReturnEngine(min_obs=min_obs).estimate(
        returns, exposures, mask=mask, min_obs=min_obs
    )
