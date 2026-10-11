#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格暴露与截面残差。

分数中性化：对完整 X_T（风格+NLSIZE）等权 OLS。
因子中性化：只对 Size+NLSIZE。选股等权，暴露不做市值加权回归权重。
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from FactorEvaluates.exposure_engine import (
    NLSIZE_NAME,
    RAW_STYLE_NAMES,
    SIZE_NAME,
    ExposureEngine,
    transform_style_arrays,
)

logger = logging.getLogger("Strategies.style_neutralize")

MIN_OBS = 40
_CACHE = "_lgbm_style_expo"
_CACHE_SN = "_lgbm_size_nl_expo"


def neutralize_scores(
    score: np.ndarray,
    expo: np.ndarray,
    universe: np.ndarray,
    *,
    min_obs: int = MIN_OBS,
) -> np.ndarray:
    """score = a + Xβ + e，返回 e。缺暴露的股票不参选。"""
    y = np.asarray(score, dtype=np.float64)
    x = np.asarray(expo, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    out = y.copy()
    if x.shape[0] != y.shape[0]:
        return out
    uni = np.asarray(universe, dtype=bool)
    finite_x = np.all(np.isfinite(x), axis=1)
    ok = uni & np.isfinite(y) & finite_x
    missing = uni & np.isfinite(y) & ~finite_x
    out[missing] = np.nan
    n = int(ok.sum())
    k = int(x.shape[1])
    if n < max(int(min_obs), k + 8):
        return out
    yy = y[ok]
    xx = np.column_stack([np.ones(n, dtype=np.float64), x[ok]])
    try:
        coef, *_ = np.linalg.lstsq(xx, yy, rcond=None)
    except np.linalg.LinAlgError:
        return out
    out[ok] = yy - xx @ coef
    return out


def style_expo_on_store(store) -> np.ndarray:
    """date × stock × 风格（含 NLSIZE，不含行业）。缓存在 store 上。"""
    cached = getattr(store, _CACHE, None)
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)
    raw = _style_frames(store)
    if SIZE_NAME not in raw:
        raise ValueError("风格中性化需要 style_size（CNE5 风格因子须已入库）")
    dates = pd.Index([str(d) for d in store.dates])
    symbols = pd.Index([str(s) for s in store.symbols])
    mask = pd.DataFrame(np.asarray(store.mask, dtype=bool), index=dates, columns=symbols)
    logger.info("风格中性化组装 X_T %s 日 × %s 只", len(dates), len(symbols))
    expo_mat = ExposureEngine().build(raw, mask)
    names = [n for n in expo_mat.names if n in expo_mat.panels and not str(n).startswith("ind_")]
    cols = [
        np.asarray(expo_mat.panels[n].to_numpy(dtype=np.float64, copy=False))
        for n in names
    ]
    if not cols:
        raise ValueError("风格中性化暴露矩阵为空")
    expo = np.stack(cols, axis=2)
    setattr(store, _CACHE, expo)
    logger.info("风格中性化列 %s", ",".join(names))
    return expo


def extra_style_factors(existing: list[str]) -> list[str]:
    have = {str(n) for n in existing}
    return [name for name in RAW_STYLE_NAMES if name not in have]


def size_nlsize_expo_on_store(store) -> np.ndarray:
    """date × stock × 2：股票池上标准化 Size 与 NLSIZE。缓存在 store 上。"""
    cached = getattr(store, _CACHE_SN, None)
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)
    try:
        size = np.asarray(store.panel_for(SIZE_NAME), dtype=np.float64)
    except KeyError as exc:
        raise ValueError("Size/NLSIZE 中性化需要 style_size（CNE5 风格因子须已入库）") from exc
    mask = np.asarray(store.mask, dtype=bool)
    if size.shape != mask.shape:
        raise ValueError(f"style_size {size.shape} 与 mask {mask.shape} 不一致")
    z = transform_style_arrays({SIZE_NAME: size}, mask)
    expo = np.stack(
        [
            np.asarray(z[SIZE_NAME], dtype=np.float64),
            np.asarray(z[NLSIZE_NAME], dtype=np.float64),
        ],
        axis=2,
    )
    setattr(store, _CACHE_SN, expo)
    logger.info("Size+NLSIZE 中性化暴露已缓存 shape=%s", expo.shape)
    return expo


def neutralize_panel_on_expo(
    panel: np.ndarray,
    expo: np.ndarray,
    mask: np.ndarray,
    *,
    min_obs: int = MIN_OBS,
) -> np.ndarray:
    """逐日：截面 z(因子) 对截距+暴露 OLS，返回残差。"""
    arr = np.asarray(panel, dtype=np.float64)
    x_all = np.asarray(expo, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"因子面板须为 (日, 股)，收到 {arr.shape}")
    if x_all.ndim != 3 or x_all.shape[:2] != arr.shape:
        raise ValueError(f"暴露 {x_all.shape} 与因子 {arr.shape} 不一致")
    if m.shape != arr.shape:
        raise ValueError(f"mask {m.shape} 与因子 {arr.shape} 不一致")
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    k = int(x_all.shape[2])
    floor = max(int(min_obs), k + 8)
    for t in range(arr.shape[0]):
        uni = m[t]
        y = np.where(uni, arr[t], np.nan)[:, None]
        cols = [np.where(uni, 1.0, np.nan)]
        for j in range(k):
            cols.append(np.where(uni, x_all[t, :, j], np.nan))
        x = np.column_stack(cols)
        _beta, resid = ExposureEngine.regress_matrix_day(y, x, min_obs=floor)
        out[t] = resid[:, 0]
    return out


def _style_frames(store) -> Dict[str, pd.DataFrame]:
    dates = pd.Index([str(d) for d in store.dates])
    symbols = pd.Index([str(s) for s in store.symbols])
    out: Dict[str, pd.DataFrame] = {}
    for name in RAW_STYLE_NAMES:
        try:
            arr = np.asarray(store.panel_for(name), dtype=np.float64)
        except KeyError:
            continue
        if arr.shape != (len(dates), len(symbols)):
            continue
        out[str(name)] = pd.DataFrame(arr, index=dates, columns=symbols)
    return out
