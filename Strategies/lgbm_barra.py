#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Barra 标签 + 滚动 LightGBM 排序（多头）。

用法：--strategy lgbm_barra --factor a,b,c
因子名前加 '-' 表示取负。

--label：
  resid  持有期收益对当日 X_T 做市值 WLS 的残差，再 2.5σ 截尾+截面 z（默认）
  raw    持有期收益
  util   U = r_{t→t+H} − λ · σ²_barra(t) · H
  sharpe r_{t→t+H} / σ_barra(t)
  style  风格收割 x_t' f̂_t · H

resid 不把 X_T 喂进树。--resid-x 才把因子也对 X_T 残差化（这组旧因子上会把信号洗没）。

每 --refit 个交易日重训一次，同一模型沿用到下次重训；调仓日只用冻结模型打分。

--neutral 是组合层再对树分回归取残差，和 resid 标签不是一回事；resid 时不要开。
--harvest / --pos-style 会把风格加回组合，resid 时默认不要开。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from FactorEvaluates.exposure_engine import RAW_STYLE_NAMES, SIZE_NAME, ExposureEngine
from FactorEvaluates.industry_panel import load_l1_code_panel
from FactorEvaluates.market_panel_loader import MarketPanelLoader
from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy, cli_field, n_field, rebalance_field
from Strategies._factor_combo import (
    FactorLeg,
    cross_section_z,
    parse_factor_list,
)
from Strategies.lgbm import (
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    DEFAULT_MIN_ROWS,
    DEFAULT_N,
    DEFAULT_REFIT,
    MAX_GROUP,
    MAX_GROUPS,
    MIN_GROUPS,
    N_GRADES,
    _cap_group,
    _day_stride,
    _fit_ranker,
    _try_lightgbm,
)
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

logger = logging.getLogger("Strategies.lgbm_barra")

DEFAULT_LABEL = "resid"
DEFAULT_BLEND = 1.0
DEFAULT_SMOOTH = 0.8
DEFAULT_VOL_WINDOW = 60
DEFAULT_BARRA_LAMBDA = 2.0
DEFAULT_HARVEST = 0.0
MIN_RESID_OBS = 40
WINSOR_SIGMA = 2.5
VOL_FLOOR = 0.008
LABELS = frozenset({"resid", "raw", "util", "sharpe", "style"})



@dataclass
class _BarraBook:
    label: np.ndarray
    harvest: np.ndarray
    expo: np.ndarray
    xt: np.ndarray
    weights: np.ndarray


def _window2(panel: np.ndarray, t: int, need: int) -> np.ndarray:
    arr = np.asarray(panel)
    t1 = int(t) + 1
    t0 = max(0, t1 - int(need))
    sl = arr[t0:t1]
    if sl.shape[0] == int(need):
        return sl
    pad_shape = (int(need) - sl.shape[0],) + sl.shape[1:]
    pad = np.full(pad_shape, np.nan, dtype=np.float64)
    return np.concatenate([pad, sl], axis=0)


def _wls_fit(
    y: np.ndarray, x: np.ndarray, w: np.ndarray, min_obs: int
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    resid = np.full(y.shape, np.nan, dtype=np.float64)
    beta = np.full(int(x.shape[1]) + 1, np.nan, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.all(np.isfinite(x), axis=1)
    n = int(ok.sum())
    k = int(x.shape[1]) + 1
    if n < max(int(min_obs), k + 5):
        return resid, beta
    yy = y[ok]
    mu = float(yy.mean())
    sd = float(yy.std(ddof=0))
    if sd > 1e-12:
        yy = np.clip(yy, mu - WINSOR_SIGMA * sd, mu + WINSOR_SIGMA * sd)
    xx = np.column_stack([np.ones(n, dtype=np.float64), x[ok]])
    sw = np.sqrt(w[ok])
    try:
        coef, *_ = np.linalg.lstsq(xx * sw[:, None], yy * sw, rcond=None)
    except np.linalg.LinAlgError:
        return resid, beta
    beta[: coef.size] = coef
    resid[ok] = yy - xx @ coef
    return resid, beta


def _residualize_panel(
    y_panel: np.ndarray,
    expo: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    min_obs: int,
) -> np.ndarray:
    """每天把一列因子对当日 X_T 做市值 WLS，返回残差面板。"""
    y = np.asarray(y_panel, dtype=np.float64)
    x = np.asarray(expo, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    floor = max(int(min_obs), int(x.shape[2]) + 6)
    for t in range(int(y.shape[0])):
        if int(m[t].sum()) < floor:
            continue
        out[t], _ = _wls_fit(y[t], x[t], w[t], floor)
    return out


def residual_factor_panel(store, book: _BarraBook, name: str) -> np.ndarray:
    bag = getattr(store, "_lgbm_barra_xresid", None)
    if bag is None:
        bag = {}
        setattr(store, "_lgbm_barra_xresid", bag)
    key = (id(book.xt), str(name))
    cached = bag.get(key)
    if cached is None:
        raw = np.asarray(store.panel_for(name), dtype=np.float64)
        mask = np.asarray(store.mask, dtype=bool)
        floor = max(MIN_RESID_OBS, int(book.xt.shape[2]) + 6)
        logger.info("lgbm_barra 因子残差化 %s", name)
        cached = _residualize_panel(raw, book.xt, book.weights, mask, floor)
        bag[key] = cached
    return cached


def _holding_specific(resid_1d: np.ndarray, horizon: int) -> np.ndarray:
    """label[t] = sum(u[t:t+H])，至少要有 80% 交易日有残差。"""
    u = np.asarray(resid_1d, dtype=np.float64)
    n_dates, n_stocks = u.shape
    h = max(1, int(horizon))
    out = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    valid = np.isfinite(u)
    filled = np.where(valid, u, 0.0)
    pad = np.zeros((1, n_stocks), dtype=np.float64)
    c1 = np.vstack([pad, np.cumsum(filled, axis=0)])
    c0 = np.vstack([pad, np.cumsum(valid.astype(np.float64), axis=0)])
    need_n = max(1, int(np.ceil(h * 0.8)))
    width = n_dates + 1 - h
    if width <= 0:
        return out
    total = c1[h:] - c1[:-h]
    count = c0[h:] - c0[:-h]
    take = min(int(total.shape[0]), n_dates)
    out[:take] = np.where(count[:take] >= float(need_n), total[:take], np.nan)
    return out


def _cs_winsor_z(
    values: np.ndarray, mask: np.ndarray, sigma: float = WINSOR_SIGMA
) -> np.ndarray:
    """截面 2.5σ 截尾后再等权 z-score。"""
    out = np.full(np.asarray(values).shape, np.nan, dtype=np.float64)
    raw = np.asarray(values, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool) & np.isfinite(raw)
    n = int(sel.sum())
    if n < 2:
        if n == 1:
            out[sel] = 0.0
        return out
    x = raw[sel]
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd < 1e-12:
        out[sel] = 0.0
        return out
    x = np.clip(x, mu - float(sigma) * sd, mu + float(sigma) * sd)
    mu2 = float(x.mean())
    sd2 = float(x.std(ddof=0))
    if sd2 < 1e-12:
        out[sel] = 0.0
    else:
        out[sel] = (x - mu2) / sd2
    return out


def _equal_winsor_z(
    panels: Sequence[np.ndarray], signs: Sequence[float], universe: np.ndarray
) -> np.ndarray:
    if not panels:
        return np.full(universe.shape, np.nan, dtype=np.float64)
    zs = [
        _cs_winsor_z(np.asarray(panel, dtype=np.float64) * float(sign), universe)
        for panel, sign in zip(panels, signs)
    ]
    stacked = np.vstack(zs)
    count = np.sum(np.isfinite(stacked), axis=0)
    total = np.nansum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    score = np.full(universe.shape, np.nan, dtype=np.float64)
    ok = count > 0
    score[ok] = total[ok] / count[ok]
    return score


def _shrink_cov(sample: np.ndarray, n_obs: int) -> np.ndarray:
    s = np.asarray(sample, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        return s
    diag = np.diag(np.diag(s))
    alpha = float(min(0.85, max(0.2, s.shape[0] / max(int(n_obs), 1))))
    return (1.0 - alpha) * s + alpha * diag


def _rolling_nanmean(arr: np.ndarray, window: int, min_obs: int = 20) -> np.ndarray:
    """out[t] = nanmean(arr[t-window:t])，只用 t 之前的样本。"""
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("rolling nanmean 需要二维数组")
    n_dates, n_col = x.shape
    win = max(1, int(window))
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    c1 = np.cumsum(filled, axis=0)
    cnt = np.cumsum(valid.astype(np.float64), axis=0)
    pad = np.zeros((1, n_col), dtype=np.float64)
    c1p = np.vstack([pad, c1])
    n1 = np.vstack([pad, cnt])
    idx = np.arange(n_dates)
    lo = np.maximum(idx - win, 0)
    s1 = c1p[idx] - c1p[lo]
    n = n1[idx] - n1[lo]
    out = np.full((n_dates, n_col), np.nan, dtype=np.float64)
    ok = n >= float(min_obs)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = s1[ok] / n[ok]
    return out


def _style_harvest(
    x_style: np.ndarray, factor_ret: np.ndarray, window: int, horizon: int
) -> np.ndarray:
    """x_t' f̂_t · H；f̂ 只用 t 之前的风格收益。"""
    fhat = _rolling_nanmean(factor_ret, window, min_obs=20)
    xf = np.einsum("nsk,nk->ns", x_style, fhat)
    ok = np.all(np.isfinite(x_style), axis=2) & np.all(np.isfinite(fhat), axis=1)[:, None]
    return np.where(ok, xf * float(horizon), np.nan)


def _residualize_score(y: np.ndarray, x: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """等权 OLS：y = a + Xb + e，返回 e。X 为当日完整 X_T。样本不足则保持原分。"""
    out = np.asarray(y, dtype=np.float64).copy()
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    uni = np.asarray(universe, dtype=bool)
    ok = uni & np.isfinite(out) & np.all(np.isfinite(xx), axis=1)
    n = int(ok.sum())
    k = int(xx.shape[1])
    if n < max(k + 8, 40):
        return out
    raw = out[ok]
    mu = float(raw.mean())
    sd = float(raw.std(ddof=0))
    yy = raw
    if sd > 1e-12:
        yy = np.clip(raw, mu - WINSOR_SIGMA * sd, mu + WINSOR_SIGMA * sd)
    design = np.column_stack([np.ones(n, dtype=np.float64), xx[ok]])
    try:
        coef, *_ = np.linalg.lstsq(design, yy, rcond=None)
    except np.linalg.LinAlgError:
        return out
    fitted = design @ coef
    out[ok] = raw - fitted
    sst = float(np.dot(yy - yy.mean(), yy - yy.mean()))
    sse = float(np.dot(yy - fitted, yy - fitted))
    r2 = 1.0 - sse / sst if sst > 1e-18 else np.nan
    logger.info("lgbm_barra 中性化 R²=%.3f n=%s k=%s", r2, n, k)
    return out


def _mix_harvest(tree: np.ndarray, harvest: np.ndarray, universe: np.ndarray, weight: float) -> np.ndarray:
    """截面 z(树分) 与 z(x'f̂) 加权。缺收割时保持树分。"""
    w = float(np.clip(weight, 0.0, 1.0))
    out = np.asarray(tree, dtype=np.float64).copy()
    if w <= 0:
        return out
    uni = np.asarray(universe, dtype=bool)
    tz = cross_section_z(out, uni & np.isfinite(out))
    hz = cross_section_z(np.asarray(harvest, dtype=np.float64), uni & np.isfinite(harvest))
    both = np.isfinite(tz) & np.isfinite(hz)
    out[both] = (1.0 - w) * tz[both] + w * hz[both]
    only_h = np.isfinite(hz) & ~np.isfinite(tz)
    out[only_h] = hz[only_h]
    return out


def _rolling_nanstd(arr: np.ndarray, window: int, min_obs: int = 20) -> np.ndarray:
    """spec[t] = nanstd(arr[t-window:t])，只用 t 之前的样本。"""
    x = np.asarray(arr, dtype=np.float64)
    n_dates, n_stocks = x.shape
    win = max(1, int(window))
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    c1 = np.cumsum(filled, axis=0)
    c2 = np.cumsum(filled * filled, axis=0)
    cnt = np.cumsum(valid.astype(np.float64), axis=0)
    pad = np.zeros((1, n_stocks), dtype=np.float64)
    c1p = np.vstack([pad, c1])
    c2p = np.vstack([pad, c2])
    n1 = np.vstack([pad, cnt])
    idx = np.arange(n_dates)
    lo = np.maximum(idx - win, 0)
    s1 = c1p[idx] - c1p[lo]
    s2 = c2p[idx] - c2p[lo]
    n = n1[idx] - n1[lo]
    out = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    ok = n >= float(min_obs)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / np.maximum(n - 1.0, 1.0)
    out[ok] = np.sqrt(np.maximum(var[ok], 0.0))
    return out


def _frames(store, names: Sequence[str]) -> Dict[str, pd.DataFrame]:
    dates = pd.Index([str(d) for d in store.dates])
    symbols = pd.Index([str(s) for s in store.symbols])
    out: Dict[str, pd.DataFrame] = {}
    for name in names:
        try:
            arr = np.asarray(store.panel_for(name), dtype=np.float64)
        except KeyError:
            continue
        if arr.shape != (len(dates), len(symbols)):
            continue
        out[str(name)] = pd.DataFrame(arr, index=dates, columns=symbols)
    return out


def _build_book(
    store, horizon: int, vol_window: int, label: str, barra_lambda: float
) -> _BarraBook:
    raw = _frames(store, RAW_STYLE_NAMES)
    if SIZE_NAME not in raw:
        raise ValueError("lgbm_barra 缺少 style_size，无法组 Barra 暴露（需已入库 CNE5 风格因子）")
    dates = pd.Index([str(d) for d in store.dates])
    symbols = pd.Index([str(s) for s in store.symbols])
    mask = pd.DataFrame(np.asarray(store.mask, dtype=bool), index=dates, columns=symbols)
    industry = None
    labels = None
    try:
        industry, labels = load_l1_code_panel(MarketPanelLoader().storage, dates, symbols)
        if industry is None or industry.empty:
            industry, labels = None, None
            logger.warning("lgbm_barra 无申万一级，仅对风格回归")
    except Exception:
        logger.exception("lgbm_barra 加载行业失败，仅对风格回归")
        industry, labels = None, None

    logger.info("lgbm_barra 组装 Barra 暴露 %s 日 × %s 只", len(dates), len(symbols))
    expo_mat = ExposureEngine().build(
        raw, mask, industry_codes=industry, industry_labels=labels
    )
    names = [n for n in expo_mat.names if n in expo_mat.panels]
    cols = [
        np.asarray(expo_mat.panels[n].to_numpy(dtype=np.float64, copy=False))
        for n in names
    ]
    if not cols:
        raise ValueError("lgbm_barra 暴露矩阵为空")
    expo = np.stack(cols, axis=2)
    style_k = sum(1 for n in names if not str(n).startswith("ind_"))
    style_k = max(1, min(style_k, expo.shape[2]))
    weights = np.asarray(expo_mat.weights.to_numpy(dtype=np.float64), dtype=np.float64)
    close = np.asarray(store.panel_for("close"), dtype=np.float64)
    mask_arr = np.asarray(store.mask, dtype=bool)
    n_dates, n_stocks, n_k = expo.shape
    h = int(horizon)
    fwd = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    daily = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if h >= 1:
            c0 = close[: n_dates - h]
            c1 = close[h:]
            fwd[: n_dates - h] = np.where(
                np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0),
                c1 / c0 - 1.0,
                np.nan,
            )
        c0 = close[: n_dates - 1]
        c1 = close[1:]
        daily[: n_dates - 1] = np.where(
            np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0),
            c1 / c0 - 1.0,
            np.nan,
        )
    resid_1d = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    factor_ret = np.full((n_dates, style_k), np.nan, dtype=np.float64)
    floor = max(MIN_RESID_OBS, n_k + 6)
    for t in range(n_dates):
        if int(mask_arr[t].sum()) < floor or not np.isfinite(daily[t]).any():
            continue
        resid, beta = _wls_fit(daily[t], expo[t], weights[t], floor)
        resid_1d[t] = resid
        if beta.size > 1:
            take = min(style_k, beta.size - 1)
            factor_ret[t, :take] = beta[1 : 1 + take]

    win = max(20, int(vol_window))
    x_style = expo[:, :, :style_k]
    harvest = _style_harvest(x_style, factor_ret, win, h)
    if label == "style":
        logger.info("lgbm_barra 标签=风格收割 x'f̂·H window=%s", win)
        return _BarraBook(
            label=harvest, harvest=harvest, expo=x_style, xt=expo, weights=weights
        )
    if label == "resid":
        resid_h = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        for t in range(n_dates):
            if int(mask_arr[t].sum()) < floor or not np.isfinite(fwd[t]).any():
                continue
            resid_h[t], _ = _wls_fit(fwd[t], expo[t], weights[t], floor)
        logger.info("lgbm_barra 标签=持有期特异收益 residual(r, X_T) H=%s", h)
        return _BarraBook(
            label=resid_h, harvest=harvest, expo=x_style, xt=expo, weights=weights
        )
    if label == "raw":
        logger.info("lgbm_barra 标签=持有期收益 window=%s", win)
        return _BarraBook(
            label=fwd, harvest=harvest, expo=x_style, xt=expo, weights=weights
        )

    spec = _rolling_nanstd(resid_1d, win, min_obs=20)
    spec = np.where(np.isfinite(spec) & (spec >= VOL_FLOOR), spec, VOL_FLOOR)
    tot_var = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    for t in range(n_dates):
        a = max(0, t - win)
        block = factor_ret[a:t]
        okf = np.isfinite(block).all(axis=1)
        if int(okf.sum()) < 20:
            continue
        sample = np.cov(block[okf], rowvar=False)
        if np.ndim(sample) == 0:
            sample = np.array([[float(sample)]], dtype=np.float64)
        cov = _shrink_cov(sample, int(okf.sum()))
        x = x_style[t]
        finite = np.all(np.isfinite(x), axis=1)
        if not finite.any():
            continue
        xt = x[finite]
        sys = np.einsum("nk,kl,nl->n", xt, cov, xt)
        tot_var[t, finite] = np.maximum(sys, 0.0) + np.square(spec[t, finite])

    h_var = tot_var * float(h)
    if label == "sharpe":
        with np.errstate(invalid="ignore", divide="ignore"):
            label_arr = fwd / np.sqrt(np.maximum(h_var, VOL_FLOOR**2))
        label_arr = np.where(np.isfinite(label_arr), label_arr, fwd)
        logger.info("lgbm_barra 标签=持有期收益/Barra总风险 window=%s", win)
    else:
        label_arr = fwd - float(barra_lambda) * h_var
        label_arr = np.where(np.isfinite(h_var), label_arr, fwd)
        logger.info(
            "lgbm_barra 标签=效用 r-λσ² Barra λ=%s window=%s",
            barra_lambda,
            win,
        )
    return _BarraBook(
        label=label_arr, harvest=harvest, expo=x_style, xt=expo, weights=weights
    )


def barra_book(
    store, horizon: int, vol_window: int, label: str, barra_lambda: float
) -> _BarraBook:
    key = (int(horizon), int(vol_window), str(label), round(float(barra_lambda), 6))
    bag = getattr(store, "_lgbm_barra_books", None)
    if bag is None:
        bag = {}
        setattr(store, "_lgbm_barra_books", bag)
    book = bag.get(key)
    if book is None:
        book = _build_book(
            store, int(horizon), int(vol_window), str(label), float(barra_lambda)
        )
        bag[key] = book
    return book


def _feature_z(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    uni: np.ndarray,
    i: int,
    extra: Optional[np.ndarray] = None,
) -> np.ndarray:
    z_cols = []
    for panel, sign in zip(factor_hist, signs):
        raw = np.asarray(panel[i], dtype=np.float64) * float(sign)
        z_cols.append(_cs_winsor_z(raw, uni))
    if extra is not None:
        extra_arr = np.asarray(extra, dtype=np.float64)
        if extra_arr.ndim == 1:
            extra_arr = extra_arr[:, None]
        for j in range(int(extra_arr.shape[1])):
            z_cols.append(_cs_winsor_z(extra_arr[:, j], uni))
    if not z_cols:
        return np.empty((int(uni.shape[0]), 0), dtype=np.float64)
    return np.column_stack(z_cols)


def stack_barra_xy(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    label_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    seed: int = 42,
    expo_hist: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = _day_stride(horizon)
    rng = np.random.default_rng(int(seed) % (2**31))
    mask = np.asarray(mask_hist, dtype=bool)
    lab = np.asarray(label_hist, dtype=np.float64)
    expo = None if expo_hist is None else np.asarray(expo_hist, dtype=np.float64)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[int] = []
    last = int(lookback) - 1
    i = last
    extra_k = 0 if expo is None or expo.ndim < 3 else int(expo.shape[2])
    n_f = len(factor_hist) + extra_k
    while i >= 0:
        uni = mask[i]
        y = _cs_winsor_z(lab[i], uni)
        extra = None if expo is None else expo[i]
        z = _feature_z(factor_hist, signs, uni, i, extra)
        row_ok = uni & np.isfinite(y) & np.all(np.isfinite(z), axis=1)
        if int(row_ok.sum()) < 8:
            i -= step
            continue
        xx, yy = _cap_group(z[row_ok], y[row_ok], MAX_GROUP, rng)
        xs.append(xx)
        ys.append(yy)
        groups.append(int(yy.shape[0]))
        i -= step
    if not xs:
        return (
            np.empty((0, n_f), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    if len(xs) > MAX_GROUPS:
        xs = xs[:MAX_GROUPS]
        ys = ys[:MAX_GROUPS]
        groups = groups[:MAX_GROUPS]
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups, dtype=np.int32)


def combine_scores_lgbm_barra(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    label_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    min_rows: int = DEFAULT_MIN_ROWS,
    blend: float = DEFAULT_BLEND,
    expo_hist: Optional[np.ndarray] = None,
    model: Optional[object] = None,
) -> Tuple[np.ndarray, Optional[object]]:
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    eq = _equal_winsor_z(today_panels, signs, today_uni)
    empty = np.full(today_uni.shape, np.nan, dtype=np.float64)
    if not factor_hist:
        return empty, None
    fitted = model
    if fitted is None:
        need = int(lookback) + int(horizon)
        if any(np.asarray(p).shape[0] < need for p in factor_hist):
            return eq, None
        lgb = _try_lightgbm()
        if lgb is None:
            logger.warning("未安装 lightgbm，lgbm_barra 回退等权 z-score")
            return eq, None
        x, y, groups = stack_barra_xy(
            factor_hist,
            signs,
            label_hist,
            mask_hist,
            lookback=lookback,
            horizon=horizon,
            seed=42 + int(lookback) + int(horizon),
            expo_hist=expo_hist,
        )
        need_rows = max(int(min_rows), 80 * len(factor_hist))
        if groups.size < MIN_GROUPS or x.shape[0] < need_rows:
            return eq, None
        try:
            fitted = _fit_ranker(lgb, x, y, groups)
        except Exception:
            logger.exception("lgbm_barra 排序拟合失败，回退等权")
            return eq, None

    today_x = None if expo_hist is None else np.asarray(expo_hist, dtype=np.float64)[-1]
    z = _feature_z(factor_hist, signs, today_uni, -1, today_x)
    tree = np.full(today_uni.shape, np.nan, dtype=np.float64)
    ok = today_uni & np.all(np.isfinite(z), axis=1)
    if int(ok.sum()) == 0:
        return eq, fitted
    tree[ok] = np.asarray(fitted.predict(z[ok]), dtype=np.float64)
    tz = cross_section_z(tree, ok)
    b = float(np.clip(blend, 0.0, 1.0))
    both = np.isfinite(tz) & np.isfinite(eq)
    out = eq.copy()
    out[both] = (1.0 - b) * eq[both] + b * tz[both]
    only_tree = np.isfinite(tz) & ~np.isfinite(eq)
    out[only_tree] = tz[only_tree]
    return out, fitted


class LgbmBarraTopK(Strategy):
    name = "lgbm_barra"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("lgbm_barra 需要 --factor，逗号分隔，如 a,b,c")
        try:
            legs = parse_factor_list(args.factor)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        n = args.n if args.n is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or "20")
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = int(
            getattr(args, "refit", None)
            if getattr(args, "refit", None) is not None
            else DEFAULT_REFIT
        )
        blend = float(
            getattr(args, "blend", None) if getattr(args, "blend", None) is not None else DEFAULT_BLEND
        )
        smooth = float(
            getattr(args, "smooth", None) if getattr(args, "smooth", None) is not None else DEFAULT_SMOOTH
        )
        label = str(getattr(args, "label", None) or DEFAULT_LABEL).strip().lower()
        vol_window = int(getattr(args, "vol_window", None) or DEFAULT_VOL_WINDOW)
        barra_lambda = float(
            getattr(args, "barra_lambda", None)
            if getattr(args, "barra_lambda", None) is not None
            else DEFAULT_BARRA_LAMBDA
        )
        if lookback < 60:
            raise SystemExit("lgbm_barra --lookback 至少为 60（默认 252）")
        if horizon < 1:
            raise SystemExit("lgbm_barra --horizon 至少为 1（默认 20）")
        if refit < 1:
            raise SystemExit("lgbm_barra --refit 至少为 1（默认 20，同一模型沿用天数）")
        if not 0.0 <= blend <= 1.0:
            raise SystemExit("lgbm_barra --blend 须在 0 与 1 之间（树分权重）")
        if not 0.0 <= smooth < 1.0:
            raise SystemExit("lgbm_barra --smooth 须在 [0, 1)")
        harvest = float(
            getattr(args, "harvest", None)
            if getattr(args, "harvest", None) is not None
            else DEFAULT_HARVEST
        )
        if label not in LABELS:
            raise SystemExit("lgbm_barra --label 须是 resid、raw、util、sharpe 或 style")
        if vol_window < 20:
            raise SystemExit("lgbm_barra --vol-window 至少为 20")
        if barra_lambda < 0:
            raise SystemExit("lgbm_barra --barra-lambda 须 >= 0")
        if not 0.0 <= harvest <= 1.0:
            raise SystemExit("lgbm_barra --harvest 须在 0 与 1 之间（树外再叠收割的权重）")
        pos_style = bool(getattr(args, "pos_style", False))
        neutral = bool(getattr(args, "neutral", False))
        resid_x = bool(getattr(args, "resid_x", False))
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
            refit=refit,
            blend=blend,
            smooth=smooth,
            label=label,
            vol_window=vol_window,
            barra_lambda=barra_lambda,
            harvest=harvest,
            pos_style=pos_style,
            neutral=neutral,
            resid_x=resid_x,
        )

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("blend", "树分权重", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
            cli_field("label", "Barra标签 resid/raw/util/sharpe/style", "str", default=DEFAULT_LABEL),
            cli_field("vol_window", "风险估计窗口", "int", default=DEFAULT_VOL_WINDOW),
            cli_field("barra_lambda", "效用风险厌恶", "float", default=DEFAULT_BARRA_LAMBDA),
            cli_field("harvest", "树外再叠收割的权重，默认0", "float", default=DEFAULT_HARVEST),
            cli_field("pos_style", "只在正风格（x'f̂>0）里按树选", "bool", default=False),
            cli_field("neutral", "树分对X_T回归取残差再排序", "bool", default=False),
            cli_field("resid_x", "resid时把因子也对X_T残差化", "bool", default=False),
        ]

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        legs = parse_factor_list(args.factor)
        names = [name for name, _sign in legs]
        extra = [name for name in RAW_STYLE_NAMES if name not in names]
        return {"factors": names + extra}

    @classmethod
    def warmup(cls, args) -> int:
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        vol_window = int(getattr(args, "vol_window", None) or DEFAULT_VOL_WINDOW)
        return lookback + horizon + max(vol_window, 60) + 5

    @classmethod
    def run_tag(cls, args) -> str:
        legs = parse_factor_list(args.factor)
        parts = [("m" if s < 0 else "") + n for n, s in legs]
        tag = "lgbmbarra_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "20")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = DEFAULT_REFIT if getattr(args, "refit", None) is None else int(args.refit)
        blend = getattr(args, "blend", None)
        b = DEFAULT_BLEND if blend is None else float(blend)
        label = str(getattr(args, "label", None) or DEFAULT_LABEL).strip().lower()
        lam = getattr(args, "barra_lambda", None)
        lam_v = DEFAULT_BARRA_LAMBDA if lam is None else float(lam)
        hv = getattr(args, "harvest", None)
        hv_v = DEFAULT_HARVEST if hv is None else float(hv)
        pos = bool(getattr(args, "pos_style", False))
        neu = bool(getattr(args, "neutral", False))
        rx = bool(getattr(args, "resid_x", False))
        tag += (
            f"_lgb{lookback}h{horizon}rf{refit}rk{int(round(b * 100))}{label}"
            f"l{int(round(lam_v * 10))}xt1hv{int(round(hv_v * 100))}"
            f"{'xr' if rx else ''}"
            f"{'ps' if pos else ''}{'nt' if neu else ''}"
        )
        return tag

    def __init__(
        self,
        factors: Sequence[FactorLeg] | Sequence[str],
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = "20",
        lookback: int = DEFAULT_LOOKBACK,
        horizon: int = DEFAULT_HORIZON,
        refit: int = DEFAULT_REFIT,
        min_rows: int = DEFAULT_MIN_ROWS,
        blend: float = DEFAULT_BLEND,
        smooth: float = DEFAULT_SMOOTH,
        label: str = DEFAULT_LABEL,
        vol_window: int = DEFAULT_VOL_WINDOW,
        barra_lambda: float = DEFAULT_BARRA_LAMBDA,
        harvest: float = DEFAULT_HARVEST,
        pos_style: bool = False,
        neutral: bool = False,
        resid_x: bool = False,
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("lgbm_barra 至少需要一个因子")
        self.legs = legs
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self.lookback = max(60, int(lookback))
        self.horizon = max(1, int(horizon))
        self.refit = max(1, int(refit))
        self.min_rows = max(80, int(min_rows))
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.label = str(label or DEFAULT_LABEL).strip().lower()
        self.vol_window = max(20, int(vol_window))
        self.barra_lambda = max(0.0, float(barra_lambda))
        self.harvest = float(np.clip(harvest, 0.0, 1.0))
        self.pos_style = bool(pos_style)
        self.neutral = bool(neutral)
        self.resid_x = bool(resid_x)
        self._gate = RebalanceGate(self.rebalance)
        self.last_model: Optional[object] = None
        self._fitted_t: Optional[int] = None
        self._prev_score: Optional[np.ndarray] = None
        self._last_harvest: Optional[np.ndarray] = None

    @property
    def factor_names(self) -> List[str]:
        return [name for name, _ in self.legs]

    def _need_refit(self, t: int) -> bool:
        if self.last_model is None or self._fitted_t is None:
            return True
        return int(t) - int(self._fitted_t) >= int(self.refit)

    def _lgbm_score(self, ctx: DayContext) -> np.ndarray:
        need = self.lookback + self.horizon
        book = barra_book(
            ctx._store, self.horizon, self.vol_window, self.label, self.barra_lambda
        )
        t = int(ctx._t)
        mask = _window2(ctx._store.mask, t, need)
        label_hist = _window2(book.label, t, need)
        signs = [sign for _, sign in self.legs]
        if self.label == "resid" and self.resid_x:
            factor_hist = [
                _window2(residual_factor_panel(ctx._store, book, name), t, need)
                for name, _ in self.legs
            ]
        else:
            factor_hist = [ctx.history(name, need) for name, _ in self.legs]
        use_xt = self.label != "resid"
        if self.label == "resid" and (self.neutral or self.harvest > 0 or self.pos_style or self.resid_x):
            logger.warning(
                "lgbm_barra resid 是纯残差标签，--neutral/--harvest/--pos-style 会把风格加回组合"
            )
        expo_hist = _window2(book.expo, t, need) if use_xt else None
        reuse = None if self._need_refit(t) else self.last_model
        score, fitted = combine_scores_lgbm_barra(
            factor_hist,
            signs,
            label_hist,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
            min_rows=self.min_rows,
            blend=self.blend,
            expo_hist=expo_hist,
            model=reuse,
        )
        if self._need_refit(t):
            if fitted is not None:
                self.last_model = fitted
                self._fitted_t = t
            elif self.last_model is not None:
                score, _ = combine_scores_lgbm_barra(
                    factor_hist,
                    signs,
                    label_hist,
                    mask,
                    lookback=self.lookback,
                    horizon=self.horizon,
                    min_rows=self.min_rows,
                    blend=self.blend,
                    expo_hist=expo_hist,
                    model=self.last_model,
                )
        else:
            self.last_model = fitted if fitted is not None else self.last_model
        today_h = np.asarray(book.harvest[t], dtype=np.float64)
        self._last_harvest = today_h
        today = np.asarray(mask[-1], dtype=bool)
        score = _mix_harvest(score, today_h, today, self.harvest)
        if self.neutral:
            xt_today = np.asarray(book.xt[t], dtype=np.float64)
            score = _residualize_score(score, xt_today, today)
        if self._prev_score is not None and self._prev_score.shape == score.shape and self.smooth > 0:
            both = np.isfinite(score) & np.isfinite(self._prev_score)
            mixed = score.copy()
            mixed[both] = (1.0 - self.smooth) * score[both] + self.smooth * self._prev_score[both]
            score = mixed
        self._prev_score = np.asarray(score, dtype=np.float64)
        return score

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held

        values = self._lgbm_score(ctx)
        if self.pos_style and self._last_harvest is not None:
            hv = np.asarray(self._last_harvest, dtype=np.float64)
            if hv.shape == values.shape:
                values = np.where(np.isfinite(hv) & (hv > 0), values, np.nan)
        tradable = ctx.universe() & np.isfinite(values)
        ranked = np.where(tradable, values, -np.inf)
        k = int(min(self.n, int(tradable.sum())))
        if k <= 0:
            weights = np.zeros(len(ctx.symbols))
            return self._gate.remember(
                TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
            )
        pick = np.argpartition(ranked, -k)[-k:]
        pick = pick[tradable[pick]]
        weights = self.allocator.size(ctx, pick)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )
