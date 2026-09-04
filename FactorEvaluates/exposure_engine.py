#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险暴露引擎：在当前股票池 mask 上组 X_T。

原料是 Bin 里的 9 个 style_* 原始描述子。NLSIZE 不落盘，这里用当天
z(Size) 的三次项对 Size 残差化得到。申万一级哑变量同样 mask 后现铺，
丢掉出现最多的一个行业做参照，避免和截距共线。

不做收益、不 shift open。截面 z-score / 残差都依赖传入的 mask。
回归因子前先在当日样本上等权标准化，β 与因子量纲无关。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("FactorEvaluates")

RAW_STYLE_NAMES: Tuple[str, ...] = (
    "style_size",
    "style_beta",
    "style_momentum",
    "style_resvol",
    "style_liquidity",
    "style_btop",
    "style_ey",
    "style_growth",
    "style_leverage",
)
SIZE_NAME = "style_size"
NLSIZE_NAME = "nlsize"
INDUSTRY_PREFIX = "ind_"
STYLE_LABELS: Dict[str, str] = {
    "style_size": "Size",
    "nlsize": "NLSIZE",
    "style_beta": "Beta",
    "style_momentum": "Momentum",
    "style_resvol": "ResVol",
    "style_liquidity": "Liquidity",
    "style_btop": "BTOP",
    "style_ey": "EY",
    "style_growth": "Growth",
    "style_leverage": "Leverage",
}


def is_industry_col(name: str) -> bool:
    return str(name).startswith(INDUSTRY_PREFIX)


def industry_col(code: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(code))
    return f"{INDUSTRY_PREFIX}{safe}"


def _industry_dummy_panels(
    codes: pd.DataFrame,
    mask_arr: np.ndarray,
    labels: Mapping[str, str],
) -> Tuple[Dict[str, pd.DataFrame], Tuple[str, ...], Dict[str, str], Optional[str]]:
    """一次 factorize，再按整数编号铺哑变量。避免对整表做 30 次 pandas 比较。"""
    flat = pd.Series(codes.to_numpy().ravel(), dtype=object)
    flat = flat.mask(~mask_arr.ravel())
    ids, uniques = pd.factorize(flat, use_na_sentinel=True)
    if uniques.size == 0:
        return {}, (), dict(labels), None
    ids = np.asarray(ids, dtype=np.int32).reshape(codes.shape)
    valid = ids >= 0
    if not valid.any():
        return {}, (), dict(labels), None
    counts = np.bincount(ids[valid], minlength=int(uniques.size))
    ref_id = int(np.argmax(counts))
    ref_code = str(uniques[ref_id])
    label_map = {str(k): str(v) for k, v in dict(labels).items()}
    panels: Dict[str, pd.DataFrame] = {}
    names: list[str] = []
    order = sorted(range(int(uniques.size)), key=lambda i: str(uniques[i]))
    for idx in order:
        code = str(uniques[idx])
        if idx == ref_id:
            continue
        dummy = np.where(valid, (ids == idx).astype(np.float64), np.nan)
        col = industry_col(code)
        panels[col] = pd.DataFrame(dummy, index=codes.index, columns=codes.columns)
        names.append(col)
        label_map.setdefault(col, label_map.get(code, code))
    label_map.setdefault(industry_col(ref_code), label_map.get(ref_code, ref_code))
    label_map.setdefault(ref_code, label_map.get(ref_code, ref_code))
    return panels, tuple(names), label_map, ref_code


def _align(
    frame: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out.columns = [str(col) for col in out.columns]
    return out.reindex(index=index, columns=columns)


def _finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def _winsorize_rows(arr: np.ndarray, p: float) -> np.ndarray:
    out = arr.copy()
    valid_n = np.isfinite(out).sum(axis=1)
    ok = valid_n >= 10
    if not ok.any():
        return out
    lo = np.nanquantile(out[ok], p, axis=1)
    hi = np.nanquantile(out[ok], 1.0 - p, axis=1)
    clipped = np.clip(out[ok], lo[:, None], hi[:, None])
    out[ok] = clipped
    return out


def _row_weights(values: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    finite = _finite_mask(values)
    if weights is None:
        w = finite.astype(np.float64)
    else:
        w = np.where(finite & np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = w.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, np.nan)
    return w / total


def _zscore_sample(values: np.ndarray) -> Optional[np.ndarray]:
    """等权 z-score；方差过小则无法标准化，返回 None。"""
    mu = float(np.mean(values))
    sd = float(np.std(values))
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    return (values - mu) / sd


def _weighted_zscore(arr: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    w = _row_weights(arr, weights)
    mu = np.nansum(np.where(np.isfinite(arr), arr * w, 0.0), axis=1, keepdims=True)
    centered = np.where(np.isfinite(arr), arr - mu, np.nan)
    var = np.nansum(np.where(np.isfinite(centered), (centered ** 2) * w, 0.0), axis=1, keepdims=True)
    sd = np.sqrt(var)
    sd = np.where(sd > 1e-12, sd, np.nan)
    return centered / sd


def _residualize_on_one(
    y: np.ndarray, x: np.ndarray, weights: Optional[np.ndarray]
) -> np.ndarray:
    """Row-wise y = a + b x + e, equal or cap weights. Returns e."""
    valid = _finite_mask(y) & _finite_mask(x)
    work_y = np.where(valid, y, np.nan)
    work_x = np.where(valid, x, np.nan)
    w = _row_weights(work_x, None if weights is None else np.where(valid, weights, np.nan))
    mx = np.nansum(np.where(np.isfinite(work_x), work_x * w, 0.0), axis=1, keepdims=True)
    my = np.nansum(np.where(np.isfinite(work_y), work_y * w, 0.0), axis=1, keepdims=True)
    xc = np.where(np.isfinite(work_x), work_x - mx, np.nan)
    yc = np.where(np.isfinite(work_y), work_y - my, np.nan)
    cov = np.nansum(np.where(np.isfinite(xc) & np.isfinite(yc), xc * yc * w, 0.0), axis=1, keepdims=True)
    var = np.nansum(np.where(np.isfinite(xc), (xc ** 2) * w, 0.0), axis=1, keepdims=True)
    b = np.divide(cov, var, out=np.full_like(cov, np.nan), where=var > 1e-18)
    a = my - b * mx
    return y - a - b * x


@dataclass
class ExposureMatrix:
    """date × symbol panels; ``names`` is OLS 列顺序（含 nlsize）。"""

    panels: Dict[str, pd.DataFrame]
    names: Tuple[str, ...]
    weighted: bool
    winsor_p: float
    weights: Optional[pd.DataFrame] = None
    industry_names: Tuple[str, ...] = ()
    industry_labels: Dict[str, str] = field(default_factory=dict)
    industry_ref: Optional[str] = None

    def align(self, index: pd.Index, columns: pd.Index) -> "ExposureMatrix":
        panels = {
            name: _align(frame, index, columns) for name, frame in self.panels.items()
        }
        weight_frame = (
            _align(self.weights, index, columns) if self.weights is not None else None
        )
        return ExposureMatrix(
            panels=panels,
            names=self.names,
            weighted=self.weighted,
            winsor_p=self.winsor_p,
            weights=weight_frame,
            industry_names=self.industry_names,
            industry_labels=dict(self.industry_labels),
            industry_ref=self.industry_ref,
        )


class ExposureEngine:
    def __init__(self, winsor_p: float = 0.05, min_obs: int = 20):
        if not 0.0 < float(winsor_p) < 0.5:
            raise ValueError(f"winsor_p 应在 (0, 0.5)，收到 {winsor_p!r}")
        self.winsor_p = float(winsor_p)
        self.min_obs = int(min_obs)

    def build(
        self,
        raw_styles: Mapping[str, pd.DataFrame],
        mask: pd.DataFrame,
        weights: Optional[pd.DataFrame] = None,
        industry_codes: Optional[pd.DataFrame] = None,
        industry_labels: Optional[Mapping[str, str]] = None,
    ) -> ExposureMatrix:
        if SIZE_NAME not in raw_styles:
            raise ValueError("ExposureEngine 需要 style_size 才能计算 NLSIZE")
        mask_n = mask.copy()
        mask_n.index = pd.to_datetime(mask_n.index).strftime("%Y-%m-%d")
        mask_n.columns = [str(c) for c in mask_n.columns]
        dates = mask_n.index
        symbols = mask_n.columns
        mask_arr = mask_n.fillna(False).astype(bool).to_numpy()

        raw_size = _align(raw_styles[SIZE_NAME], dates, symbols)
        if weights is None:
            weight_arr = np.exp(raw_size.to_numpy(dtype=np.float64, copy=False))
            weighted = True
        else:
            weight_arr = _align(weights, dates, symbols).to_numpy(dtype=np.float64)
            weighted = True
        weight_arr = np.where(mask_arr, weight_arr, np.nan)

        z_panels: Dict[str, pd.DataFrame] = {}
        ordered: list[str] = []
        for name in RAW_STYLE_NAMES:
            frame = raw_styles.get(name)
            if frame is None or frame.empty:
                continue
            arr = _align(frame, dates, symbols).to_numpy(dtype=np.float64)
            arr = np.where(mask_arr, arr, np.nan)
            arr = _winsorize_rows(arr, self.winsor_p)
            z = _weighted_zscore(arr, weight_arr)
            z_panels[name] = pd.DataFrame(z, index=dates, columns=symbols)
            ordered.append(name)

        size_z = z_panels[SIZE_NAME].to_numpy(dtype=np.float64, copy=False)
        cubed = size_z ** 3
        nlsize = _residualize_on_one(cubed, size_z, weight_arr)
        nlsize = np.where(mask_arr, nlsize, np.nan)
        nlsize = _weighted_zscore(nlsize, weight_arr)
        z_panels[NLSIZE_NAME] = pd.DataFrame(nlsize, index=dates, columns=symbols)

        names: list[str] = []
        for name in ordered:
            names.append(name)
            if name == SIZE_NAME:
                names.append(NLSIZE_NAME)

        industry_names: Tuple[str, ...] = ()
        label_map = {str(k): str(v) for k, v in dict(industry_labels or {}).items()}
        ref_code: Optional[str] = None
        if industry_codes is not None and not industry_codes.empty:
            code_panel = _align(industry_codes.astype(object), dates, symbols)
            dummies, industry_names, label_map, ref_code = _industry_dummy_panels(
                code_panel, mask_arr, label_map
            )
            z_panels.update(dummies)
            names.extend(industry_names)
            if industry_names:
                logger.info(
                    "X_T 加入 %s 个行业哑变量，参照=%s",
                    len(industry_names),
                    label_map.get(ref_code, ref_code),
                )

        return ExposureMatrix(
            panels=z_panels,
            names=tuple(names),
            weighted=weighted,
            winsor_p=self.winsor_p,
            weights=pd.DataFrame(weight_arr, index=dates, columns=symbols),
            industry_names=tuple(industry_names),
            industry_labels=label_map,
            industry_ref=ref_code,
        )

    def residualize(
        self,
        factor: pd.DataFrame,
        exposures: ExposureMatrix,
        mask: Optional[pd.DataFrame] = None,
        min_obs: Optional[int] = None,
    ) -> pd.DataFrame:
        resid, _ = self.regress(factor, exposures, mask=mask, min_obs=min_obs)
        return resid

    def regress(
        self,
        factor: pd.DataFrame,
        exposures: ExposureMatrix,
        mask: Optional[pd.DataFrame] = None,
        min_obs: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """截面 OLS：z(F) = a + Xβ + e。

        每天在进入回归的股票上对因子做等权 z-score，再对已标准化的 X_T 回归。
        β 是「因子 1σ / 风格 1σ」，跨因子可比；残差也在 z 空间，供纯化 RankIC。
        """
        factor_n = factor.copy()
        factor_n.index = pd.to_datetime(factor_n.index).strftime("%Y-%m-%d")
        factor_n.columns = [str(c) for c in factor_n.columns]
        dates = factor_n.index
        symbols = factor_n.columns
        aligned = exposures.align(dates, symbols)
        y = factor_n.to_numpy(dtype=np.float64)
        if mask is None:
            m = np.isfinite(y)
        else:
            m = _align(mask.astype(float), dates, symbols).fillna(0).to_numpy() > 0
            m = m & np.isfinite(y)
        style_names = [name for name in aligned.names if name in aligned.panels]
        cols = [aligned.panels[name].to_numpy(dtype=np.float64) for name in style_names]
        beta_names = ["intercept", *style_names]
        beta_mat = np.full((y.shape[0], len(beta_names)), np.nan, dtype=np.float64)
        out = np.full(y.shape, np.nan, dtype=np.float64)
        if not cols:
            return factor_n, pd.DataFrame(beta_mat, index=dates, columns=beta_names)
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
            yt = _zscore_sample(y[t, valid])
            if yt is None:
                continue
            beta, *_ = np.linalg.lstsq(x, yt, rcond=None)
            beta_mat[t, : beta.size] = beta
            resid = np.full(y.shape[1], np.nan, dtype=np.float64)
            resid[valid] = yt - x @ beta
            out[t] = resid
        resid_df = pd.DataFrame(out, index=dates, columns=symbols)
        beta_df = pd.DataFrame(beta_mat, index=dates, columns=beta_names)
        return resid_df, beta_df

    @staticmethod
    def regress_matrix_day(
        y: np.ndarray,
        x: np.ndarray,
        min_obs: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """一天多因子 OLS。y (n_stocks, n_factors)，x (n_stocks, k) 已含有效行的设计阵。
        返回 beta (k, n_factors) 与残差 (n_stocks, n_factors)，无效行为 NaN。"""
        n, p = y.shape
        k = x.shape[1]
        beta = np.full((k, p), np.nan, dtype=np.float64)
        resid = np.full((n, p), np.nan, dtype=np.float64)
        x_ok = np.isfinite(x).all(axis=1)
        if not x_ok.any():
            return beta, resid

        # 每列各自的有效行。把缺失值填 0 再回归，会让缺失股票贡献一个
        # 「正好等于均值」的假观测，并产出 -fitted 的假残差。
        y_ok = np.isfinite(y) & x_ok[:, None]
        counts = y_ok.sum(axis=0)

        cnt = np.maximum(counts, 1)
        mu = np.where(y_ok, y, 0.0).sum(axis=0) / cnt
        dev = np.where(y_ok, y - mu, 0.0)
        sd = np.sqrt((dev * dev).sum(axis=0) / cnt)

        usable = (counts >= max(min_obs, k + 5)) & (sd > 1e-12)
        if not usable.any():
            return beta, resid
        yz = np.where(y_ok, dev / np.where(sd > 1e-12, sd, 1.0), np.nan)

        # 相同缺失模式的列合并成一次 lstsq；最坏退化成每列一次，规模也很小。
        groups: Dict[bytes, List[int]] = {}
        for j in np.flatnonzero(usable):
            groups.setdefault(y_ok[:, j].tobytes(), []).append(int(j))
        for key, idx in groups.items():
            rows = np.frombuffer(key, dtype=bool)
            xv = x[rows]
            yg = yz[np.ix_(rows, idx)]
            coef, *_ = np.linalg.lstsq(xv, yg, rcond=None)
            beta[:, idx] = coef
            resid[np.ix_(rows, idx)] = yg - xv @ coef
        return beta, resid
