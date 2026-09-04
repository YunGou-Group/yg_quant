#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格收益与因子归因：f_t 以及 β_k f_k。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..exposure_engine import (
    NLSIZE_NAME,
    STYLE_LABELS,
    ExposureEngine,
    ExposureMatrix,
    is_industry_col,
)
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..style_return_engine import StyleReturnEngine
from ..matrix_utils import nan_cumsum, rolling_nanmean

MIN_OBS_PARAM = ParamSpec(
    name="min_obs",
    type="int",
    label="每日最少有效股票数",
    default=20,
    min=3,
    max=500,
    scope="metric",
)

BETA_WINDOW_PARAM = ParamSpec(
    name="beta_window",
    type="int",
    label="暴露均线窗口（交易日）",
    default=20,
    min=2,
    max=120,
    scope="metric",
)


class AttributionMetric(BaseMetric):
    name = "attribution"
    dimension = "风格归因"
    description = (
        "同一张 X_T 上，用契约 B 远期收益估风格/行业收益 f_t，再与暴露 β 相乘做归因。"
        "行业只汇总成一条腿；看亏在风格、行业还是残差 alpha"
    )
    cost = "panel"
    produces = ("style_returns",)
    requires = ("style_betas",)
    engine_requires = ("style_exposures",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM, BETA_WINDOW_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        docs = [
            FieldDoc(
                "n_days",
                "样本",
                "有效天数",
                "风格收益或因子组合收益非空的交易日数",
            ),
            FieldDoc(
                "n_styles",
                "参数",
                "风格列数",
                "进入 r=Xf 回归的风格列数（含 NLSIZE，不含行业）",
            ),
            FieldDoc(
                "n_industries",
                "样本",
                "行业哑变量",
                "申万一级进入 r=Xf 的列数",
            ),
            FieldDoc(
                "cum_attr_industry",
                "风格归因",
                "累计行业归因",
                "所有行业 β×f 之和的累计；从残差里拆出来的行业腿",
            ),
            FieldDoc(
                "horizon",
                "参数",
                "持有期 N",
                "与本次评估共用的远期收益窗口；N>1 时 f_t 重叠",
            ),
            FieldDoc(
                "cum_factor",
                "风格归因",
                "累计因子收益",
                "因子组合收益 λ_t 之和；λ 是 r 对 z(因子) 的当日斜率",
            ),
            FieldDoc(
                "cum_explained",
                "风格归因",
                "累计风格归因",
                "Σ β_k f_k 之和，因子收益里能用风格解释的部分",
            ),
            FieldDoc(
                "cum_residual",
                "风格归因",
                "累计残差",
                "λ − Σβf 之和；持续为负说明纯化后也在亏，不只是风格逆风",
            ),
            FieldDoc(
                "residual_share",
                "风格归因",
                "残差占比",
                "累计残差 / 累计因子收益；接近 0 说明盈亏主要是风格，接近 1 是残差",
            ),
        ]
        for key, label in STYLE_LABELS.items():
            docs.append(
                FieldDoc(
                    f"mean_f_{key}",
                    "风格收益",
                    f"f {label}均值",
                    f"评估窗内{label}风格收益的平均；正=市场在奖这条腿",
                )
            )
            docs.append(
                FieldDoc(
                    f"ma_last_f_{key}",
                    "风格收益",
                    f"maf {label}末值",
                    f"{label}风格收益均线最新值；变号表示风格刚切换",
                )
            )
            docs.append(
                FieldDoc(
                    f"cum_attr_{key}",
                    "风格归因",
                    f"累计归因 {label}",
                    f"β_{label} × f_{label} 的累计；这个因子因{label}赚或亏了多少",
                )
            )
        return tuple(docs)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        x = batch_ctx.intermediates.get("x_t")
        names = [str(n) for n in (batch_ctx.intermediates.get("x_names") or [])]
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = int(f.shape[1])
        out: Dict[str, np.ndarray] = {
            "factor_return": np.full(p, np.nan, dtype=np.float64),
            "explained": np.full(p, np.nan, dtype=np.float64),
            "residual": np.full(p, np.nan, dtype=np.float64),
            "attr_industry": np.full(p, np.nan, dtype=np.float64),
        }
        for name in names:
            if _is_style_leg(name):
                out[f"f_{name}"] = np.full(p, np.nan, dtype=np.float64)
                out[f"attr_{name}"] = np.full(p, np.nan, dtype=np.float64)
        if x is None or r is None or not names:
            return out
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        beta, _resid = ExposureEngine.regress_matrix_day(f, x, min_obs=min_obs)
        engine = StyleReturnEngine(min_obs=min_obs)
        ft = engine.estimate_day(x, r, min_obs=min_obs)
        lam = engine.factor_return_day(f, r, min_obs=min_obs)
        out["factor_return"] = lam
        style_parts = []
        industry_parts = []
        for i, name in enumerate(names):
            if i >= ft.size or i >= beta.shape[0] or name == "intercept":
                continue
            contrib = beta[i] * ft[i]
            if is_industry_col(name):
                industry_parts.append(contrib)
            else:
                out[f"f_{name}"] = np.full(p, ft[i], dtype=np.float64)
                out[f"attr_{name}"] = contrib
                style_parts.append(contrib)
        all_parts = style_parts + industry_parts
        if all_parts:
            out["explained"] = _nansum_mincount(all_parts)
        if industry_parts:
            out["attr_industry"] = _nansum_mincount(industry_parts)
        out["residual"] = lam - out["explained"]
        return out

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        window = int(params.get("beta_window", 20))
        produced: Dict[str, Any] = {}
        for src, dest in (
            ("factor_return", "cum_factor"),
            ("explained", "cum_explained"),
            ("residual", "cum_residual"),
            ("attr_industry", "cum_attr_industry"),
        ):
            arr = arrays.get(src)
            if arr is not None:
                produced[dest] = nan_cumsum(np.asarray(arr, dtype=np.float64))
        for key, arr in arrays.items():
            if _leg_from_key(key, "attr_"):
                produced[f"cum_{key}"] = nan_cumsum(np.asarray(arr, dtype=np.float64))
            if _leg_from_key(key, "f_"):
                produced[f"ma_{key}"] = rolling_nanmean(np.asarray(arr, dtype=np.float64), window)
        return produced

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        exposures = ctx.intermediates.get("style_exposures")
        betas = ctx.intermediates.get("style_betas")
        if not isinstance(exposures, ExposureMatrix):
            raise ValueError("attribution 需要 style_exposures")
        if not isinstance(betas, pd.DataFrame):
            raise ValueError("attribution 需要 style_betas，请先运行 exposure")
        min_obs = int(params.get("min_obs", 20))
        window = int(params.get("beta_window", 20))
        engine = StyleReturnEngine(min_obs=min_obs)
        returns = ctx.masked_fwd_ret()
        factors = engine.estimate(
            returns, exposures, mask=ctx.universe_mask, min_obs=min_obs
        )
        lam = engine.factor_return(
            ctx.masked_factor(), returns, mask=ctx.universe_mask, min_obs=min_obs
        )
        ctx.intermediates["style_returns"] = factors
        aligned_beta = betas.reindex(index=factors.index)
        all_cols = [c for c in factors.columns if c != "intercept" and c in aligned_beta.columns]
        style_cols = [c for c in all_cols if not is_industry_col(c)]
        industry_cols = [c for c in all_cols if is_industry_col(c)]
        series: dict[str, pd.Series] = {
            "factor_return": lam.reindex(factors.index),
        }
        attr_parts = []
        industry_parts = []
        for col in all_cols:
            f_s = pd.to_numeric(factors[col], errors="coerce")
            b_s = pd.to_numeric(aligned_beta[col], errors="coerce")
            attr = b_s * f_s
            attr_parts.append(attr)
            rolled = f_s.rolling(window=window, min_periods=window).mean()
            if is_industry_col(col):
                industry_parts.append(attr)
            else:
                series[f"f_{col}"] = f_s
                series[f"ma_f_{col}"] = rolled
                series[f"attr_{col}"] = attr
                series[f"cum_attr_{col}"] = attr.cumsum()
        if attr_parts:
            explained = pd.concat(attr_parts, axis=1).sum(axis=1, min_count=1)
        else:
            explained = pd.Series(np.nan, index=factors.index)
        if industry_parts:
            industry_attr = pd.concat(industry_parts, axis=1).sum(axis=1, min_count=1)
        else:
            industry_attr = pd.Series(np.nan, index=factors.index)
        series["attr_industry"] = industry_attr
        series["cum_attr_industry"] = industry_attr.cumsum()
        residual = series["factor_return"] - explained
        series["explained"] = explained
        series["residual"] = residual
        series["cum_factor"] = series["factor_return"].cumsum()
        series["cum_explained"] = explained.cumsum()
        series["cum_residual"] = residual.cumsum()

        scalars: dict[str, Any] = {
            "n_days": int(factors.dropna(how="all").shape[0]),
            "n_styles": len(style_cols),
            "n_industries": len(industry_cols),
            "horizon": ctx.horizon,
            "beta_window": window,
        }
        for key, cum_name in (
            ("factor_return", "cum_factor"),
            ("explained", "cum_explained"),
            ("residual", "cum_residual"),
        ):
            clean = series[key].dropna()
            scalars[f"mean_{key}"] = float(clean.mean()) if len(clean) else float("nan")
            scalars[cum_name] = float(clean.sum()) if len(clean) else float("nan")
        cum_f = scalars["cum_factor"]
        cum_r = scalars["cum_residual"]
        if np.isfinite(cum_f) and abs(cum_f) > 1e-12:
            scalars["residual_share"] = float(cum_r / cum_f)
        else:
            scalars["residual_share"] = float("nan")

        for col in style_cols:
            f_clean = series[f"f_{col}"].dropna()
            ma_clean = series[f"ma_f_{col}"].dropna()
            attr_clean = series[f"attr_{col}"].dropna()
            scalars[f"mean_f_{col}"] = float(f_clean.mean()) if len(f_clean) else float("nan")
            scalars[f"ma_last_f_{col}"] = (
                float(ma_clean.iloc[-1]) if len(ma_clean) else float("nan")
            )
            scalars[f"mean_attr_{col}"] = (
                float(attr_clean.mean()) if len(attr_clean) else float("nan")
            )
            scalars[f"cum_attr_{col}"] = (
                float(attr_clean.sum()) if len(attr_clean) else float("nan")
            )
        ind_clean = industry_attr.dropna()
        scalars["mean_attr_industry"] = (
            float(ind_clean.mean()) if len(ind_clean) else float("nan")
        )
        scalars["cum_attr_industry"] = (
            float(ind_clean.sum()) if len(ind_clean) else float("nan")
        )
        return MetricResult(scalars=scalars, series=series)


def _is_style_leg(name: str) -> bool:
    return bool(name) and name != "intercept" and not is_industry_col(name)


def _leg_from_key(key: str, prefix: str) -> Optional[str]:
    if not key.startswith(prefix):
        return None
    name = key[len(prefix):]
    if not _is_style_leg(name):
        return None
    if name in STYLE_LABELS or name.startswith("style_") or name == NLSIZE_NAME:
        return name
    return None


def _nansum_mincount(parts: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([np.asarray(item, dtype=np.float64) for item in parts])
    total = np.nansum(stacked, axis=0)
    count = np.isfinite(stacked).sum(axis=0)
    return np.where(count > 0, total, np.nan)
