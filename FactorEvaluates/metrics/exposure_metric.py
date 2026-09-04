#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子对 CNE5-lite 风格暴露的逐日截面回归系数 β_t。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..exposure_engine import (
    STYLE_LABELS,
    ExposureEngine,
    ExposureMatrix,
    is_industry_col,
)
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec

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

class ExposureMetric(BaseMetric):
    name = "exposure"
    dimension = "风格暴露"
    description = (
        "因子当日截面 z-score 对 X_T（风格 + NLSIZE + 申万一级哑变量）回归的 β_t。"
        "β 是因子 1σ 对风格 1σ，跨因子可比；行业只汇总，不逐个展开"
    )
    cost = "panel"
    produces = ("factor_pure", "style_betas")
    requires = ()
    engine_requires = ("style_exposures",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM, BETA_WINDOW_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        docs = [
            FieldDoc(
                "n_days",
                "样本",
                "有效天数",
                "回归系数非空的交易日数",
            ),
            FieldDoc(
                "n_styles",
                "参数",
                "风格列数",
                "进入回归的风格列数（含 NLSIZE，不含行业）",
            ),
            FieldDoc(
                "n_industries",
                "样本",
                "行业哑变量",
                "申万一级进入回归的列数；丢掉出现最多的一个作参照",
            ),
            FieldDoc(
                "industry_ref_name",
                "样本",
                "行业参照",
                "被丢掉的参照行业，行业系数都相对它",
            ),
            FieldDoc(
                "ma_abs_industry",
                "风格暴露",
                "|maβ| 行业",
                "各行业 |maβ| 的平均，衡量行业腿总体有多厚",
            ),
            FieldDoc(
                "top_industry",
                "风格暴露",
                "主行业",
                "|maβ| 最大的那个申万一级",
            ),
            FieldDoc(
                "ma_abs_top_industry",
                "风格暴露",
                "|maβ| 主行业",
                "主行业均线绝对值，和风格 |maβ| 同一量纲",
            ),
            FieldDoc(
                "beta_window",
                "参数",
                "均线窗口",
                "对逐日截面 β 做滚动均值的交易日数；回归本身仍是每天单独估",
            ),
        ]
        for key, label in STYLE_LABELS.items():
            docs.append(
                FieldDoc(
                    f"mean_abs_{key}",
                    "风格暴露",
                    f"|β| {label}",
                    f"标准化后日度 |β| 平均；约 0.3 表示主要贴{label}，跨因子可比",
                )
            )
            docs.append(
                FieldDoc(
                    f"ma_abs_{key}",
                    "风格暴露",
                    f"|maβ| {label}",
                    f"标准化 β 均线绝对值的平均，比日频 |β| 更稳，用来排主风险腿",
                )
            )
            docs.append(
                FieldDoc(
                    f"ma_last_{key}",
                    "风格暴露",
                    f"maβ {label}末值",
                    f"均线最新值；看符号判断是否还偏{label}，变号才谈翻转",
                )
            )
        return tuple(docs)

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        exposures = ctx.intermediates.get("style_exposures")
        if not isinstance(exposures, ExposureMatrix):
            raise ValueError("exposure 需要 style_exposures，请由 ExposureEngine 在评估前组 X_T")
        min_obs = int(params.get("min_obs", 20))
        window = int(params.get("beta_window", 20))
        resid, betas = ExposureEngine(min_obs=min_obs).regress(
            ctx.masked_factor(),
            exposures,
            mask=ctx.universe_mask,
            min_obs=min_obs,
        )
        ctx.intermediates["factor_pure"] = resid
        ctx.intermediates["style_betas"] = betas
        series = {}
        style_names = [name for name in exposures.names if not is_industry_col(name)]
        industry_names = list(exposures.industry_names) or [
            name for name in exposures.names if is_industry_col(name)
        ]
        scalars: dict[str, Any] = {
            "n_days": int(betas.dropna(how="all").shape[0]),
            "n_styles": len(style_names),
            "n_industries": len(industry_names),
            "horizon": ctx.horizon,
            "beta_window": window,
        }
        if exposures.industry_ref:
            scalars["industry_ref"] = exposures.industry_ref
            scalars["industry_ref_name"] = exposures.industry_labels.get(
                exposures.industry_ref,
                exposures.industry_labels.get(
                    f"ind_{exposures.industry_ref}", exposures.industry_ref
                ),
            )
        for col in betas.columns:
            if col == "intercept":
                continue
            values = pd.to_numeric(betas[col], errors="coerce")
            rolled = values.rolling(window=window, min_periods=window).mean()
            if not is_industry_col(col):
                series[f"beta_{col}"] = values
                series[f"ma_beta_{col}"] = rolled
            clean = values.dropna()
            ma_clean = rolled.dropna()
            if clean.empty:
                scalars[f"mean_{col}"] = float("nan")
                scalars[f"mean_abs_{col}"] = float("nan")
            else:
                scalars[f"mean_{col}"] = float(clean.mean())
                scalars[f"mean_abs_{col}"] = float(clean.abs().mean())
            if ma_clean.empty:
                scalars[f"ma_abs_{col}"] = float("nan")
                scalars[f"ma_last_{col}"] = float("nan")
            else:
                scalars[f"ma_abs_{col}"] = float(ma_clean.abs().mean())
                scalars[f"ma_last_{col}"] = float(ma_clean.iloc[-1])
        ma_abs_ind = []
        top_key = None
        top_abs = float("-inf")
        for col in industry_names:
            value = scalars.get(f"ma_abs_{col}")
            if value is None or not pd.notna(value):
                continue
            ma_abs_ind.append(float(value))
            if float(value) > top_abs:
                top_abs = float(value)
                top_key = col
        if ma_abs_ind:
            scalars["ma_abs_industry"] = float(sum(ma_abs_ind) / len(ma_abs_ind))
        if top_key is not None:
            scalars["top_industry"] = exposures.industry_labels.get(top_key, top_key)
            scalars["top_industry_key"] = top_key
            scalars["ma_abs_top_industry"] = top_abs
        return MetricResult(scalars=scalars, series=series)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> dict:
        x = batch_ctx.intermediates.get("x_t")
        names = batch_ctx.intermediates.get("x_names") or []
        f = batch_ctx.masked_factors()
        p = f.shape[1]
        out = {}
        if x is None or not len(names):
            return out
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        beta, resid = ExposureEngine.regress_matrix_day(f, x, min_obs=min_obs)
        batch_ctx.intermediates["factor_pure"] = resid
        for i, name in enumerate(names):
            if i >= beta.shape[0]:
                break
            out[f"beta_{name}"] = beta[i]
        return out
