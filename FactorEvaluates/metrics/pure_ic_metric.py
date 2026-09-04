#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯化 RankIC：先对 ExposureEngine 的 X_T 回归，再对残差做截面 Spearman。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..cross_section_ic_calculator import CrossSectionICCalculator
from ..context import EvalContext
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


class PureICMetric(BaseMetric):
    name = "pure_ic"
    dimension = "预测力"
    description = (
        "纯化 RankIC：因子先做截面 z-score，再对当天 X_T（风格 + NLSIZE + 申万一级）回归，"
        "残差与远期收益做 Spearman。用来看剥掉市值/动量和行业之后还有没有 alpha"
    )
    cost = "panel"
    produces = ("daily_pure_rank_ic",)
    requires = ("factor_pure",)
    engine_requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "预测力", "纯化IC均值", "残差因子的日度 RankIC 平均值"),
            FieldDoc("std", "预测力", "纯化IC标准差", "纯化 RankIC 的波动"),
            FieldDoc(
                "icir",
                "预测力",
                "纯化ICIR",
                "mean/std；和原始 ICIR 对比，可看预测力有多少来自风格",
            ),
            FieldDoc("positive_ratio", "稳定性", "纯化IC正值比例", "纯化 RankIC>0 的交易日占比"),
            FieldDoc("n_days", "样本", "有效天数", "纯化 RankIC 非空的交易日数"),
            FieldDoc("n_styles", "参数", "风格列数", "进入回归的 X_T 列数（含 NLSIZE）"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        pure = ctx.intermediates.get("factor_pure")
        if not isinstance(pure, pd.DataFrame):
            raise ValueError("pure_ic 需要中间量 factor_pure，请先运行 exposure")
        min_obs = int(params.get("min_obs", 20))
        calculator = CrossSectionICCalculator()
        series = calculator.daily_rank_ic(pure, ctx.masked_fwd_ret(), min_obs=min_obs)
        ctx.intermediates["daily_pure_rank_ic"] = series
        summary = calculator.summary(series)
        styles = ctx.intermediates.get("style_betas")
        n_styles = 0
        if isinstance(styles, pd.DataFrame):
            n_styles = max(0, len(styles.columns) - (1 if "intercept" in styles.columns else 0))
        return MetricResult(
            scalars={
                "mean": summary["mean"],
                "std": summary["std"],
                "icir": summary["icir"],
                "positive_ratio": summary["positive_ratio"],
                "n_days": summary["n_days"],
                "n_styles": n_styles,
                "horizon": ctx.horizon,
            },
            series={"daily_pure_rank_ic": series, "cumsum": series.cumsum()},
        )

    def compute_matrix(self, batch_ctx, params):
        from ..matrix_utils import spearman_pairwise

        pure = batch_ctx.intermediates.get("factor_pure")
        r = batch_ctx.masked_returns()
        p = batch_ctx.masked_factors().shape[1]
        if pure is None or r is None:
            return {"pure_ic": np.full(p, np.nan)}
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        return {"pure_ic": spearman_pairwise(pure, r, min_obs=min_obs)}
