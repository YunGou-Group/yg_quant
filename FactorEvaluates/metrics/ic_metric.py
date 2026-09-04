#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截面 Pearson IC 与 RankIC。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ..base_metric import BaseMetric
from ..cross_section_ic_calculator import CrossSectionICCalculator
from ..context import BatchEvalContext, EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import pearson_pairwise, spearman_pairwise, summarize_daily

MIN_OBS_PARAM = ParamSpec(
    name="min_obs",
    type="int",
    label="每日最少有效股票数",
    default=20,
    min=3,
    max=500,
    scope="metric",
)


class ICMetric(BaseMetric):
    name = "ic"
    dimension = "预测力"
    description = "日度 Pearson IC：因子与远期收益的截面线性相关"
    cost = "panel"
    produces = ("daily_ic",)
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "预测力", "IC均值", "区间内日度 Pearson IC 的平均值"),
            FieldDoc("std", "预测力", "IC标准差", "日度 Pearson IC 的波动"),
            FieldDoc("positive_ratio", "稳定性", "IC正值比例", "区间内 Pearson IC>0 的交易日占比"),
            FieldDoc("n_days", "样本", "有效天数", "IC 非空的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        if r is None:
            return {"ic": np_nan(f)}
        return {"ic": pearson_pairwise(f, r, min_obs=min_obs)}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        min_obs = int(params.get("min_obs", 20))
        calculator = CrossSectionICCalculator()
        series = calculator.daily_pearson_ic(
            ctx.masked_factor(), ctx.masked_fwd_ret(), min_obs=min_obs
        )
        ctx.intermediates["daily_ic"] = series
        summary = summarize_daily(series)
        return MetricResult(
            scalars={
                "mean": summary["mean"],
                "std": summary["std"],
                "positive_ratio": summary["positive_ratio"],
                "n_days": summary["n_days"],
                "horizon": ctx.horizon,
            },
            series={"daily_ic": series, "cumsum": series.cumsum()},
        )


class RankICMetric(BaseMetric):
    name = "rank_ic"
    dimension = "预测力"
    description = "日度 RankIC（Spearman）：因子与远期收益的截面秩相关"
    cost = "panel"
    produces = ("daily_rank_ic",)
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "预测力", "RankIC均值", "区间内日度 RankIC 的平均值，衡量平均预测方向与强度"),
            FieldDoc("std", "预测力", "RankIC标准差", "日度 RankIC 的波动；越大说明预测力越不稳定"),
            FieldDoc(
                "positive_ratio",
                "稳定性",
                "RankIC正值比例",
                "区间内 RankIC>0 的交易日占比，衡量方向是否稳定",
            ),
            FieldDoc("n_days", "样本", "有效天数", "RankIC 非空的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        if r is None:
            return {"rank_ic": np_nan(f)}
        return {"rank_ic": spearman_pairwise(f, r, min_obs=min_obs)}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        min_obs = int(params.get("min_obs", 20))
        calculator = CrossSectionICCalculator()
        series = calculator.daily_rank_ic(
            ctx.masked_factor(), ctx.masked_fwd_ret(), min_obs=min_obs
        )
        ctx.intermediates["daily_rank_ic"] = series
        summary = summarize_daily(series)
        return MetricResult(
            scalars={
                "mean": summary["mean"],
                "std": summary["std"],
                "positive_ratio": summary["positive_ratio"],
                "n_days": summary["n_days"],
                "horizon": ctx.horizon,
            },
            series={"daily_rank_ic": series, "cumsum": series.cumsum()},
        )


def np_nan(factors):
    import numpy as np

    p = 1 if factors.ndim == 1 else factors.shape[1]
    return np.full(p, np.nan, dtype=np.float64)
