#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20 日持有期 IC：对 open[T+1+20]/open[T+1]-1，不是 ic_decay 那种把 5 日标签往后挪。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..cross_section_ic_calculator import CrossSectionICCalculator
from ..field_doc import FieldDoc
from ..matrix_utils import expanding_icir, pearson_pairwise, rank_ic_pairwise, summarize_daily
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec

HOLD_HORIZON = 20
FWD_HOLD_KEY = "fwd_20"

MIN_OBS_PARAM = ParamSpec(
    name="min_obs",
    type="int",
    label="每日最少有效股票数",
    default=10,
    min=3,
    max=500,
    scope="metric",
)

HOLD_HORIZON_PARAM = ParamSpec(
    name="hold_horizon",
    type="int",
    label="持有交易日（20 日频）",
    default=HOLD_HORIZON,
    min=2,
    max=60,
    scope="metric",
)


def hold_returns(batch_ctx: BatchEvalContext) -> Optional[np.ndarray]:
    extra = batch_ctx.intermediates.get(FWD_HOLD_KEY)
    if extra is None:
        return None
    return np.asarray(extra, dtype=np.float64)


class IC20Metric(BaseMetric):
    name = "ic_20"
    dimension = "预测力"
    description = "日度 Pearson IC，标签为 20 日远期收益 open[T+1+20]/open[T+1]-1，供 20 日调仓用"
    cost = "panel"
    produces = ("daily_ic_20",)
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM, HOLD_HORIZON_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "预测力", "IC_20 均值", "因子与 20 日远期收益的日度 Pearson IC 平均"),
            FieldDoc("std", "预测力", "IC_20 标准差", "日度 IC_20 波动"),
            FieldDoc("positive_ratio", "稳定性", "IC_20 正值比例", "IC_20>0 的交易日占比"),
            FieldDoc("n_days", "样本", "有效天数", "IC_20 非空交易日数"),
            FieldDoc("hold_horizon", "参数", "持有期", "20 日远期收益的持有交易日数"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        f = batch_ctx.masked_factors()
        r = hold_returns(batch_ctx)
        if r is None:
            return {"ic_20": np.full(f.shape[1], np.nan, dtype=np.float64)}
        return {"ic_20": pearson_pairwise(f, r, min_obs=min_obs)}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        min_obs = int(params.get("min_obs", 10))
        hold = int(params.get("hold_horizon", HOLD_HORIZON))
        series = CrossSectionICCalculator().daily_pearson_ic(
            ctx.masked_factor(), ctx.fwd_ret_for(hold), min_obs=min_obs
        )
        ctx.intermediates["daily_ic_20"] = series
        summary = summarize_daily(series)
        return MetricResult(
            scalars={
                "mean": summary["mean"],
                "std": summary["std"],
                "positive_ratio": summary["positive_ratio"],
                "n_days": summary["n_days"],
                "hold_horizon": hold,
            },
            series={"daily_ic_20": series, "cumsum": series.cumsum()},
        )


class RankIC20Metric(BaseMetric):
    name = "rank_ic_20"
    dimension = "预测力"
    description = "日度 RankIC，标签为 20 日远期收益，不是把 IC_5 标签滞后 20 日"
    cost = "panel"
    produces = ("daily_rank_ic_20",)
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM, HOLD_HORIZON_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "预测力", "RankIC_20 均值", "因子与 20 日远期收益的日度 RankIC 平均"),
            FieldDoc("std", "预测力", "RankIC_20 标准差", "日度 RankIC_20 波动"),
            FieldDoc("positive_ratio", "稳定性", "RankIC_20 正值比例", "RankIC_20>0 的交易日占比"),
            FieldDoc("n_days", "样本", "有效天数", "RankIC_20 非空交易日数"),
            FieldDoc("hold_horizon", "参数", "持有期", "20 日远期收益的持有交易日数"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        f = batch_ctx.masked_factors()
        r = hold_returns(batch_ctx)
        if r is None:
            return {"rank_ic_20": np.full(f.shape[1], np.nan, dtype=np.float64)}
        return {"rank_ic_20": rank_ic_pairwise(f, r, min_obs=min_obs)}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        min_obs = int(params.get("min_obs", 10))
        hold = int(params.get("hold_horizon", HOLD_HORIZON))
        series = CrossSectionICCalculator().daily_rank_ic(
            ctx.masked_factor(), ctx.fwd_ret_for(hold), min_obs=min_obs
        )
        ctx.intermediates["daily_rank_ic_20"] = series
        summary = summarize_daily(series)
        return MetricResult(
            scalars={
                "mean": summary["mean"],
                "std": summary["std"],
                "positive_ratio": summary["positive_ratio"],
                "n_days": summary["n_days"],
                "hold_horizon": hold,
            },
            series={"daily_rank_ic_20": series, "cumsum": series.cumsum()},
        )


class PearsonICIR20Metric(BaseMetric):
    name = "ic_ir_20"
    dimension = "预测力"
    description = "IC_20 的信息比率 mean(IC_20)/std"
    cost = "derived"
    produces = ()
    requires = ("daily_ic_20",)

    def params(self) -> Sequence[ParamSpec]:
        return (HOLD_HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("icir", "预测力", "IC_20 IR", "mean(IC_20)/std(IC_20)"),
            FieldDoc("n_days", "样本", "有效天数", "IC_20 非空交易日数"),
            FieldDoc("hold_horizon", "参数", "持有期", "20 日远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = ctx.intermediates.get("daily_ic_20")
        if daily is None:
            raise ValueError("ic_ir_20 需要中间量 daily_ic_20，请先运行 ic_20")
        if not isinstance(daily, pd.Series):
            daily = pd.Series(daily)
        expanding = expanding_icir(daily.to_numpy(dtype=np.float64)[:, None])[:, 0]
        summary = CrossSectionICCalculator().summary(daily)
        return MetricResult(
            scalars={
                "icir": summary["icir"],
                "ic_mean": summary["mean"],
                "ic_std": summary["std"],
                "n_days": summary["n_days"],
                "hold_horizon": int(params.get("hold_horizon", HOLD_HORIZON)),
            },
            series={"ic_ir_20": pd.Series(expanding, index=daily.index, dtype="float64")},
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        src = arrays.get("ic_20")
        if src is None:
            return {}
        return {"ic_ir_20": expanding_icir(np.asarray(src, dtype=np.float64)).astype(np.float32)}


class RankICIR20Metric(BaseMetric):
    name = "rank_ic_ir_20"
    dimension = "预测力"
    description = "RankIC_20 的 expanding 信息比率，给 20 日调仓筛因子用"
    cost = "derived"
    produces = ()
    requires = ("daily_rank_ic_20",)

    def params(self) -> Sequence[ParamSpec]:
        return (HOLD_HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("icir", "预测力", "RankIC_20 IR", "expanding mean(RankIC_20)/std 的样本末值"),
            FieldDoc("n_days", "样本", "有效天数", "RankIC_20 非空交易日数"),
            FieldDoc("hold_horizon", "参数", "持有期", "20 日远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = ctx.intermediates.get("daily_rank_ic_20")
        if daily is None:
            raise ValueError("rank_ic_ir_20 需要中间量 daily_rank_ic_20，请先运行 rank_ic_20")
        if not isinstance(daily, pd.Series):
            daily = pd.Series(daily)
        expanding = expanding_icir(daily.to_numpy(dtype=np.float64)[:, None])[:, 0]
        summary = CrossSectionICCalculator().summary(daily)
        return MetricResult(
            scalars={
                "icir": summary["icir"],
                "ic_mean": summary["mean"],
                "ic_std": summary["std"],
                "n_days": summary["n_days"],
                "hold_horizon": int(params.get("hold_horizon", HOLD_HORIZON)),
            },
            series={"rank_ic_ir_20": pd.Series(expanding, index=daily.index, dtype="float64")},
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        src = arrays.get("rank_ic_20")
        if src is None:
            return {}
        return {"rank_ic_ir_20": expanding_icir(np.asarray(src, dtype=np.float64)).astype(np.float32)}
