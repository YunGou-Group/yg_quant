#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滚动 RankIC 均值。只读 daily_rank_ic。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from ..base_metric import BaseMetric
from ..context import EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, ParamSpec
from ..matrix_utils import rolling_nanmean

WINDOW_PARAM = ParamSpec(
    name="window",
    type="int",
    label="滚动窗口（交易日）",
    default=20,
    min=2,
    max=250,
    scope="metric",
)


class RollingICMetric(BaseMetric):
    name = "rolling_ic"
    dimension = "有效期"
    description = "日度 RankIC 的滚动均值，用来看预测力是否随时间漂移"
    cost = "derived"
    produces = ()
    requires = ("daily_rank_ic",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, WINDOW_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "有效期", "滚动IC均值", "滚动 RankIC 在区间上的平均，概括整段滚动曲线的位置"),
            FieldDoc("window", "参数", "滚动窗口", "滚动均值所用的交易日数"),
            FieldDoc("n_days", "样本", "有效天数", "滚动均值非空的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = ctx.intermediates.get("daily_rank_ic")
        if daily is None:
            raise ValueError("rolling_ic 需要中间量 daily_rank_ic，请先运行 rank_ic")
        if not isinstance(daily, pd.Series):
            daily = pd.Series(daily)
        window = int(params.get("window", 20))
        rolled = daily.rolling(window=window, min_periods=window).mean()
        clean = rolled.dropna()
        mean = float(clean.mean()) if len(clean) else float("nan")
        return MetricResult(
            scalars={
                "mean": mean,
                "window": window,
                "n_days": int(len(clean)),
                "horizon": ctx.horizon,
            },
            series={"rolling_ic": rolled},
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rank = arrays.get("rank_ic")
        if rank is None:
            return {}
        window = int(params.get("window", 20))
        return {"rolling_ic": rolling_nanmean(rank, window)}
