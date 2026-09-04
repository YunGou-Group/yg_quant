#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长短窗 RankIC 方向一致性（替换原 ic_trend）。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, ParamSpec
from ..matrix_utils import rolling_nanmean, summarize_daily

FAST_WINDOW_PARAM = ParamSpec(
    name="fast_window",
    type="int",
    label="短窗（交易日）",
    default=20,
    min=2,
    max=120,
    scope="metric",
)

SLOW_WINDOW_PARAM = ParamSpec(
    name="slow_window",
    type="int",
    label="长窗（交易日）",
    default=250,
    min=5,
    max=500,
    scope="metric",
)


class LongShortTermConsistencyMetric(BaseMetric):
    name = "long_short_term_consistency"
    dimension = "失效风险"
    description = "RankIC 短窗与长窗滚动均值是否同号；expanding mean 衡量方向是否长期一致"
    cost = "derived"
    produces = ()
    requires = ("daily_rank_ic",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, FAST_WINDOW_PARAM, SLOW_WINDOW_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "失效风险", "一致性均值", "短窗与长窗 RankIC 同号的 expanding 均值"),
            FieldDoc("last", "失效风险", "一致性最新值", "最新一天的 expanding 一致性"),
            FieldDoc("fast_window", "参数", "短窗", "短窗滚动交易日数，默认 20"),
            FieldDoc("slow_window", "参数", "长窗", "长窗滚动交易日数，默认 250"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = ctx.intermediates.get("daily_rank_ic")
        if daily is None:
            raise ValueError("long_short_term_consistency 需要中间量 daily_rank_ic，请先运行 rank_ic")
        if not isinstance(daily, pd.Series):
            daily = pd.Series(daily)
        fast_window = int(params.get("fast_window", 20))
        slow_window = int(params.get("slow_window", 250))
        if slow_window <= fast_window:
            raise ValueError("slow_window 必须大于 fast_window")
        fast = daily.rolling(window=fast_window, min_periods=fast_window).mean()
        slow = daily.rolling(window=slow_window, min_periods=slow_window).mean()
        same = (fast * slow > 0).astype("float64")
        same[(fast.isna()) | (slow.isna())] = float("nan")
        expanding = same.expanding(min_periods=1).mean()
        summary = summarize_daily(expanding)
        last = expanding.dropna()
        return MetricResult(
            scalars={
                "mean": summary["mean"],
                "last": float(last.iloc[-1]) if len(last) else float("nan"),
                "fast_window": fast_window,
                "slow_window": slow_window,
                "n_days": summary["n_days"],
                "horizon": ctx.horizon,
            },
            series={
                "consistency": expanding.astype("float64"),
                "ma_fast": fast.astype("float64"),
                "ma_slow": slow.astype("float64"),
            },
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rank = arrays.get("rank_ic")
        if rank is None:
            return {}
        fast_window = int(params.get("fast_window", 20))
        slow_window = int(params.get("slow_window", 250))
        fast = rolling_nanmean(rank, fast_window)
        slow = rolling_nanmean(rank, slow_window)
        same = ((fast * slow) > 0).astype(np.float64)
        same[~(np.isfinite(fast) & np.isfinite(slow))] = np.nan
        csum = np.nancumsum(np.where(np.isfinite(same), same, 0.0), axis=0)
        ccnt = np.cumsum(np.isfinite(same).astype(np.float64), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            expand = csum / np.maximum(ccnt, 1.0)
            expand[ccnt < 1] = np.nan
        return {"long_short_term_consistency": expand}