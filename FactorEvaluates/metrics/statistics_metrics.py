#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截面统计与因子自相关。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import (
    apply_daily_matrix,
    nan_kurtosis,
    nan_skew,
    pearson_cols,
    pearson_pairwise,
    rank_cols,
    result_from_daily,
)

AUTOCORR_LAGS: Tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60)


class FactorStatisticsMetric(BaseMetric):
    name = "factor_stats"
    dimension = "暴露度"
    description = "截面均值/波动/偏度/峰度"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "暴露度", "截面均值的平均", "日度 factor_mean 的区间平均"),
            FieldDoc("std_mean", "暴露度", "截面波动均值", "日度 factor_std 的区间平均"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        return {
            "factor_mean": np.nanmean(f, axis=0),
            "factor_std": np.nanstd(f, axis=0, ddof=1),
            "factor_median": np.nanmedian(f, axis=0),
            "factor_skewness": nan_skew(f).reshape(-1),
            "factor_kurtosis": nan_kurtosis(f).reshape(-1),
        }

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="factor_mean", extra_scalars={"horizon": ctx.horizon})


class FactorMinMetric(BaseMetric):
    name = "factor_min"
    dimension = "暴露度"
    description = "截面因子最小值"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "截面最小均值", "日度 factor_min 的区间平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        with np.errstate(all="ignore"):
            vals = np.nanmin(f, axis=0)
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        vals[~np.isfinite(f).any(axis=0)] = np.nan
        return {"factor_min": vals}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="factor_min", extra_scalars={"horizon": ctx.horizon})


class FactorMaxMetric(BaseMetric):
    name = "factor_max"
    dimension = "暴露度"
    description = "截面因子最大值"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "截面最大均值", "日度 factor_max 的区间平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        with np.errstate(all="ignore"):
            vals = np.nanmax(f, axis=0)
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        vals[~np.isfinite(f).any(axis=0)] = np.nan
        return {"factor_max": vals}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="factor_max", extra_scalars={"horizon": ctx.horizon})


class FactorAutocorrMetric(BaseMetric):
    name = "factor_autocorr"
    dimension = "有效期"
    description = "因子截面百分位秩与滞后日秩的 Pearson 相关"
    cost = "derived"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return tuple(
            FieldDoc(f"mean_{lag}d", "有效期", f"自相关 {lag}d", f"滞后 {lag} 日的因子秩自相关均值")
            for lag in AUTOCORR_LAGS
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        p = int(batch_ctx.factors.shape[1])
        out = {f"factor_autocorr_{lag}d": np.full(p, np.nan, dtype=np.float64) for lag in AUTOCORR_LAGS}
        ranks = batch_ctx.intermediates.get("factor_ranks_full")
        ring = batch_ctx.intermediates.get("rank_ring")
        if ranks is None or ring is None:
            return out
        min_obs = int(params.get("min_obs", batch_ctx.min_obs or 20))
        for lag in AUTOCORR_LAGS:
            if lag > int(ring.shape[0]):
                continue
            out[f"factor_autocorr_{lag}d"] = pearson_cols(ranks, ring[lag - 1], min_obs=min_obs)
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        factor = ctx.masked_factor()
        values = factor.to_numpy(dtype=np.float64, copy=False)
        # rank_cols 按列排名；日截面秩 = 对 (股票 × 日) 转置后再转回
        ranks = rank_cols(values.T).T
        series: Dict[str, pd.Series] = {}
        scalars: Dict[str, Any] = {"horizon": ctx.horizon}
        for lag in AUTOCORR_LAGS:
            if lag >= ranks.shape[0]:
                continue
            daily = np.full(ranks.shape[0], np.nan)
            for i in range(lag, ranks.shape[0]):
                daily[i] = pearson_pairwise(ranks[i, :, None], ranks[i - lag], min_obs=20)[0]
            s = pd.Series(daily, index=factor.index, name=f"factor_autocorr_{lag}d")
            series[s.name] = s
            clean = s.dropna()
            scalars[f"mean_{lag}d"] = float(clean.mean()) if len(clean) else float("nan")
        return MetricResult(scalars=scalars, series=series)