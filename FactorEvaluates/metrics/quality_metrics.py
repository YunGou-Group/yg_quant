#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子质量：覆盖率、缺失、唯一值、极端值；保留 IQR。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import apply_daily_matrix, quality_stats, result_from_daily, summarize_daily


class CoverageRateMetric(BaseMetric):
    name = "coverage_rate"
    dimension = "暴露度"
    description = "有效覆盖率与截面四分位距：有多少股票能用这个因子、值是否挤成一团"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "暴露度", "日均覆盖率", "每日有效样本 / 股票池股票数的平均"),
            FieldDoc("iqr_mean", "暴露度", "截面IQR均值", "每日因子值 75 分位减 25 分位的平均"),
            FieldDoc("n_stocks_mean", "暴露度", "日均有效股票数", "每日有限因子值的平均股票数"),
            FieldDoc("n_days", "样本", "有效天数", "股票池非空的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "与本次评估共用的持有期（覆盖本身不依赖收益）"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        stats = quality_stats(batch_ctx.masked_factors())
        f = batch_ctx.masked_factors()
        n_valid = np.isfinite(f).sum(axis=0).astype(np.float64)
        return {
            "coverage_rate": stats["coverage_rate"],
            "iqr": stats["iqr"],
            "valid_samples": n_valid,
        }

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        factor = ctx.masked_factor()
        values = factor.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        n_stocks = finite.sum(axis=1).astype(np.float64)
        if ctx.universe_mask is not None:
            universe = ctx.universe_mask.to_numpy(dtype=bool, copy=False)
            denom = universe.sum(axis=1).astype(np.float64)
        else:
            denom = np.full(values.shape[0], float(values.shape[1]))
        denom = np.where(denom > 0, denom, np.nan)
        coverage = n_stocks / denom
        iqr = np.full(values.shape[0], np.nan, dtype=np.float64)
        for i in range(values.shape[0]):
            row = values[i, finite[i]]
            if row.size < 8:
                continue
            q75, q25 = np.percentile(row, [75.0, 25.0])
            iqr[i] = float(q75 - q25)
        index = factor.index
        coverage_s = pd.Series(coverage, index=index, name="coverage_rate")
        n_stocks_s = pd.Series(n_stocks, index=index, name="n_stocks")
        iqr_s = pd.Series(iqr, index=index, name="iqr")
        ok = np.isfinite(coverage)
        summary = summarize_daily(coverage_s)
        return MetricResult(
            scalars={
                "mean": summary["mean"],
                "coverage_mean": summary["mean"],
                "n_stocks_mean": float(np.mean(n_stocks[ok])) if ok.any() else float("nan"),
                "iqr_mean": float(np.nanmean(iqr)) if np.isfinite(iqr).any() else float("nan"),
                "n_days": int(ok.sum()),
                "horizon": ctx.horizon,
            },
            series={
                "coverage_rate": coverage_s.astype("float64"),
                "n_stocks": n_stocks_s.astype("float64"),
                "iqr": iqr_s.astype("float64"),
            },
        )


class MissingRateMetric(BaseMetric):
    name = "missing_rate"
    dimension = "暴露度"
    description = "股票池内因子缺失率"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "日均缺失率", "每日缺失样本占比的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        return {"missing_rate": quality_stats(batch_ctx.masked_factors())["missing_rate"]}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="missing_rate", extra_scalars={"horizon": ctx.horizon})


class UniqueRateMetric(BaseMetric):
    name = "unique_rate"
    dimension = "暴露度"
    description = "有效样本中唯一值占比，过低说明因子几乎是常数"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "日均唯一值比率", "unique_count / valid_samples 的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        stats = quality_stats(batch_ctx.masked_factors())
        return {"unique_rate": stats["unique_rate"], "unique_count": stats["unique_count"]}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="unique_rate", extra_scalars={"horizon": ctx.horizon})


class ExtremeValueRatioMetric(BaseMetric):
    name = "extreme_value_ratio"
    dimension = "暴露度"
    description = "截面 |z|>3 的样本占比"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "日均极端值占比", "|z|>3 占比的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        return {"extreme_value_ratio": quality_stats(batch_ctx.masked_factors())["extreme_value_ratio"]}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="extreme_value_ratio", extra_scalars={"horizon": ctx.horizon})


class MissingSamplesMetric(BaseMetric):
    name = "missing_samples"
    dimension = "暴露度"
    description = "股票池内因子缺失样本绝对个数"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "日均缺失样本数", "每日缺失个数的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        return {"missing_samples": quality_stats(batch_ctx.masked_factors())["missing_samples"]}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="missing_samples", extra_scalars={"horizon": ctx.horizon})


class TotalSamplesMetric(BaseMetric):
    name = "total_samples"
    dimension = "暴露度"
    description = "股票池当日样本总数（含缺失）"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "日均样本总数", "股票池每日股票数的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        return {"total_samples": quality_stats(batch_ctx.masked_factors())["total_samples"]}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="total_samples", extra_scalars={"horizon": ctx.horizon})
