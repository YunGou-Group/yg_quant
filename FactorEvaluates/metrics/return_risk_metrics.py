#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收益、多头、加权 PnL、风格秩相关。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..exposure_engine import RAW_STYLE_NAMES
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import apply_daily_matrix, result_from_daily, spearman_pairwise

MIN_OBS = 10


class FactorReturnsMetric(BaseMetric):
    name = "factor_returns"
    dimension = "预测力"
    description = "Cov(f,r)/Var(f) 截面回归斜率，即因子收益"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "因子收益均值", "日度回归斜率的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        out = np.full(p, np.nan)
        if r is None:
            return {"factor_returns": out}
        valid = np.isfinite(f) & np.isfinite(r)[:, None]
        count = valid.sum(axis=0)
        fz = np.where(valid, f, 0.0)
        rz = np.where(valid, r[:, None], 0.0)
        f_mean = fz.sum(axis=0) / np.maximum(count, 1)
        r_mean = rz.sum(axis=0) / np.maximum(count, 1)
        fc = np.where(valid, fz - f_mean, 0.0)
        rc = np.where(valid, rz - r_mean, 0.0)
        cov = (fc * rc).sum(axis=0) / np.maximum(count - 1, 1)
        var_f = (fc ** 2).sum(axis=0) / np.maximum(count - 1, 1)
        safe = (var_f > 1e-15) & (count >= MIN_OBS)
        np.divide(cov, var_f, out=out, where=safe)
        return {"factor_returns": out}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="factor_returns", extra_scalars={"horizon": ctx.horizon})


class LongOnlyReturnMetric(BaseMetric):
    name = "long_only_return"
    dimension = "预测力"
    description = "factor>0 等权多头与 factor<0 等权空头的截面均收益"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "多头收益均值", "factor>0 等权篮子日均收益"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        valid_f = np.isfinite(f)
        long_cnt = (valid_f & (f > 0)).sum(axis=0).astype(np.float64)
        short_cnt = (valid_f & (f < 0)).sum(axis=0).astype(np.float64)
        if r is None:
            nan = np.full(p, np.nan)
            return {
                "long_only_return": nan,
                "short_only_return": nan,
                "long_count": long_cnt,
                "short_count": short_cnt,
            }
        valid = valid_f & np.isfinite(r)[:, None]

        def _leg(side):
            cnt = side.sum(axis=0)
            ret_sum = np.where(side, r[:, None], 0.0).sum(axis=0)
            return np.where(cnt >= MIN_OBS, ret_sum / np.maximum(cnt, 1), np.nan), cnt.astype(np.float64)

        long_ret, _ = _leg(valid & (f > 0))
        short_ret, _ = _leg(valid & (f < 0))
        return {
            "long_only_return": long_ret,
            "short_only_return": short_ret,
            "long_count": long_cnt,
            "short_count": short_cnt,
        }

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="long_only_return", extra_scalars={"horizon": ctx.horizon})


class WeightedPnLMetric(BaseMetric):
    name = "weighted_pnl"
    dimension = "预测力"
    description = "有效样本上 Σ(f·r)；多头 Σ(max(f,0)·r)，空头 Σ((-min(f,0))·r)"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "加权PnL均值", "日度截面 mean(f·r) 的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        if r is None:
            nan = np.full(p, np.nan)
            return {"weighted_pnl": nan, "weighted_long_pnl": nan, "weighted_short_pnl": nan}
        valid = np.isfinite(f) & np.isfinite(r)[:, None]
        contrib = np.where(valid, f * r[:, None], 0.0)
        long_m = valid & (f > 0)
        short_m = valid & (f < 0)
        long = np.where(long_m, f * r[:, None], 0.0)
        short = np.where(short_m, -f * r[:, None], 0.0)
        return {
            "weighted_pnl": contrib.sum(axis=0),
            "weighted_long_pnl": long.sum(axis=0),
            "weighted_short_pnl": short.sum(axis=0),
        }

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="weighted_pnl", extra_scalars={"horizon": ctx.horizon})


class FactorStyleCorrelationMetric(BaseMetric):
    name = "factor_style_correlation"
    dimension = "风格暴露"
    description = "因子与 Barra 风格的截面 Spearman 相关（与 exposure β 并存）"
    cost = "panel"
    produces = ()
    requires = ()
    engine_requires = ("style_raw",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "风格暴露", "与Size相关均值", "因子 vs style_size 的日均秩相关"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        b = batch_ctx.masked_barra()
        p = f.shape[1]
        names = RAW_STYLE_NAMES
        out = {}
        if b is None or b.size == 0:
            for name in names:
                out[f"factor_style_correlation_{name}"] = np.full(p, np.nan)
            return out
        n_styles = min(b.shape[1], len(names))
        for si in range(n_styles):
            out[f"factor_style_correlation_{names[si]}"] = spearman_pairwise(
                f, b[:, si], min_obs=MIN_OBS
            )
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(
            daily,
            primary="factor_style_correlation_style_size",
            extra_scalars={"horizon": ctx.horizon},
        )


class LongCountMetric(BaseMetric):
    name = "long_count"
    dimension = "预测力"
    description = "factor>0 / factor<0 的股票数，不依赖收益"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "多头广度均值", "factor>0 日均股票数"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        valid = np.isfinite(f)
        return {
            "long_count": (valid & (f > 0)).sum(axis=0).astype(np.float64),
            "short_count": (valid & (f < 0)).sum(axis=0).astype(np.float64),
        }

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="long_count", extra_scalars={"horizon": ctx.horizon})