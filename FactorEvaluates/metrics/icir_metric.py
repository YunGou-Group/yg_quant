#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICIR = mean(RankIC) / std(RankIC)。只读 daily_rank_ic，不再扫面板。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..cross_section_ic_calculator import CrossSectionICCalculator
from ..context import EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, ParamSpec
from ..matrix_utils import expanding_icir, icir_row


class ICIRMetric(BaseMetric):
    name = "icir"
    dimension = "预测力"
    description = "信息比率 ICIR = mean(RankIC)/std(RankIC)，衡量预测力是否盖过噪声"
    cost = "derived"
    produces = ()
    requires = ("daily_rank_ic",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("icir", "预测力", "ICIR", "mean(RankIC)/std(RankIC)；绝对值越大，平均预测力相对波动越强"),
            FieldDoc("ic_mean", "预测力", "IC均值", "与 ic.mean 相同，便于摘要单独列出"),
            FieldDoc("ic_std", "预测力", "IC标准差", "日度 RankIC 标准差"),
            FieldDoc("n_days", "样本", "有效天数", "RankIC 非空的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = ctx.intermediates.get("daily_rank_ic")
        if daily is None:
            raise ValueError("icir 需要中间量 daily_rank_ic，请先运行 rank_ic")
        if not isinstance(daily, pd.Series):
            daily = pd.Series(daily)
        summary = CrossSectionICCalculator().summary(daily)
        return MetricResult(
            scalars={
                "icir": summary["icir"],
                "ic_mean": summary["mean"],
                "ic_std": summary["std"],
                "n_days": summary["n_days"],
                "horizon": ctx.horizon,
            }
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rank = arrays.get("rank_ic")
        if rank is None:
            return {}
        return {"icir": icir_row(rank)}


class PearsonICIRMetric(BaseMetric):
    name = "ic_ir"
    dimension = "预测力"
    description = "Pearson IC 的信息比率 = mean(IC)/std(IC)"
    cost = "derived"
    produces = ()
    requires = ("daily_ic",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("icir", "预测力", "IC IR", "mean(Pearson IC)/std(Pearson IC)"),
            FieldDoc("ic_mean", "预测力", "IC均值", "与 ic.mean 相同"),
            FieldDoc("ic_std", "预测力", "IC标准差", "日度 Pearson IC 标准差"),
            FieldDoc("n_days", "样本", "有效天数", "IC 非空的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = ctx.intermediates.get("daily_ic")
        if daily is None:
            raise ValueError("ic_ir 需要中间量 daily_ic，请先运行 ic")
        if not isinstance(daily, pd.Series):
            daily = pd.Series(daily)
        summary = CrossSectionICCalculator().summary(daily)
        return MetricResult(
            scalars={
                "icir": summary["icir"],
                "ic_mean": summary["mean"],
                "ic_std": summary["std"],
                "n_days": summary["n_days"],
                "horizon": ctx.horizon,
            }
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        ic = arrays.get("ic")
        if ic is None:
            return {}
        return {"ic_ir": icir_row(ic)}


class NonlinearICIRMetric(BaseMetric):
    name = "nonlinear_ic_ir"
    dimension = "预测力"
    description = "非线性 IC 的 expanding 信息比率 mean/std（与参考实现对齐的逐日序列）"
    cost = "derived"
    produces = ()
    requires = ("daily_nonlinear_ic",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("icir", "预测力", "非线性IC IR", "全样本 mean(nonlinear_ic)/std"),
            FieldDoc("n_days", "样本", "有效天数", "非线性 IC 非空的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = ctx.intermediates.get("daily_nonlinear_ic")
        if daily is None:
            raise ValueError("nonlinear_ic_ir 需要中间量 daily_nonlinear_ic，请先运行 nonlinear_ic")
        if not isinstance(daily, pd.Series):
            daily = pd.Series(daily)
        values = daily.to_numpy(dtype=np.float64)
        expanding = expanding_icir(values[:, None])[:, 0]
        series = pd.Series(expanding, index=daily.index, name="nonlinear_ic_ir", dtype="float64")
        summary = CrossSectionICCalculator().summary(daily)
        return MetricResult(
            scalars={
                "icir": summary["icir"],
                "ic_mean": summary["mean"],
                "ic_std": summary["std"],
                "n_days": summary["n_days"],
                "horizon": ctx.horizon,
            },
            series={"nonlinear_ic_ir": series},
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        src = arrays.get("nonlinear_ic")
        if src is None:
            return {}
        return {"nonlinear_ic_ir": expanding_icir(np.asarray(src, dtype=np.float64)).astype(np.float32)}
