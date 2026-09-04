#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拥挤度：估值溢价（B/P 端点差）与端点收益离散度。缺 B/P 则跳过估值项。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..exposure_engine import RAW_STYLE_NAMES
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import apply_daily_matrix, assign_quantiles, result_from_daily

N_QUANTILES_PARAM = ParamSpec(
    name="n_quantiles",
    type="int",
    label="分层组数",
    default=5,
    min=2,
    max=20,
    scope="metric",
)

_BTOP = "style_btop"


class FactorCrowdingMetric(BaseMetric):
    name = "factor_crowding"
    dimension = "暴露度"
    description = "低分端减高分端的 B/P 差（估值拥挤）以及两端收益方差的合并尺度"
    cost = "panel"
    produces = ()
    requires = ()
    engine_requires = ("style_raw",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "暴露度", "估值拥挤溢价", "mean(B/P|Q1)-mean(B/P|Qn) 的区间平均"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        b = batch_ctx.masked_barra()
        p = f.shape[1]
        valuation = np.full(p, np.nan)
        dispersion = np.full(p, np.nan)
        labels = batch_ctx.intermediates.get("quantile_labels")
        if labels is None:
            labels = assign_quantiles(f, n_q)
            batch_ctx.intermediates["quantile_labels"] = labels
        btop = None
        if b is not None and b.ndim == 2:
            try:
                idx = list(RAW_STYLE_NAMES).index(_BTOP)
            except ValueError:
                idx = -1
            if 0 <= idx < b.shape[1]:
                btop = b[:, idx]
        for j in range(p):
            low = labels[:, j] == 1
            high = labels[:, j] == n_q
            if btop is not None:
                low_bp = btop[low & np.isfinite(btop)]
                high_bp = btop[high & np.isfinite(btop)]
                if low_bp.size and high_bp.size:
                    valuation[j] = float(low_bp.mean() - high_bp.mean())
            if r is None:
                continue
            low_r = r[low & np.isfinite(r)]
            high_r = r[high & np.isfinite(r)]
            if low_r.size < 2 or high_r.size < 2:
                continue
            dispersion[j] = float(np.sqrt((np.var(high_r, ddof=1) + np.var(low_r, ddof=1)) / 2.0))
        return {
            "crowding_valuation_premium": valuation,
            "crowding_endpoint_dispersion": dispersion,
        }

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(
            daily, primary="crowding_valuation_premium", extra_scalars={"horizon": ctx.horizon}
        )