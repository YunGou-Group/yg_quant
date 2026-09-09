#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分位组 Barra 暴露 / 组内风格相关，及多空端派生。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..exposure_engine import RAW_STYLE_NAMES
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import apply_daily_matrix, result_from_daily, spearman_pairwise
from .quantile_metric import N_QUANTILES_PARAM, _ensure_labels

MIN_OBS = 10

BARRA_DIRECTION_PARAM = ParamSpec(
    name="barra_direction",
    type="enum",
    label="多空组判定",
    default="linear",
    choices=("linear", "nonlinear"),
    scope="metric",
)


def _resolve_long_short_q(
    arrays: Mapping[str, Any],
    n_q: int,
    n_factors: int,
    direction: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """按全样本 RankIC / 分位收益确定每个因子的多空组分位标签。"""
    direction = str(direction or "linear").strip().lower()
    if direction == "nonlinear":
        q_means = np.full((n_q, n_factors), np.nan, dtype=np.float64)
        for qi in range(1, n_q + 1):
            qr = arrays.get(f"quantile_returns_Q{qi}")
            if qr is None:
                continue
            q_means[qi - 1] = np.nanmean(np.asarray(qr, dtype=np.float64), axis=0)
        has_data = ~np.all(np.isnan(q_means), axis=0)
        long_q = np.ones(n_factors, dtype=np.int32)
        short_q = np.full(n_factors, n_q, dtype=np.int32)
        if has_data.any():
            with np.errstate(invalid="ignore"):
                best = np.nanargmax(q_means[:, has_data], axis=0) + 1
                worst = np.nanargmin(q_means[:, has_data], axis=0) + 1
            long_q[has_data] = best.astype(np.int32)
            short_q[has_data] = worst.astype(np.int32)
        no_data = ~has_data
        if no_data.any():
            rank = arrays.get("rank_ic")
            if rank is not None:
                ic_mean = np.nanmean(np.asarray(rank, dtype=np.float64), axis=0)
                ic_pos = ic_mean >= 0
                long_q[no_data] = np.where(ic_pos[no_data], n_q, 1)
                short_q[no_data] = np.where(ic_pos[no_data], 1, n_q)
        return long_q, short_q

    rank = arrays.get("rank_ic")
    if rank is None:
        return (
            np.full(n_factors, n_q, dtype=np.int32),
            np.ones(n_factors, dtype=np.int32),
        )
    ic_mean = np.nanmean(np.asarray(rank, dtype=np.float64), axis=0)
    ic_pos = np.where(np.isfinite(ic_mean), ic_mean >= 0, True)
    long_q = np.where(ic_pos, n_q, 1).astype(np.int32)
    short_q = np.where(ic_pos, 1, n_q).astype(np.int32)
    return long_q, short_q


def _derive_long_short_series(
    arrays: Mapping[str, Any],
    *,
    in_prefix: str,
    out_prefix: str,
    n_q: int,
    styles: Sequence[str],
    direction: str,
) -> Dict[str, np.ndarray]:
    sample: Optional[np.ndarray] = None
    for style in styles:
        for qi in range(1, n_q + 1):
            arr = arrays.get(f"{in_prefix}{qi}_{style}")
            if arr is not None:
                sample = np.asarray(arr)
                break
        if sample is not None:
            break
    if sample is None:
        return {}
    n_dates = int(sample.shape[0])
    n_factors = int(sample.shape[1]) if sample.ndim == 2 else 1
    long_q, short_q = _resolve_long_short_q(arrays, n_q, n_factors, direction)
    out: Dict[str, np.ndarray] = {}
    for style in styles:
        long_mat = np.full((n_dates, n_factors), np.nan, dtype=np.float64)
        short_mat = np.full((n_dates, n_factors), np.nan, dtype=np.float64)
        any_src = False
        for qi in range(1, n_q + 1):
            src = arrays.get(f"{in_prefix}{qi}_{style}")
            if src is None:
                continue
            any_src = True
            vals = np.asarray(src, dtype=np.float64)
            if vals.ndim == 1:
                vals = vals[:, None]
            long_sel = long_q == qi
            short_sel = short_q == qi
            if long_sel.any():
                long_mat[:, long_sel] = vals[:, long_sel]
            if short_sel.any():
                short_mat[:, short_sel] = vals[:, short_sel]
        if not any_src:
            continue
        out[f"{out_prefix}_long_{style}"] = long_mat.astype(np.float32, copy=False)
        out[f"{out_prefix}_short_{style}"] = short_mat.astype(np.float32, copy=False)
    return out


class LongShortBarraExposureMetric(BaseMetric):
    name = "long_short_barra_exposure"
    dimension = "风格暴露"
    description = "分位组内 Barra 风格均值；并按 RankIC 方向派生多空端暴露"
    cost = "panel"
    produces = ()
    requires = ()
    engine_requires = ("style_raw",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM, BARRA_DIRECTION_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "风格暴露", "多头 Size 暴露均值", "long_short_barra_exposure_long_style_size 均值"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f = batch_ctx.masked_factors()
        b = batch_ctx.masked_barra()
        p = f.shape[1]
        names = RAW_STYLE_NAMES
        out: Dict[str, Any] = {}
        for qi in range(1, n_q + 1):
            for name in names:
                out[f"barra_exposure_Q{qi}_{name}"] = np.full(p, np.nan)
        if b is None or b.size == 0:
            return out
        labels = _ensure_labels(batch_ctx, n_q)
        n_styles = min(b.shape[1], len(names))
        for si in range(n_styles):
            style = names[si]
            col = b[:, si]
            for qi in range(1, n_q + 1):
                # 风格 NaN 不参与均值：按因子列掩码
                vals = np.full(p, np.nan, dtype=np.float64)
                for j in range(p):
                    sel = (labels[:, j] == qi) & np.isfinite(col) & np.isfinite(f[:, j])
                    if int(sel.sum()) <= 0:
                        continue
                    vals[j] = float(np.mean(col[sel]))
                out[f"barra_exposure_Q{qi}_{style}"] = vals
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        n_q = int(params.get("n_quantiles", 5))
        direction = str(params.get("barra_direction", "linear"))
        # 交互路径：用日度 RankIC 均值定多空，再抽多空端序列
        arrays = {k: v.to_numpy(dtype=np.float64)[:, None] for k, v in daily.items()}
        rank = ctx.intermediates.get("daily_rank_ic")
        if isinstance(rank, pd.Series):
            arrays["rank_ic"] = rank.to_numpy(dtype=np.float64)[:, None]
        derived = _derive_long_short_series(
            arrays,
            in_prefix="barra_exposure_Q",
            out_prefix="long_short_barra_exposure",
            n_q=n_q,
            styles=RAW_STYLE_NAMES,
            direction=direction,
        )
        extra = {
            key: pd.Series(np.asarray(arr)[:, 0], index=ctx.masked_factor().index, dtype="float64")
            for key, arr in derived.items()
        }
        merged = {**daily, **extra}
        primary = "long_short_barra_exposure_long_style_size"
        if primary not in merged:
            primary = next(iter(merged), None)
        return result_from_daily(
            merged,
            primary=primary,
            extra_scalars={"horizon": ctx.horizon, "barra_direction": direction},
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", 5))
        direction = str(params.get("barra_direction", "linear"))
        return _derive_long_short_series(
            arrays,
            in_prefix="barra_exposure_Q",
            out_prefix="long_short_barra_exposure",
            n_q=n_q,
            styles=RAW_STYLE_NAMES,
            direction=direction,
        )


class BarraQuantileCorrelationMetric(BaseMetric):
    name = "barra_quantile_correlation"
    dimension = "风格暴露"
    description = "各分位组内因子与 Barra 风格的 Spearman 秩相关"
    cost = "panel"
    produces = ("barra_quantile_correlation",)
    requires = ()
    engine_requires = ("style_raw",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "风格暴露", "顶组 vs Size 相关", "barra_corr_Qn_style_size 均值"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f = batch_ctx.masked_factors()
        b = batch_ctx.masked_barra()
        p = f.shape[1]
        names = RAW_STYLE_NAMES
        out: Dict[str, Any] = {}
        for qi in range(1, n_q + 1):
            for name in names:
                out[f"barra_corr_Q{qi}_{name}"] = np.full(p, np.nan)
        if b is None or b.size == 0:
            return out
        labels = _ensure_labels(batch_ctx, n_q)
        n_styles = min(b.shape[1], len(names))
        min_obs = max(MIN_OBS, int(params.get("min_obs", MIN_OBS)))
        for si in range(n_styles):
            style = names[si]
            col = b[:, si]
            for qi in range(1, n_q + 1):
                vals = np.full(p, np.nan, dtype=np.float64)
                for j in range(p):
                    sel = (labels[:, j] == qi) & np.isfinite(col) & np.isfinite(f[:, j])
                    if int(sel.sum()) < min_obs:
                        continue
                    vals[j] = spearman_pairwise(f[sel, j : j + 1], col[sel], min_obs=min_obs)[0]
                out[f"barra_corr_Q{qi}_{style}"] = vals
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        ctx.intermediates["barra_quantile_correlation"] = True
        n_q = int(params.get("n_quantiles", 5))
        return result_from_daily(
            daily,
            primary=f"barra_corr_Q{n_q}_style_size",
            extra_scalars={"horizon": ctx.horizon},
        )


class LongShortBarraCorrMetric(BaseMetric):
    name = "long_short_barra_corr"
    dimension = "风格暴露"
    description = "从分位组内风格相关派生多空端 barra_corr"
    cost = "derived"
    produces = ()
    requires = ("barra_quantile_correlation",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, N_QUANTILES_PARAM, BARRA_DIRECTION_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("mean", "风格暴露", "多头 Size 相关均值", "long_short_barra_corr_long_style_size 均值"),
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        # 交互：先算分位相关，再派生
        n_q = int(params.get("n_quantiles", 5))
        direction = str(params.get("barra_direction", "linear"))
        daily = apply_daily_matrix(BarraQuantileCorrelationMetric(), ctx, params)
        arrays = {k: v.to_numpy(dtype=np.float64)[:, None] for k, v in daily.items()}
        rank = ctx.intermediates.get("daily_rank_ic")
        if isinstance(rank, pd.Series):
            arrays["rank_ic"] = rank.to_numpy(dtype=np.float64)[:, None]
        derived = _derive_long_short_series(
            arrays,
            in_prefix="barra_corr_Q",
            out_prefix="long_short_barra_corr",
            n_q=n_q,
            styles=RAW_STYLE_NAMES,
            direction=direction,
        )
        index = ctx.masked_factor().index
        series = {
            **daily,
            **{
                key: pd.Series(np.asarray(arr)[:, 0], index=index, dtype="float64")
                for key, arr in derived.items()
            },
        }
        primary = "long_short_barra_corr_long_style_size"
        if primary not in series:
            primary = next(iter(series), None)
        return result_from_daily(
            series,
            primary=primary,
            extra_scalars={"horizon": ctx.horizon, "barra_direction": direction},
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", 5))
        direction = str(params.get("barra_direction", "linear"))
        return _derive_long_short_series(
            arrays,
            in_prefix="barra_corr_Q",
            out_prefix="long_short_barra_corr",
            n_q=n_q,
            styles=RAW_STYLE_NAMES,
            direction=direction,
        )
