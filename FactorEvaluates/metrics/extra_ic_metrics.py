#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""非线性 IC、互信息 IC、方向调整 IC、IC 衰减。"""

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
    auto_direction_signs,
    distance_correlation,
    quantile_bin,
    rank_ic_pairwise,
    result_from_daily,
    summarize_daily,
)

MIN_OBS_PARAM = ParamSpec(
    name="min_obs",
    type="int",
    label="每日最少有效股票数",
    default=10,
    min=3,
    max=500,
    scope="metric",
)

# 与 doc/evaluator CROSSDAY_HORIZONS 一致：同一套日频标签往后挪 lag 天，不是 N 日累计收益。
DECAY_HORIZONS: Tuple[int, ...] = (1, 2, 3, 4, 5, 10, 20, 40, 60)


def shift_return_panel(fwd: pd.DataFrame, lag: int) -> pd.DataFrame:
    """行 t 换成 t+lag 的收益标签。lag<1 时原样返回。"""
    n = int(lag)
    if n < 1:
        return fwd
    return fwd.shift(-n)


def lag_fwd_stack(main: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """(lag, date, stock)：第 t 日放 main[t+lag]；越界为 NaN。"""
    panel = np.asarray(main)
    if panel.ndim != 2:
        raise ValueError(f"收益面板须为 (日期, 股票)，收到 shape={panel.shape}")
    n_dates, n_stocks = panel.shape
    keys = tuple(int(n) for n in lags)
    out = np.full((len(keys), n_dates, n_stocks), np.nan, dtype=np.float32)
    for i, lag in enumerate(keys):
        if lag < 1 or lag >= n_dates:
            continue
        out[i, : n_dates - lag] = panel[lag:]
    return out


class NonlinearICMetric(BaseMetric):
    name = "nonlinear_ic"
    dimension = "预测力"
    description = "距离相关 dcor：捕捉 U 型等非线性预测关系；覆盖率过滤后最多 1000 样本"
    cost = "panel"
    produces = ("daily_nonlinear_ic",)
    requires = ()
    _MAX_SAMPLES = 1000

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "非线性IC均值", "日度距离相关的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        out = np.full(p, np.nan, dtype=np.float64)
        if r is None:
            return {"nonlinear_ic": out}
        valid_r = np.isfinite(r)
        if int(valid_r.sum()) < 10:
            return {"nonlinear_ic": out}
        f_sub, r_sub = f[valid_r], r[valid_r]
        col_valid = np.isfinite(f_sub).sum(axis=0)
        row_cov = np.isfinite(f_sub).sum(axis=1)
        good = row_cov >= int(p * 0.80)
        if int(good.sum()) < 10:
            return {"nonlinear_ic": out}
        f_batch = f_sub[good].copy()
        r_batch = r_sub[good]
        nans = np.isnan(f_batch)
        if nans.any():
            col_means = np.nan_to_num(np.nanmean(f_batch, axis=0), nan=0.0)
            f_batch[nans] = np.broadcast_to(col_means, f_batch.shape)[nans]
        n_good = len(r_batch)
        rng = np.random.RandomState(42)
        if n_good > self._MAX_SAMPLES:
            idx = rng.choice(n_good, self._MAX_SAMPLES, replace=False)
            f_batch, r_batch = f_batch[idx], r_batch[idx]
        for j in range(p):
            if int(col_valid[j]) < 10:
                continue
            out[j] = distance_correlation(f_batch[:, j], r_batch)
        return {"nonlinear_ic": out}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        series = daily.get("nonlinear_ic")
        if series is not None:
            ctx.intermediates["daily_nonlinear_ic"] = series
        return result_from_daily(daily, primary="nonlinear_ic", extra_scalars={"horizon": ctx.horizon})


class MutualInfoICMetric(BaseMetric):
    name = "mi_ic"
    dimension = "预测力"
    description = "等频分箱互信息，归一化为 NMI = MI/√(H_f·H_r)"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "互信息IC均值", "日度归一化互信息的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_bins = int(params.get("n_bins", 30))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        out = np.full(p, np.nan, dtype=np.float64)
        if r is None:
            return {"mi_ic": out}
        valid_r = np.isfinite(r)
        for j in range(p):
            sel = valid_r & np.isfinite(f[:, j])
            nv = int(sel.sum())
            if nv < n_bins * 2:
                continue
            fv, rv = f[sel, j], r[sel]
            f_bins = quantile_bin(fv, n_bins)
            r_bins = quantile_bin(rv, n_bins)
            joint = np.bincount(f_bins * n_bins + r_bins, minlength=n_bins * n_bins).reshape(
                n_bins, n_bins
            ).astype(np.float64) / nv
            marg_f = joint.sum(axis=1)
            marg_r = joint.sum(axis=0)
            outer = marg_f[:, None] * marg_r[None, :]
            nz = (joint > 0) & (outer > 0)
            mi = float(np.sum(joint[nz] * np.log(joint[nz] / outer[nz])))
            h_f = -float(np.sum(marg_f[marg_f > 0] * np.log(marg_f[marg_f > 0])))
            h_r = -float(np.sum(marg_r[marg_r > 0] * np.log(marg_r[marg_r > 0])))
            den = np.sqrt(h_f * h_r)
            out[j] = mi / den if den > 0 else np.nan
        return {"mi_ic": out}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="mi_ic", extra_scalars={"horizon": ctx.horizon})


class WeightedICMetric(BaseMetric):
    name = "weighted_ic"
    dimension = "预测力"
    description = "用 RankIC 过去 250 日（不含当天）均值判方向后，对 Pearson IC 翻转符号"
    cost = "derived"
    produces = ()
    requires = ("daily_ic", "daily_rank_ic")

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "方向调整IC均值", "按长窗 RankIC 符号对齐后的 Pearson IC 均值"),)

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        ic = ctx.intermediates.get("daily_ic")
        rank_ic = ctx.intermediates.get("daily_rank_ic")
        if ic is None or rank_ic is None:
            raise ValueError("weighted_ic 需要 daily_ic 与 daily_rank_ic")
        ic = pd.Series(ic)
        rank_ic = pd.Series(rank_ic)
        direction = pd.Series(
            auto_direction_signs(rank_ic.to_numpy(dtype=np.float64), lookback=250, min_periods=20),
            index=rank_ic.index,
        )
        aligned = ic * direction
        summary = summarize_daily(aligned)
        return MetricResult(
            scalars={**summary, "horizon": ctx.horizon},
            series={"weighted_ic": aligned.astype("float64")},
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rank = arrays.get("rank_ic")
        ic = arrays.get("ic")
        if rank is None or ic is None:
            return {}
        rank = np.asarray(rank, dtype=np.float64)
        ic = np.asarray(ic, dtype=np.float64)
        if rank.ndim == 1:
            direction = auto_direction_signs(rank, lookback=250, min_periods=20)
            return {"weighted_ic": ic * direction}
        direction = np.column_stack(
            [auto_direction_signs(rank[:, j], lookback=250, min_periods=20) for j in range(rank.shape[1])]
        )
        return {"weighted_ic": ic * direction}


class ICDecayMetric(BaseMetric):
    name = "ic_decay"
    dimension = "有效期"
    description = (
        "Corr(Rank(f_t), Rank(r_{t+lag}))，r 与主 IC 同一套收益标签；"
        "lag=1/2/3/4/5/10/20/40/60 交易日"
    )
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, MIN_OBS_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return tuple(
            FieldDoc(
                f"mean_{n}d",
                "有效期",
                f"RankIC lag {n}d",
                f"因子当日与滞后 {n} 个交易日收益标签的 RankIC 均值",
            )
            for n in DECAY_HORIZONS
        )

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        min_obs = int(params.get("min_obs", 10))
        factor = ctx.masked_factor()
        fwd = ctx.fwd_ret.reindex(index=factor.index, columns=factor.columns)
        mask = None
        if ctx.universe_mask is not None:
            mask = ctx.universe_mask.reindex(index=factor.index, columns=factor.columns)
        series: Dict[str, pd.Series] = {}
        scalars: Dict[str, Any] = {"horizon": ctx.horizon}
        from ..cross_section_ic_calculator import CrossSectionICCalculator

        calc = CrossSectionICCalculator()
        for n in DECAY_HORIZONS:
            aligned = shift_return_panel(fwd, n)
            if mask is not None:
                aligned = aligned.where(mask)
            daily = calc.daily_rank_ic(factor, aligned, min_obs=min_obs)
            series[f"ic_decay_{n}d"] = daily
            summary = summarize_daily(daily)
            scalars[f"mean_{n}d"] = summary["mean"]
        if series:
            first = next(iter(series.values()))
            scalars.update({k: v for k, v in summarize_daily(first).items() if k != "mean"})
        return MetricResult(scalars=scalars, series=series)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        p = f.shape[1]
        horizons = tuple(
            int(n) for n in (batch_ctx.intermediates.get("decay_horizons") or DECAY_HORIZONS)
        )
        out = {f"ic_decay_{n}d": np.full(p, np.nan, dtype=np.float64) for n in horizons}
        fwd_decay = batch_ctx.intermediates.get("fwd_decay")
        if fwd_decay is None:
            return out
        min_obs = int(params.get("min_obs", batch_ctx.min_obs))
        stacked = np.asarray(fwd_decay, dtype=np.float64)
        for hi, n in enumerate(horizons):
            if hi >= stacked.shape[0]:
                continue
            out[f"ic_decay_{n}d"] = rank_ic_pairwise(f, stacked[hi], min_obs=min_obs)
        return out