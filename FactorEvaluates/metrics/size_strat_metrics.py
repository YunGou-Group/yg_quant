#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""市值分层多空：等数量层 + 排名断点层。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple
import warnings

import numpy as np

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import apply_daily_matrix, assign_quantiles, result_from_daily, winsorize_1d

N_SIZE = 5
N_FACTOR_Q = 5
RANK_CUTS: Tuple[int, ...] = (300, 800, 1800, 3800)
MIN_COUNT = 5


def _layer_ls(f_sub: np.ndarray, r_sub: np.ndarray, n_q: int) -> Tuple[np.ndarray, np.ndarray]:
    q = assign_quantiles(f_sub, n_q)
    p = f_sub.shape[1]
    long = np.full(p, np.nan)
    short = np.full(p, np.nan)
    valid_r = np.isfinite(r_sub)
    for j in range(p):
        hi = (q[:, j] == n_q) & valid_r
        lo = (q[:, j] == 1) & valid_r
        if int(hi.sum()) >= MIN_COUNT:
            long[j] = float(np.mean(r_sub[hi]))
        if int(lo.sum()) >= MIN_COUNT:
            short[j] = float(np.mean(r_sub[lo]))
    return long, short


def _mean_layers(rows: Sequence[np.ndarray], n: int) -> np.ndarray:
    if not rows:
        return np.full(n, np.nan)
    stacked = np.vstack(rows)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(stacked, axis=0)
    return np.asarray(out, dtype=np.float64)


class SizeStratifiedMetric(BaseMetric):
    name = "size_stratified_long_short"
    dimension = "预测力"
    description = "按 style_size 等数量分层，层内因子高低组多空，再跨层等权"
    cost = "panel"
    produces = ()
    requires = ()
    engine_requires = ("style_raw",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "市值中性多空", "跨层等权 size_ls 均值"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        size = batch_ctx.masked_size()
        p = f.shape[1]
        out: Dict[str, np.ndarray] = {}
        nan = np.full(p, np.nan)
        for s in range(1, N_SIZE + 1):
            out[f"size_long_S{s}"] = nan.copy()
            out[f"size_short_S{s}"] = nan.copy()
            out[f"size_ls_S{s}"] = nan.copy()
        out["size_long"] = nan.copy()
        out["size_short"] = nan.copy()
        out["size_ls"] = nan.copy()
        if r is None or size is None:
            return out
        r = winsorize_1d(r)
        size_q = assign_quantiles((-size)[:, None], N_SIZE)[:, 0]  # S1 = 最大市值
        longs, shorts = [], []
        for s in range(1, N_SIZE + 1):
            sel = size_q == s
            if int(sel.sum()) < N_FACTOR_Q * MIN_COUNT:
                continue
            long, short = _layer_ls(f[sel], r[sel], N_FACTOR_Q)
            out[f"size_long_S{s}"] = long
            out[f"size_short_S{s}"] = short
            out[f"size_ls_S{s}"] = long - short
            longs.append(long)
            shorts.append(short)
        if longs:
            out["size_long"] = _mean_layers(longs, p)
            out["size_short"] = _mean_layers(shorts, p)
            out["size_ls"] = out["size_long"] - out["size_short"]
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="size_ls", extra_scalars={"horizon": ctx.horizon})


class SizeRankCutMetric(BaseMetric):
    name = "size_rank_cut_long_short"
    dimension = "预测力"
    description = "按 style_size 降序排名断点分层（Top300/800/…）后层内多空"
    cost = "panel"
    produces = ()
    requires = ()
    engine_requires = ("style_raw",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "排名分层多空", "跨层等权 size_rank_ls 均值"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        size = batch_ctx.masked_size()
        p = f.shape[1]
        n_layers = len(RANK_CUTS) + 1
        out: Dict[str, np.ndarray] = {}
        nan = np.full(p, np.nan)
        for i in range(1, n_layers + 1):
            out[f"size_rank_long_R{i}"] = nan.copy()
            out[f"size_rank_short_R{i}"] = nan.copy()
            out[f"size_rank_ls_R{i}"] = nan.copy()
        out["size_rank_long"] = nan.copy()
        out["size_rank_short"] = nan.copy()
        out["size_rank_ls"] = nan.copy()
        if r is None or size is None:
            return out
        order = np.argsort(-np.where(np.isfinite(size), size, -np.inf))
        ranks = np.empty(size.shape[0], dtype=np.int32)
        ranks[order] = np.arange(1, size.shape[0] + 1)
        bounds = (0,) + RANK_CUTS + (10 ** 9,)
        longs, shorts = [], []
        for i in range(n_layers):
            lo, hi = bounds[i], bounds[i + 1]
            sel = (ranks > lo) & (ranks <= hi) & np.isfinite(size)
            if int(sel.sum()) < N_FACTOR_Q * MIN_COUNT:
                continue
            long, short = _layer_ls(f[sel], r[sel], N_FACTOR_Q)
            out[f"size_rank_long_R{i + 1}"] = long
            out[f"size_rank_short_R{i + 1}"] = short
            out[f"size_rank_ls_R{i + 1}"] = long - short
            longs.append(long)
            shorts.append(short)
        if longs:
            out["size_rank_long"] = _mean_layers(longs, p)
            out["size_rank_short"] = _mean_layers(shorts, p)
            out["size_rank_ls"] = out["size_rank_long"] - out["size_rank_short"]
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="size_rank_ls", extra_scalars={"horizon": ctx.horizon})