#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""市值分层多空：等数量层 + 排名断点层。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM
from ..matrix_utils import apply_daily_matrix, assign_quantiles, result_from_daily

N_SIZE = 5
N_FACTOR_Q = 5
RANK_CUTS: Tuple[int, ...] = (300, 800, 1800, 3800)
MIN_COUNT = 10
MIN_LAYER_COUNT = 100


def _layer_ls(
    f_sub: np.ndarray, r_sub: np.ndarray, n_q: int, min_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = assign_quantiles(f_sub, n_q)
    p = f_sub.shape[1]
    long = np.full(p, np.nan)
    short = np.full(p, np.nan)
    long_c = np.zeros(p)
    short_c = np.zeros(p)
    valid_r = np.isfinite(r_sub)
    for j in range(p):
        hi = (q[:, j] == n_q) & valid_r
        lo = (q[:, j] == 1) & valid_r
        n_hi, n_lo = int(hi.sum()), int(lo.sum())
        if n_hi >= min_count:
            long[j] = float(np.mean(r_sub[hi]))
            long_c[j] = n_hi
        if n_lo >= min_count:
            short[j] = float(np.mean(r_sub[lo]))
            short_c[j] = n_lo
    return long, short, long_c, short_c


def _layer_equal_weight_mean(r_sub: np.ndarray) -> float:
    valid = np.isfinite(r_sub)
    if not valid.any():
        return float("nan")
    return float(np.mean(r_sub[valid]))


def _relative_to_weighted_benchmark(
    layer_returns: np.ndarray, layer_counts: np.ndarray
) -> np.ndarray:
    valid = ~np.isnan(layer_returns)
    weights = np.where(valid, layer_counts, 0.0)
    denom = weights.sum(axis=0)
    numer = np.where(valid, layer_returns * weights, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        benchmark = np.where(denom > 0, numer / denom, np.nan)
    return layer_returns - benchmark[None, :]


def _leg_vs_baseline(leg_layers: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return leg_layers - baseline[:, None]


def _emit_leg_pair(
    out: Dict[str, np.ndarray],
    long_mat: np.ndarray,
    short_mat: np.ndarray,
    long_prefix: str,
    short_prefix: str,
    token: str,
) -> None:
    n_layers = long_mat.shape[0]
    for i in range(n_layers):
        tag = f"{token}{i + 1}"
        out[f"{long_prefix}_{tag}"] = long_mat[i]
        out[f"{short_prefix}_{tag}"] = short_mat[i]


def _rank_cut_layer_labels(size: np.ndarray, cuts: Sequence[int]) -> np.ndarray:
    labels = np.zeros(size.shape[0], dtype=np.int32)
    valid = np.isfinite(size)
    nv = int(valid.sum())
    if nv == 0:
        return labels
    order = np.argsort(-size[valid], kind="stable")
    ranks = np.empty(nv, dtype=np.int64)
    ranks[order] = np.arange(1, nv + 1, dtype=np.int64)
    cuts_arr = np.asarray(list(cuts), dtype=np.int64)
    labels[valid] = np.searchsorted(cuts_arr, ranks, side="left").astype(np.int32) + 1
    return labels


class SizeStratifiedMetric(BaseMetric):
    name = "size_stratified_long_short"
    dimension = "预测力"
    description = "按 style_size 等数量分层，层内因子高低组多空，再跨层等权"
    cost = "panel"
    produces = ()
    requires = ()
    engine_requires = ("style_raw",)

    def params(self):
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self):
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
        size_q = assign_quantiles(size[:, None], N_SIZE).ravel()
        size_layer = np.where(size_q > 0, N_SIZE + 1 - size_q, 0)
        long_layers = np.full((N_SIZE, p), np.nan)
        short_layers = np.full((N_SIZE, p), np.nan)
        long_counts = np.zeros((N_SIZE, p))
        short_counts = np.zeros((N_SIZE, p))
        layer_means = np.full(N_SIZE, np.nan)
        for s in range(1, N_SIZE + 1):
            sel = size_layer == s
            if int(sel.sum()) >= N_FACTOR_Q:
                long, short, long_c, short_c = _layer_ls(f[sel], r[sel], N_FACTOR_Q, MIN_COUNT)
                layer_means[s - 1] = _layer_equal_weight_mean(r[sel])
            else:
                long = nan.copy()
                short = nan.copy()
                long_c = np.zeros(p)
                short_c = np.zeros(p)
            long_layers[s - 1] = long
            short_layers[s - 1] = short
            long_counts[s - 1] = long_c
            short_counts[s - 1] = short_c
            out[f"size_long_S{s}"] = long
            out[f"size_short_S{s}"] = short
            out[f"size_ls_S{s}"] = long - short
        ls_layers = long_layers - short_layers
        with np.errstate(all="ignore"):
            out["size_long"] = np.nanmean(long_layers, axis=0)
            out["size_short"] = np.nanmean(short_layers, axis=0)
            out["size_ls"] = np.nanmean(ls_layers, axis=0)
        _emit_leg_pair(
            out,
            _relative_to_weighted_benchmark(long_layers, long_counts),
            _relative_to_weighted_benchmark(short_layers, short_counts),
            "size_long_rel",
            "size_short_rel",
            "S",
        )
        _emit_leg_pair(
            out,
            _leg_vs_baseline(long_layers, layer_means),
            _leg_vs_baseline(short_layers, layer_means),
            "size_long_vs_layer_mean",
            "size_short_vs_layer_mean",
            "S",
        )
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

    def params(self):
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self):
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
        layer = _rank_cut_layer_labels(size, RANK_CUTS)
        long_layers = np.full((n_layers, p), np.nan)
        short_layers = np.full((n_layers, p), np.nan)
        long_counts = np.zeros((n_layers, p))
        short_counts = np.zeros((n_layers, p))
        layer_means = np.full(n_layers, np.nan)
        for s in range(1, n_layers + 1):
            sel = layer == s
            if int(sel.sum()) >= max(N_FACTOR_Q, MIN_LAYER_COUNT):
                long, short, long_c, short_c = _layer_ls(f[sel], r[sel], N_FACTOR_Q, MIN_COUNT)
                layer_means[s - 1] = _layer_equal_weight_mean(r[sel])
            else:
                long = nan.copy()
                short = nan.copy()
                long_c = np.zeros(p)
                short_c = np.zeros(p)
            long_layers[s - 1] = long
            short_layers[s - 1] = short
            long_counts[s - 1] = long_c
            short_counts[s - 1] = short_c
            out[f"size_rank_long_R{s}"] = long
            out[f"size_rank_short_R{s}"] = short
            out[f"size_rank_ls_R{s}"] = long - short
        ls_layers = long_layers - short_layers
        with np.errstate(all="ignore"):
            out["size_rank_long"] = np.nanmean(long_layers, axis=0)
            out["size_rank_short"] = np.nanmean(short_layers, axis=0)
            out["size_rank_ls"] = np.nanmean(ls_layers, axis=0)
        _emit_leg_pair(
            out,
            _relative_to_weighted_benchmark(long_layers, long_counts),
            _relative_to_weighted_benchmark(short_layers, short_counts),
            "size_rank_long_rel",
            "size_rank_short_rel",
            "R",
        )
        _emit_leg_pair(
            out,
            _leg_vs_baseline(long_layers, layer_means),
            _leg_vs_baseline(short_layers, layer_means),
            "size_rank_long_vs_layer_mean",
            "size_rank_short_vs_layer_mean",
            "R",
        )
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="size_rank_ls", extra_scalars={"horizon": ctx.horizon})
