#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按日计算核：主进程与 worker 共用，不在这里读 bin。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import warnings

import numpy as np

from ..context import BatchEvalContext
from ..exposure_engine import RAW_STYLE_NAMES
from ..metric_discoverer import MetricDiscoverer
from ..matrix_utils import assign_quantiles, rank_cols, spearman_corr_matrix
from ..metrics.statistics_metrics import AUTOCORR_LAGS
from .shared_mem import attach_pack, close_attached

_STATE: Dict[str, Any] = {}


def evaluate_date_range(
    *,
    cube: np.ndarray,
    mask: np.ndarray,
    fwd: np.ndarray,
    fwd_decay: Optional[np.ndarray],
    decay_horizons: Sequence[int],
    style: Optional[np.ndarray],
    x_t: Optional[np.ndarray],
    x_names: Sequence[str],
    out: np.ndarray,
    labels: Optional[np.ndarray],
    key_index: Mapping[str, int],
    metrics: Sequence,
    params: Mapping[str, Any],
    t0: int,
    t1: int,
    min_obs: int,
    n_quantiles: int,
    size_col: int,
    need_daily_corr: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """写入 out[:, t0:t1, :] 与 labels[t0:t1]。返回本批因子相关累加与指标异常。"""
    n_f = cube.shape[2]
    corr_sum = np.zeros((n_f, n_f), dtype=np.float64)
    corr_n = np.zeros((n_f, n_f), dtype=np.float64)
    errors: list = []
    need_corr = bool(need_daily_corr) and n_f >= 2
    need_ranks = any(getattr(m, "name", "") == "factor_autocorr" for m in metrics)
    max_lag = max(AUTOCORR_LAGS) if need_ranks else 0
    rank_ring = _warmup_rank_ring(cube, mask, t0, max_lag) if need_ranks else None
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in range(t0, t1):
            ranks_full = _full_day_ranks(cube, mask, t) if need_ranks else None
            row_mask = mask[t]
            n_valid = int(row_mask.sum())
            if n_valid >= min_obs:
                f = cube[t, row_mask, :].astype(np.float64, copy=False)
                r = fwd[t, row_mask].astype(np.float64, copy=False)
                batch = BatchEvalContext(
                    factors=f,
                    returns=r,
                    mask=np.ones(n_valid, dtype=bool),
                    barra=_style_day(style, t, row_mask),
                    size=_size_day(style, t, row_mask, size_col),
                    n_quantiles=n_quantiles,
                    min_obs=min_obs,
                )
                if x_t is not None:
                    batch.intermediates["x_t"] = x_t[t, row_mask, :].astype(np.float64, copy=False)
                    batch.intermediates["x_names"] = list(x_names)
                if ranks_full is not None and rank_ring is not None:
                    batch.intermediates["factor_ranks_full"] = ranks_full
                    batch.intermediates["rank_ring"] = rank_ring
                q_labels = assign_quantiles(f, n_quantiles)
                batch.intermediates["quantile_labels"] = q_labels
                if fwd_decay is not None:
                    batch.intermediates["fwd_decay"] = fwd_decay[:, t, row_mask].astype(
                        np.float64, copy=False
                    )
                    batch.intermediates["decay_horizons"] = tuple(int(h) for h in decay_horizons)
                if labels is not None:
                    labels[t, row_mask, :] = q_labels.astype(np.int8, copy=False)
                for metric in metrics:
                    try:
                        day = metric.compute_matrix(batch, params)
                    except Exception as exc:
                        name = metric.get_name() if hasattr(metric, "get_name") else type(metric).__name__
                        if len(errors) < 200:
                            errors.append(
                                {
                                    "metric": str(name),
                                    "t": int(t),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        continue
                    for key, vec in day.items():
                        idx = key_index.get(key)
                        if idx is None:
                            continue
                        values = np.asarray(vec, dtype=np.float32).reshape(-1)
                        n = min(values.size, n_f)
                        out[idx, t, :n] = values[:n]
                if need_corr:
                    corr = spearman_corr_matrix(f, min_obs=min_obs)
                    finite = np.isfinite(corr)
                    corr_sum = np.where(finite, corr_sum + corr, corr_sum)
                    corr_n += finite.astype(np.float64)
            if rank_ring is not None and ranks_full is not None:
                rank_ring = np.concatenate([ranks_full[None], rank_ring[:-1]], axis=0)
    return corr_sum, corr_n, errors


def init_worker_panel(spec: Dict[str, Any]) -> None:
    """跨因子批次复用：只 attach 面板并解析 metrics。"""
    from ..repo import ensure_repo_root

    ensure_repo_root()
    attached = attach_pack(spec["panel"])
    _STATE["panel_views"] = {key: item.array for key, item in attached.items()}
    _STATE["panel_attached"] = attached
    discoverer = MetricDiscoverer()
    ordered = discoverer.resolve(spec["metric_names"], eval_scope="single")
    _STATE["metrics"] = [m for m in ordered if m.has_matrix()]
    _STATE["base_spec"] = spec


def run_chunk_batch(task: Tuple[Tuple[int, int], Dict[str, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """跨批复用 worker：每任务 attach 当批 cube/out/labels。"""
    bounds, batch_meta, task_spec = task
    t0, t1 = bounds
    batch_attached = attach_pack(batch_meta)
    try:
        views = dict(_STATE["panel_views"])
        views.update({key: item.array for key, item in batch_attached.items()})
        return _evaluate_with_views(views, task_spec, t0, t1)
    finally:
        close_attached(batch_attached)


def _evaluate_with_views(
    views: Dict[str, np.ndarray],
    spec: Dict[str, Any],
    t0: int,
    t1: int,
) -> Dict[str, Any]:
    metrics = _STATE.get("metrics") or []
    corr_sum, corr_n, errors = evaluate_date_range(
        cube=views["cube"],
        mask=views["mask"],
        fwd=views["fwd"],
        fwd_decay=views.get("fwd_decay") if spec.get("has_decay") else None,
        decay_horizons=spec.get("decay_horizons") or (),
        style=views.get("style") if spec.get("has_style") else None,
        x_t=views.get("x_t") if spec.get("has_xt") else None,
        x_names=spec.get("x_names") or (),
        out=views["out"],
        labels=views.get("labels") if spec.get("has_labels") else None,
        key_index=spec["key_index"],
        metrics=metrics,
        params=spec["params"],
        t0=t0,
        t1=t1,
        min_obs=int(spec["min_obs"]),
        n_quantiles=int(spec["n_quantiles"]),
        size_col=int(spec.get("size_col", 0)),
        need_daily_corr=bool(spec.get("need_daily_corr")),
    )
    return {
        "t0": t0,
        "t1": t1,
        "corr_sum": corr_sum,
        "corr_n": corr_n,
        "errors": errors,
    }


def shutdown_worker() -> None:
    attached = _STATE.pop("panel_attached", None)
    if attached:
        close_attached(attached)
    _STATE.clear()


def _style_day(style: Optional[np.ndarray], t: int, row_mask: np.ndarray) -> Optional[np.ndarray]:
    if style is None:
        return None
    n_styles = style.shape[2]
    wanted = len(RAW_STYLE_NAMES)
    cols = []
    for i in range(wanted):
        if i < n_styles:
            cols.append(style[t, row_mask, i].astype(np.float64, copy=False))
        else:
            cols.append(np.full(int(row_mask.sum()), np.nan, dtype=np.float64))
    return np.column_stack(cols)


def _size_day(
    style: Optional[np.ndarray], t: int, row_mask: np.ndarray, size_col: int
) -> Optional[np.ndarray]:
    if style is None or size_col < 0 or size_col >= style.shape[2]:
        return None
    return style[t, row_mask, size_col].astype(np.float64, copy=False)


def _full_day_ranks(cube: np.ndarray, mask: np.ndarray, t: int) -> np.ndarray:
    values = cube[t].astype(np.float64, copy=True)
    values[~mask[t]] = np.nan
    return rank_cols(values)


def _warmup_rank_ring(
    cube: np.ndarray, mask: np.ndarray, t0: int, max_lag: int
) -> np.ndarray:
    n_stocks, n_f = cube.shape[1], cube.shape[2]
    ring = np.full((max_lag, n_stocks, n_f), np.nan, dtype=np.float64)
    if max_lag <= 0:
        return ring
    start = max(0, t0 - max_lag)
    hist = [_full_day_ranks(cube, mask, t) for t in range(start, t0)]
    for lag in range(1, max_lag + 1):
        if lag <= len(hist):
            ring[lag - 1] = hist[-lag]
    return ring
