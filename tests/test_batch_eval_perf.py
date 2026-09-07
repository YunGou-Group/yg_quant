#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import numpy as np
import pytest

from FactorEvaluates.batch.batch_eval_runner import BatchEvalRunner
from FactorEvaluates.matrix_utils import rank_cols


def _legacy_rank_cols(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    n, p = values.shape
    for j in range(p):
        col = values[:, j]
        valid = np.isfinite(col)
        m = int(valid.sum())
        if m < 2:
            continue
        vals = col[valid]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(m, dtype=np.float64)
        ranks[order] = np.arange(1, m + 1, dtype=np.float64)
        sorted_vals = vals[order]
        i = 0
        while i < m:
            k = i + 1
            while k < m and sorted_vals[k] == sorted_vals[i]:
                k += 1
            if k - i > 1:
                ranks[order[i:k]] = 0.5 * ((i + 1) + k)
            i = k
        out[valid, j] = ranks
    return out[:, 0] if squeeze else out


def test_rank_cols_matches_legacy():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(200, 8))
    mat[rng.random((200, 8)) < 0.05] = np.nan
    mat[10:15, 2] = 1.0
    new = rank_cols(mat)
    old = _legacy_rank_cols(mat)
    assert np.allclose(new, old, equal_nan=True)


def test_plan_metrics_need_daily_corr():
    planned = BatchEvalRunner.plan_metrics(BatchEvalRunner().discoverer, None)
    assert planned["need_daily_corr"] is True
    planned2 = BatchEvalRunner.plan_metrics(
        BatchEvalRunner().discoverer,
        ["ic", "rank_ic"],
    )
    assert planned2["need_daily_corr"] is False


def test_fit_workers_respects_explicit_request():
    class _Store:
        dates = ["d"] * 4000
        symbols = ["s"] * 5000
        fwd = {5: None}
        style = {f"s{i}": None for i in range(9)}
        x_t = None

    # 显式指定不再被内存预算压掉
    assert BatchEvalRunner._fit_workers(_Store(), batch_size=32, requested=12) == 12


def test_fit_workers_auto_respects_ram_budget():
    class _Store:
        dates = ["d"] * 4000
        symbols = ["s"] * 5000
        fwd = {5: None}
        style = {f"s{i}": None for i in range(9)}
        x_t = np.zeros((4000, 5000, 40), dtype=np.float32)

    workers = BatchEvalRunner._fit_workers(_Store(), batch_size=32, requested=0)
    assert 1 <= workers <= (os.cpu_count() or 4)
    # 自动模式在带 X_T 时不应默认拉满
    assert workers <= 16


def test_available_ram_bytes_positive():
    assert BatchEvalRunner._available_ram_bytes() > 1024 ** 3


def test_resolve_metrics_skips_xt_by_default():
    from FactorEvaluates.run_batch import XT_METRICS, resolve_metrics

    names, skipped = resolve_metrics(xt=False)
    assert not any(name in XT_METRICS for name in names)
    assert set(skipped) == set(XT_METRICS)
    # 默认仍应包含库级指标
    assert "factor_corr" in names


def test_resolve_metrics_with_xt_keeps_exposure():
    from FactorEvaluates.run_batch import resolve_metrics

    names, skipped = resolve_metrics(
        metrics="ic,exposure,attribution",
        xt=True,
    )
    assert names == ["ic", "exposure", "attribution"]
    assert skipped == []


def test_plan_metrics_accepts_library_names_in_selected():
    runner = BatchEvalRunner()
    planned = BatchEvalRunner.plan_metrics(
        runner.discoverer,
        ["ic", "rank_ic", "factor_corr", "exposure"],
        allow_xt=True,
    )
    assert {m.get_name() for m in planned["daily"]} >= {"ic", "rank_ic", "exposure"}
    assert {m.get_name() for m in planned["library"]} >= {"factor_corr"}
    assert planned["need_xt"] is True


def test_plan_metrics_allow_xt_false_strips_exposure():
    runner = BatchEvalRunner()
    planned = BatchEvalRunner.plan_metrics(
        runner.discoverer,
        ["ic", "exposure", "pure_ic", "attribution"],
        allow_xt=False,
    )
    names = {m.get_name() for m in planned["daily"]}
    assert "ic" in names
    assert not names & {"exposure", "pure_ic", "attribution"}
    assert planned["need_xt"] is False


def test_detach_store_from_shm_keeps_mask_after_unlink():
    """回归：close/unlink 后 store.mask 仍可索引，跨批相关才不会静默崩。"""
    from FactorEvaluates.batch.daily_matrix_engine import DailyMatrixEngine
    from FactorEvaluates.batch.shared_mem import ShmArray, ShmPack

    class _Store:
        def __init__(self):
            self.mask = np.ones((8, 5), dtype=bool)
            self.fwd = {5: np.zeros((8, 5), dtype=np.float32)}
            self.style = {"style_size": np.zeros((8, 5), dtype=np.float32)}
            self.x_t = np.zeros((8, 5, 2), dtype=np.float32)
            self.x_names = ["a", "b"]
            self.open = np.zeros((8, 5), dtype=np.float32)
            self.horizon = 5

    store = _Store()
    engine = DailyMatrixEngine.__new__(DailyMatrixEngine)
    engine.store = store
    engine._style = None
    engine._fwd_decay = None
    engine._decay_h = ()
    engine._pool = None
    pack = ShmPack()
    pack.add("mask", ShmArray.create(store.mask))
    store.mask = pack.items["mask"].array
    engine._panel_pack = pack
    engine.close()
    assert store.mask.shape == (8, 5)
    assert bool(store.mask[3, 2]) is True
    assert store.fwd == {}
    assert store.x_t is None
