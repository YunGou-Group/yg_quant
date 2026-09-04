#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

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
