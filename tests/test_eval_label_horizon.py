#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评价标签：先在完整 open 上算远期收益再切窗；分位 spread 需齐全。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from FactorEvaluates.return_calculator import ReturnCalculator
from FactorEvaluates.batch.batch_eval_runner import BatchEvalRunner
from FactorEvaluates.context import BatchEvalContext
from FactorEvaluates.matrix_utils import assign_quantiles
from FactorEvaluates.metrics.ic_metric import RankICMetric
from FactorEvaluates.metrics.quantile_metric import QuantileMetric


def test_fwd_on_full_open_keeps_last_eval_day():
    """切窗后再算标签会丢掉末日；先算再切则末日仍有 T+n 开盘。"""
    dates = pd.bdate_range("2020-01-02", periods=12).strftime("%Y-%m-%d")
    open_px = pd.DataFrame(
        10.0 + np.arange(12)[:, None] + np.array([0.0, 0.1, 0.2]),
        index=dates,
        columns=["A", "B", "C"],
    )
    horizon = 2
    eval_idx = list(dates[:8])
    full_then_slice = (
        ReturnCalculator(open_px).get(horizon).reindex(eval_idx)
    )
    slice_then_calc = ReturnCalculator(open_px.loc[eval_idx]).get(horizon)
    assert np.isfinite(full_then_slice.iloc[-1].to_numpy()).all()
    assert not np.isfinite(slice_then_calc.iloc[-1].to_numpy()).any()
    n_full = int(np.isfinite(full_then_slice.iloc[:, 0]).sum())
    n_trunc = int(np.isfinite(slice_then_calc.iloc[:, 0]).sum())
    assert n_full == len(eval_idx)
    assert n_trunc < n_full


def test_fwd_return_clips_bad_ticks():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    open_px = pd.DataFrame({"A": [1.0, 1.0, 1000.0]}, index=dates)
    got = ReturnCalculator(open_px).get(1).loc["2020-01-02", "A"]
    assert got == 10.0


def test_weighted_pnl_is_cross_section_mean():
    from FactorEvaluates.metrics.return_risk_metrics import WeightedPnLMetric

    n = 12
    f = np.ones((n, 1), dtype=np.float64)
    r = np.full(n, 0.1, dtype=np.float64)
    batch = BatchEvalContext(
        factors=f,
        returns=r,
        mask=np.ones(n, dtype=bool),
        n_quantiles=5,
        min_obs=5,
    )
    out = WeightedPnLMetric().compute_matrix(batch, {})
    np.testing.assert_allclose(out["weighted_pnl"][0], 0.1)
    np.testing.assert_allclose(out["weighted_long_pnl"][0], 0.1)


def test_quantile_spread_requires_all_bins():
    n, p = 40, 2
    factors = np.zeros((n, p), dtype=np.float64)
    factors[:, 0] = np.linspace(-2, 2, n)
    factors[:, 1] = np.r_[np.zeros(35), np.ones(5)]
    returns = np.linspace(-0.1, 0.1, n)
    labels = assign_quantiles(factors, 5)
    batch = BatchEvalContext(
        factors=factors,
        returns=returns,
        mask=np.ones(n, dtype=bool),
        n_quantiles=5,
        min_obs=5,
    )
    batch.intermediates["quantile_labels"] = labels
    out = QuantileMetric().compute_matrix(
        batch, {"n_quantiles": 5, "min_stock_per_bin": 5}
    )
    assert np.isfinite(out["quantile_spread"][0])
    assert not np.isfinite(out["quantile_spread"][1])


def test_probe_keys_only_selected_metrics():
    keys = BatchEvalRunner._probe_keys([], {"min_obs": 20}, n_q=5)
    assert keys == set()
    keys = BatchEvalRunner._probe_keys(
        [RankICMetric()], {"min_obs": 20, "horizon": 5}, n_q=5
    )
    assert "rank_ic" in keys
    assert "quantile_spread" not in keys
    assert "ic" not in keys
    assert "coverage_rate" not in keys


def test_selected_metric_exception_is_recorded():
    from FactorEvaluates.batch.day_worker import evaluate_date_range

    class Boom:
        def get_name(self):
            return "boom"

        def compute_matrix(self, *_a, **_k):
            raise ValueError("metric exploded")

    n_dates, n_stocks, n_f = 2, 25, 1
    cube = np.ones((n_dates, n_stocks, n_f), dtype=np.float64)
    mask = np.ones((n_dates, n_stocks), dtype=bool)
    fwd = np.ones((n_dates, n_stocks), dtype=np.float64)
    out = np.full((1, n_dates, n_f), np.nan, dtype=np.float32)
    _, _, errors = evaluate_date_range(
        cube=cube,
        mask=mask,
        fwd=fwd,
        fwd_decay=None,
        decay_horizons=(),
        style=None,
        x_t=None,
        x_names=(),
        out=out,
        labels=None,
        key_index={"boom": 0},
        metrics=[Boom()],
        params={},
        t0=0,
        t1=n_dates,
        min_obs=20,
        n_quantiles=5,
        size_col=0,
    )
    assert errors
    assert errors[0]["metric"] == "boom"
    assert "exploded" in errors[0]["error"]
