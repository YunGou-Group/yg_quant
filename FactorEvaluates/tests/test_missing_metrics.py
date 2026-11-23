#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新增完全缺失指标的冒烟测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from FactorEvaluates.context import BatchEvalContext, EvalContext
from FactorEvaluates.matrix_utils import expanding_icir, quality_stats
from FactorEvaluates.metrics.barra_quantile_metrics import (
    BarraQuantileCorrelationMetric,
    LongShortBarraExposureMetric,
    _resolve_long_short_q,
)
from FactorEvaluates.metrics.icir_metric import NonlinearICIRMetric
from FactorEvaluates.metrics.quantile_metric import (
    QuantileFactorMeanMetric,
    QuantileReturnsAfterCostMetric,
    QuantileTurnoverMetric,
)
from FactorEvaluates.metrics.quality_metrics import MissingSamplesMetric, TotalSamplesMetric
from FactorEvaluates.metrics.statistics_metrics import FactorMaxMetric, FactorMinMetric


def _panel(n_dates: int = 8, n_stocks: int = 40, seed: int = 0):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    cols = [f"{i:06d}.SH" for i in range(n_stocks)]
    factor = pd.DataFrame(rng.randn(n_dates, n_stocks), index=dates, columns=cols)
    fwd = pd.DataFrame(rng.randn(n_dates, n_stocks) * 0.01, index=dates, columns=cols)
    return factor, fwd


def test_quality_stats_has_missing_and_total():
    f = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
    stats = quality_stats(f)
    assert "missing_samples" in stats
    assert np.allclose(stats["missing_samples"], [1.0, 1.0])
    assert np.allclose(stats["total_samples"], [3.0, 3.0])


def test_factor_min_max_matrix():
    f = np.array([[1.0, 5.0], [3.0, 2.0], [np.nan, 4.0]])
    ctx = BatchEvalContext(factors=f, mask=np.ones(3, dtype=bool))
    assert np.allclose(FactorMinMetric().compute_matrix(ctx, {})["factor_min"], [1.0, 2.0])
    assert np.allclose(FactorMaxMetric().compute_matrix(ctx, {})["factor_max"], [3.0, 5.0])


def test_missing_total_samples_matrix():
    f = np.array([[1.0, np.nan], [2.0, 3.0]])
    ctx = BatchEvalContext(factors=f, mask=np.ones(2, dtype=bool))
    assert np.allclose(
        MissingSamplesMetric().compute_matrix(ctx, {})["missing_samples"], [0.0, 1.0]
    )
    assert np.allclose(
        TotalSamplesMetric().compute_matrix(ctx, {})["total_samples"], [2.0, 2.0]
    )


def test_quantile_factor_mean_matrix():
    f = np.arange(20, dtype=np.float64).reshape(20, 1)
    ctx = BatchEvalContext(factors=f, n_quantiles=5, mask=np.ones(20, dtype=bool))
    out = QuantileFactorMeanMetric().compute_matrix(ctx, {"n_quantiles": 5})
    assert "quantile_factor_mean_Q1" in out
    assert np.isfinite(out["quantile_factor_mean_Q5"][0])


def test_after_cost_crossday():
    n_dates, n_f = 5, 2
    arrays = {
        "quantile_returns_Q1": np.full((n_dates, n_f), 0.01),
        "quantile_turnover_Q1": np.full((n_dates, n_f), 0.5),
    }
    arrays["quantile_turnover_Q1"][0] = np.nan
    out = QuantileReturnsAfterCostMetric().compute_crossday(
        arrays, {"n_quantiles": 1, "transaction_cost_bps": 10.0, "transaction_buy_rate": 0.0003}
    )
    net = out["quantile_returns_after_cost_Q1"]
    assert np.allclose(net[0], 0.01 - 0.0003)
    assert np.allclose(net[1], 0.01 - 0.5 * 0.001)


def test_expanding_icir_and_nonlinear_crossday():
    daily = np.array([[0.1, np.nan], [0.2, 0.1], [0.0, 0.2], [0.3, 0.0]])
    ir = expanding_icir(daily)
    assert ir.shape == daily.shape
    assert np.isnan(ir[0, 0])
    assert np.isfinite(ir[-1, 0])
    out = NonlinearICIRMetric().compute_crossday({"nonlinear_ic": daily}, {})
    assert out["nonlinear_ic_ir"].shape == daily.shape


def test_resolve_long_short_linear():
    arrays = {"rank_ic": np.array([[0.1, -0.2], [0.2, -0.1]])}
    long_q, short_q = _resolve_long_short_q(arrays, n_q=5, n_factors=2, direction="linear")
    assert list(long_q) == [5, 1]
    assert list(short_q) == [1, 5]


def test_barra_exposure_matrix_empty_style():
    f = np.random.randn(30, 1)
    ctx = BatchEvalContext(factors=f, barra=None, n_quantiles=5, mask=np.ones(30, dtype=bool))
    out = LongShortBarraExposureMetric().compute_matrix(ctx, {"n_quantiles": 5})
    assert "barra_exposure_Q1_style_size" in out
    assert np.all(np.isnan(out["barra_exposure_Q1_style_size"]))


def test_barra_corr_with_style():
    rng = np.random.RandomState(1)
    f = rng.randn(50, 1)
    b = np.column_stack([f[:, 0] + rng.randn(50) * 0.1] + [rng.randn(50) for _ in range(8)])
    ctx = BatchEvalContext(factors=f, barra=b, n_quantiles=5, mask=np.ones(50, dtype=bool))
    out = BarraQuantileCorrelationMetric().compute_matrix(ctx, {"n_quantiles": 5, "min_obs": 5})
    assert any(np.isfinite(out[k]).any() for k in out if k.startswith("barra_corr_Q"))


def test_turnover_produces_marker():
    factor, _ = _panel()
    labels = pd.DataFrame(
        np.tile(np.arange(1, 6), (len(factor), factor.shape[1] // 5 + 1))[:, : factor.shape[1]],
        index=factor.index,
        columns=factor.columns,
        dtype=float,
    )
    ctx = EvalContext(
        factor=factor,
        fwd_ret=factor * 0,
        universe_mask=None,
        horizon=5,
        label="t",
        asof="t",
        intermediates={"quantile_labels": labels},
    )
    result = QuantileTurnoverMetric().compute(ctx, {"n_quantiles": 5})
    assert "quantile_turnover" in ctx.intermediates
    assert "quantile_turnover_Q5" in result.series


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
