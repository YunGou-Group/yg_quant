"""分层只按 T 日因子分桶，未来收益缺失不改分位边界。"""

import numpy as np

from FactorEvaluates.matrix_utils import assign_quantiles, assign_quantiles_on_returns


def test_missing_future_return_keeps_stock_in_quantile():
    factors = np.array([1.0, 2.0, 3.0, 4.0])
    returns = np.array([0.1, np.nan, 0.2, 0.3])
    labels = assign_quantiles(factors, 4)
    assert labels[1] != 0
    assert set(labels.tolist()) == {1, 2, 3, 4}
    masked = np.where(np.isfinite(factors) & np.isfinite(returns), factors, np.nan)
    leaked = assign_quantiles(masked, 4)
    assert leaked[1] == 0
    np.testing.assert_array_equal(assign_quantiles_on_returns(factors, returns, 4), labels)
