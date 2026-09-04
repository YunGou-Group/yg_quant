#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多头最小方差：对齐 doc/optimize 的 sample_cov + EfficientFrontier.min_volatility。

窗口用 asof 前 252 个交易日（配置里的 minimum_observations / annualization_factor），
不把全样本未来收益算进协方差。μ 不进目标。单票上限默认 1；要对齐 doc 的
weight_bounds=(0, 0.2) 时显式传入 max_weight=0.2。
"""

from __future__ import annotations

import numpy as np

from ..context import DayContext
from .common import min_volatility_weights, sample_cov
from .window import WindowAllocator


class MinVol(WindowAllocator):
    label = "min_vol"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        cap = min(self.max_weight, 1.0)
        return min_volatility_weights(sample_cov(returns, self.frequency), cap)
