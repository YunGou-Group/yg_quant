#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prior-Z 与因果分位；复用 FactorEvaluates.matrix_utils。"""

from __future__ import annotations

import pandas as pd

from FactorEvaluates.matrix_utils import causal_percentile, prior_rolling_zscore

__all__ = ["prior_rolling_zscore", "causal_percentile", "rolling_mean"]


def rolling_mean(series: pd.Series, window: int, min_observations: int) -> pd.Series:
    out = series.rolling(window, min_periods=min_observations).mean()
    out.iloc[: max(window - 1, 0)] = float("nan")
    return out
