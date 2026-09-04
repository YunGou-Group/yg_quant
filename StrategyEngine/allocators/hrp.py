#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hierarchical Risk Parity（Lopez de Prado / pypfopt HRPOpt.optimize）。

不需要 cvxpy；单票上限在求解后再收口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DayContext
from .common import pack_weights
from .window import WindowAllocator


def hrp_weights(returns: np.ndarray, cap: float) -> np.ndarray:
    try:
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd
    except ImportError as exc:
        raise ImportError("hrp 需要 scipy，请在当前环境执行: python -m pip install scipy") from exc

    df = pd.DataFrame(np.asarray(returns, dtype=np.float64))
    corr = df.corr()
    cov = df.cov()
    matrix = np.sqrt(np.clip((1.0 - corr.to_numpy(dtype=np.float64)) / 2.0, 0.0, 1.0))
    dist = ssd.squareform(matrix, checks=False)
    link = sch.linkage(dist, method="single")
    order = sch.to_tree(link, rd=False).pre_order()
    ordered = corr.index[order].tolist()
    raw = _raw_hrp_allocation(cov, ordered)
    packed = raw.reindex(corr.index).to_numpy(dtype=np.float64)
    return pack_weights(packed, cap, name="hrp")


def _raw_hrp_allocation(cov: pd.DataFrame, ordered_tickers: list) -> pd.Series:
    w = pd.Series(1.0, index=ordered_tickers)
    cluster_items = [ordered_tickers]
    while cluster_items:
        cluster_items = [
            i[j:k]
            for i in cluster_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(cluster_items), 2):
            first = cluster_items[i]
            second = cluster_items[i + 1]
            first_var = _cluster_var(cov, first)
            second_var = _cluster_var(cov, second)
            alpha = 1.0 - first_var / (first_var + second_var)
            w[first] *= alpha
            w[second] *= 1.0 - alpha
    return w


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    slice_ = cov.loc[items, items]
    weights = 1.0 / np.diag(slice_)
    weights = weights / weights.sum()
    return float(np.linalg.multi_dot((weights, slice_, weights)))


class HRP(WindowAllocator):
    label = "hrp"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        return hrp_weights(returns, min(self.max_weight, 1.0))
