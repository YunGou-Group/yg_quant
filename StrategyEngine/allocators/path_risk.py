#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径风险：EfficientSemivariance / EfficientCVaR / EfficientCDaR 的最小化目标。

日频每个调仓日对当前候选解一次；efficient_return / efficient_risk 变体不单独挂名，
需要目标收益/风险时用均值方差那一组。
"""

from __future__ import annotations

import numpy as np

from ..context import DayContext
from .common import _EPS, pack_weights, require_cvxpy
from .window import WindowAllocator


def _long_only(cp, n: int, cap: float):
    if cap * n + _EPS < 1.0:
        raise ValueError("weight_bounds 使问题不可行")
    w = cp.Variable(n)
    return w, [cp.sum(w) == 1, w >= 0, w <= cap]


def min_semivariance_weights(returns: np.ndarray, cap: float, benchmark: float = 0.0) -> np.ndarray:
    """EfficientSemivariance.min_semivariance。"""
    cp = require_cvxpy("min_semivariance")
    r = np.asarray(returns, dtype=np.float64)
    t, n = r.shape
    w, cons = _long_only(cp, n, cap)
    p = cp.Variable(t, nonneg=True)
    dn = cp.Variable(t, nonneg=True)
    b = (r - float(benchmark)) / np.sqrt(t)
    cons.append(b @ w - p + dn == 0)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(dn)), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"min_semivariance 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="min_semivariance")


def min_cvar_weights(returns: np.ndarray, cap: float, beta: float) -> np.ndarray:
    """Rockafellar–Uryasev：EfficientCVaR.min_cvar。"""
    cp = require_cvxpy("min_cvar")
    r = np.asarray(returns, dtype=np.float64)
    t, n = r.shape
    w, cons = _long_only(cp, n, cap)
    alpha = cp.Variable()
    u = cp.Variable(t)
    cons.extend([u >= 0, r @ w + alpha + u >= 0])
    obj = alpha + (1.0 / (t * (1.0 - float(beta)))) * cp.sum(u)
    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"min_cvar 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="min_cvar")


def min_cdar_weights(returns: np.ndarray, cap: float, beta: float) -> np.ndarray:
    """Chekhlov et al.：EfficientCDaR.min_cdar。"""
    cp = require_cvxpy("min_cdar")
    r = np.asarray(returns, dtype=np.float64)
    t, n = r.shape
    w, cons = _long_only(cp, n, cap)
    alpha = cp.Variable()
    u = cp.Variable(t + 1)
    z = cp.Variable(t)
    cons.extend(
        [
            z >= u[1:] - alpha,
            u[1:] >= u[:-1] - r @ w,
            u[0] == 0,
            z >= 0,
            u[1:] >= 0,
        ]
    )
    obj = alpha + (1.0 / (t * (1.0 - float(beta)))) * cp.sum(z)
    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"min_cdar 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="min_cdar")


class MinSemivariance(WindowAllocator):
    label = "min_semivariance"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        return min_semivariance_weights(returns, min(self.max_weight, 1.0))


class MinCVaR(WindowAllocator):
    label = "min_cvar"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        return min_cvar_weights(returns, min(self.max_weight, 1.0), self.tail_confidence)


class MinCDaR(WindowAllocator):
    label = "min_cdar"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        return min_cdar_weights(returns, min(self.max_weight, 1.0), self.tail_confidence)
