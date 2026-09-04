#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EfficientFrontier 其余目标：μ 来自 asof 前窗口，Σ 仍是 sample_cov。

对齐 doc/optimize 的 _run_ef_objective（不含 nonconvex_max_sharpe）。
"""

from __future__ import annotations

import numpy as np

from ..context import DayContext
from .common import (
    _EPS,
    default_targets,
    expected_return,
    pack_weights,
    require_cvxpy,
    sample_cov,
)
from .window import WindowAllocator


def _mu_cov(returns: np.ndarray, allocator: WindowAllocator) -> tuple[np.ndarray, np.ndarray]:
    mu = expected_return(returns, allocator.frequency, allocator.mu_model, allocator.ema_span)
    cov = sample_cov(returns, allocator.frequency)
    return np.asarray(mu, dtype=np.float64).reshape(-1), cov


def _long_only(cp, n: int, cap: float):
    if cap * n + _EPS < 1.0:
        raise ValueError("weight_bounds 使问题不可行")
    w = cp.Variable(n)
    return w, [cp.sum(w) == 1, w >= 0, w <= cap]


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, cap: float, risk_free_rate: float) -> np.ndarray:
    """pypfopt EfficientFrontier.max_sharpe：变量替换后的凸问题。"""
    cp = require_cvxpy("max_sharpe")
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(cov, dtype=np.float64)
    n = mu.size
    cap = float(cap)
    rf = float(risk_free_rate)
    if float(np.max(mu)) <= rf:
        raise ValueError("至少一只资产的预期收益须高于无风险利率")
    if cap * n + _EPS < 1.0:
        raise ValueError("weight_bounds 使问题不可行")
    order = np.argsort(-mu)
    greedy = np.zeros(n, dtype=np.float64)
    remain = 1.0
    for i in order:
        take = min(cap, remain)
        greedy[i] = take
        remain -= take
        if remain <= _EPS:
            break
    if float((mu - rf) @ greedy) <= _EPS:
        raise ValueError("约束下组合无法跑赢无风险利率")
    w = cp.Variable(n)
    k = cp.Variable()
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, sigma, assume_PSD=True)),
        [
            (mu - rf) @ w == 1,
            cp.sum(w) == k,
            k >= 0,
            w >= 0,
            w <= cap * k,
        ],
    )
    problem.solve(verbose=False)
    if w.value is None or k.value is None or float(k.value) <= _EPS:
        raise RuntimeError(f"max_sharpe 未解出 status={problem.status}")
    return pack_weights(np.asarray(w.value, dtype=np.float64) / float(k.value), cap, name="max_sharpe")


def max_quadratic_utility_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    cap: float,
    risk_aversion: float,
) -> np.ndarray:
    """max μ'w − (δ/2) w'Σw。"""
    cp = require_cvxpy("max_quadratic_utility")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion 必须为正")
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(cov, dtype=np.float64)
    w, cons = _long_only(cp, mu.size, cap)
    utility = mu @ w - 0.5 * float(risk_aversion) * cp.quad_form(w, sigma, assume_PSD=True)
    problem = cp.Problem(cp.Maximize(utility), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"max_quadratic_utility 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="max_quadratic_utility")


def efficient_return_weights(mu: np.ndarray, cov: np.ndarray, cap: float, target_return: float) -> np.ndarray:
    """给定目标收益，最小化方差。"""
    cp = require_cvxpy("efficient_return")
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(cov, dtype=np.float64)
    w, cons = _long_only(cp, mu.size, cap)
    cons.append(mu @ w >= float(target_return))
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, sigma, assume_PSD=True)), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"efficient_return 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="efficient_return")


def efficient_risk_weights(mu: np.ndarray, cov: np.ndarray, cap: float, target_volatility: float) -> np.ndarray:
    """给定目标波动上限，最大化收益。"""
    cp = require_cvxpy("efficient_risk")
    if target_volatility < 0:
        raise ValueError("target_volatility 必须非负")
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(cov, dtype=np.float64)
    w, cons = _long_only(cp, mu.size, cap)
    cons.append(cp.quad_form(w, sigma, assume_PSD=True) <= float(target_volatility) ** 2)
    problem = cp.Problem(cp.Maximize(mu @ w), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"efficient_risk 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="efficient_risk")


def multiobjective_weights(
    cov: np.ndarray,
    cap: float,
    l2_gamma: float,
    w_prev: np.ndarray,
    tc_rate: float,
) -> np.ndarray:
    """min_vol + L2 + 换手；跟踪误差需要基准权重，日频上下文没有，不接。"""
    cp = require_cvxpy("multiobjective")
    sigma = np.asarray(cov, dtype=np.float64)
    prev = np.asarray(w_prev, dtype=np.float64).reshape(-1)
    w, cons = _long_only(cp, sigma.shape[0], cap)
    obj = cp.quad_form(w, sigma, assume_PSD=True)
    obj = obj + float(l2_gamma) * cp.sum_squares(w)
    obj = obj + float(tc_rate) * cp.norm(w - prev, 1)
    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"multiobjective 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="multiobjective")


def min_l2_weights(n: int, cap: float, l2_gamma: float) -> np.ndarray:
    """custom_convex_l2：min γ||w||²，sum w=1，0≤w≤cap。"""
    cp = require_cvxpy("min_l2")
    w, cons = _long_only(cp, n, cap)
    problem = cp.Problem(cp.Minimize(float(l2_gamma) * cp.sum_squares(w)), cons)
    problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError(f"min_l2 未解出 status={problem.status}")
    return pack_weights(w.value, cap, name="min_l2")


class MaxSharpe(WindowAllocator):
    label = "max_sharpe"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        mu, cov = _mu_cov(returns, self)
        return max_sharpe_weights(mu, cov, min(self.max_weight, 1.0), self.risk_free_rate)


class MaxQuadraticUtility(WindowAllocator):
    label = "max_quadratic_utility"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        mu, cov = _mu_cov(returns, self)
        return max_quadratic_utility_weights(mu, cov, min(self.max_weight, 1.0), self.risk_aversion)


class EfficientReturn(WindowAllocator):
    label = "efficient_return"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        mu, cov = _mu_cov(returns, self)
        target, _ = default_targets(mu, cov, self.target_return, self.target_volatility)
        return efficient_return_weights(mu, cov, min(self.max_weight, 1.0), target)


class EfficientRisk(WindowAllocator):
    label = "efficient_risk"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        mu, cov = _mu_cov(returns, self)
        _, vol = default_targets(mu, cov, self.target_return, self.target_volatility)
        return efficient_risk_weights(mu, cov, min(self.max_weight, 1.0), vol)


class MultiObjective(WindowAllocator):
    label = "multiobjective"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        cov = sample_cov(returns, self.frequency)
        prev = np.asarray(ctx.position, dtype=np.float64).reshape(-1)[idx]
        return multiobjective_weights(
            cov,
            min(self.max_weight, 1.0),
            self.l2_gamma,
            prev,
            self.transaction_cost_rate,
        )


class MinL2(WindowAllocator):
    label = "min_l2"

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        return min_l2_weights(returns.shape[1], min(self.max_weight, 1.0), self.l2_gamma)
