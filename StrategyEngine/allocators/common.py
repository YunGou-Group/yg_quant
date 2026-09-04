#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仓位分配共用：候选清洗、等权回退、与 PyPortfolioOpt 对齐的样本协方差 / 最小方差。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..context import DayContext

_LOG = logging.getLogger("StrategyEngine")
_EPS = 1e-12
FREQUENCY = 252
LOOKBACK = FREQUENCY
FILL_LIMIT = 5
MIN_COVERAGE = 0.95
MAX_WEIGHT = 1.0
RISK_FREE_RATE = 0.02
RISK_AVERSION = 1.0
L2_GAMMA = 0.1
TRANSACTION_COST_RATE = 0.001
TAIL_CONFIDENCE = 0.95
EMA_SPAN = 500
TARGET_RETURN_QUANTILE = 0.60
TARGET_VOL_MULT = 1.15


def candidate_index(ctx: DayContext, candidates: np.ndarray) -> np.ndarray:
    n = len(ctx.symbols)
    idx = np.unique(np.asarray(candidates, dtype=int).reshape(-1))
    return idx[(idx >= 0) & (idx < n)]


def equal_weights(n_symbols: int, idx: np.ndarray) -> np.ndarray:
    out = np.zeros(int(n_symbols), dtype=np.float64)
    if idx.size:
        out[idx] = 1.0 / idx.size
    return out


def unique_long_only(n: int, cap: float) -> np.ndarray | None:
    """单票上限把满仓可行集收成一个点时直接给出：每只 cap（凑不满 1）或等权。

    小市值默认约 5 只时，只有 max_weight=0.2 才会把可行集收成等权。
    """
    n = int(n)
    cap = float(cap)
    if n < 1:
        return None
    if cap * n + _EPS < 1.0:
        return np.full(n, cap, dtype=np.float64)
    if cap * n <= 1.0 + 1e-9:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return None


def close_returns(ctx: DayContext, idx: np.ndarray, lookback: int) -> np.ndarray | None:
    """asof 前 lookback 根收盘。与 doc 一致：非正作缺失、最多前向填 5 日，再删仍缺的行。

    不要求 252/252 天同时有价（五只小盘里停一天就会整段作废）。
    填完后完整天数须达到 lookback × 0.95。
    """
    if idx.size < 1:
        return None
    need = max(3, int(lookback))
    hist = ctx.history("close", need)
    px = np.asarray(hist[:, idx], dtype=np.float64)
    px = np.where(np.isfinite(px) & (px > 0), px, np.nan)
    filled = pd.DataFrame(px).ffill(limit=FILL_LIMIT).dropna(axis=0, how="any")
    min_rows = max(3, int(np.ceil(need * MIN_COVERAGE)))
    if filled.shape[0] < min_rows:
        return None
    prices = filled.to_numpy(dtype=np.float64)
    with np.errstate(all="ignore"):
        ret = prices[1:] / prices[:-1] - 1.0
    ret = ret[np.isfinite(ret).all(axis=1)]
    if ret.shape[0] < 2:
        return None
    return np.asarray(ret, dtype=np.float64)


def _is_psd(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """pypfopt.risk_models.fix_nonpositive_semidefinite(..., fix_method='spectral')。"""
    sigma = np.asarray(matrix, dtype=np.float64)
    if _is_psd(sigma):
        return sigma
    eigval, eigvec = np.linalg.eigh(sigma)
    eigval = np.where(eigval > 0, eigval, 0.0)
    return eigvec @ np.diag(eigval) @ eigvec.T


def sample_cov(returns: np.ndarray, frequency: int = FREQUENCY) -> np.ndarray:
    """pypfopt.risk_models.sample_cov(..., returns_data=True, frequency=252)。"""
    cov = pd.DataFrame(np.asarray(returns, dtype=np.float64)).cov().to_numpy(dtype=np.float64)
    return fix_nonpositive_semidefinite(cov * float(frequency))


def require_cvxpy(name: str):
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(f"{name} 需要 cvxpy，请在当前环境执行: python -m pip install cvxpy") from exc
    return cp


def pack_weights(value, cap: float, *, name: str = "allocator") -> np.ndarray:
    """归一化到 sum==1 且逐项 <= cap。

    先 clip 再归一化会把 cap 重新放大（例如 cap=0.2 时 clip 后总和 0.6，
    除完每项变成 0.33）。这里改成迭代投影：把顶到上限的权重钉住，剩余额度
    在其余标的上重新分配，直到没有新的越界项。
    """
    cap = float(cap)
    out = np.clip(np.asarray(value, dtype=np.float64).reshape(-1), 0.0, np.inf)
    n = out.size
    if n == 0 or float(out.sum()) <= _EPS:
        raise RuntimeError(f"{name} 权重和为 0")
    if cap <= 0 or cap * n < 1.0 - 1e-12:
        # cap * n < 1 时无解：满仓和上限不可能同时满足。退化成等权（1/n 是
        # 离可行域最近的点），并告警，因为这说明 max_weight 与持仓数配置冲突。
        _LOG.warning(
            "%s: cap=%.4f × %d 只 < 1，无法同时满足满仓与上限，退化为等权 %.4f",
            name,
            cap,
            n,
            1.0 / n,
        )
        return np.full(n, 1.0 / n)

    free = np.ones(n, dtype=bool)
    pinned = np.zeros(n, dtype=np.float64)
    for _ in range(n + 1):
        budget = 1.0 - float(pinned.sum())
        pool = float(out[free].sum())
        if pool <= _EPS:
            # 剩余标的原始权重全为 0：把余额平摊到它们身上
            share = budget / max(1, int(free.sum()))
            pinned[free] = share
            return pinned
        scaled = np.zeros(n, dtype=np.float64)
        scaled[free] = out[free] / pool * budget
        over = free & (scaled > cap + 1e-15)
        if not over.any():
            pinned[free] = scaled[free]
            return pinned
        pinned[over] = cap
        free &= ~over
        if not free.any():
            return pinned
    return pinned


def expected_return(
    returns: np.ndarray,
    frequency: int = FREQUENCY,
    model: str = "geometric",
    ema_span: int = EMA_SPAN,
) -> np.ndarray:
    """asof 窗口上的年化 μ，对齐 pypfopt.expected_returns（returns_data=True）。"""
    r = np.asarray(returns, dtype=np.float64)
    key = str(model or "geometric").strip().lower()
    freq = float(frequency)
    if key in {"geometric", "mean_geometric"}:
        t = max(1, r.shape[0])
        return np.prod(1.0 + r, axis=0) ** (freq / t) - 1.0
    if key in {"arithmetic", "mean_arithmetic"}:
        return r.mean(axis=0) * freq
    if key == "ema":
        last = (
            pd.DataFrame(r)
            .ewm(span=max(2, int(ema_span)))
            .mean()
            .iloc[-1]
            .to_numpy(dtype=np.float64)
        )
        return (1.0 + last) ** freq - 1.0
    raise ValueError(f"未知 mu_model {model!r}，可选 geometric / arithmetic / ema")


def default_targets(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float | None = None,
    target_volatility: float | None = None,
) -> tuple[float, float]:
    """doc `_targets`：μ 的 60% 分位；等权波动 × 1.15。"""
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov, dtype=np.float64)
    if target_return is None:
        target_return = float(np.quantile(mu, TARGET_RETURN_QUANTILE))
    if target_volatility is None:
        n = mu.size
        equal = np.full(n, 1.0 / n, dtype=np.float64)
        equal_vol = float(np.sqrt(equal @ cov @ equal))
        target_volatility = equal_vol * TARGET_VOL_MULT
    return float(target_return), float(target_volatility)


def min_volatility_weights(cov: np.ndarray, cap: float) -> np.ndarray:
    """EfficientFrontier.min_volatility：min w'Σw，sum w = 1，0 ≤ w ≤ cap。"""
    cp = require_cvxpy("min_vol")
    sigma = np.asarray(cov, dtype=np.float64)
    n = sigma.shape[0]
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    cap = float(cap)
    if cap * n + _EPS < 1.0:
        raise ValueError("weight_bounds 使问题不可行")
    weights = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(weights, sigma, assume_PSD=True)),
        [cp.sum(weights) == 1, weights >= 0, weights <= cap],
    )
    problem.solve(verbose=False)
    if weights.value is None:
        raise RuntimeError(f"min_volatility 未解出 status={problem.status}")
    return pack_weights(weights.value, cap, name="min_volatility")
