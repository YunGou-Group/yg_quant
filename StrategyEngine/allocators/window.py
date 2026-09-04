#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""带 lookback 窗口的仓位分配：样本不足或求解失败回退等权。CLI 多余参数忽略。"""

from __future__ import annotations

import logging

import numpy as np

from ..context import DayContext
from .common import (
    EMA_SPAN,
    FILL_LIMIT,
    FREQUENCY,
    L2_GAMMA,
    LOOKBACK,
    MAX_WEIGHT,
    MIN_COVERAGE,
    RISK_AVERSION,
    RISK_FREE_RATE,
    TAIL_CONFIDENCE,
    TRANSACTION_COST_RATE,
    candidate_index,
    close_returns,
    equal_weights,
    unique_long_only,
)

logger = logging.getLogger("StrategyEngine")


class WindowAllocator:
    label = "allocator"

    def __init__(
        self,
        *,
        lookback: int = LOOKBACK,
        max_weight: float = MAX_WEIGHT,
        frequency: int = FREQUENCY,
        risk_free_rate: float = RISK_FREE_RATE,
        risk_aversion: float = RISK_AVERSION,
        target_return: float | None = None,
        target_volatility: float | None = None,
        l2_gamma: float = L2_GAMMA,
        transaction_cost_rate: float = TRANSACTION_COST_RATE,
        tail_confidence: float = TAIL_CONFIDENCE,
        mu_model: str = "geometric",
        ema_span: int = EMA_SPAN,
        **_kwargs,
    ):
        self.lookback = max(5, int(lookback))
        cap = float(max_weight)
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError("max_weight 必须为正数")
        self.max_weight = min(1.0, cap)
        self.frequency = max(1, int(frequency))
        self.risk_free_rate = float(risk_free_rate)
        self.risk_aversion = float(risk_aversion)
        self.target_return = None if target_return is None else float(target_return)
        self.target_volatility = None if target_volatility is None else float(target_volatility)
        self.l2_gamma = float(l2_gamma)
        self.transaction_cost_rate = float(transaction_cost_rate)
        beta = float(tail_confidence)
        if not 0.0 <= beta < 1.0:
            raise ValueError("tail_confidence 必须在 [0, 1)")
        self.tail_confidence = beta
        self.mu_model = str(mu_model or "geometric")
        self.ema_span = max(2, int(ema_span))

    def size(self, ctx: DayContext, candidates: np.ndarray) -> np.ndarray:
        idx = candidate_index(ctx, candidates)
        n = len(ctx.symbols)
        if idx.size == 0:
            return np.zeros(n, dtype=np.float64)
        if idx.size == 1:
            return equal_weights(n, idx)
        unique = unique_long_only(idx.size, min(self.max_weight, 1.0))
        if unique is not None:
            out = np.zeros(n, dtype=np.float64)
            out[idx] = unique
            return out
        returns = close_returns(ctx, idx, self.lookback)
        if returns is None:
            logger.warning(
                "%s 样本不足：asof=%s 前 %s 日、%s 只在前向填 %s 日后仍凑不齐 %.0f%% 完整收盘，回退等权",
                self.label,
                ctx.asof,
                self.lookback,
                idx.size,
                FILL_LIMIT,
                100 * MIN_COVERAGE,
            )
            return equal_weights(n, idx)
        try:
            packed = self.allocate(returns, ctx, idx)
        except ImportError:
            raise
        except Exception as exc:
            logger.warning("%s 求解失败，回退等权 asof=%s：%s", self.label, ctx.asof, exc)
            return equal_weights(n, idx)
        out = np.zeros(n, dtype=np.float64)
        out[idx] = packed
        return out

    def allocate(self, returns: np.ndarray, ctx: DayContext, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError
