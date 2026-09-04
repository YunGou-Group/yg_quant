#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目标持仓：策略意图，不是买卖动作。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger("StrategyEngine")

_EPS = 1e-9


@dataclass
class TargetHoldings:
    asof: str
    execute_on: Optional[str]
    weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned: Dict[str, float] = {}
        for key, raw in (self.weights or {}).items():
            name = str(key)
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"目标权重非有限数: {name}={raw!r}")
            if value < -_EPS:
                raise ValueError(f"首版只支持多头，收到负权重 {name}={value}")
            if value <= _EPS:
                continue
            cleaned[name] = value
        total = float(sum(cleaned.values()))
        if total > 1.0 + 1e-6:
            raise ValueError(f"目标权重和为 {total}，不能超过 1")
        self.weights = cleaned
        self.asof = str(self.asof)
        if self.execute_on is not None:
            self.execute_on = str(self.execute_on)
            if self.execute_on < self.asof:
                raise ValueError(f"execute_on {self.execute_on} 早于 asof {self.asof}")

    @classmethod
    def from_array(
        cls,
        asof: str,
        execute_on: Optional[str],
        symbols: Sequence[str],
        weights: np.ndarray,
    ) -> "TargetHoldings":
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if arr.size != len(symbols):
            raise ValueError(f"权重长度 {arr.size} 与股票数 {len(symbols)} 不一致")
        sparse = {str(sym): float(w) for sym, w in zip(symbols, arr) if np.isfinite(w) and w > _EPS}
        return cls(asof=str(asof), execute_on=execute_on, weights=sparse)

    def aligned(self, symbols: Sequence[str], *, drop_unknown: bool = True) -> np.ndarray:
        """按 symbols 展开为向量；未出现的票为 0。"""
        out = np.zeros(len(symbols), dtype=np.float64)
        index = {str(sym): i for i, sym in enumerate(symbols)}
        unknown = []
        for name, weight in self.weights.items():
            loc = index.get(name)
            if loc is None:
                unknown.append(name)
                continue
            out[loc] = weight
        if unknown:
            msg = f"目标含宇宙外标的，已忽略: {unknown[:8]}"
            if drop_unknown:
                logger.warning(msg)
            else:
                raise ValueError(msg)
        return out
