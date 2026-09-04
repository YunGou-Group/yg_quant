#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指定因子多头 Top N；仓位由 allocator 决定（默认等权）。"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np

from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.strategy import Strategy
from Strategies.small_cap import week_end_asofs


class FactorTopK(Strategy):
    name = "topk"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("topk 需要 --factor")
        n = args.n if args.n is not None else 50
        rebalance = str(getattr(args, "rebalance", None) or "daily").lower()
        if rebalance not in {"daily", "weekly"}:
            raise SystemExit("topk --rebalance 仅支持 daily / weekly")
        return cls(args.factor, n=n, allocator=allocator, rebalance=rebalance)

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        return {"factors": [args.factor]}

    @classmethod
    def run_tag(cls, args) -> str:
        tag = f"topk_{args.factor}"
        rebalance = str(getattr(args, "rebalance", None) or "daily").lower()
        if rebalance == "weekly":
            tag += "_weekly"
        return tag

    def __init__(
        self,
        factor: str,
        n: int = 50,
        allocator: Allocator | None = None,
        rebalance: str = "daily",
    ):
        self.factor = str(factor)
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = "weekly" if str(rebalance).lower() == "weekly" else "daily"
        self._held: Optional[dict] = None
        self._days: Optional[Set[str]] = None

    def score(self, ctx: DayContext) -> TargetHoldings:
        if self.rebalance == "weekly":
            if self._days is None:
                self._days = week_end_asofs(list(ctx._store.dates))
            if ctx.asof not in self._days:
                if self._held:
                    return TargetHoldings(ctx.asof, ctx.execute_on, dict(self._held))
                return TargetHoldings.from_array(
                    ctx.asof, ctx.execute_on, ctx.symbols, np.zeros(len(ctx.symbols))
                )
        values = ctx.factor(self.factor)
        tradable = ctx.universe() & np.isfinite(values)
        ranked = np.where(tradable, values, -np.inf)
        k = int(min(self.n, int(tradable.sum())))
        if k <= 0:
            weights = np.zeros(len(ctx.symbols))
            holdings = TargetHoldings.from_array(
                ctx.asof, ctx.execute_on, ctx.symbols, weights
            )
            if self.rebalance == "weekly":
                self._held = dict(holdings.weights)
            return holdings
        pick = np.argpartition(ranked, -k)[-k:]
        pick = pick[tradable[pick]]
        weights = self.allocator.size(ctx, pick)
        holdings = TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        if self.rebalance == "weekly":
            self._held = dict(holdings.weights)
        return holdings
