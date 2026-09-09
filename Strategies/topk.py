#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指定因子多头 Top N；仓位由 allocator 决定（默认等权）。"""

from __future__ import annotations

import numpy as np

from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.strategy import Strategy
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli


class FactorTopK(Strategy):
    name = "topk"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("topk 需要 --factor")
        n = args.n if args.n is not None else 50
        spec = spec_from_cli(getattr(args, "rebalance", None) or "daily")
        return cls(args.factor, n=n, allocator=allocator, rebalance=spec)

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        return {"factors": [args.factor]}

    @classmethod
    def run_tag(cls, args) -> str:
        spec = parse_rebalance(getattr(args, "rebalance", None) or "daily")
        return f"topk_{args.factor}{spec.tag_suffix()}"

    def __init__(
        self,
        factor: str,
        n: int = 50,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = "daily",
    ):
        self.factor = str(factor)
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        self._gate = RebalanceGate(self.rebalance)

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held
        values = ctx.factor(self.factor)
        tradable = ctx.universe() & np.isfinite(values)
        ranked = np.where(tradable, values, -np.inf)
        k = int(min(self.n, int(tradable.sum())))
        if k <= 0:
            weights = np.zeros(len(ctx.symbols))
            return self._gate.remember(
                TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
            )
        pick = np.argpartition(ranked, -k)[-k:]
        pick = pick[tradable[pick]]
        weights = self.allocator.size(ctx, pick)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )
