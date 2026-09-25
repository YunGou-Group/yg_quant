#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多因子截面 z 等权合成 + Top N。

用法：--strategy equal --factor a,b,-c
因子名前加 '-' 表示取负。当日对每个因子做截面 z-score，等权平均后选 Top N。
不估计权重，作为 icir / lgbm 的基准。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy, cli_field, n_field, rebalance_field
from Strategies._factor_combo import FactorLeg, equal_weight_scores, parse_factor_list
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

DEFAULT_N = 50


class EqualWeightTopK(Strategy):
    name = "equal"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("equal 需要 --factor，逗号分隔，如 a,b,c")
        try:
            legs = parse_factor_list(args.factor)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        n = args.n if args.n is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or "20")
        return cls(legs, n=n, allocator=allocator, rebalance=spec)

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / weekly / 20"),
        ]

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        legs = parse_factor_list(args.factor)
        return {"factors": [name for name, _sign in legs]}

    @classmethod
    def warmup(cls, args) -> int:
        return 5

    @classmethod
    def run_tag(cls, args) -> str:
        legs = parse_factor_list(args.factor)
        parts = [("m" if s < 0 else "") + n for n, s in legs]
        spec = parse_rebalance(getattr(args, "rebalance", None) or "20")
        return "equal_" + "_".join(parts) + spec.tag_suffix()

    def __init__(
        self,
        factors: Sequence[FactorLeg] | Sequence[str],
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = "20",
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("equal 至少需要一个因子")
        self.legs = legs
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self._gate = RebalanceGate(self.rebalance)

    @property
    def factor_names(self) -> List[str]:
        return [name for name, _ in self.legs]

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held

        uni = ctx.universe()
        panels = [ctx.factor(name) for name, _ in self.legs]
        signs = [sign for _, sign in self.legs]
        values = equal_weight_scores(panels, signs, uni)
        tradable = uni & np.isfinite(values)
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
