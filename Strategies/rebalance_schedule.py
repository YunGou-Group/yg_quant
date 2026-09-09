#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""topk / multifactor 调仓日程：日频、周频、每 N 个交易日。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Set

import numpy as np
import pandas as pd

from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings


@dataclass(frozen=True)
class RebalanceSpec:
    """mode=daily|weekly|every；every 仅在 mode=every 时生效（交易日间隔）。"""

    mode: str
    every: int = 1

    def tag_suffix(self) -> str:
        if self.mode == "weekly":
            return "_weekly"
        if self.mode == "every":
            return f"_every{int(self.every)}"
        return ""

    @property
    def holds_between(self) -> bool:
        return self.mode != "daily"


def parse_rebalance(value: object) -> RebalanceSpec:
    """解析 daily / weekly / 5 / 5d / every5。"""
    if value is None or value == "":
        return RebalanceSpec("daily", 1)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = int(value)
        if n <= 1:
            return RebalanceSpec("daily", 1)
        return RebalanceSpec("every", n)
    text = str(value).strip().lower()
    if not text or text in {"daily", "day", "1", "1d"}:
        return RebalanceSpec("daily", 1)
    if text in {"weekly", "week"}:
        return RebalanceSpec("weekly", 1)
    raw = text
    for prefix in ("every", "each", "n"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :].lstrip("_-")
            break
    if raw.endswith("d"):
        raw = raw[:-1]
    if raw.isdigit():
        n = int(raw)
        if n <= 1:
            return RebalanceSpec("daily", 1)
        return RebalanceSpec("every", n)
    raise ValueError(
        f"无效调仓频率 {value!r}，请用 daily、weekly，或交易日间隔如 5 / 5d / every5"
    )


def week_end_asofs(dates: Sequence[str]) -> Set[str]:
    """调仓日：周五收盘；周五休市则用当周最后一个交易日。

    引擎按 T+1 开盘成交，所以正常是周一换仓。
    """
    days = [str(d) for d in dates]
    out: Set[str] = set()
    for i, day in enumerate(days):
        ts = pd.Timestamp(day)
        weekday = int(ts.weekday())
        if weekday == 4:
            out.add(day)
            continue
        if weekday > 4:
            continue
        if i + 1 >= len(days):
            continue
        nxt = pd.Timestamp(days[i + 1])
        if (int(nxt.isocalendar().year), int(nxt.isocalendar().week)) != (
            int(ts.isocalendar().year),
            int(ts.isocalendar().week),
        ):
            out.add(day)
    return out


def every_n_asofs(dates: Sequence[str], every: int) -> Set[str]:
    """从序列首日开始，每隔 every 个交易日调仓一次。"""
    n = max(1, int(every))
    days = [str(d) for d in dates]
    if n <= 1:
        return set(days)
    return {days[i] for i in range(0, len(days), n)}


def schedule_asofs(
    dates: Sequence[str],
    spec: RebalanceSpec,
    *,
    active_index: int = 0,
) -> Optional[Set[str]]:
    """返回需要重新打分的 asof 集合；daily 返回 None（每日都调）。"""
    if not spec.holds_between:
        return None
    days = [str(d) for d in dates]
    start = max(0, int(active_index))
    active = days[start:] if start < len(days) else days
    if spec.mode == "weekly":
        return week_end_asofs(active)
    if spec.mode == "every":
        return every_n_asofs(active, spec.every)
    return None


def spec_from_cli(value: object) -> RebalanceSpec:
    try:
        return parse_rebalance(value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


class RebalanceGate:
    """非调仓日沿用上次目标仓；调仓日返回 None 让策略重新打分。"""

    def __init__(self, spec: RebalanceSpec):
        self.spec = spec
        self._held: Optional[dict] = None
        self._days: Optional[Set[str]] = None

    def skip(self, ctx: DayContext) -> Optional[TargetHoldings]:
        if not self.spec.holds_between:
            return None
        if self._days is None:
            active = int(getattr(ctx._store, "active_index", 0) or 0)
            self._days = schedule_asofs(
                list(ctx._store.dates), self.spec, active_index=active
            ) or set()
        if ctx.asof in self._days:
            return None
        if self._held:
            return TargetHoldings(ctx.asof, ctx.execute_on, dict(self._held))
        return TargetHoldings.from_array(
            ctx.asof, ctx.execute_on, ctx.symbols, np.zeros(len(ctx.symbols))
        )

    def remember(self, holdings: TargetHoldings) -> TargetHoldings:
        if self.spec.holds_between:
            self._held = dict(holdings.weights)
        return holdings
