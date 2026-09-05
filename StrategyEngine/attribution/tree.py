#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收益分解树：对齐 RiceQuant RQPAttr 的 returns_decomposition。

总收益 → 交易 / 杠杆 / 持仓 → 主动（配置、选择、残差）与基准持仓（基准、穿透）。
有日收益序列时整棵树走同一套组合财富链，子项之和等于父项。
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from .linking import additive_tree_values


def returns_tree(
    summary: Mapping[str, Any],
    *,
    periods: Optional[Sequence[Any]] = None,
    effect_rows: Optional[Sequence[Any]] = None,
) -> List[dict]:
    """从上到下列出收益层级。金额一律小数收益（0.01 = 1%）。"""
    linked = additive_tree_values(periods, effect_rows) if periods else None
    source: Mapping[str, Any] = linked if linked else summary

    portfolio = _num(source.get("portfolio_return"))
    benchmark = _num(source.get("benchmark_return"))
    trade = _num(source.get("trade_return"))
    holding = _num(source.get("holding_return"))
    leverage = _num(source.get("leverage_return"))
    holding_active = _num(source.get("holding_active_return"))
    bench_holding = _num(source.get("benchmark_holding_return"))
    allocation = _num(source.get("allocation_effect"))
    selection = _num(source.get("selection_effect"))
    residual = _num(source.get("attribution_residual"))
    if residual is None and holding_active is not None and allocation is not None and selection is not None:
        residual = holding_active - allocation - selection
    penetration = _num(source.get("penetration"))
    if penetration is None and bench_holding is not None and benchmark is not None:
        penetration = bench_holding - benchmark

    return [
        _row("总收益", portfolio, 0, tone="positive"),
        _row("交易收益", trade, 1),
        _row("杠杆收益", leverage, 1),
        _row("持仓收益", holding, 1),
        _row("主动收益", holding_active, 2, tone="positive"),
        _row("股票配置收益", allocation, 3),
        _row("股票选择收益", selection, 3),
        _row("归因残差", residual, 3),
        _row("基准持仓收益", bench_holding, 2, tone="muted"),
        _row("基准收益", benchmark, 3, tone="muted"),
        _row("穿透效应", penetration, 3, tone="muted"),
    ]


def _row(label: str, value: Optional[float], level: int, *, tone: Optional[str] = None) -> dict:
    item: dict = {"label": label, "value": value, "level": int(level), "kind": "row"}
    if tone:
        item["tone"] = tone
    return item


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
