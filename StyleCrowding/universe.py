#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按目标 style 构造 H/L 暴露组。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from sys import intern
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .config import StyleCrowdingSettings


@dataclass(frozen=True)
class GroupSnapshot:
    factor_id: str
    signal_date: str
    universe_size: int
    valid_factor_size: int
    factor_coverage: float
    high: Tuple[str, ...]
    low: Tuple[str, ...]
    valid: bool
    reason: str


SIZE_REASON = "GROUP_SIZE_BELOW_THRESHOLD"
COVERAGE_REASON = "COVERAGE_BELOW_THRESHOLD"


def reverse_group_snapshots(
    groups: Dict[str, Dict[str, GroupSnapshot]],
) -> Dict[str, Dict[str, GroupSnapshot]]:
    return {
        factor_id: {
            date: replace(
                snapshot,
                high=tuple(reversed(snapshot.low)),
                low=tuple(reversed(snapshot.high)),
            )
            for date, snapshot in by_date.items()
        }
        for factor_id, by_date in groups.items()
    }


def build_group_snapshot(
    signal_date: str,
    exposure: pd.Series,
    factor_id: str,
    settings: StyleCrowdingSettings,
    *,
    group_count: int,
    universe_symbols: Optional[pd.Index] = None,
) -> GroupSnapshot:
    universe_size = int(len(universe_symbols)) if universe_symbols is not None else int(exposure.size)
    ranked = pd.to_numeric(exposure, errors="coerce")
    valid_mask = np.isfinite(ranked.to_numpy(dtype="float64"))
    symbols = ranked.index.astype(str).to_numpy()[valid_mask]
    values = ranked.to_numpy(dtype="float64")[valid_mask]
    valid_size = int(values.size)
    coverage = valid_size / universe_size if universe_size else 0.0
    group_n = valid_size // int(group_count) if group_count else 0

    if group_n < settings.min_group_size:
        return GroupSnapshot(
            factor_id,
            signal_date,
            universe_size,
            valid_size,
            coverage,
            (),
            (),
            False,
            SIZE_REASON,
        )
    order = np.lexsort((symbols, values))
    low = tuple(intern(str(s)) for s in symbols[order[:group_n]])
    high = tuple(intern(str(s)) for s in symbols[order[-group_n:]])
    # 组够大但因子只覆盖了一小撮股票时，分位口子代表的不是这个风格，而是
    # 「恰好有该因子值的那批股票」。分组照样留着（article 口径门槛更松），
    # 但主口径标记为不可用。
    ok = coverage >= float(settings.min_group_coverage)
    return GroupSnapshot(
        factor_id,
        signal_date,
        universe_size,
        valid_size,
        coverage,
        high,
        low,
        ok,
        "OK" if ok else COVERAGE_REASON,
    )


def snapshot_usable(
    snapshot: Optional[GroupSnapshot],
    *,
    min_coverage: Optional[float] = None,
) -> bool:
    """主口径看 ``valid``；传 min_coverage 时按更松的覆盖度门槛放行。"""
    if snapshot is None or not snapshot.high or not snapshot.low:
        return False
    if snapshot.valid:
        return True
    if min_coverage is None or snapshot.reason != COVERAGE_REASON:
        return False
    return float(snapshot.factor_coverage) >= float(min_coverage)


def build_groups_for_style(
    exposure_panel: pd.DataFrame,
    factor_id: str,
    settings: StyleCrowdingSettings,
    *,
    group_count: int,
    mask: Optional[pd.DataFrame] = None,
) -> Dict[str, GroupSnapshot]:
    output: Dict[str, GroupSnapshot] = {}
    for date in exposure_panel.index:
        row = exposure_panel.loc[date]
        if mask is not None and date in mask.index:
            valid = mask.loc[date].fillna(False).astype(bool)
            row = row.where(valid)
        universe = row.index[valid.to_numpy()] if mask is not None and date in mask.index else row.index
        output[str(date)] = build_group_snapshot(
            str(date),
            row,
            factor_id,
            settings,
            group_count=group_count,
            universe_symbols=universe,
        )
    return output
