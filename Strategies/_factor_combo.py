#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""icir / lgbm 共用的因子列表解析与截面标准化。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

FactorLeg = Tuple[str, float]

DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 5


def parse_factor_list(text: str) -> List[FactorLeg]:
    """解析 'a,b,-c' → [('a',1),('b',1),('c',-1)]。"""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("需要至少一个因子，例如 --factor a,b,c")
    legs: List[FactorLeg] = []
    seen = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        sign = 1.0
        if item.startswith("-"):
            sign = -1.0
            item = item[1:].strip()
        elif item.startswith("+"):
            item = item[1:].strip()
        if not item:
            raise ValueError(f"无效因子项: {token!r}")
        if item in seen:
            raise ValueError(f"重复因子: {item}")
        seen.add(item)
        legs.append((item, sign))
    if not legs:
        raise ValueError("需要至少一个因子，例如 --factor a,b,c")
    return legs


def cross_section_z(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """在 mask 有效样本上做截面 z-score；无效处置 NaN。"""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool) & np.isfinite(values)
    n = int(sel.sum())
    if n < 2:
        if n == 1:
            out[sel] = 0.0
        return out
    x = values[sel]
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd < 1e-12:
        out[sel] = 0.0
    else:
        out[sel] = (x - mu) / sd
    return out


def equal_weight_scores(
    panels: Sequence[np.ndarray],
    signs: Sequence[float],
    universe: np.ndarray,
) -> np.ndarray:
    """各因子截面 z 后等权 nanmean。"""
    if not panels:
        return np.full(universe.shape, np.nan, dtype=np.float64)
    zs = []
    for panel, sign in zip(panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        zs.append(cross_section_z(raw, universe))
    stacked = np.vstack(zs)
    count = np.sum(np.isfinite(stacked), axis=0)
    total = np.nansum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    score = np.full(universe.shape, np.nan, dtype=np.float64)
    ok = count > 0
    score[ok] = total[ok] / count[ok]
    return score