#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则筛选预览。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .factors import _summary_frame

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def preview(request: Mapping[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
    frame = _summary_frame(run_id)
    rules = list(request.get("rules") or [])
    mask = pd.Series(True, index=frame.index)
    applied = []
    for rule in rules:
        col = str(rule.get("metric") or "")
        op = str(rule.get("op") or ">=")
        if col not in frame.columns or op not in _OPS:
            continue
        try:
            thresh = float(rule.get("value"))
        except (TypeError, ValueError):
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        mask = mask & _OPS[op](values, thresh)
        applied.append({"metric": col, "op": op, "value": thresh, "passed": int(mask.sum())})
    ranked = frame.loc[mask].copy()
    rank_by = str(request.get("rank_by") or "rank_ic_ir")
    ascending = bool(request.get("ascending", False))
    if rank_by in ranked.columns:
        ranked = ranked.sort_values(rank_by, ascending=ascending, na_position="last")
    limit = int(request.get("limit") or 50)
    items = []
    for row in ranked.head(limit).to_dict(orient="records"):
        items.append({k: (None if _nan(v) else v) for k, v in row.items()})
    return {
        "n_total": int(len(frame)),
        "n_passed": int(mask.sum()),
        "funnel": applied,
        "rank_by": rank_by,
        "items": items,
    }


def _nan(value: Any) -> bool:
    try:
        return value != value
    except Exception:
        return False
