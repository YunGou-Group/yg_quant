#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""家族成员列表。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..batch.family_key import family_of
from .factors import list_factors


_MEMBER_FIELDS = (
    "factor_name",
    "rank_ic_mean",
    "rank_ic_ir",
    "coverage_rate_mean",
    "quantile_spread_mean",
)


def list_families(run_id: Optional[str] = None) -> Dict[str, Any]:
    data = list_factors(run_id)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in data["items"]:
        name = str(item.get("factor_name") or "")
        buckets[family_of(name)].append(item)
    families = []
    for key, members in sorted(buckets.items()):
        slim = [{field: row.get(field) for field in _MEMBER_FIELDS} for row in members]
        irs = [row.get("rank_ic_ir") for row in members if _finite(row.get("rank_ic_ir"))]
        ics = [row.get("rank_ic_mean") for row in members if _finite(row.get("rank_ic_mean"))]
        covs = [
            row.get("coverage_rate_mean")
            for row in members
            if _finite(row.get("coverage_rate_mean"))
        ]
        best = max(
            (row for row in members if _finite(row.get("rank_ic_ir"))),
            key=lambda row: float(row["rank_ic_ir"]),
            default=None,
        )
        families.append(
            {
                "family": key,
                "n": len(members),
                "rank_ic_ir_mean": _mean(irs),
                "rank_ic_mean": _mean(ics),
                "coverage_rate_mean": _mean(covs),
                "best_member": None if best is None else best.get("factor_name"),
                "best_rank_ic_ir": None if best is None else best.get("rank_ic_ir"),
                "members": slim,
            }
        )
    families.sort(key=lambda row: (row["rank_ic_ir_mean"] is None, -(row["rank_ic_ir_mean"] or 0)))
    return {"items": families}


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number


def _mean(values: List[Any]) -> Optional[float]:
    if not values:
        return None
    return float(sum(float(v) for v in values) / len(values))
