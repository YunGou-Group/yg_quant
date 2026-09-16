#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全库 run 的库级结果（因子相关矩阵等）。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from .datasets import read_meta, run_dir

LIBRARY_FILES = ("factor_corr", "ic_corr", "family_redundancy")
CORR_METRICS = ("factor_corr", "ic_corr")
MAX_SUBSET = 80
_SPLIT = re.compile(r"[\s,;，、；|]+")


def library_overview(run_id: Optional[str] = None) -> Dict[str, Any]:
    meta = read_meta(run_id)
    folder = run_dir(run_id) / "library"
    items: Dict[str, Any] = {}
    for name in LIBRARY_FILES:
        path = folder / f"{name}.json"
        if path.is_file():
            items[name] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": meta.get("run_id"),
        "library_metrics": list(meta.get("library_metrics") or []),
        "items": items,
    }


def parse_factor_names(raw: Union[str, Sequence[str], None]) -> List[str]:
    if raw is None:
        tokens: List[str] = []
    elif isinstance(raw, str):
        tokens = [part for part in _SPLIT.split(raw) if part]
    else:
        tokens = []
        for item in raw:
            tokens.extend(parse_factor_names(str(item)))
    seen = set()
    names: List[str] = []
    for token in tokens:
        name = str(token).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def library_corr_subset(
    run_id: Optional[str] = None,
    factors: Union[str, Sequence[str], None] = None,
    metric: str = "factor_corr",
) -> Dict[str, Any]:
    metric_name = str(metric or "factor_corr").strip()
    if metric_name not in CORR_METRICS:
        raise ValueError(f"metric 只能是 {', '.join(CORR_METRICS)}")
    names = parse_factor_names(factors)
    if not names:
        raise ValueError("请输入因子名")
    if len(names) > MAX_SUBSET:
        raise ValueError(f"最多 {MAX_SUBSET} 个因子，收到 {len(names)} 个")
    meta = read_meta(run_id)
    path = run_dir(run_id) / "library" / f"{metric_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"这次 run 没有 {metric_name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("matrix")
    if not matrix or not matrix.get("index") or not matrix.get("data"):
        raise FileNotFoundError(f"这次 run 的 {metric_name} 没有矩阵")
    sliced = slice_corr_matrix(matrix, names)
    return {
        "run_id": meta.get("run_id"),
        "metric": metric_name,
        **sliced,
    }


def slice_corr_matrix(matrix: Dict[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    index = [str(item) for item in matrix.get("index") or []]
    columns = [str(item) for item in (matrix.get("columns") or index)]
    data = matrix.get("data") or []
    pos = {name: i for i, name in enumerate(index)}
    lower = {name.lower(): name for name in index}
    found: List[str] = []
    idxs: List[int] = []
    missing: List[str] = []
    seen = set()
    for raw in names:
        name = str(raw).strip()
        canon = name if name in pos else lower.get(name.lower())
        if canon is None:
            missing.append(name)
            continue
        if canon in seen:
            continue
        seen.add(canon)
        found.append(canon)
        idxs.append(pos[canon])
    sub = [[_cell(data, i, j) for j in idxs] for i in idxs]
    return {
        "names": found,
        "missing": missing,
        "matrix": {"index": found, "columns": [columns[i] for i in idxs], "data": sub} if found else None,
        "scalars": _offdiag_scalars(sub),
    }


def _cell(data: Sequence[Any], i: int, j: int) -> Any:
    if i >= len(data):
        return None
    row = data[i]
    if not isinstance(row, (list, tuple)) or j >= len(row):
        return None
    return row[j]


def _offdiag_scalars(data: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    vals: List[float] = []
    n = len(data)
    for i in range(n):
        row = data[i]
        for j in range(i + 1, n):
            if j >= len(row):
                continue
            try:
                number = float(row[j])
            except (TypeError, ValueError):
                continue
            if number != number:
                continue
            vals.append(abs(number))
    return {
        "mean_abs": (sum(vals) / len(vals)) if vals else None,
        "max_abs": max(vals) if vals else None,
        "n_factors": n,
    }
