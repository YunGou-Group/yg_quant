#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全库 summary / 单因子时序。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ..batch.batch_result_writer import read_table
from .datasets import run_dir

_METRIC_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def _summary_frame(run_id: Optional[str] = None) -> pd.DataFrame:
    folder = run_dir(run_id) / "summary"
    return read_table(folder / "summary.parquet")


def list_factors(run_id: Optional[str] = None, q: Optional[str] = None) -> Dict[str, Any]:
    frame = _summary_frame(run_id)
    if q:
        mask = frame["factor_name"].astype(str).str.contains(str(q), case=False, na=False)
        frame = frame.loc[mask]
    records = []
    for row in frame.to_dict(orient="records"):
        clean = {k: (None if _is_nan(v) else v) for k, v in row.items()}
        records.append(clean)
    return {"items": records, "columns": list(frame.columns)}


def factor_summary(factor_name: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    frame = _summary_frame(run_id)
    hit = frame.loc[frame["factor_name"].astype(str) == factor_name]
    if hit.empty:
        raise FileNotFoundError(factor_name)
    row = hit.iloc[0].to_dict()
    return {k: (None if _is_nan(v) else v) for k, v in row.items()}


def available_metrics(run_id: Optional[str] = None) -> List[str]:
    daily = run_dir(run_id) / "daily"
    return sorted({p.stem for p in daily.glob("*.parquet")})


def factor_series(
    factor_name: str,
    metric: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    metric = str(metric or "")
    # metric 直接拼进文件名，必须限定字符集并与该 run 实际产出的指标对齐，
    # 否则 ../../ 之类的输入能读到 runs 目录以外的文件。
    if not _METRIC_RE.fullmatch(metric):
        raise FileNotFoundError(f"非法指标名 {metric!r}")
    if metric not in set(available_metrics(run_id)):
        raise FileNotFoundError(f"该数据集没有指标 {metric}")
    daily = run_dir(run_id) / "daily"
    path = daily / f"{metric}.parquet"
    try:
        frame = pd.read_parquet(path, columns=[factor_name])
    except (ValueError, KeyError, OSError) as exc:
        raise FileNotFoundError(f"{metric} 没有因子 {factor_name}") from exc
    series = pd.to_numeric(frame[factor_name], errors="coerce")
    return {
        "metric": metric,
        "factor": factor_name,
        "dates": [str(i) for i in series.index],
        "values": [None if _is_nan(v) else float(v) for v in series.to_numpy()],
    }


def _is_nan(value: Any) -> bool:
    try:
        return value != value
    except Exception:
        return False
