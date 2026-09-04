#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原子发布 manifest 与指标 parquet 序列。"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .catalog import INDICATORS, indicator_key
from .config import StyleCrowdingSettings

# 约 36 个交易月；序列有限观测少于此则尚未就绪。
READY_MIN_OBS = 756


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=path.suffix + ".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_series(path: Path, series: pd.Series) -> None:
    frame = pd.DataFrame({"trade_date": series.index.astype(str), "value": series.astype("float64")})
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    _atomic_write_bytes(path, buf.getvalue())


def series_path(
    root: Path,
    style: str,
    group_count: int,
    orientation: str,
    indicator: str,
) -> Path:
    return root / style / f"group_{group_count}" / orientation / f"{indicator_key(indicator)}.parquet"


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return str(value)


def _dumps(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(dict(payload)),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def _series_readiness(path: Path) -> Dict[str, Any]:
    frame = pd.read_parquet(path)
    if "value" not in frame.columns:
        return {"first_valid_date": None, "valid_count": 0, "ready": False}
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(values)
    valid_count = int(ok.sum())
    first_valid = None
    if valid_count and "trade_date" in frame.columns:
        first_valid = str(frame.loc[ok, "trade_date"].iloc[0])
    return {
        "first_valid_date": first_valid,
        "valid_count": valid_count,
        "ready": valid_count >= READY_MIN_OBS,
    }


def scan_published(root: Path) -> Dict[str, Any]:
    """只根据实际写出的 parquet 汇总分组 / 方向 / 指标与就绪状态。"""
    styles = []
    group_counts = []
    orientations = []
    indicators = []
    series: Dict[str, Any] = {}
    for path in sorted(root.glob("*/group_*/*/*.parquet")):
        if not path.is_file():
            continue
        style = path.parents[2].name
        group_name = path.parents[1].name
        if not group_name.startswith("group_"):
            continue
        try:
            group_count = int(group_name.replace("group_", "", 1))
        except ValueError:
            continue
        orientation = path.parent.name
        indicator = path.stem
        styles.append(style)
        group_counts.append(group_count)
        orientations.append(orientation)
        indicators.append(indicator)
        key = f"{style}/group_{group_count}/{orientation}/{indicator}"
        try:
            series[key] = _series_readiness(path)
        except Exception:
            series[key] = {"first_valid_date": None, "valid_count": 0, "ready": False}
    unique_styles = list(dict.fromkeys(styles))
    unique_groups = sorted(set(group_counts))
    unique_orients = list(dict.fromkeys(orientations))
    unique_inds = list(dict.fromkeys(indicators))
    ready = bool(series) and all(item.get("ready") for item in series.values())
    return {
        "styles": unique_styles,
        "group_counts": unique_groups,
        "orientations": unique_orients,
        "indicators": unique_inds,
        "series": series,
        "ready": ready,
    }


def publish_manifest(
    root: Path,
    settings: StyleCrowdingSettings,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "schema": "qmt-style-crowding-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_date": settings.start_date,
        "end_date": settings.end_date or None,
        "skip": list(settings.skip_indicators),
        "notes": {
            "style_returns": "StyleReturnEngine WLS on same-day close returns",
            "specific_returns": "Local WLS residuals e=r-X beta",
            "factor_hl_returns": "Same-day close H-L from style exposure groups",
        },
    }
    if extra:
        payload.update(dict(extra))
    scanned = scan_published(root)
    payload["styles"] = scanned["styles"] or list(settings.style_targets())
    payload["group_counts"] = scanned["group_counts"] or list(settings.group_counts)
    payload["orientations"] = scanned["orientations"] or list(settings.orientations)
    payload["indicators"] = scanned["indicators"] or [item.key for item in INDICATORS]
    payload["series"] = scanned["series"]
    payload["ready"] = scanned["ready"]
    payload["skip"] = list(settings.skip_indicators)
    path = root / "manifest.json"
    _atomic_write_bytes(path, _dumps(payload))
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _dumps(payload))
