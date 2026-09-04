#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读 query：bootstrap、序列、拥挤温度。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from FactorEvaluates.exposure_engine import RAW_STYLE_NAMES, STYLE_LABELS
from FactorEvaluates.matrix_utils import causal_percentile, prior_rolling_zscore

from ..catalog import CORE_INDICATORS, INDICATOR_BY_KEY, INDICATORS
from ..config import StyleCrowdingSettings, default_settings
from ..writer import series_path

__all__ = [
    "crowding_temperature",
    "last_finite",
    "load_series",
    "load_manifest",
    "bootstrap",
    "fetch_series",
]


def last_finite(series: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    """最后一个有限观测及其日期；全缺失则 (None, None)。"""
    if series is None or len(series) == 0:
        return None, None
    values = pd.to_numeric(series, errors="coerce")
    arr = values.to_numpy(dtype="float64", copy=False)
    ok = np.isfinite(arr)
    if not ok.any():
        return None, None
    idx = int(np.flatnonzero(ok)[-1])
    return float(arr[idx]), str(values.index[idx])


def crowding_temperature(percentile: Optional[float], direction: str) -> Optional[float]:
    if percentile is None or not (0.0 <= percentile <= 1.0):
        return None
    if direction == "low":
        return 1.0 - percentile
    if direction == "both":
        return max(percentile, 1.0 - percentile)
    return percentile


def load_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    settings = default_settings()
    path = (root or settings.published_root()) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_series(
    root: Path,
    style: str,
    group_count: int,
    orientation: str,
    indicator: str,
) -> pd.Series:
    path = series_path(root, style, group_count, orientation, indicator)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_parquet(path)
    return pd.Series(frame["value"].to_numpy(dtype="float64"), index=frame["trade_date"].astype(str))


def bootstrap(root: Optional[Path] = None) -> Dict[str, Any]:
    settings = default_settings()
    published = root or settings.published_root()
    manifest = load_manifest(published)
    styles = manifest.get("styles") or list(RAW_STYLE_NAMES)
    group_counts = list(manifest.get("group_counts") or [10])
    orientations = list(manifest.get("orientations") or ["positive"])
    matrix: List[Dict[str, Any]] = []
    group_count = int(group_counts[0]) if group_counts else 10
    orientation = "positive" if "positive" in orientations else (orientations[0] if orientations else "positive")
    published_keys = set(manifest.get("indicators") or [])
    for style in styles:
        row: Dict[str, Any] = {"style": style, "label": STYLE_LABELS.get(style, style), "indicators": {}}
        for key in CORE_INDICATORS:
            spec = INDICATOR_BY_KEY.get(key)
            if spec is None:
                continue
            try:
                raw = load_series(published, style, group_count, orientation, key)
            except FileNotFoundError:
                continue
            pct = causal_percentile(raw, min_prior=settings.percentile_window)
            last_raw, as_of = last_finite(raw)
            last_pct, _ = last_finite(pct)
            row["indicators"][key] = {
                "title": spec.title,
                "raw": last_raw,
                "percentile": last_pct,
                "temperature": crowding_temperature(last_pct, spec.crowding_direction),
                "direction": spec.crowding_direction,
                "as_of": as_of,
            }
        matrix.append(row)

    market_path = published / "market_exposure.json"
    events_path = published / "event_analysis.json"
    market = json.loads(market_path.read_text(encoding="utf-8")) if market_path.is_file() else {}
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.is_file() else {}

    return {
        "manifest": manifest,
        "styles": styles,
        "style_labels": STYLE_LABELS,
        "group_counts": group_counts,
        "orientations": orientations,
        "indicators": [
            {"key": i.key, "title": i.title, "direction": i.crowding_direction}
            for i in INDICATORS
            if not published_keys or i.key in published_keys
        ],
        "core_indicators": list(CORE_INDICATORS),
        "matrix": matrix,
        "market_exposure": market,
        "event_analysis": events,
    }


def fetch_series(
    style: str,
    indicator: str,
    *,
    group_count: int = 10,
    orientation: str = "positive",
    transform: str = "raw",
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    settings = default_settings()
    published = root or settings.published_root()
    raw = load_series(published, style, group_count, orientation, indicator)
    spec = INDICATOR_BY_KEY.get(indicator)
    direction = spec.crowding_direction if spec else "high"
    if transform == "prior_z":
        values = prior_rolling_zscore(raw, window=settings.prior_z_window, min_history=settings.prior_z_min_history)
    elif transform == "percentile":
        values = causal_percentile(raw, min_prior=settings.percentile_window)
    else:
        values = raw
    temps = [crowding_temperature(float(p), direction) if pd.notna(p) else None for p in values]
    last, as_of = last_finite(values)
    return {
        "style": style,
        "indicator": indicator,
        "group_count": group_count,
        "orientation": orientation,
        "transform": transform,
        "dates": list(values.index.astype(str)),
        "values": [float(v) if pd.notna(v) else None for v in values],
        "temperatures": temps,
        "last_finite": last,
        "as_of": as_of,
    }
