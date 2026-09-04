#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5 日 3 次尾部确认 / 10 日解除；只读已发布序列。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from FactorEvaluates.exposure_engine import STYLE_LABELS
from FactorEvaluates.matrix_utils import causal_percentile

from .catalog import INDICATORS, INDICATOR_BY_KEY
from .config import StyleCrowdingSettings
from .query.bootstrap import crowding_temperature, last_finite, load_series

SCHEMA = "qmt-style-crowding-event-analysis-v1"
CONFIRM_WINDOW = 5
CONFIRM_HITS = 3
RELEASE_DAYS = 10
THRESHOLD = 0.90

FAMOUS_EVENTS = (
    {"name": "2021核心资产抱团瓦解", "start": "2021-02-18", "end": "2021-03-19", "type": "crowding_unwind"},
    {"name": "2024年初小微盘与量化踩踏", "start": "2024-01-17", "end": "2024-02-07", "type": "liquidity_cascade"},
)


def _alarm_path(percentile: pd.Series, *, direction: str) -> np.ndarray:
    high_tail = percentile.ge(THRESHOLD)
    low_tail = percentile.le(1.0 - THRESHOLD)
    if direction == "low":
        crowded = low_tail
    elif direction == "both":
        crowded = high_tail | low_tail
    else:
        crowded = high_tail
    hits = crowded.fillna(False).to_numpy(dtype=bool)
    active = np.zeros(len(hits), dtype=bool)
    # 状态机：CONFIRM_WINDOW 内命中 CONFIRM_HITS 次触发；触发后必须连续
    # RELEASE_DAYS 天不在尾部才解除。原实现每天独立判定，警报会在阈值附近
    # 反复开关，跟 confirmation 里声明的 release_days 也对不上。
    on = False
    calm = 0
    for i in range(len(hits)):
        start = max(0, i - CONFIRM_WINDOW + 1)
        triggered = hits[start : i + 1].sum() >= CONFIRM_HITS
        if on:
            # 解除只看「是否还在尾部」。若也让确认窗口重置计时，触发后的
            # CONFIRM_WINDOW-1 天会一直被旧命中续命，解除永远晚到。
            calm = 0 if hits[i] else calm + 1
            if calm >= RELEASE_DAYS:
                on = False
                calm = 0
        elif triggered:
            on = True
            calm = 0
        active[i] = on
    return active


def build_event_analysis(root: Path, settings: StyleCrowdingSettings) -> Dict[str, Any]:
    group_count = settings.group_counts[0]
    orientation = "positive"
    layers = {
        "vulnerability_accumulation": "脆弱性积累",
        "holding_structure": "持仓结构共振",
        "risk_release_confirmation": "风险释放确认",
    }
    votes: List[Dict[str, Any]] = []
    for spec in INDICATORS:
        if not spec.independent_vote:
            continue
        for style in settings.style_targets():
            try:
                series = load_series(root, style, group_count, orientation, spec.key)
            except FileNotFoundError:
                continue
            pct = causal_percentile(series, min_prior=settings.percentile_window)
            active = _alarm_path(pct, direction=spec.crowding_direction)
            last_pct, as_of = last_finite(pct)
            last_active = bool(active[-1]) if active.size else False
            if as_of is not None and as_of in pct.index:
                loc = list(pct.index.astype(str)).index(as_of)
                last_active = bool(active[loc])
            votes.append(
                {
                    "style": style,
                    "style_label": STYLE_LABELS.get(style, style),
                    "indicator": spec.key,
                    "title": spec.title,
                    "layer": spec.layer,
                    "layer_title": layers.get(spec.layer, spec.layer),
                    "active": last_active,
                    "last_percentile": last_pct,
                    "temperature": crowding_temperature(last_pct, spec.crowding_direction),
                    "as_of": as_of,
                }
            )

    by_style: Dict[str, int] = {}
    for item in votes:
        if item["active"]:
            by_style[item["style"]] = by_style.get(item["style"], 0) + 1

    return {
        "schema": SCHEMA,
        "confirmation": {"window": CONFIRM_WINDOW, "hits": CONFIRM_HITS, "release_days": RELEASE_DAYS},
        "threshold": THRESHOLD,
        "layers": [{"id": k, "title": v} for k, v in layers.items()],
        "indicator_votes": votes,
        "active_votes_by_style": by_style,
        "reference_events": list(FAMOUS_EVENTS),
        "indicators": [item.event_metadata() for item in INDICATORS],
    }
