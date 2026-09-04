#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全市场流通市值加权 Style 暴露 → 合成风险分。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from FactorEvaluates.exposure_engine import STYLE_LABELS
from FactorEvaluates.matrix_utils import causal_percentile

from .catalog import CORE_INDICATORS, INDICATOR_BY_KEY
from .config import StyleCrowdingSettings
from .context import StyleCrowdingContext, build_context
from .query.bootstrap import crowding_temperature, last_finite, load_series


SCHEMA = "qmt-style-crowding-market-exposure-v1"
CORE_THRESHOLD = 0.90


def build_market_exposure(
    root: Path,
    settings: StyleCrowdingSettings,
    ctx: StyleCrowdingContext | None = None,
) -> Dict[str, Any]:
    if ctx is None:
        ctx = build_context(settings)
    group_count = settings.group_counts[0]
    orientation = "positive"
    latest = None

    style_weights: Dict[str, float] = {}
    for date in reversed(ctx.dates):
        row_mv = ctx.circ_mv.loc[date]
        total = float(row_mv[np.isfinite(row_mv)].sum())
        if total <= 0:
            continue
        for style in ctx.factor_ids:
            panel = ctx.style_panels.get(style)
            if panel is None or date not in panel.index:
                continue
            exp = panel.loc[date]
            valid = np.isfinite(exp.to_numpy(dtype="float64")) & np.isfinite(row_mv.to_numpy(dtype="float64"))
            if int(valid.sum()) < 10:
                continue
            w = row_mv.to_numpy(dtype="float64")[valid]
            w = w / w.sum()
            style_weights[style] = float(np.average(exp.to_numpy(dtype="float64")[valid], weights=w))
        if style_weights:
            latest = date
            break

    indicators_out: List[Dict[str, Any]] = []
    composite_temps: List[float] = []
    for style in ctx.factor_ids:
        for ind_key in CORE_INDICATORS:
            spec = INDICATOR_BY_KEY.get(ind_key)
            if spec is None:
                continue
            try:
                series = load_series(root, style, group_count, orientation, ind_key)
            except FileNotFoundError:
                continue
            pct = causal_percentile(series, min_prior=settings.percentile_window)
            last_pct, as_of_ind = last_finite(pct)
            temp = crowding_temperature(last_pct, spec.crowding_direction)
            if temp is not None and np.isfinite(temp):
                composite_temps.append(temp)
            indicators_out.append(
                {
                    "style": style,
                    "style_label": STYLE_LABELS.get(style, style),
                    "indicator": ind_key,
                    "title": spec.title,
                    "percentile": last_pct,
                    "temperature": temp,
                    "weight": style_weights.get(style),
                    "as_of": as_of_ind,
                }
            )

    return {
        "schema": SCHEMA,
        "as_of": latest,
        "style_exposure": {k: v for k, v in style_weights.items()},
        "style_labels": {k: STYLE_LABELS.get(k, k) for k in style_weights},
        "core_indicators": indicators_out,
        "composite_temperature": float(np.mean(composite_temps)) if composite_temps else None,
        "core_alarm_count": sum(1 for item in indicators_out if (item.get("temperature") or 0) >= CORE_THRESHOLD),
    }
