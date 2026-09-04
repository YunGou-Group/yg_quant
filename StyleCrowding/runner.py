#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编排：style × group × orientation × 指标发布。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .catalog import INDICATORS
from .config import StyleCrowdingSettings, default_settings
from .context import build_context
from .event_analysis import build_event_analysis
from .indicators import compute_indicators
from .market_exposure import build_market_exposure
from .writer import publish_manifest, series_path, write_json, write_series

logger = logging.getLogger("StyleCrowding")


def _read_manifest(root: Path) -> Dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def warmup_trading_days(settings: StyleCrowdingSettings) -> int:
    """重算尾段需要的预热交易日数。

    最深的一条链是 252 日窗口再套 252 日 prior z-score，两层要叠加，
    否则拼接点附近的值和全量重算对不上。
    """
    windows = (
        settings.risk_covariance_window,
        settings.percentile_window,
        settings.prior_z_window,
        settings.relative_vol_z_min_prior,
        settings.factor_vol_window,
        settings.price_stretch_window,
        settings.pairwise_window,
        settings.factor_momentum_window,
        settings.article_window,
    )
    return int(max(windows)) + int(settings.prior_z_window) + 40


def _splice(published: Path, fresh: pd.Series, splice_from: str) -> pd.Series:
    """拼接：拼接点之前用已发布值，之后用刚算出来的值。"""
    if not published.is_file():
        return fresh
    try:
        old = pd.read_parquet(published)
    except Exception:
        logger.warning("读取已发布序列失败，改用本次重算结果: %s", published)
        return fresh
    if "trade_date" not in old.columns or "value" not in old.columns:
        return fresh
    head = pd.Series(
        old["value"].to_numpy(dtype="float64"),
        index=old["trade_date"].astype(str),
    )
    head = head[head.index < splice_from]
    tail = fresh[fresh.index.astype(str) >= splice_from]
    merged = pd.concat([head, tail])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def run_style_crowding(
    settings: Optional[StyleCrowdingSettings] = None,
    *,
    incremental: bool = False,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    settings = settings or default_settings()
    root = settings.published_root()
    root.mkdir(parents=True, exist_ok=True)

    # 真增量：只重算 [已发布末日 - warmup, 今天]，再和已发布序列拼接。
    # 原来把 start_date 设成 manifest 的最早日期，等于每次都全量重算。
    splice_from: Optional[str] = None
    published_start = ""
    if incremental:
        manifest = _read_manifest(root)
        published_start = str(manifest.get("start_date") or "")
        published_end = str(manifest.get("end_date") or "")
        base_start = published_start or settings.start_date or "2018-01-01"
        if published_end:
            warm = warmup_trading_days(settings)
            # 交易日折算自然日按 1.5 倍留足余量
            fetch_start = pd.Timestamp(published_end) - pd.Timedelta(
                days=int(warm * 1.5) + 10
            )
            if fetch_start > pd.Timestamp(base_start):
                settings.start_date = fetch_start.strftime("%Y-%m-%d")
                splice_from = published_end
                logger.info(
                    "增量模式：重算 %s 起（预热 %s 交易日），%s 之前沿用已发布值",
                    settings.start_date,
                    warm,
                    splice_from,
                )
            else:
                settings.start_date = base_start
        elif not settings.start_date:
            settings.start_date = base_start

    t0 = time.perf_counter()
    if progress:
        progress({"phase": "context", "message": "加载面板…"})
    ctx = build_context(settings)

    total = (
        len(ctx.factor_ids)
        * len(settings.group_counts)
        * len(settings.orientations)
        * len(INDICATORS)
    )
    done = 0

    for style in ctx.factor_ids:
        for group_count in settings.group_counts:
            for orientation in settings.orientations:
                if progress:
                    progress(
                        {
                            "phase": "indicators",
                            "style": style,
                            "group": group_count,
                            "orientation": orientation,
                            "done": done,
                            "total": total,
                        }
                    )
                series_map = compute_indicators(ctx, style, group_count, orientation)
                for key, series in series_map.items():
                    path = series_path(root, style, group_count, orientation, key)
                    if splice_from:
                        series = _splice(path, series, splice_from)
                    write_series(path, series)
                    done += 1

    if progress:
        progress({"phase": "market_exposure", "message": "合成全市场暴露…"})
    market = build_market_exposure(root, settings, ctx=ctx)
    write_json(root / "market_exposure.json", market)

    if progress:
        progress({"phase": "event_analysis", "message": "事件分析…"})
    events = build_event_analysis(root, settings)
    write_json(root / "event_analysis.json", events)

    manifest_extra = {
        "end_date": ctx.dates[-1] if ctx.dates else settings.end_date,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "n_dates": len(ctx.dates),
        "n_styles": len(ctx.factor_ids),
        "incremental": bool(splice_from),
    }
    if splice_from:
        # manifest 的 start_date 必须留住整段历史的起点，否则下次增量
        # 会以本次的重算窗口为准，历史被一段段截掉
        manifest_extra["start_date"] = published_start or settings.start_date
        manifest_extra["recomputed_from"] = settings.start_date
        manifest_extra["spliced_at"] = splice_from
    publish_manifest(root, settings, extra=manifest_extra)
    logger.info("StyleCrowding 发布完成 → %s (%.1fs)", root, manifest_extra["elapsed_sec"])
    return {"root": str(root), **manifest_extra}
