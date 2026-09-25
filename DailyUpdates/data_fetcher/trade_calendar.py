#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把交易所开市日写入本地 trade_calendar。策略层只读表，不绑具体数据商。"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd

from DailyUpdates.data_fetcher.data_fetcher import source_class_for_config

logger = logging.getLogger("DailyUpdates")

FORWARD_DAYS = 400


def refresh_open_calendar(
    storage,
    config: Dict,
    start: str,
    end: str,
) -> int:
    """按配置里的 data_source 拉开市日并 upsert。源不支持则返回 0。"""
    cls = source_class_for_config(config)
    if cls is None:
        logger.warning("未找到数据源 %s，跳过交易日历刷新", config.get("data_source"))
        return 0
    source = cls()
    frame = source.fetch_trade_calendar(config, start, end)
    if frame is None or frame.empty or "trade_date" not in frame.columns:
        logger.warning(
            "%s 未返回 %s ~ %s 的开市日",
            config.get("data_source"),
            start,
            end,
        )
        return 0
    days = [str(v) for v in frame["trade_date"] if str(v).strip()]
    written = int(storage.upsert_trade_calendar(days))
    logger.info(
        "交易日历已按 %s 更新 %s 日（%s ~ %s）",
        config.get("data_source"),
        written,
        start,
        end,
    )
    return written


def forward_window(from_date: Optional[str] = None, days: int = FORWARD_DAYS) -> tuple:
    start = pd.Timestamp(from_date or pd.Timestamp.today())
    end = start + timedelta(days=int(days))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
