#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""业绩预告、业绩快报、股东增减持。

这三张表是公告事件，不进 market_data。
预告的首次信息日是 first_ann_date，其后的更正落在 ann_date。
快报和增减持用 ann_date。都不要用 end_date 当信息日。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class EventColumn:
    name: str
    source: str
    kind: str  # symbol / date / text / real
    key: bool = False


@dataclass(frozen=True)
class EventTable:
    dataset_name: str
    api_name: str
    table: str
    columns: Tuple[EventColumn, ...]
    primary_key: Tuple[str, ...]
    cursor: str
    fetch: str  # period：按报告期；day：按日；range：按日期区间
    day_param: str
    earliest: str
    pause: float
    page_rows: int
    chunk_days: int = 1
    min_interval: float = 0.0

    def tushare_fields(self) -> Tuple[str, ...]:
        seen = []
        for column in self.columns:
            if column.source not in seen:
                seen.append(column.source)
        return tuple(seen)

    def create_sql(self) -> str:
        lines = []
        for column in self.columns:
            if column.kind == "real":
                suffix = " NOT NULL" if column.key else ""
                lines.append(f"{column.name} REAL{suffix}")
            elif column.kind == "symbol":
                lines.append(f"{column.name} TEXT NOT NULL")
            else:
                lines.append(f"{column.name} TEXT NOT NULL DEFAULT ''")
        keys = ", ".join(self.primary_key)
        body = ",\n                    ".join(lines)
        return f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    {body},
                    PRIMARY KEY ({keys})
                );
                CREATE INDEX IF NOT EXISTS idx_{self.table}_{self.cursor}
                    ON {self.table} ({self.cursor});
                """


EVENT_SPECS: Tuple[EventTable, ...] = (
    EventTable(
        dataset_name="earnings_forecast",
        api_name="forecast_vip",
        table="earnings_forecast",
        cursor="ann_date",
        fetch="period",
        day_param="ann_date",
        earliest="",
        pause=0.35,
        page_rows=6500,
        primary_key=("symbol", "end_date", "ann_date"),
        columns=(
            EventColumn("symbol", "ts_code", "symbol", True),
            EventColumn("ann_date", "ann_date", "date", True),
            EventColumn("end_date", "end_date", "date", True),
            EventColumn("first_ann_date", "first_ann_date", "date"),
            EventColumn("forecast_type", "type", "text"),
            EventColumn("p_change_min", "p_change_min", "real"),
            EventColumn("p_change_max", "p_change_max", "real"),
            EventColumn("net_profit_min", "net_profit_min", "real"),
            EventColumn("net_profit_max", "net_profit_max", "real"),
            EventColumn("last_parent_net", "last_parent_net", "real"),
            EventColumn("summary", "summary", "text"),
            EventColumn("change_reason", "change_reason", "text"),
        ),
    ),
    EventTable(
        dataset_name="earnings_express",
        api_name="express_vip",
        table="earnings_express",
        cursor="ann_date",
        fetch="period",
        day_param="ann_date",
        earliest="",
        pause=0.35,
        page_rows=5000,
        primary_key=("symbol", "end_date", "ann_date"),
        columns=(
            EventColumn("symbol", "ts_code", "symbol", True),
            EventColumn("ann_date", "ann_date", "date", True),
            EventColumn("end_date", "end_date", "date", True),
            EventColumn("revenue", "revenue", "real"),
            EventColumn("operate_profit", "operate_profit", "real"),
            EventColumn("total_profit", "total_profit", "real"),
            EventColumn("n_income", "n_income", "real"),
            EventColumn("diluted_eps", "diluted_eps", "real"),
            EventColumn("diluted_roe", "diluted_roe", "real"),
            EventColumn("yoy_net_profit", "yoy_net_profit", "real"),
            EventColumn("yoy_sales", "yoy_sales", "real"),
            EventColumn("bps", "bps", "real"),
        ),
    ),
    EventTable(
        dataset_name="holder_trade",
        api_name="stk_holdertrade",
        table="holder_trade",
        cursor="ann_date",
        fetch="range",
        day_param="ann_date",
        earliest="",
        pause=0.2,
        page_rows=3000,
        chunk_days=1460,
        min_interval=0.2,
        primary_key=(
            "symbol",
            "ann_date",
            "holder_name",
            "in_de",
            "begin_date",
            "change_vol",
        ),
        columns=(
            EventColumn("symbol", "ts_code", "symbol", True),
            EventColumn("ann_date", "ann_date", "date", True),
            EventColumn("holder_name", "holder_name", "text", True),
            EventColumn("holder_type", "holder_type", "text"),
            EventColumn("in_de", "in_de", "text", True),
            EventColumn("change_vol", "change_vol", "real", True),
            EventColumn("change_ratio", "change_ratio", "real"),
            EventColumn("after_share", "after_share", "real"),
            EventColumn("after_ratio", "after_ratio", "real"),
            EventColumn("avg_price", "avg_price", "real"),
            EventColumn("total_share", "total_share", "real"),
            EventColumn("begin_date", "begin_date", "date", True),
            EventColumn("close_date", "close_date", "date"),
        ),
    ),
)

EVENT_BY_API = {spec.api_name: spec for spec in EVENT_SPECS}
EVENT_BY_DATASET = {spec.dataset_name: spec for spec in EVENT_SPECS}


def event_spec(api_name: str) -> EventTable:
    spec = EVENT_BY_API.get(str(api_name))
    if spec is None:
        known = ", ".join(EVENT_BY_API)
        raise ValueError(f"未知事件接口 {api_name}，已有: {known}")
    return spec


def event_dataset_configs() -> Dict[str, Dict]:
    """调度配置。不含 token。"""
    configs = {}
    for spec in EVENT_SPECS:
        configs[spec.dataset_name] = {
            "data_source": "Tushare",
            "data_type": "event",
            "api_name": spec.api_name,
            "pause_seconds": spec.pause,
            "fields": list(spec.tushare_fields()),
            "primary_key": list(spec.primary_key),
        }
    return configs
