#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽质量 / 基础 / 成长 / 每股：财务按 ann_date 前向填充到交易日。"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from yg_quant_repo import default_db_path

from . import _panel as panel

_CACHE: Dict[int, "FinEngine"] = {}
_LOCK = threading.Lock()

PERCENT_FIELDS: Set[str] = {
    "roe",
    "roa",
    "roic",
    "npta",
    "grossprofit_margin",
    "netprofit_margin",
    "cogs_of_sales",
    "op_of_gr",
    "ebit_of_gr",
    "saleexp_to_gr",
    "adminexp_of_gr",
    "finaexp_of_gr",
    "ocf_to_or",
    "salescash_to_or",
    "ocf_to_opincome",
    "debt_to_assets",
    "debt_to_eqt",
    "assets_to_eqt",
    "longdeb_to_debt",
    "currentdebt_to_debt",
    "ca_to_assets",
    "nca_to_assets",
    "tbassets_to_totalassets",
    "ocf_to_debt",
    "ocf_to_netdebt",
    "ocf_to_shortdebt",
    "longdebt_to_workingcapital",
    "dtprofit_to_profit",
    "opincome_of_ebt",
    "n_op_profit_of_ebt",
    "investincome_of_ebt",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "or_yoy",
    "equity_yoy",
    "op_yoy",
    "ebt_yoy",
    "assets_yoy",
    "ocf_yoy",
    "basic_eps_yoy",
    "tr_yoy",
}


def scale_percent_ratio(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sample = values.to_numpy(dtype=np.float64, copy=False)
    med = np.nanmedian(np.abs(sample)) if sample.size else np.nan
    if np.isfinite(med) and med > 2.0:
        return values / 100.0
    return values


def _events_to_wide(
    events: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    empty = pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float64)
    if events is None or events.empty:
        return empty
    piece = events.dropna(subset=["value"]).copy()
    if piece.empty:
        return empty
    piece["ann_date"] = pd.to_datetime(piece["ann_date"]).dt.strftime("%Y-%m-%d")
    piece = piece.drop_duplicates(["ts_code", "ann_date"], keep="last")
    announced = piece.pivot(index="ann_date", columns="ts_code", values="value")
    calendar = pd.Index(index.astype(str))
    combined = calendar.union(announced.index).sort_values()
    wide = announced.reindex(combined).ffill().reindex(calendar)
    return wide.reindex(columns=columns).astype(np.float64)


class FinEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        financial: Optional[pd.DataFrame] = None,
    ):
        frame = panel.stock_frame(data)
        if "close" in frame.columns:
            self.template = panel.to_wide(frame, "close")
        elif "total_mv" in frame.columns:
            self.template = panel.to_wide(frame, "total_mv")
        else:
            raise ValueError("财务因子需要 close 或 total_mv 作为日历")
        self.total_mv = (
            panel.to_wide(frame, "total_mv") if "total_mv" in frame.columns else None
        )
        self.circ_mv = (
            panel.to_wide(frame, "circ_mv") if "circ_mv" in frame.columns else None
        )
        self.pe_ttm = (
            panel.to_wide(frame, "pe_ttm") if "pe_ttm" in frame.columns else None
        )
        self.ps_ttm = (
            panel.to_wide(frame, "ps_ttm") if "ps_ttm" in frame.columns else None
        )
        self.total_share = (
            panel.to_wide(frame, "total_share")
            if "total_share" in frame.columns
            else None
        )
        self._fin = self._load_financial(financial)
        self._pit: Dict[str, pd.DataFrame] = {}

    def _load_financial(self, override: Optional[pd.DataFrame]) -> pd.DataFrame:
        if override is not None:
            frame = override.copy()
        else:
            from DailyUpdates.storage.sqlite_storage import SQLiteStorage

            frame = SQLiteStorage(str(default_db_path())).read_financial_indicator()
        if frame is None or frame.empty:
            return pd.DataFrame()
        if "ts_code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "ts_code"})
        frame["ts_code"] = frame["ts_code"].astype(str)
        if "update_flag" in frame.columns:
            frame = frame.sort_values(["ts_code", "ann_date", "update_flag"])
        else:
            frame = frame.sort_values(["ts_code", "ann_date"])
        for name in PERCENT_FIELDS:
            if name in frame.columns:
                frame[name] = scale_percent_ratio(frame[name])
        return frame

    def pit(self, field: str) -> pd.DataFrame:
        if field in self._pit:
            return self._pit[field]
        if self._fin.empty or field not in self._fin.columns:
            wide = pd.DataFrame(
                np.nan, index=self.template.index, columns=self.template.columns
            )
        else:
            events = self._fin[["ts_code", "ann_date", field]].rename(
                columns={field: "value"}
            )
            events["value"] = pd.to_numeric(events["value"], errors="coerce")
            if "update_flag" in self._fin.columns:
                events = events.assign(update_flag=self._fin["update_flag"])
                events = events.sort_values(["ts_code", "ann_date", "update_flag"])
            wide = _events_to_wide(events, self.template.index, self.template.columns)
        self._pit[field] = wide
        return wide

    def compute(self, key: str) -> pd.DataFrame:
        if key.startswith("pit:"):
            return self.pit(key.split(":", 1)[1])
        fn = getattr(self, f"_{key}", None)
        if fn is None:
            raise KeyError(f"未知财务因子: {key}")
        return fn()

    def _operating_cycle(self) -> pd.DataFrame:
        return self.pit("invturn_days") + self.pit("arturn_days")

    def _equity_to_asset(self) -> pd.DataFrame:
        debt = self.pit("debt_to_assets")
        return 1.0 - debt

    def _market_cap(self) -> pd.DataFrame:
        if self.total_mv is None:
            return self.template * np.nan
        return self.total_mv.reindex_like(self.template)

    def _circ_cap(self) -> pd.DataFrame:
        if self.circ_mv is None:
            return self.template * np.nan
        return self.circ_mv.reindex_like(self.template)

    def _sales_to_price(self) -> pd.DataFrame:
        if self.ps_ttm is None:
            return self.template * np.nan
        ps = self.ps_ttm.reindex_like(self.template)
        return (1.0 / ps).where(ps > 0)

    def _cash_flow_to_price(self) -> pd.DataFrame:
        ocfps = self.pit("ocfps")
        if self.total_mv is None or self.total_share is None:
            return self.template * np.nan
        mv = self.total_mv.reindex_like(self.template)
        shares = self.total_share.reindex_like(self.template)
        return (ocfps * shares / mv).where((mv > 0) & (shares > 0))

    def _peg(self) -> pd.DataFrame:
        if self.pe_ttm is None:
            return self.template * np.nan
        pe = self.pe_ttm.reindex_like(self.template)
        growth = self.pit("netprofit_yoy")
        # 聚宽：PE / (增长率 * 100)；增长率为小数。PE 或增长为负记缺失。
        return (pe / (growth * 100.0)).where((pe > 0) & (growth > 0))


def get_fin_engine(
    data: pd.DataFrame, financial: Optional[pd.DataFrame] = None
) -> FinEngine:
    key = id(data) if financial is None else (id(data), id(financial))
    with _LOCK:
        engine = _CACHE.get(key)
        if engine is None:
            engine = FinEngine(data, financial=financial)
            if financial is None:
                _CACHE.clear()
                _CACHE[key] = engine
            else:
                _CACHE[key] = engine
        return engine


def extract_fin(
    data: pd.DataFrame,
    key: str,
    start_date: Optional[str] = None,
    financial: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    wide = get_fin_engine(data, financial=financial).compute(key)
    return panel.wide_to_long(wide, start_date=start_date)
