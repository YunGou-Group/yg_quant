#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNE5-lite style descriptors as raw (pre-z-score) panels.

Stored values are as-of close_T and use only T and earlier observations.
Cross-sectional z-score / NLSIZE residualization happen later, on the
current universe — they are not written to Bin.

NLSIZE is intentionally not a stored series.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from yg_quant_repo import default_db_path

logger = logging.getLogger("Style.engine")

HS300_SYMBOL = os.getenv("YG_QUANT_BETA_INDEX", "index_SH000300")

_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE: Dict[int, "StyleEngine"] = {}

_GROWTH_FIELDS = ("netprofit_yoy", "or_yoy", "dt_netprofit_yoy")
_LEVERAGE_FIELDS = ("debt_to_assets", "debt_to_eqt")
_FINANCIAL_FIELDS = tuple(dict.fromkeys(_GROWTH_FIELDS + _LEVERAGE_FIELDS))


def _resolve_db_path() -> Optional[str]:
    path = default_db_path()
    return str(path) if path.is_file() else None


def _normalize_market(data: pd.DataFrame) -> pd.DataFrame:
    frame = data
    if "ts_code" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "ts_code"})
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "trade_date"})
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["ts_code"] = frame["ts_code"].astype(str)
    mask = ~frame["ts_code"].str.startswith("index_")
    frame = frame.loc[mask]
    return frame.sort_values(["ts_code", "trade_date"])


def _to_wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in frame.columns:
        raise ValueError(f"风格描述子缺少行情字段: {field}")
    wide = frame.pivot(index="trade_date", columns="ts_code", values=field)
    if np.issubdtype(wide.values.dtype, np.floating):
        wide = wide.astype(np.float32)
    return wide.sort_index()


def _wide_to_long(wide: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
    if start_date:
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        wide = wide.loc[wide.index >= start]
    stacked = wide.stack(future_stack=True).rename("factor_value").reset_index()
    stacked.columns = ["trade_date", "ts_code", "factor_value"]
    stacked["factor_value"] = stacked["factor_value"].replace(
        [np.inf, -np.inf], np.nan
    )
    stacked = stacked.dropna(subset=["factor_value"])
    return stacked.reset_index(drop=True)


def _daily_return(pct_chg: pd.DataFrame) -> pd.DataFrame:
    """Tushare ``pct_chg`` is percent; convert to decimal returns."""
    return pct_chg / np.float32(100.0)


def _rolling_beta(
    stock_ret: pd.DataFrame,
    mkt_ret: pd.Series,
    window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    market = mkt_ret.reindex(stock_ret.index).astype(np.float64)
    stock = stock_ret.astype(np.float64)
    mean_s = stock.rolling(window, min_periods=min_periods).mean()
    mean_m = market.rolling(window, min_periods=min_periods).mean()
    mean_sm = stock.mul(market, axis=0).rolling(window, min_periods=min_periods).mean()
    mean_m2 = (market * market).rolling(window, min_periods=min_periods).mean()
    cov = mean_sm.sub(mean_s.mul(mean_m, axis=0))
    var_m = mean_m2 - mean_m * mean_m
    return cov.divide(var_m.replace(0.0, np.nan), axis=0).astype(np.float32)


class StyleEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        financial: Optional[pd.DataFrame] = None,
        index_returns: Optional[pd.Series] = None,
    ):
        self.frame = _normalize_market(data)
        self._wides: Dict[str, pd.DataFrame] = {}
        self._financial_override = financial
        self._index_override = index_returns
        self._fin_wides: Optional[Dict[str, pd.DataFrame]] = None
        self._index_ret: Optional[pd.Series] = None

    def wide(self, field: str) -> pd.DataFrame:
        cached = self._wides.get(field)
        if cached is not None:
            return cached
        panel = _to_wide(self.frame, field)
        self._wides[field] = panel
        return panel

    def _template(self) -> pd.DataFrame:
        for field in ("close", "circ_mv", "pct_chg", "pb", "pe", "pe_ttm"):
            if field in self.frame.columns:
                return self.wide(field)
        raise ValueError("风格引擎需要至少一列行情字段作为日历模板")

    def compute(self, key: str) -> pd.DataFrame:
        if key == "size":
            circ = self.wide("circ_mv")
            valid = circ.where(circ > 0)
            return np.log(valid)
        if key == "momentum":
            close = self.wide("close")
            lagged = close.shift(21)
            base = close.shift(252)
            return lagged.divide(base.where(base > 0)) - 1.0
        if key == "liquidity":
            primary = self.wide("turnover_rate_f") if "turnover_rate_f" in self.frame.columns else None
            fallback = self.wide("turnover_rate") if "turnover_rate" in self.frame.columns else None
            if primary is None and fallback is None:
                raise ValueError("style_liquidity 需要 turnover_rate_f 或 turnover_rate")
            if primary is None:
                series = fallback
            elif fallback is None:
                series = primary
            else:
                series = primary.fillna(fallback)
            return series.rolling(21, min_periods=10).mean()
        if key == "resvol":
            returns = _daily_return(self.wide("pct_chg"))
            return returns.rolling(60, min_periods=20).std()
        if key == "btop":
            pb = self.wide("pb")
            return (1.0 / pb.where(pb > 0)).astype(np.float32)
        if key == "ey":
            pe_ttm = self.wide("pe_ttm") if "pe_ttm" in self.frame.columns else None
            pe = self.wide("pe") if "pe" in self.frame.columns else None
            if pe_ttm is None and pe is None:
                raise ValueError("style_ey 需要 pe_ttm 或 pe")
            if pe_ttm is None:
                return (1.0 / pe.where(pe > 0)).astype(np.float32)
            ey = 1.0 / pe_ttm.where(pe_ttm > 0)
            if pe is not None:
                ey = ey.fillna(1.0 / pe.where(pe > 0))
            return ey.astype(np.float32)
        if key == "beta":
            stock_ret = _daily_return(self.wide("pct_chg"))
            market = self._index_returns().reindex(stock_ret.index)
            market = market / np.float32(100.0)
            return _rolling_beta(stock_ret, market)
        if key == "growth":
            panels = self._financial_wides()
            growth = panels.get("netprofit_yoy")
            if growth is None:
                raise ValueError("style_growth 缺少 netprofit_yoy")
            for alt in ("or_yoy", "dt_netprofit_yoy"):
                if alt in panels:
                    growth = growth.fillna(panels[alt])
            return growth
        if key == "leverage":
            panels = self._financial_wides()
            lev = panels.get("debt_to_assets")
            if lev is None:
                raise ValueError("style_leverage 缺少 debt_to_assets")
            if "debt_to_eqt" in panels:
                lev = lev.fillna(panels["debt_to_eqt"])
            return lev
        raise KeyError(f"未知风格描述子: {key}")

    def _index_returns(self) -> pd.Series:
        if self._index_ret is not None:
            return self._index_ret
        if self._index_override is not None:
            series = self._index_override.copy()
            series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
            self._index_ret = series.sort_index().astype(np.float32)
            return self._index_ret
        db_path = _resolve_db_path()
        if not db_path:
            raise FileNotFoundError(f"style_beta 找不到数据库: {default_db_path()}")
        from DailyUpdates.storage.sqlite_storage import SQLiteStorage

        storage = SQLiteStorage(db_path)
        frame = storage.read_market_data(
            fields=["pct_chg"],
            symbols=[HS300_SYMBOL],
            include_indexes=True,
            ordered=True,
        )
        if frame.empty:
            raise ValueError(f"未找到指数行情: {HS300_SYMBOL}")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
        series = frame.set_index("trade_date")["pct_chg"].astype(np.float32)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        logger.info(
            "加载指数收益 %s: %s -> %s rows=%s",
            HS300_SYMBOL,
            series.index.min(),
            series.index.max(),
            len(series),
        )
        self._index_ret = series
        return series

    def _load_financial(self) -> pd.DataFrame:
        if self._financial_override is not None:
            frame = self._financial_override.copy()
        else:
            db_path = _resolve_db_path()
            if not db_path:
                raise FileNotFoundError(f"风格财务描述子找不到数据库: {default_db_path()}")
            from DailyUpdates.storage.sqlite_storage import SQLiteStorage

            storage = SQLiteStorage(db_path)
            frame = storage.read_financial_indicator()
        if frame is None or frame.empty:
            return pd.DataFrame()
        if "ts_code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "ts_code"})
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="coerce")
        frame = frame.dropna(subset=["ts_code", "ann_date"])
        if "end_date" in frame.columns:
            frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
        if "update_flag" not in frame.columns:
            frame["update_flag"] = ""
        sort_cols = ["ts_code", "ann_date"]
        if "end_date" in frame.columns:
            sort_cols.append("end_date")
        sort_cols.append("update_flag")
        frame = frame.sort_values(sort_cols)
        return frame.drop_duplicates(["ts_code", "ann_date"], keep="last")

    def _financial_wides(self) -> Dict[str, pd.DataFrame]:
        if self._fin_wides is not None:
            return self._fin_wides
        template = self._template()
        fin = self._load_financial()
        calendar = pd.Index(template.index.astype(str))
        if fin.empty:
            empty = template.copy()
            empty.loc[:, :] = np.nan
            self._fin_wides = {name: empty for name in _FINANCIAL_FIELDS}
            return self._fin_wides

        panels: Dict[str, pd.DataFrame] = {}
        for field in _FINANCIAL_FIELDS:
            if field not in fin.columns:
                continue
            piece = fin[["ts_code", "ann_date", field]].dropna(subset=[field]).copy()
            if piece.empty:
                empty = template.copy()
                empty.loc[:, :] = np.nan
                panels[field] = empty
                continue
            piece["ann_date"] = pd.to_datetime(piece["ann_date"]).dt.strftime("%Y-%m-%d")
            piece = piece.drop_duplicates(["ts_code", "ann_date"], keep="last")
            announced = piece.pivot(index="ann_date", columns="ts_code", values=field)
            combined = calendar.union(announced.index).sort_values()
            wide = announced.reindex(combined).ffill().reindex(calendar)
            wide = wide.reindex(columns=template.columns)
            if np.issubdtype(wide.values.dtype, np.floating):
                wide = wide.astype(np.float32)
            panels[field] = wide
        self._fin_wides = panels
        logger.info("财务 PIT 宽表完成: fields=%s", list(panels))
        return panels


def get_prepared_engine(
    data: pd.DataFrame,
    *,
    financial: Optional[pd.DataFrame] = None,
    index_returns: Optional[pd.Series] = None,
) -> StyleEngine:
    key = id(data)
    with _CACHE_LOCK:
        cached = _ENGINE_CACHE.get(key)
        if cached is not None and financial is None and index_returns is None:
            return cached
        engine = StyleEngine(
            data, financial=financial, index_returns=index_returns
        )
        if financial is None and index_returns is None:
            _ENGINE_CACHE.clear()
            _ENGINE_CACHE[key] = engine
        return engine


def clear_prepared_engine() -> None:
    with _CACHE_LOCK:
        _ENGINE_CACHE.clear()


def extract_style(
    data: pd.DataFrame,
    key: str,
    start_date: Optional[str] = None,
    *,
    financial: Optional[pd.DataFrame] = None,
    index_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute one raw style descriptor as long ``ts_code, trade_date, factor_value``."""
    engine = get_prepared_engine(
        data, financial=financial, index_returns=index_returns
    )
    started = time.perf_counter()
    wide = engine.compute(key)
    frame = _wide_to_long(wide, start_date=start_date)
    logger.info(
        "风格 %s 完成 rows=%s (%.2fs)",
        key,
        len(frame),
        time.perf_counter() - started,
    )
    return frame


STYLE_KEYS: List[str] = [
    "size",
    "momentum",
    "liquidity",
    "resvol",
    "beta",
    "btop",
    "ey",
    "growth",
    "leverage",
]
