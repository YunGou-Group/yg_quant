#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行情面板加载者：从 SQLite 读 open 宽表，供 ReturnCalculator 使用。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from DailyUpdates.storage import SQLiteStorage
from yg_quant_repo import default_db_path

HS300_SYMBOL = os.getenv("YG_QUANT_BETA_INDEX", "index_SH000300")


def default_adjust() -> str:
    """价格面板默认复权口径，可用 YG_QUANT_PRICE_ADJUST=none 关闭。"""
    return (os.getenv("YG_QUANT_PRICE_ADJUST") or "hfq").strip().lower()



INDEX_LABELS = {
    "index_SH000300": "沪深300",
    "index_SH000001": "上证综指",
    "index_SZ399001": "深证成指",
    "index_SZ399006": "创业板指",
}


class MarketPanelLoader:
    def __init__(self, db_path: Optional[str] = None):
        path = db_path or str(default_db_path())
        self.storage = SQLiteStorage(str(Path(path).expanduser().resolve()))
        self._field_panels: Dict[str, pd.DataFrame] = {}
        self._index_close: Optional[pd.Series] = None
        self._index_symbol: Optional[str] = None

    def load_open(
        self,
        reload: bool = False,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.load_field(
            "open",
            reload=reload,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

    def load_field(
        self,
        field: str,
        reload: bool = False,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Optional[str] = None,
    ) -> pd.DataFrame:
        key = str(field)
        mode = (adjust or default_adjust()).strip().lower()
        cache_key = f"{key}|{start_date}|{end_date}|{mode}"
        if cache_key in self._field_panels and not reload:
            return self._field_panels[cache_key]
        data = self.storage.read_market_data(
            fields=[key],
            start_date=start_date,
            end_date=end_date,
            ordered=False,
            adjust=mode,
        )
        if data.empty or key not in data.columns:
            panel = pd.DataFrame()
        else:
            frame = data.copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
            frame["ts_code"] = frame["ts_code"].astype(str)
            panel = (
                frame.pivot(index="trade_date", columns="ts_code", values=key)
                .sort_index()
                .astype("float64")
            )
        self._field_panels[cache_key] = panel
        return panel

    def load_index_close(
        self, symbol: Optional[str] = None, reload: bool = False
    ) -> pd.Series:
        code = str(symbol or HS300_SYMBOL)
        if (
            self._index_close is not None
            and not reload
            and self._index_symbol == code
        ):
            return self._index_close
        data = self.storage.read_market_data(
            fields=["close"],
            symbols=[code],
            include_indexes=True,
            ordered=True,
        )
        if data.empty or "close" not in data.columns:
            self._index_symbol = code
            self._index_close = pd.Series(dtype="float64")
            return self._index_close
        frame = data.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
        series = pd.to_numeric(frame.set_index("trade_date")["close"], errors="coerce")
        series = series[~series.index.duplicated(keep="last")].sort_index()
        self._index_symbol = code
        self._index_close = series.astype("float64")
        return self._index_close

    @staticmethod
    def summarize_index(
        close: pd.Series, symbol: str = HS300_SYMBOL
    ) -> Optional[Dict[str, Any]]:
        """评估窗内收盘累计与日收益。不用 open.shift，不是远期收益。"""
        if close is None or close.empty:
            return None
        series = pd.to_numeric(close, errors="coerce").copy()
        series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
        series = series[~series.index.duplicated(keep="last")].sort_index()
        valid = series.dropna()
        if len(valid) < 2:
            return None
        base = float(valid.iloc[0])
        if base == 0.0:
            return None
        nav = series / base
        cum = nav - 1.0
        daily = series.pct_change()
        drawdown = nav / nav.cummax() - 1.0
        dd_clean = drawdown.dropna()
        cum_clean = cum.dropna()
        return {
            "symbol": symbol,
            "label": INDEX_LABELS.get(symbol, symbol),
            "scalars": {
                "cum_last": float(cum_clean.iloc[-1]) if len(cum_clean) else float("nan"),
                "max_drawdown": float(dd_clean.min()) if len(dd_clean) else float("nan"),
                "n_days": int(len(valid)),
            },
            "series": {
                "cum_ret": cum,
                "daily_ret": daily,
                "drawdown": drawdown,
            },
        }
