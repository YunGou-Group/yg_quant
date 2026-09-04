#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch, normalize, and persist daily market data in SQLite."""

from __future__ import annotations

import datetime
import gc
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Set

for _p in Path(__file__).resolve().parents:
    if (_p / "yg_quant_repo.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
else:
    raise RuntimeError("找不到 yg_quant 仓库根目录（缺少 yg_quant_repo.py）")
from yg_quant_repo import default_db_path, ensure_repo_root

ensure_repo_root()

import numpy as np
import pandas as pd
import tushare as ts
from loguru import logger
from tqdm import tqdm

from DailyUpdates.data_fetcher.data_processor import DataProcessor, market_uses_range_window
from DailyUpdates.data_fetcher.dataset_installer import DatasetInstaller, is_sidecar_dataset
from DailyUpdates.storage import SQLiteStorage


class StockDataUpdater:
    """Main controller for full and incremental market-data updates."""

    def __init__(
        self,
        db_path: str,
        tushare_token: str = "",
        max_workers: int = 4,
        dataset_config: Optional[Dict] = None,
    ):
        self.db_path = Path(db_path).expanduser().resolve()
        self.storage = SQLiteStorage(str(self.db_path))
        self.tushare_token = tushare_token
        self.max_workers = max_workers
        self.first_date_str = "20050101"

        self.pro = None
        if tushare_token:
            ts.set_token(tushare_token)
            self.pro = ts.pro_api()
        self.data_processor = DataProcessor(tushare_pro=self.pro, storage=self.storage)
        self.dataset_installer = DatasetInstaller(self.storage, self.data_processor)

        self.dataset_config = dataset_config or {
            "daily": {
                "data_source": "Tushare",
                "data_type": "daily",
                "token": tushare_token,
                "api_name": "daily",
                "fields": [
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "change",
                    "pct_chg",
                    "vol",
                    "amount",
                ],
                "primary_key": ["ts_code", "trade_date"],
            }
        }
        logger.info(f"更新器初始化完成，SQLite 数据库: {self.db_path}")

    def _split_configs(self) -> tuple:
        market = {}
        sidecar = {}
        for name, config in self.dataset_config.items():
            if is_sidecar_dataset(config):
                sidecar[name] = config
            else:
                market[name] = config
        return market, sidecar

    def _get_all_fields(self) -> Set[str]:
        market_config, _ = self._split_configs()
        return {
            field
            for config in market_config.values()
            for field in config.get("fields", [])
            if field not in {"ts_code", "trade_date", "symbol", "date"}
        }

    def update_sidecar_datasets(self, end_date: Optional[str] = None) -> bool:
        """日常 sidecar 更新：有存量增量，空表/行业成分全量。"""
        _, sidecar_config = self._split_configs()
        if not sidecar_config:
            return True
        failed = []
        for name, config in sidecar_config.items():
            logger.info(f"更新 sidecar 数据集: {name}")
            if not self.dataset_installer.update_sidecar_dataset(
                name, config, end_date=end_date
            ):
                failed.append(name)
        if failed:
            raise RuntimeError(f"sidecar 数据集更新失败: {failed}")
        return True

    def get_current_stock_codes(self) -> Set[str]:
        stocks = self.storage.get_instruments()
        logger.info(f"SQLite 数据库中有 {len(stocks)} 只股票")
        return stocks

    def _get_data_for_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        market_config, _ = self._split_configs()
        if not market_config:
            return pd.DataFrame()
        data = self.data_processor.process_datasets_for_date(
            market_config,
            start_date,
            end_date,
            self.get_current_stock_codes(),
            self._get_all_fields(),
        )
        if data is None or data.empty:
            logger.warning(f"{start_date} 至 {end_date} 没有数据")
            return pd.DataFrame()
        return data

    def install_new_dataset(
        self,
        dataset_name: str,
        dataset_config: Dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        return self.dataset_installer.install_new_dataset(
            dataset_name,
            dataset_config,
            start_date=start_date,
            end_date=end_date,
        )

    def update_all_stock_by_trade_day(self, trade_date_str: str) -> int:
        current_stocks = self.get_current_stock_codes()
        result = self._get_data_for_date(trade_date_str, trade_date_str)
        if result.empty:
            return 0

        existing = set(result["symbol"].astype(str).unique())
        missing = current_stocks - existing
        if missing:
            logger.info(f"{trade_date_str} 有 {len(missing)} 只停牌股票，写入空值")
            missing_rows = pd.DataFrame(
                [
                    {
                        **{
                            column: np.nan
                            for column in result.columns
                            if column not in {"symbol", "date"}
                        },
                        "symbol": symbol,
                        "date": pd.to_datetime(trade_date_str),
                    }
                    for symbol in missing
                ]
            )
            result = pd.concat([result, missing_rows], ignore_index=True)

        written = self.storage.upsert_market_data(result)
        logger.info(f"{trade_date_str} 更新完成，共写入 {written} 行")
        del result
        gc.collect()
        return written

    def _market_uses_range_window(self) -> bool:
        """按数据源类上的 FetchSlice 决定窗口，不按厂商名分支。"""
        market, _ = self._split_configs()
        return market_uses_range_window(
            market.values(), self.data_processor.data_source_classes
        )

    def _upsert_range(self, start_date: str, end_date: str) -> int:
        data = self._get_data_for_date(start_date, end_date)
        if data is None or data.empty:
            logger.warning(f"{start_date} 至 {end_date} 没有数据")
            return 0
        written = self.storage.upsert_market_data(data)
        logger.info(f"{start_date} ~ {end_date} 写入 {written} 行")
        del data
        gc.collect()
        return written

    def _trade_days(self, start: str, end: str) -> list:
        """只枚举交易所交易日。有远端日历时以 trade_cal 为准，不用本地行情日截断。"""
        digits_s = "".join(ch for ch in str(start) if ch.isdigit())[:8]
        digits_e = "".join(ch for ch in str(end) if ch.isdigit())[:8]
        start_iso = f"{digits_s[:4]}-{digits_s[4:6]}-{digits_s[6:8]}"
        end_iso = f"{digits_e[:4]}-{digits_e[4:6]}-{digits_e[6:8]}"
        if self.pro is not None:
            frame = self.pro.trade_cal(
                exchange="SSE",
                start_date=digits_s,
                end_date=digits_e,
                is_open="1",
            )
            if frame is None:
                raise RuntimeError(
                    f"trade_cal 失败，无法覆盖 {start_iso} ~ {end_iso}"
                )
            if frame.empty:
                return []
            col = "cal_date" if "cal_date" in frame.columns else frame.columns[0]
            return [pd.Timestamp(str(v)).strftime("%Y-%m-%d") for v in frame[col]]
        days = [
            str(d)
            for d in self.storage.list_trade_dates(
                start_date=start_iso, end_date=end_iso
            )
        ]
        if days:
            last = max(str(d).replace("-", "")[:8] for d in days)
            if last < digits_e:
                raise RuntimeError(
                    f"本地交易日历只到 {last}，覆盖不到 {end_iso}，且没有远端日历"
                )
            return days
        raise RuntimeError(
            f"无交易日历覆盖 {start_iso} ~ {end_iso}，拒绝按自然日探测"
        )

    def rebuild_market_data(self, end_date: Optional[str] = None) -> bool:
        end_date = end_date or datetime.date.today().strftime("%Y%m%d")
        if self._market_uses_range_window():
            logger.info("行情源按区间拉取（BY_SYMBOL/BY_PANEL），一次写入整个窗口")
            return self._upsert_range(self.first_date_str, end_date) > 0
        wrote_any = False
        for trade_date in tqdm(
            self._trade_days(self.first_date_str, end_date), desc="重建历史数据"
        ):
            wrote_any = (
                self.update_all_stock_by_trade_day(trade_date.replace("-", "")) > 0
                or wrote_any
            )
        return wrote_any

    def update_market_data(self, end_date: Optional[str] = None, start_date: Optional[str] = None):
        end_date = end_date or datetime.date.today().strftime("%Y%m%d")
        latest = self.storage.get_latest_market_date()
        if latest is None:
            logger.info("SQLite 数据库为空，将执行完整重建")
            if start_date:
                self.first_date_str = "".join(ch for ch in str(start_date) if ch.isdigit())[:8]
            return self.rebuild_market_data(end_date)

        start = pd.Timestamp(latest) + pd.Timedelta(days=1)
        if start_date:
            start = max(start, pd.Timestamp(start_date))
        if start > pd.Timestamp(end_date):
            return True
        start_ymd = start.strftime("%Y%m%d")
        days = self._trade_days(start_ymd, end_date)
        if not days:
            logger.info("%s ~ %s 没有交易日，跳过增量", start_ymd, end_date)
            return True
        if self._market_uses_range_window():
            written = self._upsert_range(start_ymd, end_date)
            if written <= 0:
                logger.error(
                    "%s ~ %s 有 %s 个交易日但写入 0 行", start_ymd, end_date, len(days)
                )
                return False
            return True
        failed = []
        for trade_date in tqdm(days, desc="增量更新"):
            ymd = trade_date.replace("-", "")
            if self.update_all_stock_by_trade_day(ymd) <= 0:
                failed.append(trade_date)
        if failed:
            logger.error("预期交易日写入 0 行: %s", failed)
            return False
        return True

    def update_all(self, end_date: Optional[str] = None, start_date: Optional[str] = None):
        market_config, sidecar_config = self._split_configs()
        ok = True
        if market_config:
            original = self.dataset_config
            self.dataset_config = market_config
            try:
                ok = bool(self.update_market_data(end_date, start_date=start_date)) and ok
            finally:
                self.dataset_config = original
        if sidecar_config:
            try:
                ok = bool(self.update_sidecar_datasets(end_date)) and ok
            except Exception:
                logger.exception("sidecar 更新失败")
                ok = False
        return ok


def main():
    token = os.getenv("TUSHARE_TOKEN", "")
    updater = StockDataUpdater(
        db_path=str(default_db_path()),
        tushare_token=token,
    )
    updater.update_all()


if __name__ == "__main__":
    main()
