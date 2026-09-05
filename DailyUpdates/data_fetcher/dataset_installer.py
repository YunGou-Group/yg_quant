#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install newly configured datasets into SQLite (market + sidecar tables)."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger
from tqdm import tqdm

from DailyUpdates.data_fetcher.data_fetcher import source_class_for_config
from DailyUpdates.data_fetcher.data_source_base import FetchSlice
from DailyUpdates.preprocessing.market_bars import load_prev_adj

SIDECAR_DATA_TYPES = {"industry", "stock_info", "financial", "index_constituent"}
_TRUNCATED_BY_DATE_APIS = {"stk_limit", "daily_basic", "adj_factor"}


def is_industry_dataset(config: Dict) -> bool:
    return config.get("data_type") == "industry"


def is_sidecar_dataset(config: Dict) -> bool:
    """非行情宽表路径：行业 / 股票基础 / 财务 / 指数成分。"""
    return config.get("data_type") in SIDECAR_DATA_TYPES


def always_full_sidecar(config: Dict) -> bool:
    """整表替换的 sidecar（含历史调样 / 更正），不做增量。"""
    if (
        config.get("data_type") == "industry"
        and config.get("api_name") == "index_member_all"
    ):
        return True
    return (
        config.get("data_type") == "stock_info"
        and config.get("api_name") == "namechange"
    )


class DatasetInstaller:
    def __init__(self, storage, data_fetcher, data_processor):
        self.storage = storage
        self.data_fetcher = data_fetcher
        self.data_processor = data_processor

    def install_new_dataset(
        self,
        dataset_name: str,
        dataset_config: Dict,
        stock_codes: Optional[List[str]] = None,
        max_workers: int = 4,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        del max_workers  # SQLite writes are intentionally serialized.
        if is_sidecar_dataset(dataset_config):
            return self._write_sidecar_dataset(
                dataset_name,
                dataset_config,
                start_date,
                end_date,
                mode="full",
            )

        if self._should_install_by_trade_calendar(dataset_config):
            return self._install_by_trade_calendar(
                dataset_name, dataset_config, start_date, end_date
            )

        is_index = dataset_config.get("data_type") == "index" or bool(
            dataset_config.get("index_list")
        )
        stocks = set(stock_codes or self.storage.get_instruments())
        if not stocks and not is_index:
            logger.error("数据库中没有可用于历史回填的股票")
            return False

        cal_start, cal_end = self.storage.get_calendar_range()
        start_date = (start_date or cal_start or "").replace("-", "")
        end_date = (end_date or cal_end or "").replace("-", "")
        if not start_date or not end_date:
            logger.error("数据库中没有交易日历，无法回填新字段")
            return False

        logger.info(
            f"回填数据集 {dataset_name}: {start_date} 至 {end_date}"
            + (
                f", 指数 {dataset_config.get('index_list')}"
                if is_index
                else f", {len(stocks)} 只股票"
            )
        )
        frames = self.data_fetcher.fetch_market_datasets(
            {dataset_name: dataset_config},
            start_date,
            end_date,
        )
        prev_adj = (
            load_prev_adj(self.storage, start_date)
            if "adj_factor" in (dataset_config.get("fields") or [])
            else None
        )
        data = self.data_processor.build_panel(
            frames,
            {dataset_name: dataset_config},
            start_date,
            stocks,
            set(),
            prev_adj=prev_adj,
        )
        if data is None or data.empty:
            logger.error(f"数据集 {dataset_name} 未获取到数据")
            return False

        written = self.storage.upsert_market_data(data)
        logger.info(f"数据集 {dataset_name} 历史回填完成，共写入 {written} 行")
        return written > 0

    def _should_install_by_trade_calendar(self, dataset_config: Dict) -> bool:
        """仅 BY_DATE 源的宽截面接口需要按交易日切片，避免一次返回被截断。"""
        if dataset_config.get("api_name") not in _TRUNCATED_BY_DATE_APIS:
            return False
        classes = getattr(self.data_fetcher, "data_source_classes", None)
        cls = source_class_for_config(dataset_config, classes)
        slice_kind = getattr(cls, "fetch_slice", FetchSlice.BY_DATE)
        return slice_kind == FetchSlice.BY_DATE

    def update_sidecar_dataset(
        self,
        dataset_name: str,
        dataset_config: Dict,
        end_date: Optional[str] = None,
    ) -> bool:
        """日常更新：有存量则增量，空表/行业成分则全量。"""
        return self._write_sidecar_dataset(
            dataset_name,
            dataset_config,
            start_date=None,
            end_date=end_date,
            mode="auto",
        )

    def _install_by_trade_calendar(
        self,
        dataset_name: str,
        dataset_config: Dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """按交易日历逐日拉取，避免 BY_DATE 源大区间一次返回被截断。"""
        cal_start, cal_end = self.storage.get_calendar_range()
        start = pd.Timestamp(start_date or cal_start).strftime("%Y-%m-%d")
        end = pd.Timestamp(end_date or cal_end).strftime("%Y-%m-%d")
        value_fields = [
            field
            for field in dataset_config.get("fields") or []
            if field not in {"ts_code", "trade_date", "symbol", "date"}
        ]
        dates = self._trade_dates_missing_fields(start, end, value_fields)
        if not dates:
            logger.info(
                f"{dataset_name} 在 {start} ~ {end} 无需回填（目标字段已有数据）"
            )
            return True

        pause = float(dataset_config.get("pause_seconds") or 0.15)
        logger.info(
            f"按日回填 {dataset_name}({dataset_config.get('api_name')}): "
            f"{dates[0]} ~ {dates[-1]} 共 {len(dates)} 个交易日 "
            f"fields={value_fields}"
        )
        written_total = 0
        empty_days = 0
        for trade_date in tqdm(dates, desc=f"回填{dataset_name}"):
            day = trade_date.replace("-", "")
            raw = self.data_fetcher.fetch_dataset(dataset_config, day, day)
            if raw is None or raw.empty:
                empty_days += 1
                continue
            prev_adj = (
                load_prev_adj(self.storage, day)
                if "adj_factor" in (dataset_config.get("fields") or [])
                else None
            )
            data = self.data_processor.normalize_dataframe(
                raw, dataset_name, dataset_config, prev_adj=prev_adj
            )
            if data is None or data.empty:
                empty_days += 1
                continue
            written_total += self.storage.upsert_market_data(data)
            if pause > 0:
                time.sleep(pause)
        logger.info(
            f"{dataset_name} 按日回填完成，写入 {written_total} 行，"
            f"空日 {empty_days}/{len(dates)}"
        )
        if empty_days:
            logger.error(
                f"{dataset_name} 按日回填有 {empty_days}/{len(dates)} 个交易日为空，视为失败"
            )
            return False
        return True

    def _trade_dates_missing_fields(
        self, start: str, end: str, fields: Sequence[str]
    ) -> List[str]:
        """日历中任一目标字段未覆盖当天全部主键行的日期。

        指数行没有 adj_factor，核验该字段时排除 symbol LIKE 'index_%'。
        """
        with self.storage._connect() as connection:
            calendar = [
                row["trade_date"]
                for row in connection.execute(
                    "SELECT trade_date FROM trade_calendar "
                    "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                    (start, end),
                )
            ]
            if not calendar or not fields:
                return calendar
            existing = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(market_data)")
            }
            usable = [field for field in fields if field in existing]
            if not usable:
                return calendar
            complete = None
            for field in usable:
                extra = ""
                if field == "adj_factor":
                    extra = " AND symbol NOT LIKE 'index_%'"
                    if "close" in existing:
                        extra += " AND close IS NOT NULL"
                filled = {
                    row["trade_date"]
                    for row in connection.execute(
                        f'SELECT trade_date FROM market_data '
                        f"WHERE trade_date BETWEEN ? AND ?{extra} "
                        f'GROUP BY trade_date HAVING COUNT("{field}") >= COUNT(*)',
                        (start, end),
                    )
                }
                complete = filled if complete is None else (complete & filled)
        complete = complete or set()
        return [day for day in calendar if day not in complete]

    def _resolve_sidecar_mode(
        self, dataset_config: Dict, mode: str
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (effective_mode, incremental_start_yyyymmdd_or_none).

        effective_mode: 'full' | 'incremental' | 'skip'
        """
        data_type = dataset_config.get("data_type")
        api_name = dataset_config.get("api_name")
        src = dataset_config.get("src", "SW2021")

        if data_type == "index_constituent" and api_name == "index_weight":
            latest = self.storage.get_index_constituent_latest_date(
                dataset_config.get("index_list")
            )
            if not latest:
                return "full", None
            # INSTALL 在注册表未保存时会反复以 mode=full 进来；表里已有截面则只补增量。
            # 新加入 index_list 但库里还没有的指数，由 _write_sidecar_dataset 单独全量回填。
            return "incremental", latest.replace("-", "")

        if mode == "full" or always_full_sidecar(dataset_config):
            return "full", None

        if data_type == "industry" and api_name == "index_classify":
            if self.storage.count_industry_classify(src) == 0:
                return "full", None
            # 分类维表无自然日序；日常增量跳过，配置变更走 INSTALL 全量。
            return "skip", None

        if data_type == "stock_info" and api_name == "stock_basic":
            if self.storage.count_stock_basic() == 0:
                return "full", None
            return "incremental", None

        if data_type == "stock_info" and api_name == "namechange":
            # 历史名称表本身很小，且历史行可能被更正，每次整表重拉最省心。
            return "full", None

        if data_type == "financial" and api_name in (
            "fina_indicator",
            "fina_indicator_vip",
        ):
            latest = self.storage.get_financial_latest_end_date()
            if not latest:
                return "full", None
            # 含最新报告期，便于公告更正覆盖
            return "incremental", latest.replace("-", "")

        # 未知 sidecar：保守全量
        return "full", None

    def _missing_index_codes(self, dataset_config: Dict) -> List[str]:
        wanted = [str(code) for code in (dataset_config.get("index_list") or [])]
        if not wanted or not hasattr(self.storage, "list_index_constituent_codes"):
            return wanted
        present = {str(code) for code in self.storage.list_index_constituent_codes()}
        return [code for code in wanted if code not in present]

    def _write_sidecar_dataset(
        self,
        dataset_name: str,
        dataset_config: Dict,
        start_date: Optional[str],
        end_date: Optional[str],
        mode: str,
    ) -> bool:
        data_type = dataset_config.get("data_type")
        api_name = dataset_config.get("api_name")
        cal_start, cal_end = self.storage.get_calendar_range()
        end = (end_date or cal_end or "").replace("-", "")

        effective, incr_start = self._resolve_sidecar_mode(dataset_config, mode)
        if effective == "skip":
            logger.info(
                f"sidecar {dataset_name} 已有存量且无需按日增量，跳过"
            )
            return True

        if effective == "full":
            start = (start_date or cal_start or "").replace("-", "")
            write_mode = "full"
        else:
            start = (incr_start or start_date or cal_start or "").replace("-", "")
            write_mode = "incremental"

        if (
            write_mode == "incremental"
            and data_type == "index_constituent"
            and api_name == "index_weight"
        ):
            missing = self._missing_index_codes(dataset_config)
            if missing:
                miss_cfg = dict(dataset_config)
                miss_cfg["index_list"] = missing
                start_full = (start_date or cal_start or "").replace("-", "")
                logger.info(
                    f"指数成分缺历史 {missing}，先全量回填 {start_full}->{end}"
                )
                try:
                    miss_data = self.data_fetcher.fetch_dataset(
                        miss_cfg, start_full, end
                    )
                except Exception:
                    logger.exception(
                        f"数据集 {dataset_name} 回填缺失指数失败: {missing}"
                    )
                    return False
                if miss_data is None or miss_data.empty:
                    logger.error(
                        f"数据集 {dataset_name} 缺失指数 {missing} 未获取到数据"
                    )
                    return False
                written = self.storage.upsert_index_constituents(miss_data)
                if written <= 0:
                    return False
                logger.info(
                    f"缺失指数 {missing} 已回填 {written} 行，继续增量 {start}->{end}"
                )

        logger.info(
            f"{'全量' if write_mode == 'full' else '增量'}更新 sidecar "
            f"{dataset_name}: type={data_type}, api={api_name}, "
            f"range={start}->{end}"
        )
        try:
            data = self.data_fetcher.fetch_dataset(dataset_config, start, end)
        except Exception:
            logger.exception(f"数据集 {dataset_name} 拉取失败")
            return False
        if data is None or data.empty:
            if write_mode == "incremental":
                logger.info(f"数据集 {dataset_name} 增量区间无新数据")
                return True
            logger.error(f"数据集 {dataset_name} 未获取到数据")
            return False

        logger.info(
            f"开始写入 {dataset_name} 到 SQLite，原始 {len(data)} 行 "
            f"(mode={write_mode})"
        )
        written = self._persist_sidecar(
            dataset_config, data, write_mode=write_mode
        )
        if written < 0:
            return False
        logger.info(f"数据集 {dataset_name} 写入完成，共 {written} 行")
        return True

    def _persist_sidecar(
        self, dataset_config: Dict, data: pd.DataFrame, write_mode: str
    ) -> int:
        data_type = dataset_config.get("data_type")
        api_name = dataset_config.get("api_name")
        src = dataset_config.get("src", "SW2021")

        if data_type == "industry":
            if api_name == "index_classify":
                return self.storage.replace_industry_classify(data, src=src)
            if api_name == "index_member_all":
                return self.storage.replace_industry_members(data, src=src)
            logger.error(f"不支持的行业 api_name: {api_name}")
            return -1

        if data_type == "stock_info" and api_name == "stock_basic":
            if write_mode == "full":
                return self.storage.replace_stock_basic(data)
            return self.storage.upsert_stock_basic(data)

        if data_type == "stock_info" and api_name == "namechange":
            if write_mode == "full":
                return self.storage.replace_stock_namechange(data)
            return self.storage.upsert_stock_namechange(data)

        if data_type == "financial" and api_name in (
            "fina_indicator",
            "fina_indicator_vip",
        ):
            if write_mode == "full":
                return self.storage.replace_financial_indicator(data)
            return self.storage.upsert_financial_indicator(data)

        if data_type == "index_constituent" and api_name == "index_weight":
            if write_mode == "full":
                return self.storage.replace_index_constituents(
                    data, index_codes=dataset_config.get("index_list")
                )
            return self.storage.upsert_index_constituents(data)

        logger.error(
            f"不支持的 sidecar 组合: data_type={data_type}, api={api_name}"
        )
        return -1
