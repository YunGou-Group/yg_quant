#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按配置发现数据源并拉取原始表。不做标准化或跨表合并。"""

from functools import lru_cache
from typing import Dict, Optional, Type
import importlib
import inspect
from pathlib import Path

import pandas as pd

from DailyUpdates.data_fetcher.data_source_base import DataSourceBase

_SIDECAR_TYPES = {"industry", "stock_info", "financial", "index_constituent"}


@lru_cache(maxsize=1)
def discover_data_source_classes() -> Dict[str, Type[DataSourceBase]]:
    """扫描 data_sources 目录，键为类名去掉 DataSource 后缀。"""
    data_source_classes: Dict[str, Type[DataSourceBase]] = {}
    data_sources_dir = Path(__file__).parent / "data_sources"
    try:
        for py_file in data_sources_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            module_name = f"DailyUpdates.data_fetcher.data_sources.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, DataSourceBase)
                        and obj is not DataSourceBase
                    ):
                        class_name = name
                        if class_name.endswith("DataSource"):
                            class_name = class_name[:-10]
                        data_source_classes[class_name] = obj
                        print(f"发现数据源类: {name} -> {class_name}")
            except Exception as e:
                print(f"导入模块 {module_name} 失败: {e}")
                continue
        print(f"总共发现 {len(data_source_classes)} 个数据源类")
        return data_source_classes
    except Exception as e:
        print(f"扫描数据源目录失败: {e}")
        return {}


def source_class_for_config(
    config: Dict, classes: Optional[Dict[str, Type[DataSourceBase]]] = None
) -> Optional[Type[DataSourceBase]]:
    name = str(config.get("data_source") or "Tushare")
    registry = classes if classes is not None else discover_data_source_classes()
    return registry.get(name)


def market_uses_range_window(
    configs, classes: Optional[Dict[str, Type[DataSourceBase]]] = None
) -> bool:
    """全部行情源都声明按区间拉取时，引擎走一次 start/end，而不是按自然日循环。"""
    registry = classes if classes is not None else discover_data_source_classes()
    resolved = []
    for config in configs:
        cls = source_class_for_config(config, registry)
        if cls is None:
            return False
        resolved.append(cls)
    return bool(resolved) and all(cls.uses_range_window() for cls in resolved)


class DataFetcher:
    """发现数据源并按配置拉取原始 DataFrame。交易日由调用方用 trade_cal 排队。"""

    def __init__(self, tushare_pro=None, storage=None):
        self.data_sources = {}
        self.pro = tushare_pro
        self.storage = storage
        self.data_source_classes = discover_data_source_classes()
        print("DataFetcher 初始化完成")
        print(f"发现的数据源类: {list(self.data_source_classes.keys())}")

    def register_data_source(self, name: str, data_source_instance):
        self.data_sources[name] = data_source_instance
        print(f"注册数据源: {name}")

    def fetch_dataset(
        self, config: Dict, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame:
        """按单条数据集配置拉取原始数据（含行业维表）。"""
        return self._create_data_source_and_fetch(config, start_date, end_date)

    def fetch_market_datasets(
        self, dataset_config: Dict, start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """拉取行情配置对应的原始表。必填为空则拒绝。"""
        print(f"拉取日期范围 {start_date} 到 {end_date} 的行情数据集...")
        if start_date == end_date:
            print(f"增量更新模式：日期 {start_date}")
        else:
            print(f"新因子安装模式：日期范围 {start_date} 到 {end_date}")

        dataset_dfs: Dict[str, pd.DataFrame] = {}
        empty_required = []
        for dataset_name, config in dataset_config.items():
            if config.get("data_type") in _SIDECAR_TYPES:
                print(f"跳过 sidecar 数据集（非行情路径）: {dataset_name}")
                continue
            print(f"拉取数据集: {dataset_name}")
            try:
                df = self._create_data_source_and_fetch(config, start_date, end_date)
            except Exception as exc:
                raise RuntimeError(
                    f"数据集 {dataset_name} 在 {start_date} ~ {end_date} 拉取失败: {exc}"
                ) from exc
            if df is None or df.empty:
                if config.get("optional"):
                    print(f"可选数据集 {dataset_name} 在 {start_date} ~ {end_date} 没有数据，跳过")
                    continue
                print(f"数据集 {dataset_name} 在日期范围 {start_date} 到 {end_date} 没有数据")
                empty_required.append(dataset_name)
                continue
            dataset_dfs[dataset_name] = df
            print(f"数据集 {dataset_name} 获取到 {len(df)} 条记录")

        if empty_required:
            raise RuntimeError(
                f"{start_date} ~ {end_date} 以下数据集为空，拒绝写入残缺行情: "
                f"{empty_required}（确认为非交易日或数据源确实无数据时，"
                f"可在配置里给该数据集加 'optional': True）"
            )
        if not dataset_dfs:
            print(f"日期范围 {start_date} 到 {end_date} 所有数据集都没有数据")
        return dataset_dfs

    def _create_data_source_and_fetch(
        self, config: Dict, start_date: str, end_date: str
    ) -> pd.DataFrame:
        data_source_type = config.get("data_source", "Tushare")
        if data_source_type not in self.data_source_classes:
            print(f"不支持的数据源类型: {data_source_type}")
            print(f"可用的数据源类型: {list(self.data_source_classes.keys())}")
            return pd.DataFrame()
        data_source_class = self.data_source_classes[data_source_type]
        data_source = data_source_class()
        return data_source.fetch_data(config, start_date, end_date)
