#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理类，负责协调多种数据源获取数据并进行标准化处理

主要职责：
1. 协调多种数据源获取数据
2. 标准化数据格式（股票代码、时间格式）
3. 合并多个数据集并填充缺失值
4. 交易日历检查
"""

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Type
import importlib
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd

from DailyUpdates.data_fetcher.data_source_base import DataSourceBase


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


class DataProcessor:
    """数据处理类，负责协调多种数据源获取数据并进行标准化处理"""
    
    def __init__(self, tushare_pro=None, storage=None):
        """
        初始化数据处理器
        
        Parameters
        ----------
        tushare_pro : ts.pro_api(), optional
            Tushare API实例，用于交易日历检查
            storage : SQLiteStorage, optional
            已覆盖区间内的交易日查库；日历末日之后的日期仍走数据源，否则无法拉当天
        """
        self.data_sources = {}
        self.pro = tushare_pro
        self.storage = storage
        self.data_source_classes = discover_data_source_classes()
        print("DataProcessor 初始化完成")
        print(f"发现的数据源类: {list(self.data_source_classes.keys())}")

    def _discover_data_sources(self) -> Dict[str, type]:
        return discover_data_source_classes()

    def register_data_source(self, name: str, data_source_instance):
        """注册数据源"""
        self.data_sources[name] = data_source_instance
        print(f"注册数据源: {name}")
    
    def getTushareInfo(
        self, getter: Callable, paras: Dict, fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        从tushare获取数据
        :param getter: tushare的接口函数
        :param paras: 接口函数的参数
        :param fields: 需要获取的字段
        :return->pd.DataFrame: 获取到的数据
        """
        df = pd.DataFrame()
        bFinishFetch = False
        while not bFinishFetch:
            tmp = None
            last_error = None
            for _ in range(61):
                try:
                    current_paras = paras.copy()
                    current_paras["offset"] = len(df)
                    tmp = getter(**current_paras, fields=fields)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(1)
            if last_error is not None:
                raise RuntimeError(
                    f"Tushare 接口连续 61 次失败 paras={paras}: {last_error}"
                ) from last_error
            if tmp is not None and len(tmp) > 0:
                df = tmp if len(df) == 0 else pd.concat([df, tmp], ignore_index=True)
            else:
                bFinishFetch = True
        return df
    
    def fetch_dataset(
        self, config: Dict, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame:
        """按单条数据集配置拉取原始数据（含行业维表）。"""
        return self._create_data_source_and_fetch(config, start_date, end_date)

    def process_datasets_for_date(self, dataset_config: Dict, start_date: str, end_date: str, 
                                current_stocks: set, all_fields: set) -> pd.DataFrame:
        """
        处理指定日期范围的所有数据集配置，返回合并后的DataFrame
        
        Parameters
        ----------
        dataset_config : Dict
            数据集配置字典
        start_date : str
            开始日期 (YYYYMMDD)
        end_date : str
            结束日期 (YYYYMMDD)
        current_stocks : set
            当前数据库中的股票集合
        all_fields : set
            所有字段集合
            
        Returns
        -------
        pd.DataFrame
            合并后的数据，包含所有配置的数据集字段
        """
        print(f"处理日期范围 {start_date} 到 {end_date} 的所有数据集配置...")
        
        # 检查是否为交易日（只有在增量更新时才检查）
        if start_date == end_date:
            # 增量更新模式：检查是否为交易日
            if not self._is_trading_day(start_date):
                print(f"日期 {start_date} 不是交易日，跳过数据处理")
                return pd.DataFrame()
            print(f"增量更新模式：日期 {start_date}")
        else:
            # 新因子安装模式：不需要检查交易日
            print(f"新因子安装模式：日期范围 {start_date} 到 {end_date}")
        
        # 存储各个数据集的数据
        dataset_dfs = {}
        # 拉空的必填数据集。局部失败绝不能静默合并，否则会把已有数据覆盖成 NULL。
        empty_required = []

        # 拉取每个数据集的数据（sidecar 维表走独立路径，不进入行情合并）
        for dataset_name, config in dataset_config.items():
            if config.get("data_type") in {
                "industry",
                "stock_info",
                "financial",
                "index_constituent",
            }:
                print(f"跳过 sidecar 数据集（非行情路径）: {dataset_name}")
                continue
            print(f"拉取数据集: {dataset_name}")

            try:
                # 动态创建数据源实例
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

            # 标准化股票代码和时间
            df = self._normalize_dataframe(df, dataset_name, config)

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
            return pd.DataFrame()


        # 合并所有数据集并填充缺失值
        result_df = self._merge_and_fill_data(dataset_dfs, start_date, current_stocks, all_fields)
        print(result_df)
        print(f"合并后获取到 {len(result_df)} 条记录，字段: {list(result_df.columns)}")
        return result_df
    
    def _create_data_source_and_fetch(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        根据配置动态创建数据源实例并获取数据
        
        Parameters
        ----------
        config : Dict
            数据源配置
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        pd.DataFrame
            获取到的数据
        """
        data_source_type = config.get('data_source', 'Tushare')
        
        # 使用动态发现的数据源类
        if data_source_type not in self.data_source_classes:
            print(f"不支持的数据源类型: {data_source_type}")
            print(f"可用的数据源类型: {list(self.data_source_classes.keys())}")
            return pd.DataFrame()
        
        # 动态创建数据源实例
        data_source_class = self.data_source_classes[data_source_type]
        data_source = data_source_class()
        return data_source.fetch_data(config, start_date, end_date)
    
    def _merge_and_fill_data(self, dataset_dfs: Dict[str, pd.DataFrame], trade_date_str: str, 
                            current_stocks: set, all_fields: set) -> pd.DataFrame:
        """
        合并所有数据集并填充缺失值
        
        Parameters
        ----------
        dataset_dfs : Dict
            数据集字典 {dataset_name: DataFrame}
        trade_date_str : str
            交易日期
        current_stocks : set
            当前数据库中的股票集合
        all_fields : set
            所有字段集合
            
        Returns
        -------
        pd.DataFrame
            合并并填充后的数据
        """
        if not dataset_dfs:
            return pd.DataFrame()
        
        # 以第一个数据集为基础进行左连接
        base_dataset = list(dataset_dfs.keys())[0]
        result_df = dataset_dfs[base_dataset].copy()
        
        # 分别处理股票数据和指数数据
        stock_datasets = {}
        index_datasets = {}
        
        for dataset_name, df in dataset_dfs.items():
            if df.empty:
                print(f"数据集 {dataset_name} 为空，跳过合并")
                continue
            
            # 判断是否为指数数据
            if 'symbol' in df.columns and any(str(s).startswith('index_') for s in df['symbol']):
                index_datasets[dataset_name] = df
                print(f"数据集 {dataset_name} 识别为指数数据")
            else:
                stock_datasets[dataset_name] = df
                print(f"数据集 {dataset_name} 识别为股票数据")
        
        # 先合并股票数据
        for dataset_name, df in stock_datasets.items():
            if dataset_name == base_dataset:
                continue
            
            # 动态确定合并键
            merge_keys = []
            if 'symbol' in result_df.columns and 'symbol' in df.columns:
                merge_keys.append('symbol')
            elif 'ts_code' in result_df.columns and 'ts_code' in df.columns:
                merge_keys.append('ts_code')
            
            if 'date' in result_df.columns and 'date' in df.columns:
                merge_keys.append('date')
            elif 'trade_date' in result_df.columns and 'trade_date' in df.columns:
                merge_keys.append('trade_date')
            
            if not merge_keys:
                print(f"警告：数据集 {dataset_name} 无法确定合并键，跳过合并")
                continue
            
            print(f"合并股票数据集 {dataset_name}，使用合并键: {merge_keys}")
            result_df = pd.merge(result_df, df, on=merge_keys, how='left', suffixes=('', f'_{dataset_name}'))
            print(f"合并数据集 {dataset_name}，结果行数: {len(result_df)}")
        
        # 然后处理指数数据（先合并内部数据集，再追加到结果中）
        if index_datasets:
            print("处理指数数据，先合并内部数据集...")
            # 先合并指数数据内部的多个数据集
            merged_index_df = None
            for dataset_name, df in index_datasets.items():
                if merged_index_df is None:
                    merged_index_df = df.copy()
                else:
                    # 动态确定合并键
                    merge_keys = []
                    if 'symbol' in merged_index_df.columns and 'symbol' in df.columns:
                        merge_keys.append('symbol')
                    elif 'ts_code' in merged_index_df.columns and 'ts_code' in df.columns:
                        merge_keys.append('ts_code')
                    
                    if 'date' in merged_index_df.columns and 'date' in df.columns:
                        merge_keys.append('date')
                    elif 'trade_date' in merged_index_df.columns and 'trade_date' in df.columns:
                        merge_keys.append('trade_date')
                    
                    if merge_keys:
                        print(f"合并指数数据集 {dataset_name}，使用合并键: {merge_keys}")
                        merged_index_df = pd.merge(merged_index_df, df, on=merge_keys, how='outer', suffixes=('', f'_{dataset_name}'))
                        print(f"合并指数数据集 {dataset_name}，结果行数: {len(merged_index_df)}")
                    else:
                        print(f"警告：指数数据集 {dataset_name} 无法确定合并键，跳过合并")
            
            # 将合并后的指数数据追加到结果中
            if merged_index_df is not None:
                print(f"将合并后的指数数据追加到结果中，指数数据行数: {len(merged_index_df)}")
                result_df = pd.concat([result_df, merged_index_df], ignore_index=True)
                print(f"指数数据已追加，总结果行数: {len(result_df)}")
        
        # 为所有缺失的股票和因子填充NaN
        if not result_df.empty and current_stocks:
            # 动态确定股票代码列名
            stock_col = None
            if 'symbol' in result_df.columns:
                stock_col = 'symbol'
            elif 'ts_code' in result_df.columns:
                stock_col = 'ts_code'
            
            if stock_col:
                # 获取当前数据中的股票
                existing_stocks = set(result_df[stock_col].unique())
                missing_stocks = current_stocks - existing_stocks
                
                if missing_stocks:
                    print(f"为缺失的 {len(missing_stocks)} 只股票填充NaN")
                    
                    # 动态确定日期列名
                    date_col = None
                    if 'date' in result_df.columns:
                        date_col = 'date'
                    elif 'trade_date' in result_df.columns:
                        date_col = 'trade_date'
                    
                    if date_col:
                        # 为缺失的股票创建数据行
                        missing_data = []
                        for stock_code in missing_stocks:
                            missing_row = {stock_col: stock_code}
                            
                            # 设置日期
                            if date_col == 'date':
                                missing_row[date_col] = pd.to_datetime(trade_date_str)
                            else:
                                missing_row[date_col] = trade_date_str
                            
                            # 为所有字段填充NaN
                            for field in all_fields:
                                if field not in [stock_col, date_col]:
                                    missing_row[field] = np.nan
                            
                            missing_data.append(missing_row)
                        
                        # 将缺失数据添加到结果中
                        missing_df = pd.DataFrame(missing_data)
                        result_df = pd.concat([result_df, missing_df], ignore_index=True)
                        print(f"已为缺失股票填充NaN，总行数: {len(result_df)}")
        
        # 确保所有字段都存在，缺失的字段填充NaN
        if not result_df.empty:
            for field in all_fields:
                if field not in result_df.columns:
                    print(f"为缺失字段 {field} 填充NaN")
                    result_df[field] = np.nan
        
        # 按股票代码和日期排序
        if not result_df.empty:
            # 动态确定排序字段
            sort_cols = []
            if 'symbol' in result_df.columns:
                sort_cols.append('symbol')
            elif 'ts_code' in result_df.columns:
                sort_cols.append('ts_code')
            
            if 'date' in result_df.columns:
                sort_cols.append('date')
            elif 'trade_date' in result_df.columns:
                sort_cols.append('trade_date')
            
            if sort_cols:
                result_df = result_df.sort_values(sort_cols).reset_index(drop=True)
        
        return result_df
    
    def _is_trading_day(self, trade_date_str: str) -> bool:
        """检查指定日期是否为交易日。

        库内日历只对已写入区间可信：在 min~max 之间且不在表里，才是休市。
        晚于日历末日的日期（增量补当天）必须再问 Tushare，不能当成非交易日跳过。
        """
        day = str(trade_date_str or "").replace("-", "")
        if len(day) >= 8:
            iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        else:
            iso = str(trade_date_str)
        if self.storage is not None:
            try:
                calendar = self.storage.list_trade_dates()
            except Exception:
                calendar = []
            if calendar:
                iso_set = {str(d) for d in calendar}
                ymd_set = {str(d).replace("-", "")[:8] for d in calendar}
                if day[:8] in ymd_set or iso in iso_set:
                    return True
                last = max(ymd_set)
                first = min(ymd_set)
                if first <= day[:8] <= last:
                    return False
        if self.pro is None:
            for cls in self.data_source_classes.values():
                hit = cls.is_trading_day(trade_date_str)
                if hit is not None:
                    return hit
            return True
        df = self.getTushareInfo(
            getter=self.pro.daily,
            paras={'trade_date': trade_date_str},
            fields=['ts_code']
        )
        if df.empty:
            print(f"日期 {trade_date_str} 不是交易日（daily数据为空）")
            return False
        print(f"日期 {trade_date_str} 是交易日，获取到 {len(df)} 条daily数据")
        return True
    
    def _normalize_dataframe(self, df: pd.DataFrame, dataset_name: str, config: dict) -> pd.DataFrame:
        """
        标准化 DataFrame：统一股票代码、日期和字段格式
        
        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        dataset_name : str
            数据集名称
        config : dict
            数据集配置
            
        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 转换股票代码格式 (600000.SH -> SH600000)
        if 'ts_code' in df.columns:
            def convert_stock_code(ts_code):
                try:
                    if '.' in ts_code:
                        # 标准格式：600000.SH -> SH600000
                        parts = ts_code.split('.')
                        if len(parts) == 2:
                            return f"{parts[1]}{parts[0]}"
                    # 如果没有点号或格式不对，直接返回原值
                    return ts_code
                except:
                    return ts_code
            
            df['symbol'] = df['ts_code'].apply(convert_stock_code)
            
            # 为指数数据添加index前缀（只给股票代码，不给字段名）
            if config.get('data_type') == 'index':
                df['symbol'] = df['symbol'].apply(lambda x: f"index_{x}")
                # 注意：这里不给字段名添加index前缀，保持原样
        
        # 转换时间格式
        if 'trade_date' in df.columns:
            df['date'] = pd.to_datetime(df['trade_date'])
        
        # 构建保留字段列表：symbol, date + 业务字段
        keep_columns = []
        if 'symbol' in df.columns:
            keep_columns.append('symbol')
        if 'date' in df.columns:
            keep_columns.append('date')
        
        # 添加业务字段（排除已处理的字段）
        for field in config['fields']:
            if field not in ['ts_code', 'trade_date'] and field in df.columns:
                keep_columns.append(field)
        
        selected = df.loc[:, keep_columns]
        return selected if isinstance(selected, pd.DataFrame) else selected.to_frame()