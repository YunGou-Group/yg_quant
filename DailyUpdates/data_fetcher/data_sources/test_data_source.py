#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据源类，用于生成测试因子
"""

import pandas as pd
from typing import Dict
import sqlite3
import os
from DailyUpdates.data_fetcher.data_source_base import DataSourceBase, FetchSlice


class TestDataSource(DataSourceBase):
    """测试数据源类，用于生成测试因子"""

    __test__ = False
    fetch_slice = FetchSlice.BY_SYMBOL
    requires_token = False
    
    def __init__(self):
        """初始化测试数据源"""
        pass
    
    def fetch_data(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成测试数据，为每只股票创建source_test因子，值为1
        
        Parameters
        ----------
        config : Dict
            数据源配置
        start_date : str
            开始日期，格式：YYYYMMDD
        end_date : str
            结束日期，格式：YYYYMMDD
            
        Returns
        -------
        pd.DataFrame
            包含source_test因子的数据
        """
        # 获取股票列表
        stock_list = config.get('stock_list', [])
        
        # 如果没有提供股票列表，尝试从 SQLite 数据库获取
        db_path = config.get('db_path')
        if not stock_list and db_path:
            try:
                with sqlite3.connect(db_path) as connection:
                    rows = connection.execute(
                        "SELECT symbol FROM instruments WHERE is_index = 0"
                    ).fetchall()
                stock_list = [row[0] for row in rows]
                print(f"TestDataSource: 从 SQLite 获取到 {len(stock_list)} 只股票")
            except Exception as e:
                print(f"TestDataSource: 读取股票列表失败: {e}")
        
        if not stock_list:
            print("TestDataSource: 未提供股票列表，返回空DataFrame")
            return pd.DataFrame()
        
        # 生成指定日期范围的数据
        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d')
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        # 过滤交易日（简单实现，实际应该使用交易日历）
        # 这里假设周末不是交易日
        trading_dates = [d for d in date_range if d.weekday() < 5]
        
        data = []
        for stock_code in stock_list:
            for date in trading_dates:
                data.append({
                    'ts_code': stock_code,  # 修改：使用ts_code而不是symbol，保持列名一致
                    'trade_date': date.strftime('%Y%m%d'),  # 修改：使用trade_date而不是date，保持列名一致
                    'source_test': 1  # 修改：所有股票的source_test因子值都为1
                })
        
        df = pd.DataFrame(data)
        print(f"TestDataSource: 为 {len(stock_list)} 只股票生成了 {len(trading_dates)} 个交易日的source_test因子，值均为1")
        return df
