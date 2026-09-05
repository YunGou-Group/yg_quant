#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源基类，所有数据源都必须继承此类
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, Dict, Iterator

import pandas as pd


class FetchSlice(str, Enum):
    """HTTP/API 切片方式。引擎按类型选窗口，不按厂商名分支。

    BY_DATE: 每次一个交易日（Tushare 全市场 daily / 部分基本面截面）。
    BY_SYMBOL: 每只股票 + 日期区间（Akshare / BaoStock / TqSdk 历史）。
    BY_PANEL: codes[] + start/end（Ricequant / JoinQuant K 线 / Wind wsd）。
    """

    BY_DATE = "by_date"
    BY_SYMBOL = "by_symbol"
    BY_PANEL = "by_panel"


class DataSourceBase(ABC):
    """数据源基类，所有数据源都必须继承此类"""

    fetch_slice: ClassVar[FetchSlice] = FetchSlice.BY_DATE
    requires_token: ClassVar[bool] = False

    @classmethod
    def uses_range_window(cls) -> bool:
        """引擎应把 start/end 一次交给本源，由源内部按股票或面板切片。"""
        return cls.fetch_slice in (FetchSlice.BY_SYMBOL, FetchSlice.BY_PANEL)

    @abstractmethod
    def fetch_data(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取数据的抽象方法

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
            获取到的数据
        """
        pass

    def iter_chunks(
        self, config: Dict, start_date: str, end_date: str
    ) -> Iterator[pd.DataFrame]:
        """按块产出，引擎可分次 upsert。默认整段一次。"""
        frame = self.fetch_data(config, start_date, end_date)
        if frame is not None and not frame.empty:
            yield frame
