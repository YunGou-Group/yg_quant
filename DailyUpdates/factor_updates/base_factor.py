#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子抽象基类，提供计算和依赖管理的统一接口。"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import logging


class BaseFactor(ABC):
    """
    因子抽象基类
    
    提供因子计算和依赖管理的统一接口
    因子类只负责计算，读取与保存由 FactorUpdater 统一管理

    role: risk 进暴露矩阵；alpha 当信号。
    stage: production 可进实盘合成；candidate 仅研究。
    lookback_days: 增量更新时 FactorUpdater 额外加载的自然日。
    """

    name: str = ""
    description: str = ""
    dependencies: List[str] = []
    role: str = "alpha"
    stage: str = "production"
    lookback_days: int = 450
    
    def __init__(self):
        """
        初始化因子
        
        子类应该在__init__中设置self.name、self.description、self.dependencies
        """
        self.logger: Optional[logging.Logger] = None
    
    def get_name(self) -> str:
        """
        返回因子名称
        因子名称务必保证唯一性且不包含特殊字符
        """
        return self.name
    
    def get_description(self) -> str:
        """
        返回因子描述
        """
        return self.description
    
    def get_dependencies(self) -> List[str]:
        """
        返回依赖的基础数据字段列表
        """
        return self.dependencies
    
    def _get_logger(self):
        """获取logger，延迟初始化"""
        if self.logger is None:
            self.logger = logging.getLogger(f"Factor.{self.get_name()}")
        return self.logger
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        计算因子
        
        Parameters
        ----------
        data : pd.DataFrame
            股票数据 DataFrame，一定包含日期和股票代码列
        start_date : Optional[str]
            更新起始日期，用于需要历史数据的因子（如移动平均线）
            格式：'YYYY-MM-DD' 或 'YYYYMMDD'
            如果为None，则使用data中的最早日期
            
        Returns
        -------
        pd.DataFrame
            因子结果 DataFrame，必须包含日期date、股票代码symbol和因子值列
        """
        pass
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Factor(name='{self.get_name()}', description='{self.description}', dependencies={self.dependencies})"
         