#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽 CNE5 风格因子。

每个类写一条 Bin。``style_size`` 是总市值对数；其余多描述子因子是
去极值、标准化并按聚宽权重合成后的值，正交也已做完。按股票池再标准化
在 ExposureEngine。calculate() 不 shift open，也不调用 ReturnCalculator。
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

from ._engine import extract_style


class StyleSizeFactor(BaseFactor):
    name = "style_size"
    description = "聚宽 size：ln(总市值)"
    dependencies: List[str] = ["total_mv"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "size", start_date=start_date)


class StyleMomentumFactor(BaseFactor):
    name = "style_momentum"
    description = "聚宽 momentum：滞后21日、504日超额对数收益的指数加权，半衰期126日"
    dependencies: List[str] = ["pct_chg", "circ_mv", "vol"]
    role = "risk"
    lookback_days = 900

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "momentum", start_date=start_date)


class StyleLiquidityFactor(BaseFactor):
    name = "style_liquidity"
    description = "聚宽 liquidity：0.35*ln(21日换手和)+0.35*ln(63日均换手)+0.30*ln(252日均换手)，再对对数市值正交"
    dependencies: List[str] = ["turnover_rate", "turnover_rate_f", "total_mv"]
    role = "risk"
    lookback_days = 450

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "liquidity", start_date=start_date)


class StyleResvolFactor(BaseFactor):
    name = "style_resvol"
    description = "聚宽 residual_volatility：0.74*日超额波动+0.16*月收益极差+0.10*回归残差波动，再对 beta 和 size 正交"
    dependencies: List[str] = ["pct_chg", "circ_mv", "total_mv", "close", "vol"]
    role = "risk"
    lookback_days = 900

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "resvol", start_date=start_date)


class StyleBetaFactor(BaseFactor):
    name = "style_beta"
    description = "聚宽 beta：252日对流通市值加权全市场收益的指数加权回归，半衰期63日"
    dependencies: List[str] = ["pct_chg", "circ_mv", "vol"]
    role = "risk"
    lookback_days = 450

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "beta", start_date=start_date)


class StyleBtopFactor(BaseFactor):
    name = "style_btop"
    description = "聚宽 book_to_price_ratio：1/pb，pb<=0 为缺失"
    dependencies: List[str] = ["pb"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "btop", start_date=start_date)


class StyleEyFactor(BaseFactor):
    name = "style_ey"
    description = "聚宽 earnings_yield：预期利润/市值缺失时，在现金流市值比与利润市值比之间重归一"
    dependencies: List[str] = ["pe_ttm", "pe", "pb", "total_mv"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "ey", start_date=start_date)


class StyleGrowthFactor(BaseFactor):
    name = "style_growth"
    description = "聚宽 growth：五年 EPS 与每股营收回归斜率；分析师预期缺失时重归一。银行和保险不计营收增长"
    dependencies: List[str] = ["total_mv"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "growth", start_date=start_date)


class StyleLeverageFactor(BaseFactor):
    name = "style_leverage"
    description = "聚宽 leverage：0.38*市场杠杆+0.35*资产负债率+0.27*账面杠杆。优先股按0，长期债务用非流动负债近似"
    dependencies: List[str] = ["pb", "total_mv"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "leverage", start_date=start_date)
