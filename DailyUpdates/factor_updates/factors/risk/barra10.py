#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNE5-lite / Barra10-style raw descriptors.

Each class writes one Bin series. Values are pre-z-score descriptors as of
close_T. calculate() must not shift open or call ReturnCalculator.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

from ._engine import extract_style


class StyleSizeFactor(BaseFactor):
    name = "style_size"
    description = "CNE5-lite Size: log(circ_mv)"
    dependencies: List[str] = ["circ_mv"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "size", start_date=start_date)


class StyleMomentumFactor(BaseFactor):
    name = "style_momentum"
    description = "CNE5-lite Momentum: close[T-21]/close[T-252]-1"
    dependencies: List[str] = ["close"]
    role = "risk"
    lookback_days = 600

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "momentum", start_date=start_date)


class StyleLiquidityFactor(BaseFactor):
    name = "style_liquidity"
    description = "CNE5-lite Liquidity: 21d mean turnover_rate_f (fallback turnover_rate)"
    dependencies: List[str] = ["turnover_rate_f", "turnover_rate"]
    role = "risk"
    lookback_days = 40

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "liquidity", start_date=start_date)


class StyleResvolFactor(BaseFactor):
    name = "style_resvol"
    description = "CNE5-lite Residual Volatility: 60d std of daily returns"
    dependencies: List[str] = ["pct_chg"]
    role = "risk"
    lookback_days = 120

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "resvol", start_date=start_date)


class StyleBetaFactor(BaseFactor):
    name = "style_beta"
    description = "CNE5-lite Beta: 252d rolling OLS vs HS300"
    dependencies: List[str] = ["pct_chg"]
    role = "risk"
    lookback_days = 600

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "beta", start_date=start_date)


class StyleBtopFactor(BaseFactor):
    name = "style_btop"
    description = "CNE5-lite Book-to-Price: 1/pb"
    dependencies: List[str] = ["pb"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "btop", start_date=start_date)


class StyleEyFactor(BaseFactor):
    name = "style_ey"
    description = "CNE5-lite Earnings Yield: 1/pe_ttm (fallback 1/pe)"
    dependencies: List[str] = ["pe_ttm", "pe"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "ey", start_date=start_date)


class StyleGrowthFactor(BaseFactor):
    name = "style_growth"
    description = "CNE5-lite Growth: PIT netprofit_yoy asof ann_date"
    dependencies: List[str] = ["close"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "growth", start_date=start_date)


class StyleLeverageFactor(BaseFactor):
    name = "style_leverage"
    description = "CNE5-lite Leverage: PIT debt_to_assets asof ann_date"
    dependencies: List[str] = ["close"]
    role = "risk"
    lookback_days = 5

    def calculate(self, data: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        return extract_style(data, "leverage", start_date=start_date)
