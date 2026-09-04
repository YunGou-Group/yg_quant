#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拥挤指标目录：与 doc/Factor_Crowding indicator_catalog 对齐。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class IndicatorSpec:
    file: str
    title: str
    layer: str
    role: str
    independent_vote: bool = True
    group_invariant: bool = False
    crowding_direction: str = "high"

    @property
    def key(self) -> str:
        return indicator_key(self.file)

    def event_metadata(self) -> dict:
        return {
            "file": self.file,
            "key": self.key,
            "title": self.title,
            "layer": self.layer,
            "role": self.role,
            "independent_vote": self.independent_vote,
            "group_invariant": self.group_invariant,
            "crowding_direction": self.crowding_direction,
        }


def indicator_key(filename: str) -> str:
    text = str(filename)
    return text[:-4] if text.endswith(".csv") else text


INDICATORS: Tuple[IndicatorSpec, ...] = (
    IndicatorSpec("01_article_valuation_spread_bp_exposure_20d.csv", "文章复现 · 估值差 · B/P 暴露（20 日）", "vulnerability_accumulation", "标准化 B/P 暴露的 H−L 均值差；低尾表示高暴露端相对更贵", crowding_direction="low"),
    IndicatorSpec("01_valuation_bp_z.csv", "扩展 · 估值贵度 · B/P（prior-Z）", "vulnerability_accumulation", "原始 PB 倒数构造的相对估值贵度"),
    IndicatorSpec("01_valuation_pe_z.csv", "扩展 · 估值贵度 · E/P（prior-Z）", "vulnerability_accumulation", "盈利估值脆弱性背景"),
    IndicatorSpec("01_valuation_ps_z.csv", "扩展 · 估值贵度 · S/P（prior-Z）", "vulnerability_accumulation", "销售估值脆弱性背景"),
    IndicatorSpec("02_price_stretch_60d.csv", "60 日价格伸展", "vulnerability_accumulation", "近期业绩追逐与价格伸展"),
    IndicatorSpec("03_article_within_group_dispersion_close_20d.csv", "文章复现 · 组内分化度 · 收盘收益（20 日）", "holding_structure", "H 组同日收盘收益横截面标准差", crowding_direction="both"),
    IndicatorSpec("03_article_within_group_dispersion_close_top50_20d.csv", "文章复现 · 组内分化度 · Top 50", "holding_structure", "固定 Top-N 双尾稳健性", False, True, "both"),
    IndicatorSpec("03_article_within_group_dispersion_close_top100_20d.csv", "文章复现 · 组内分化度 · Top 100", "holding_structure", "固定 Top-N 双尾稳健性", False, True, "both"),
    IndicatorSpec("03_article_within_group_dispersion_close_top200_20d.csv", "文章复现 · 组内分化度 · Top 200", "holding_structure", "固定 Top-N 双尾稳健性", False, True, "both"),
    IndicatorSpec("03_low_dispersion_relative_20d.csv", "扩展 · 低分化抱团 · 相对市场", "holding_structure", "高尾表示低分化抱团增强", crowding_direction="both"),
    IndicatorSpec("03_low_dispersion_relative_top50_20d.csv", "扩展 · 低分化抱团 · Top 50", "holding_structure", "固定 Top-N 相对分化双尾", False, True, "both"),
    IndicatorSpec("03_low_dispersion_relative_top100_20d.csv", "扩展 · 低分化抱团 · Top 100", "holding_structure", "固定 Top-N 相对分化双尾", False, True, "both"),
    IndicatorSpec("03_low_dispersion_relative_top200_20d.csv", "扩展 · 低分化抱团 · Top 200", "holding_structure", "固定 Top-N 相对分化双尾", False, True, "both"),
    IndicatorSpec("04_dispersion_spike_raw.csv", "分化度尖峰 · 原始分化", "risk_release_confirmation", "组内分化水平和增速双尾异常", crowding_direction="both"),
    IndicatorSpec("04_dispersion_spike_raw_top50.csv", "分化度尖峰 · Top 50", "risk_release_confirmation", "固定 Top-N 分化尖峰", False, True, "both"),
    IndicatorSpec("04_dispersion_spike_raw_top100.csv", "分化度尖峰 · Top 100", "risk_release_confirmation", "固定 Top-N 分化尖峰", False, True, "both"),
    IndicatorSpec("04_dispersion_spike_raw_top200.csv", "分化度尖峰 · Top 200", "risk_release_confirmation", "固定 Top-N 分化尖峰", False, True, "both"),
    IndicatorSpec("04_dispersion_spike_raw_momentum_20d.csv", "分化度尖峰 · 20 日动量", "risk_release_confirmation", "原始分化尖峰 20 日变化", crowding_direction="both"),
    IndicatorSpec("04_dispersion_spike_relative.csv", "分化度尖峰 · 相对分化", "risk_release_confirmation", "相对市场组内失稳双尾异常", crowding_direction="both"),
    IndicatorSpec("04_dispersion_spike_relative_top50.csv", "分化度尖峰 · 相对 Top 50", "risk_release_confirmation", "固定 Top-N 相对尖峰", False, True, "both"),
    IndicatorSpec("04_dispersion_spike_relative_top100.csv", "分化度尖峰 · 相对 Top 100", "risk_release_confirmation", "固定 Top-N 相对尖峰", False, True, "both"),
    IndicatorSpec("04_dispersion_spike_relative_top200.csv", "分化度尖峰 · 相对 Top 200", "risk_release_confirmation", "固定 Top-N 相对尖峰", False, True, "both"),
    IndicatorSpec("04_dispersion_spike_relative_momentum_20d.csv", "分化度尖峰 · 相对 20 日动量", "risk_release_confirmation", "相对分化尖峰 20 日变化", crowding_direction="both"),
    IndicatorSpec("05_factor_volatility_60d.csv", "因子自身波动率 · 60 日年化", "risk_release_confirmation", "因子收益风险水平", group_invariant=True),
    IndicatorSpec("05_factor_volatility_momentum_20d.csv", "因子自身波动率 · 20 日动量", "risk_release_confirmation", "风险加速确认", group_invariant=True),
    IndicatorSpec("05_factor_volatility_of_volatility_20d.csv", "因子自身波动率 · 波动率（20 日）", "risk_release_confirmation", "因子风险不稳定程度", group_invariant=True),
    IndicatorSpec("06_factor_cumulative_return_momentum_21d.csv", "因子累计收益率 · 21 日动量", "vulnerability_accumulation", "因子收益动量（唯一动量核心）", group_invariant=True),
    IndicatorSpec("07_pairwise_correlation_63d.csv", "组内成对相关性", "holding_structure", "持仓结构同质化"),
    IndicatorSpec("08_relative_factor_volatility.csv", "相对因子波动率", "risk_release_confirmation", "因子相对全市场风险状态", group_invariant=True),
)

INDICATOR_BY_KEY: Dict[str, IndicatorSpec] = {item.key: item for item in INDICATORS}
INDICATOR_BY_FILE: Dict[str, IndicatorSpec] = {item.file: item for item in INDICATORS}

CORE_INDICATORS = (
    "01_valuation_bp_z",
    "02_price_stretch_60d",
    "03_article_within_group_dispersion_close_top50_20d",
    "01_valuation_pe_z",
    "06_factor_cumulative_return_momentum_21d",
)

if len(INDICATOR_BY_KEY) != len(INDICATORS):
    raise RuntimeError("指标目录存在重复 key")
