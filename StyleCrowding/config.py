#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StyleCrowding 运行配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

from FactorEvaluates.exposure_engine import RAW_STYLE_NAMES, STYLE_LABELS
from yg_quant_repo import default_style_crowding_dir


@dataclass
class StyleCrowdingSettings:
    """与参考 doc/Factor_Crowding 对齐的可调参数。"""

    output_root: Path = field(default_factory=default_style_crowding_dir)
    start_date: str = "2018-01-01"
    end_date: str = ""
    universe: str = "all"
    style_names: Tuple[str, ...] = RAW_STYLE_NAMES
    include_nlsize: bool = False
    group_counts: Tuple[int, ...] = (10, 5)
    orientations: Tuple[str, ...] = ("positive", "reverse")
    min_group_size: int = 30
    min_group_coverage: float = 0.80
    prior_z_window: int = 252
    prior_z_min_history: int = 60
    percentile_window: int = 252
    valuation_z_min_prior: int = 60
    valuation_aggregation: str = "median"
    article_window: int = 20
    article_min_group_coverage: float = 0.60
    dispersion_top_counts: Tuple[int, ...] = (50, 100, 200)
    factor_vol_window: int = 60
    factor_vol_min_obs: int = 54
    factor_momentum_window: int = 21
    price_stretch_window: int = 60
    price_stretch_min_obs: int = 54
    pairwise_window: int = 63
    pairwise_min_pair_obs: int = 45
    pairwise_min_member_ratio: float = 0.80
    risk_covariance_window: int = 252
    risk_covariance_half_life: int = 63
    relative_vol_z_min_prior: int = 252
    skip_indicators: Tuple[str, ...] = ()
    min_obs_style_wls: int = 20

    def published_root(self) -> Path:
        return Path(self.output_root).expanduser().resolve() / "published"

    def style_targets(self) -> Tuple[str, ...]:
        names = list(self.style_names)
        if self.include_nlsize and "nlsize" not in names:
            names.append("nlsize")
        return tuple(names)

    def style_label(self, name: str) -> str:
        return STYLE_LABELS.get(name, name)


def default_settings(**overrides) -> StyleCrowdingSettings:
    base = StyleCrowdingSettings()
    for key, value in overrides.items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base
