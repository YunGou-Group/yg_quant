#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仓位分配：策略选出候选后调用。新增实现时在本目录加模块并登记到 REGISTRY。"""

from __future__ import annotations

from typing import Dict, Type

from .equal import EqualWeight
from .hrp import HRP
from .mean_variance import (
    EfficientReturn,
    EfficientRisk,
    MaxQuadraticUtility,
    MaxSharpe,
    MinL2,
    MultiObjective,
)
from .min_vol import MinVol
from .path_risk import MinCDaR, MinCVaR, MinSemivariance
from .protocol import Allocator

REGISTRY: Dict[str, Type] = {
    "equal": EqualWeight,
    "min_vol": MinVol,
    "max_sharpe": MaxSharpe,
    "max_quadratic_utility": MaxQuadraticUtility,
    "efficient_return": EfficientReturn,
    "efficient_risk": EfficientRisk,
    "multiobjective": MultiObjective,
    "min_l2": MinL2,
    "min_semivariance": MinSemivariance,
    "min_cvar": MinCVaR,
    "min_cdar": MinCDaR,
    "hrp": HRP,
}


def get_allocator(name: str = "equal", **kwargs) -> Allocator:
    key = str(name or "equal").strip().lower()
    cls = REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"未知仓位分配器 {name!r}，可选: {sorted(REGISTRY)}")
    return cls(**kwargs)


__all__ = [
    "Allocator",
    "EqualWeight",
    "HRP",
    "MaxSharpe",
    "MinVol",
    "REGISTRY",
    "get_allocator",
]
