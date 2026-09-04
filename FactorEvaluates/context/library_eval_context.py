#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""库级上下文：全库相关阵 / IC 序列，只给 eval_scope=library 的指标。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LibraryEvalContext:
    factor_names: list
    dates: list
    daily_rank_ic: Optional[np.ndarray] = None  # (n_dates, n_factors)
    mean_factor_corr: Optional[np.ndarray] = None  # (n_factors, n_factors)
    family_of: Optional[Dict[str, str]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
