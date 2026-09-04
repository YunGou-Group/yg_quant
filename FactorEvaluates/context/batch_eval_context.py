#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量日度上下文：一天、多因子矩阵。不进单因子 EvalContext。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BatchEvalContext:
    factors: np.ndarray  # (n_stocks, n_factors) float
    returns: Optional[np.ndarray] = None  # (n_stocks,)
    mask: Optional[np.ndarray] = None  # (n_stocks,) bool
    barra: Optional[np.ndarray] = None  # (n_stocks, n_styles)
    size: Optional[np.ndarray] = None  # (n_stocks,) style_size
    date: str = ""
    n_quantiles: int = 5
    min_obs: int = 20
    intermediates: Dict[str, Any] = field(default_factory=dict)

    def masked_factors(self) -> np.ndarray:
        if self.mask is None:
            return self.factors
        return self.factors[self.mask]

    def masked_returns(self) -> Optional[np.ndarray]:
        if self.returns is None:
            return None
        if self.mask is None:
            return self.returns
        return self.returns[self.mask]

    def masked_barra(self) -> Optional[np.ndarray]:
        if self.barra is None:
            return None
        if self.mask is None:
            return self.barra
        return self.barra[self.mask]

    def masked_size(self) -> Optional[np.ndarray]:
        if self.size is None:
            return None
        if self.mask is None:
            return self.size
        return self.size[self.mask]
