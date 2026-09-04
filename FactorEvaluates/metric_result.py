#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标输出：scalars 可落盘，series 仅本次展示。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd


@dataclass
class MetricResult:
    scalars: Dict[str, Any] = field(default_factory=dict)
    series: Dict[str, pd.Series] = field(default_factory=dict)
