#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标标量说明：评估维度 + 含义，供网页展示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FieldDoc:
    name: str
    dimension: str
    label: str
    meaning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "label": self.label,
            "meaning": self.meaning,
        }
