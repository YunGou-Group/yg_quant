#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选等权。"""

from __future__ import annotations

import numpy as np

from ..context import DayContext
from .common import candidate_index, equal_weights


class EqualWeight:
    def __init__(self, **_kwargs):
        pass

    def size(self, ctx: DayContext, candidates: np.ndarray) -> np.ndarray:
        idx = candidate_index(ctx, candidates)
        return equal_weights(len(ctx.symbols), idx)
