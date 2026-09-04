#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仓位分配合同：策略选出候选后调用，返回与 ctx.symbols 对齐的目标权重。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..context import DayContext


@runtime_checkable
class Allocator(Protocol):
    def size(self, ctx: DayContext, candidates: np.ndarray) -> np.ndarray:
        """candidates 为下标；返回长度 = len(ctx.symbols)，和不超过 1。"""
        ...
