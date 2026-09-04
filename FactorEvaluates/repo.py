#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FactorEvaluates 侧保证仓库根在 sys.path 上。"""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root
