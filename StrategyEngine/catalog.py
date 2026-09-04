#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发现 Strategies/ 下可被 CLI 加载的策略类。

Strategies 只放 score 实现；扫目录、拼 --strategy 是运行时的事。
只登记 Strategy 子类，且带 name / from_cli / panel_kwargs / run_tag。
不实例化、不在基类上统一构造参数。
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Type

from .strategy import Strategy

logger = logging.getLogger("StrategyEngine")

_SKIP = {"__init__"}


def _strategies_dir() -> Path:
    import Strategies

    return Path(Strategies.__file__).resolve().parent


def _is_cli_strategy(candidate: Any, module_name: str) -> bool:
    if not isinstance(candidate, type) or candidate.__module__ != module_name:
        return False
    if candidate is Strategy or not issubclass(candidate, Strategy):
        return False
    name = getattr(candidate, "name", None)
    if not isinstance(name, str) or not name.strip():
        return False
    return all(
        callable(getattr(candidate, attr, None))
        for attr in ("from_cli", "panel_kwargs", "run_tag")
    )


def discover() -> Dict[str, Type]:
    """`name` → 策略类。重复名直接报错。"""
    found: Dict[str, Type] = {}
    for path in sorted(_strategies_dir().glob("*.py")):
        if path.stem in _SKIP or path.name.startswith("_"):
            continue
        module_name = f"Strategies.{path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.exception("导入策略模块失败: %s", module_name)
            continue
        for attr in dir(module):
            candidate = getattr(module, attr)
            if not _is_cli_strategy(candidate, module.__name__):
                continue
            name = candidate.name.strip()
            if name in found:
                raise ValueError(
                    f"重复的策略名 {name!r}: {found[name].__name__} 与 {candidate.__name__}"
                )
            found[name] = candidate
    return found


def by_name() -> Dict[str, Type]:
    return discover()
