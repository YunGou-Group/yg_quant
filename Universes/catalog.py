#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发现 Universes/ 下可被 CLI / 评估加载的股票池类。"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import Universe

logger = logging.getLogger("Universes")

_SKIP = {"__init__", "base", "catalog", "rules", "names"}
_CACHE: Optional[Dict[str, Type[Universe]]] = None


def _universes_dir() -> Path:
    return Path(__file__).resolve().parent


def _is_universe(candidate: Any, module_name: str) -> bool:
    if not isinstance(candidate, type) or candidate.__module__ != module_name:
        return False
    if candidate is Universe or not issubclass(candidate, Universe):
        return False
    name = getattr(candidate, "name", None)
    return isinstance(name, str) and bool(name.strip())


def discover() -> Dict[str, Type[Universe]]:
    """`name` → 股票池类。重复名直接报错。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    found: Dict[str, Type[Universe]] = {}
    for path in sorted(_universes_dir().glob("*.py")):
        if path.stem in _SKIP or path.name.startswith("_"):
            continue
        module_name = f"Universes.{path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.exception("导入股票池模块失败: %s", module_name)
            continue
        for attr in dir(module):
            candidate = getattr(module, attr)
            if not _is_universe(candidate, module.__name__):
                continue
            name = candidate.name.strip()
            if name in found:
                raise ValueError(
                    f"重复的股票池名 {name!r}: {found[name].__name__} 与 {candidate.__name__}"
                )
            found[name] = candidate
    _CACHE = found
    return found


def names() -> List[str]:
    keys = sorted(discover())
    if "all" in keys:
        keys.remove("all")
        keys.insert(0, "all")
    return keys


def listing() -> List[Dict[str, str]]:
    found = discover()
    return [
        {"name": name, "description": str(getattr(found[name], "description", "") or "")}
        for name in names()
    ]


def get(name: str) -> Universe:
    found = discover()
    key = str(name or "all").strip().lower()
    if key not in found:
        raise ValueError(f"未知股票池 {name!r}，可选: {', '.join(names())}")
    return found[key]()


def build_mask(name: str, dates, symbols, open_panel, storage, **kwargs):
    return get(name).build_mask(dates, symbols, open_panel, storage, **kwargs)


def reset_cache() -> None:
    """测试用。"""
    global _CACHE
    _CACHE = None
