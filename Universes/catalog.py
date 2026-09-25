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

# 无成分指数的板块池：用最接近的公开指数当超额对照。
_BOARD_BENCHMARK = {
    "all": "000985.SH",
    "main": "000985.SH",
    "gem": "399006.SZ",
    "star": "000688.SH",
}

INDEX_LABELS = {
    "index_SH000001": "上证综指",
    "index_SH000016": "上证50",
    "index_SH000300": "沪深300",
    "index_SH000510": "中证A500",
    "index_SH000688": "科创50",
    "index_SH000852": "中证1000",
    "index_SH000905": "中证500",
    "index_SH000985": "中证全指",
    "index_CSI000985": "中证全指",
    "index_SZ399001": "深证成指",
    "index_SZ399006": "创业板指",
    "index_SZ399101": "中小100",
}

# 理想基准没有日线时，按风格接近程度回退到库里已有的指数。
BENCHMARK_FALLBACKS = {
    "index_SH000985": ("index_CSI000985", "index_SH000510", "index_SH000300"),
    "index_CSI000985": ("index_SH000985", "index_SH000510", "index_SH000300"),
    "index_SH000852": ("index_SZ399101", "index_SH000510", "index_SH000300"),
    "index_SH000905": ("index_SH000510", "index_SH000300"),
    "index_SH000016": ("index_SH000300",),
    "index_SH000688": ("index_SZ399006", "index_SH000510", "index_SH000300"),
}


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


def market_index_symbol(ts_code: str) -> str:
    """000300.SH → index_SH000300，与行情库指数代码一致。"""
    text = str(ts_code or "").strip()
    if not text:
        return ""
    if text.lower().startswith("index_"):
        return f"index_{text.split('_', 1)[1]}"
    upper = text.upper()
    if "." in upper:
        code, market = upper.split(".", 1)
        return f"index_{market}{code}"
    return f"index_{upper}"


def resolve_available_benchmark(
    preferred: Optional[str], available: Any
) -> Optional[str]:
    """在已入库的指数里挑一条能画出来的基准。"""
    avail = {str(x) for x in (available or []) if str(x).strip()}
    want = str(preferred or "").strip()
    if want and want in avail:
        return want
    for cand in BENCHMARK_FALLBACKS.get(want, ()):
        if cand in avail:
            return cand
    for cand in ("index_SH000300", "index_SH000510", "index_SH000001"):
        if cand in avail:
            return cand
    return next(iter(sorted(avail)), None) if avail else None


def index_label(symbol: str) -> str:
    code = str(symbol or "").strip()
    return INDEX_LABELS.get(code, code)


def default_benchmark(name: str) -> Optional[str]:
    """股票池默认超额基准：有成分指数用该指数，否则用板块近似指数。"""
    found = discover()
    key = str(name or "all").strip().lower()
    cls = found.get(key)
    if cls is None:
        return None
    codes = tuple(getattr(cls, "index_codes", ()) or ())
    raw = codes[0] if codes else _BOARD_BENCHMARK.get(key)
    symbol = market_index_symbol(raw) if raw else ""
    return symbol or None


def listing() -> List[Dict[str, str]]:
    found = discover()
    rows = []
    for name in names():
        bench = default_benchmark(name) or ""
        rows.append(
            {
                "name": name,
                "description": str(getattr(found[name], "description", "") or ""),
                "benchmark": bench,
                "benchmark_label": INDEX_LABELS.get(bench, bench),
            }
        )
    return rows


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
