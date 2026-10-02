#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发现 Strategies/ 下可被 CLI 加载的策略类。

Strategies 只放 score 实现；扫目录、拼 --strategy 是运行时的事。
只登记 Strategy 子类，且带 name / from_cli / panel_kwargs / run_tag。
策略参数只来自 cli_fields，run / CLI 不按策略名写死字段。
不实例化、不在基类上统一构造参数。
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type

from .strategy import Strategy

logger = logging.getLogger("StrategyEngine")

_SKIP = {"__init__", "rebalance_schedule"}
_FIELD_TYPES = frozenset({"str", "int", "float", "bool"})
ENGINE_ARG_KEYS = frozenset(
    {
        "mode",
        "strategy",
        "start",
        "end",
        "universe",
        "allocator",
        "allocator_lookback",
        "max_weight",
        "risk_free_rate",
        "risk_aversion",
        "target_return",
        "target_volatility",
        "l2_gamma",
        "tc_rate",
        "tail_confidence",
        "mu_model",
        "cash",
        "account",
        "run",
        "qmt_json",
        "commission",
        "stamp",
        "slippage",
        "min_commission",
        "out",
        "benchmark",
        "no_benchmark",
    }
)


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


def fields_of(spec: Any) -> List[Dict[str, Any]]:
    """读策略自己声明的参数。没有 cli_fields 则空表。"""
    fn = getattr(spec, "cli_fields", None)
    if not callable(fn):
        return []
    raw = fn() or ()
    return [dict(item) for item in raw]


def merge_cli_fields(specs: Optional[Mapping[str, Type]] = None) -> List[Dict[str, Any]]:
    """合并已发现策略的 cli_fields。同名第一次出现为准，类型冲突则报错。"""
    names = specs if specs is not None else by_name()
    by_key: Dict[str, Dict[str, Any]] = {}
    owners: Dict[str, List[str]] = {}
    for name, spec in names.items():
        for field in fields_of(spec):
            key = str(field.get("key") or "").strip()
            if not key:
                raise ValueError(f"策略 {name} 的 cli_fields 有空 key")
            if key in ENGINE_ARG_KEYS:
                raise ValueError(f"策略 {name} 的参数 {key} 与引擎参数重名")
            typ = str(field.get("type") or "str")
            if typ not in _FIELD_TYPES:
                raise ValueError(f"策略 {name} 的参数 {key} 类型 {typ} 不支持")
            if key in by_key:
                prev = str(by_key[key].get("type") or "str")
                if prev != typ:
                    raise ValueError(
                        f"策略参数 {key} 类型冲突: {prev} vs {typ}（{'/'.join(owners[key])} / {name}）"
                    )
                owners[key].append(name)
                continue
            by_key[key] = dict(field)
            owners[key] = [name]
    out: List[Dict[str, Any]] = []
    for key, field in by_key.items():
        item = dict(field)
        used = owners[key]
        label = str(item.get("label") or key)
        item["strategies"] = used
        item["cli_help"] = f"{label}（{', '.join(used)}）"
        out.append(item)
    return out


def coerce_cli_value(field: Mapping[str, Any], raw: Any) -> Any:
    """把网页 / CLI 原始值转成策略 from_cli 可读的类型；缺省为 None（bool 为 False）。"""
    typ = str(field.get("type") or "str")
    if raw is None or raw == "":
        return False if typ == "bool" else None
    if typ == "int":
        return int(raw)
    if typ == "float":
        return float(raw)
    if typ == "bool":
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    text = str(raw).strip()
    return text or None


def bind_strategy_fields(namespace: Any, payload: Mapping[str, Any], strategy_name: str) -> None:
    """按所选策略的 cli_fields 把 payload 写进 args，其它策略字段不出现。"""
    spec = by_name().get(str(strategy_name or ""))
    for field in fields_of(spec):
        key = str(field["key"])
        setattr(namespace, key, coerce_cli_value(field, payload.get(key)))


def add_cli_arguments(parser: Any, specs: Optional[Mapping[str, Type]] = None) -> None:
    """把已发现策略的 cli_fields 登记为 argparse 开关（下划线 → 连字符）。"""
    for field in merge_cli_fields(specs):
        key = str(field["key"])
        flag = "--" + key.replace("_", "-")
        help_text = str(field.get("cli_help") or field.get("label") or key)
        typ = str(field.get("type") or "str")
        if typ == "bool":
            parser.add_argument(flag, dest=key, action="store_true", default=False, help=help_text)
        elif typ == "int":
            parser.add_argument(flag, dest=key, type=int, default=None, help=help_text)
        elif typ == "float":
            parser.add_argument(flag, dest=key, type=float, default=None, help=help_text)
        else:
            parser.add_argument(flag, dest=key, default=None, help=help_text)

