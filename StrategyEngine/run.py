#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网页用的程序化入口。命令行回测仍是 python -m StrategyEngine，不经过这里。"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd

from FactorEvaluates.market_panel_loader import HS300_SYMBOL, MarketPanelLoader
from yg_quant_repo import default_data_dir, strategy_runs_dir
from Strategies import CANDIDATE_N, HOLD_N

from .allocators import REGISTRY, get_allocator
from .backtest import Engine, PanelStore, summarize, trades_frame, write_run_snapshot
from .backtest.report import _aligned_nav
from Universes.catalog import (
    default_benchmark,
    index_label,
    listing as universe_listing,
    resolve_available_benchmark,
)

from .catalog import by_name, fields_of

_LOG = logging.getLogger("StrategyEngine")
_SAFE_STEM = re.compile(r"^[\w.\-]+$")
_Echo = Optional[Callable[[str], None]]

WEB_DEFAULT_START = "2023-01-01"
DEFAULT_COSTS = {
    "commission": 0.0001,
    "stamp": 0.0005,
    "slippage": 0.00258,
    "min_commission": 5.0,
}
STOCK_COSTS = DEFAULT_COSTS


def cost_defaults(strategy_name: Optional[str] = None) -> Dict[str, float]:
    """默认万一佣金、0.00258 滑点、单笔最低 5 元；股票另加卖出印花税。策略可用 cost_kwargs 覆盖。"""
    out = dict(STOCK_COSTS)
    spec = by_name().get(str(strategy_name or ""))
    fn = getattr(spec, "cost_kwargs", None)
    if not callable(fn):
        return out
    raw = fn(None) or {}
    for key in STOCK_COSTS:
        if key in raw and raw[key] is not None:
            out[key] = float(raw[key])
    return out


def apply_cost_defaults(args) -> None:
    costs = cost_defaults(getattr(args, "strategy", None))
    for key, value in costs.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def _index_symbols() -> List[str]:
    try:
        return MarketPanelLoader().list_index_symbols()
    except Exception:
        return []


def _bind_benchmark(universe: str, requested: Optional[str] = None) -> Optional[str]:
    preferred = (requested or "").strip() or default_benchmark(universe) or HS300_SYMBOL
    return resolve_available_benchmark(preferred, _index_symbols()) or preferred


def catalog_meta() -> Dict[str, Any]:
    names = by_name()
    default = "small_cap" if "small_cap" in names else (sorted(names)[0] if names else None)
    available = _index_symbols()
    universes = []
    for row in universe_listing():
        item = dict(row)
        requested = item.get("benchmark") or ""
        actual = resolve_available_benchmark(requested, available) or requested
        item["benchmark_requested"] = requested
        item["benchmark"] = actual
        item["benchmark_label"] = index_label(actual)
        universes.append(item)
    benchmarks = [
        {"value": code, "label": index_label(code)}
        for code in available
    ]
    default_bench = resolve_available_benchmark(
        default_benchmark("all"), available
    ) or HS300_SYMBOL
    return {
        "strategies": [
            {
                "name": name,
                "fields": _strategy_fields(name),
                "costs": cost_defaults(name),
            }
            for name in sorted(names)
        ],
        "allocators": sorted(REGISTRY),
        "universes": universes,
        "benchmarks": benchmarks,
        "mu_models": ["geometric", "arithmetic", "ema"],
        "defaults": {
            "strategy": default,
            "start": WEB_DEFAULT_START,
            "end": None,
            "universe": "all",
            "benchmark": default_bench,
            "allocator": "equal",
            "allocator_lookback": 252,
            "max_weight": 1.0,
            "risk_free_rate": 0.02,
            "risk_aversion": 1.0,
            "l2_gamma": 0.1,
            "tc_rate": 0.001,
            "tail_confidence": 0.95,
            "mu_model": "geometric",
            "cash": 1_000_000.0,
            **DEFAULT_COSTS,
            "hold": HOLD_N,
            "n": None,
            "anti_tail": False,
        },
    }


def args_from_payload(payload: Mapping[str, Any]) -> SimpleNamespace:
    names = by_name()
    default = "small_cap" if "small_cap" in names else (sorted(names)[0] if names else "topk")
    no_benchmark = _bool(payload.get("no_benchmark"))
    strategy = str(payload.get("strategy") or default)
    costs = cost_defaults(strategy)
    universe = _opt_str(payload.get("universe")) or "all"
    if no_benchmark:
        benchmark = None
    else:
        benchmark = _bind_benchmark(universe, _opt_str(payload.get("benchmark")))
    return SimpleNamespace(
        strategy=strategy,
        start=_opt_str(payload.get("start")) or "2010-01-01",
        end=_opt_str(payload.get("end")),
        universe=universe,
        factor=_opt_str(payload.get("factor")),
        n=_opt_int(payload.get("n")),
        rebalance=_opt_str(payload.get("rebalance")) or "daily",
        lookback=_opt_int(payload.get("lookback")),
        horizon=_opt_int(payload.get("horizon")),
        hold=_opt_int(payload.get("hold")) if payload.get("hold") not in (None, "") else HOLD_N,
        anti_tail=_bool(payload.get("anti_tail")),
        allocator=str(payload.get("allocator") or "equal"),
        allocator_lookback=int(payload.get("allocator_lookback") or 252),
        max_weight=float(
            payload["max_weight"] if payload.get("max_weight") not in (None, "") else 1.0
        ),
        risk_free_rate=float(
            payload["risk_free_rate"] if payload.get("risk_free_rate") not in (None, "") else 0.02
        ),
        risk_aversion=float(
            payload["risk_aversion"] if payload.get("risk_aversion") not in (None, "") else 1.0
        ),
        target_return=_opt_float(payload.get("target_return")),
        target_volatility=_opt_float(payload.get("target_volatility")),
        l2_gamma=float(payload["l2_gamma"] if payload.get("l2_gamma") not in (None, "") else 0.1),
        tc_rate=float(payload["tc_rate"] if payload.get("tc_rate") not in (None, "") else 0.001),
        tail_confidence=float(
            payload["tail_confidence"] if payload.get("tail_confidence") not in (None, "") else 0.95
        ),
        mu_model=str(payload.get("mu_model") or "geometric"),
        cash=float(payload["cash"] if payload.get("cash") not in (None, "") else 1_000_000.0),
        commission=float(
            payload["commission"]
            if payload.get("commission") not in (None, "")
            else costs["commission"]
        ),
        stamp=float(
            payload["stamp"] if payload.get("stamp") not in (None, "") else costs["stamp"]
        ),
        slippage=float(
            payload["slippage"]
            if payload.get("slippage") not in (None, "")
            else costs["slippage"]
        ),
        min_commission=float(
            payload["min_commission"]
            if payload.get("min_commission") not in (None, "")
            else costs["min_commission"]
        ),
        out=_opt_str(payload.get("out")),
        benchmark=benchmark,
        no_benchmark=no_benchmark,
    )


def build(args):
    names = by_name()
    if args.strategy not in names:
        raise ValueError(f"未知策略 {args.strategy!r}，可选: {sorted(names)}")
    spec = names[args.strategy]
    allocator = get_allocator(
        args.allocator,
        lookback=args.allocator_lookback,
        max_weight=args.max_weight,
        risk_free_rate=args.risk_free_rate,
        risk_aversion=args.risk_aversion,
        target_return=args.target_return,
        target_volatility=args.target_volatility,
        l2_gamma=args.l2_gamma,
        transaction_cost_rate=args.tc_rate,
        tail_confidence=args.tail_confidence,
        mu_model=args.mu_model,
    )
    suffix = "" if args.allocator == "equal" else f"_{args.allocator}"
    try:
        strategy = spec.from_cli(args, allocator)
    except SystemExit as exc:
        raise ValueError(str(exc) or f"{args.strategy} 参数不完整") from exc
    store = PanelStore.load(
        start=args.start,
        end=args.end,
        universe=args.universe,
        warmup=_panel_warmup(spec, args, allocator),
        **spec.panel_kwargs(args),
    )
    return store, strategy, spec.run_tag(args) + suffix


def run_backtest(
    payload: Mapping[str, Any],
    *,
    save: bool = True,
    echo: _Echo = None,
    progress: _Echo = None,
) -> Dict[str, Any]:
    args = args_from_payload(payload)
    _say(echo, f"开始回测 strategy={args.strategy} {args.start} ~ {args.end or '最新'}")
    _phase(progress, "loading_panel")
    store, strategy, tag = build(args)
    _say(echo, "面板已加载，开始逐日撮合")
    _phase(progress, "matching")
    result = Engine(
        store,
        initial_cash=args.cash,
        commission=args.commission,
        stamp=args.stamp,
        slippage=getattr(args, "slippage", 0.0),
        min_commission=getattr(args, "min_commission", 0.0),
    ).run(strategy)
    equity_df = result.equity()
    out = Path(args.out) if args.out else (
        strategy_runs_dir("backtest") / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    trades_path = out.with_name(f"{out.stem}_trades.csv")
    snap_path = out.with_suffix(".json")
    if save:
        out.parent.mkdir(parents=True, exist_ok=True)
        equity_df.to_csv(out, encoding="utf-8")
        trades_frame(result).to_csv(trades_path, index=False, encoding="utf-8")
    bench = None
    symbol = "" if args.no_benchmark else str(args.benchmark or "").strip()
    if symbol and symbol.lower() not in {"none", "off"}:
        loader = MarketPanelLoader()
        try:
            bench = loader.load_index_close(symbol)
            if bench is None or getattr(bench, "empty", True):
                alt = _bind_benchmark(getattr(args, "universe", "all"), symbol)
                if alt and alt != symbol:
                    _LOG.warning("基准 %s 无行情，改用 %s", symbol, alt)
                    bench = loader.load_index_close(alt, reload=True)
                    symbol = alt
            if bench is None or getattr(bench, "empty", True):
                _LOG.warning("基准 %s 无行情，跳过超额", symbol)
                bench = None
                symbol = ""
        except Exception:
            _LOG.warning("基准 %s 加载失败，跳过超额", symbol)
            bench = None
            symbol = ""
    stats = summarize(result, benchmark=bench, risk_free_rate=args.risk_free_rate)
    equity = _equity_payload(result, bench)
    snapshot = {
        "id": out.stem,
        "created_at": datetime.now().astimezone().isoformat(),
        "tag": tag,
        "strategy": args.strategy,
        "allocator": args.allocator,
        "files": {"equity": str(out), "trades": str(trades_path), "snapshot": str(snap_path)},
        "params": _run_params(args),
        "metrics": stats,
        "equity": equity,
        "benchmark": symbol or None,
        "has_attribution": False,
    }
    if save:
        write_run_snapshot(snap_path, snapshot)
    return snapshot


def _has_current_attribution(snap: Path) -> bool:
    dest = Path(snap).with_name(f"{Path(snap).stem}_attribution.json")
    if not dest.is_file():
        return False
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        return False
    from StrategyEngine.attribution.evaluate import is_current_report

    return is_current_report(data)


def list_snapshots(limit: int = 40) -> List[Dict[str, Any]]:
    items = []
    seen = set()
    paths = []
    for root in _snapshot_roots():
        if root.is_dir():
            paths.extend(root.glob("*.json"))
    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    cap = max(1, int(limit))
    for path in paths:
        if path.stem in seen:
            continue
        if path.stem.endswith("_attribution") or path.stem.endswith("_qmt"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or "metrics" not in data:
            continue
        items.append(
            {
                "id": path.stem,
                "created_at": data.get("created_at"),
                "tag": data.get("tag"),
                "strategy": data.get("strategy"),
                "allocator": data.get("allocator"),
                "metrics": data.get("metrics") or {},
                "has_attribution": _has_current_attribution(path),
            }
        )
        seen.add(path.stem)
        if len(items) >= cap:
            break
    return items


def load_snapshot(stem: str) -> Dict[str, Any]:
    path = _snapshot_path(stem)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["id"] = path.stem
    data["has_attribution"] = _has_current_attribution(path)
    equity = data.get("equity") or {}
    if not equity.get("dates"):
        data["equity"] = _equity_from_csv(data, path)
    return data


def _strategy_fields(name: str) -> List[Dict[str, Any]]:
    spec = by_name().get(name)
    return fields_of(spec) if spec is not None else []


def _panel_warmup(spec, args, allocator) -> int:
    """面板预热取策略历史与仓位 lookback 的较大值，避免等权少装 252 日导致选股日历不同。"""
    alloc = int(getattr(allocator, "lookback", 0) or 0)
    fn = getattr(spec, "warmup", None)
    if callable(fn):
        strat = int(fn(args) or 0)
    else:
        strat = int(fn or 0)
    return max(alloc, strat)


def _run_params(args) -> dict:
    return {
        "start": args.start,
        "end": args.end,
        "universe": args.universe,
        "factor": args.factor,
        "n": args.n,
        "rebalance": getattr(args, "rebalance", "daily"),
        "lookback": getattr(args, "lookback", None),
        "horizon": getattr(args, "horizon", None),
        "hold": args.hold,
        "anti_tail": bool(args.anti_tail),
        "allocator_lookback": args.allocator_lookback,
        "max_weight": args.max_weight,
        "risk_free_rate": args.risk_free_rate,
        "risk_aversion": args.risk_aversion,
        "target_return": args.target_return,
        "target_volatility": args.target_volatility,
        "l2_gamma": args.l2_gamma,
        "tc_rate": args.tc_rate,
        "tail_confidence": args.tail_confidence,
        "mu_model": args.mu_model,
        "cash": args.cash,
        "commission": args.commission,
        "stamp": args.stamp,
        "slippage": getattr(args, "slippage", 0.0),
        "min_commission": getattr(args, "min_commission", 0.0),
        "benchmark": None if args.no_benchmark else args.benchmark,
    }


def _equity_payload(result, benchmark) -> Dict[str, Any]:
    nav = [float(x) for x in result.nav]
    scale = nav[0] if nav and nav[0] else 1.0
    payload = {
        "dates": list(result.dates),
        "nav": [x / scale for x in nav] if scale else nav,
    }
    if benchmark is not None and not getattr(benchmark, "empty", True):
        aligned = _aligned_nav(benchmark, result.dates)
        if aligned is not None:
            payload["benchmark"] = [float(x) for x in aligned]
    return payload


def _equity_from_csv(data: Mapping[str, Any], json_path: Path) -> Dict[str, Any]:
    csv_name = (data.get("files") or {}).get("equity")
    csv_path = Path(csv_name) if csv_name else json_path.with_suffix(".csv")
    if not csv_path.is_file():
        return {"dates": [], "nav": []}
    frame = pd.read_csv(csv_path, index_col=0)
    if "nav" not in frame.columns or frame.empty:
        return {"dates": [], "nav": []}
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    first = float(nav.iloc[0]) if len(nav) and pd.notna(nav.iloc[0]) else 1.0
    scale = first if first else 1.0
    return {
        "dates": [str(idx)[:10] for idx in frame.index],
        "nav": [float(v) / scale if pd.notna(v) else None for v in nav],
    }


def _runs_root() -> Path:
    return strategy_runs_dir("backtest")


def _snapshot_roots() -> List[Path]:
    """新路径 ``strategy_runs/backtest``，并兼容旧的扁平 ``strategy_runs/*.json``。"""
    return [strategy_runs_dir("backtest"), default_data_dir() / "strategy_runs"]


def _snapshot_path(stem: str) -> Path:
    name = str(stem or "").strip()
    if not _SAFE_STEM.fullmatch(name):
        raise FileNotFoundError(name)
    for root in _snapshot_roots():
        resolved = root.resolve()
        path = (resolved / f"{name}.json").resolve()
        if path.parent == resolved and path.is_file():
            return path
    raise FileNotFoundError(name)


def _say(echo: _Echo, message: str) -> None:
    if echo is not None:
        echo(message)


def _phase(progress: _Echo, name: str) -> None:
    if progress is not None:
        progress(name)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
