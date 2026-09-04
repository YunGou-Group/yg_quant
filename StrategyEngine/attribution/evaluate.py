#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读回测快照 + 行情库，调用 run_attribution，写出归因 JSON。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from DailyUpdates.storage import SQLiteStorage
from FactorEvaluates.industry_panel import load_l1_code_panel
from FactorEvaluates.market_panel_loader import HS300_SYMBOL, MarketPanelLoader
from yg_quant_repo import default_db_path
from StrategyEngine.live.codes import to_qmt_code
from StrategyEngine.run import _SAFE_STEM, _runs_root

from . import ledger
from .engine import run_attribution
from .factors import build_factor_payload, factor_group_totals, summarize_factor_rows
from .tree import returns_tree

_LOG = logging.getLogger("StrategyEngine")


def attribute_snapshot(
    run: str,
    *,
    out: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    snap_path = resolve_snapshot(run)
    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    snapshot.setdefault("id", snap_path.stem)
    equity_path, trades_path = _run_files(snapshot, snap_path)
    if not equity_path.is_file():
        raise FileNotFoundError(f"找不到净值文件 {equity_path}")
    if not trades_path.is_file():
        raise FileNotFoundError(f"找不到成交文件 {trades_path}")

    equity = ledger.load_equity(equity_path)
    trades = ledger.load_trades(trades_path)
    dates = [str(d) for d in equity["date"].tolist()]
    if len(dates) < 2:
        raise ValueError("回测净值少于 2 个交易日，无法归因")

    params = snapshot.get("params") or {}
    initial = float(params.get("cash") or equity["nav"].iloc[0] or 0.0)
    cash_by_date = {
        str(row.date): float(row.cash)
        for row in equity.itertuples(index=False)
        if pd.notna(row.cash)
    }
    positions = ledger.positions_by_date(dates, trades)
    start_positions = ledger.previous_positions(dates, positions)

    symbols = sorted(
        {str(s) for s in trades["symbol"].tolist() if str(s)}
        | {sym for held in positions.values() for sym in held}
    )
    storage = SQLiteStorage(str(db_path or default_db_path()))
    close_map, _open_map = _load_prices(storage, symbols, dates, adjust="none")
    bench_symbol = _benchmark_symbol(snapshot, params)
    bench_close = {}
    if bench_symbol:
        series = MarketPanelLoader(str(db_path or default_db_path())).load_index_close(bench_symbol)
        bench_close = ledger.series_to_map(series)

    returns = ledger.period_returns(
        dates,
        cash_by_date,
        positions,
        close_map,
        initial_cash=initial,
        benchmark_close=bench_close,
    )
    payload: Dict[str, Any] = {
        "asset_pool": ["stock"],
        "portfolio_name": snapshot.get("tag") or snapshot.get("id") or snap_path.stem,
        "returns": returns,
        "trades": ledger.trade_rows(trades, close_map),
        "holdings": ledger.holding_rows(dates, start_positions, close_map),
    }

    bench_w = _constituent_weights(storage, dates, bench_symbol) if bench_symbol else None
    industry_payload, industry_note = _industry_payload(
        storage,
        dates,
        start_positions,
        close_map,
        cash_by_date,
        symbols,
        bench_w,
    )
    factor_payload, factor_note = build_factor_payload(
        storage,
        dates,
        start_positions,
        close_map,
        cash_by_date,
        benchmark_weights=bench_w,
        db_path=str(db_path or default_db_path()),
    )
    payload["stock"] = {"industry": industry_payload, "factor": factor_payload}

    result = run_attribution(payload)
    dest = Path(out) if out else attribution_path(snap_path)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "run_id": snapshot.get("id") or snap_path.stem,
        "snapshot": str(snap_path),
        "files": {"equity": str(equity_path), "trades": str(trades_path), "attribution": str(dest)},
        "strategy": snapshot.get("strategy"),
        "allocator": snapshot.get("allocator"),
        "valuation": "close",
        "price_adjust": "none",
        "benchmark": bench_symbol,
        "notes": [
            "盯市与成交同为未复权价；净值按收盘还原，与回测开盘净值不同；交易收益用成交开盘价相对当日收盘。",
            industry_note,
            factor_note,
        ],
        "summary": _brief(result),
        "attribution": result,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def attribution_path(snap_path: Path) -> Path:
    return Path(snap_path).with_name(f"{Path(snap_path).stem}_attribution.json")


def is_current_report(data: Any) -> bool:
    if not isinstance(data, Mapping):
        return False
    attr = data.get("attribution")
    sections = attr.get("sections") if isinstance(attr, Mapping) else None
    if not isinstance(sections, Mapping):
        return False
    return isinstance(sections.get("factor"), Mapping) and isinstance(
        sections.get("industry"), Mapping
    )


def load_attribution_report(run: str) -> Dict[str, Any]:
    snap_path = resolve_snapshot(run)
    path = attribution_path(snap_path)
    if not path.is_file():
        raise FileNotFoundError(f"回测 {snap_path.stem} 还没有归因结果")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not is_current_report(data):
        raise ValueError(f"回测 {snap_path.stem} 的归因缺少风格分解，请重新计算")
    return data


def attribute_for_web(run: str, *, db_path: Optional[str] = None) -> Dict[str, Any]:
    """给网页用：跑归因并返回精简视图，不含逐笔成交。"""
    return web_view(attribute_snapshot(run, db_path=db_path))


def load_attribution_view(run: str) -> Dict[str, Any]:
    return web_view(load_attribution_report(run))


def web_view(report: Mapping[str, Any]) -> Dict[str, Any]:
    """去掉 pnl 逐笔，留下洋葱树、行业合计、风格因子、日收益序列。"""
    attr = report.get("attribution") if isinstance(report.get("attribution"), Mapping) else {}
    summary = dict(attr.get("summary") or {})
    sections = attr.get("sections") if isinstance(attr.get("sections"), Mapping) else {}
    industry = sections.get("industry") if isinstance(sections.get("industry"), Mapping) else {}
    factor = sections.get("factor") if isinstance(sections.get("factor"), Mapping) else {}
    pnl = attr.get("pnl") if isinstance(attr.get("pnl"), Mapping) else {}
    warnings = []
    for item in attr.get("quality_warnings") or []:
        if not isinstance(item, Mapping):
            continue
        warnings.append(
            {
                "code": item.get("code"),
                "severity": item.get("severity"),
                "message": item.get("message"),
                "scope": item.get("scope"),
            }
        )
        if len(warnings) >= 40:
            break
    by_period = []
    for row in pnl.get("by_period") or []:
        if not isinstance(row, Mapping):
            continue
        by_period.append(
            {
                "period": row.get("period"),
                "trade_return": _num(row.get("trade_return")),
                "holding_return": _num(row.get("holding_return")),
                "total_mtm_return": _num(row.get("total_mtm_return")),
                "trade_mtm": _num(row.get("trade_mtm")),
                "holding_mtm": _num(row.get("holding_mtm")),
            }
        )
    factor_rows = summarize_factor_rows(factor.get("rows") or [])
    factor_groups = factor_group_totals(factor_rows)
    return {
        "run_id": report.get("run_id"),
        "created_at": report.get("created_at"),
        "strategy": report.get("strategy"),
        "allocator": report.get("allocator"),
        "valuation": report.get("valuation"),
        "price_adjust": report.get("price_adjust"),
        "benchmark": report.get("benchmark"),
        "notes": [str(x) for x in (report.get("notes") or []) if x],
        "status": attr.get("status"),
        "summary": summary,
        "returns_tree": returns_tree(summary),
        "pnl": by_period,
        "industry": {
            "title": industry.get("title") or "股票行业归因",
            "method": industry.get("method"),
            "rows": _industry_totals(industry.get("rows") or []),
        },
        "factors": {
            "title": factor.get("title") or "股票因子归因",
            "method": factor.get("method"),
            "note": factor.get("note"),
            "groups": factor_groups,
            "rows": factor_rows,
        },
        "warnings": warnings,
        "files": report.get("files") or {},
    }


def _industry_totals(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("industry") or "未分类")
        item = buckets.setdefault(
            name,
            {
                "industry": name,
                "allocation": 0.0,
                "selection": 0.0,
                "total_effect": 0.0,
                "portfolio_weight": 0.0,
                "benchmark_weight": 0.0,
            },
        )
        for key in ("allocation", "selection", "total_effect"):
            item[key] += float(row.get(key) or 0.0)
        item["portfolio_weight"] += float(row.get("portfolio_weight") or 0.0)
        item["benchmark_weight"] += float(row.get("benchmark_weight") or 0.0)
        counts[name] = counts.get(name, 0) + 1
    for name, item in buckets.items():
        n = max(int(counts.get(name, 1)), 1)
        item["portfolio_weight"] /= n
        item["benchmark_weight"] /= n
    ordered = sorted(buckets.values(), key=lambda row: abs(row["total_effect"]), reverse=True)
    return ordered


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def resolve_snapshot(run: str) -> Path:
    text = str(run or "").strip()
    if not text:
        raise FileNotFoundError("请用 --run 指定回测快照 id 或 JSON 路径")
    path = Path(text)
    if path.is_file():
        return path.resolve()
    name = text[:-5] if text.lower().endswith(".json") else text
    if not _SAFE_STEM.fullmatch(name):
        raise FileNotFoundError(name)
    root = _runs_root().resolve()
    dest = (root / f"{name}.json").resolve()
    if dest.parent != root or not dest.is_file():
        raise FileNotFoundError(f"找不到回测快照 {name}")
    return dest


def _run_files(snapshot: Mapping[str, Any], snap_path: Path) -> tuple[Path, Path]:
    files = snapshot.get("files") or {}
    equity = Path(files["equity"]) if files.get("equity") else snap_path.with_suffix(".csv")
    trades = (
        Path(files["trades"])
        if files.get("trades")
        else snap_path.with_name(f"{snap_path.stem}_trades.csv")
    )
    if not equity.is_file():
        alt = snap_path.with_suffix(".csv")
        if alt.is_file():
            equity = alt
    if not trades.is_file():
        alt = snap_path.with_name(f"{snap_path.stem}_trades.csv")
        if alt.is_file():
            trades = alt
    return equity, trades


def _benchmark_symbol(snapshot: Mapping[str, Any], params: Mapping[str, Any]) -> Optional[str]:
    for key in ("benchmark",):
        raw = snapshot.get(key)
        if raw:
            text = str(raw).strip()
            if text.lower() not in {"none", "off"}:
                return text
    raw = params.get("benchmark")
    if raw:
        text = str(raw).strip()
        if text.lower() not in {"none", "off"}:
            return text
    return HS300_SYMBOL


def _load_prices(
    storage: SQLiteStorage,
    symbols: Sequence[str],
    dates: Sequence[str],
    *,
    adjust: str,
) -> tuple[dict, dict]:
    if not symbols:
        return {}, {}
    start = min(dates)
    end = max(dates)
    frame = storage.read_market_data(
        fields=["open", "close"],
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        symbols=list(symbols),
        ordered=False,
        adjust=adjust,
    )
    return ledger.pivot_prices(frame, "close"), ledger.pivot_prices(frame, "open")


def _industry_payload(
    storage: SQLiteStorage,
    dates: Sequence[str],
    start_positions: Mapping[str, Mapping[str, float]],
    close_map: Mapping[str, Mapping[str, float]],
    cash_by_date: Mapping[str, float],
    symbols: Sequence[str],
    benchmark_weights: Optional[Dict[str, Dict[str, float]]],
) -> tuple[dict, str]:
    held = sorted({sym for qty in start_positions.values() for sym in qty} | set(symbols))
    if not held:
        raise ValueError("无持仓，无法做行业归因。")
    try:
        panel, labels = load_l1_code_panel(
            storage, pd.Index(list(dates)), pd.Index(held)
        )
    except Exception as exc:
        raise ValueError("行业表加载失败，无法做行业归因。") from exc
    if panel.empty or panel.isna().all().all():
        raise ValueError("行业成分为空，无法做行业归因。")
    name_by_date: Dict[str, Dict[str, str]] = {}
    for day in dates:
        if day not in panel.index:
            continue
        row = panel.loc[day]
        mapping = {}
        for symbol, code in row.items():
            if code is None or (isinstance(code, float) and pd.isna(code)):
                continue
            code_s = str(code)
            mapping[str(symbol)] = str(labels.get(code_s) or code_s)
        name_by_date[day] = mapping
    sides = ledger.industry_sides(
        dates,
        start_positions,
        close_map,
        cash_by_date,
        name_by_date,
        benchmark_weights=benchmark_weights,
    )
    if not sides["portfolio"]:
        raise ValueError("行业权重为空，无法做行业归因。")
    note = "行业 BF 用申万一级；基准为指数成分 as-of 权重。"
    if not sides["benchmark"]:
        note = "无指数成分，行业归因只有组合侧，Brison 会把单边行业记入配置。"
    return sides, note


def _constituent_weights(
    storage: SQLiteStorage,
    dates: Sequence[str],
    bench_symbol: Optional[str],
) -> Optional[Dict[str, Dict[str, float]]]:
    codes = _index_code_candidates(bench_symbol)
    frame = pd.DataFrame()
    for code in codes:
        try:
            part = storage.read_index_constituents(
                index_code=code,
                start_date=min(dates).replace("-", ""),
                end_date=max(dates).replace("-", ""),
            )
        except Exception:
            part = pd.DataFrame()
        if part is not None and not part.empty:
            frame = part
            break
    if frame is None or frame.empty:
        for code in codes:
            try:
                part = storage.read_index_constituents(index_code=code)
            except Exception:
                continue
            if part is not None and not part.empty:
                frame = part
                break
    if frame is None or frame.empty:
        return None
    work = frame.copy()
    work["day"] = work["trade_date"].map(ledger.normalize_day)
    work["symbol"] = work["symbol"].astype(str)
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work = work.dropna(subset=["day", "symbol", "weight"])
    rebalance_days = sorted(work["day"].unique())
    by_day = {
        day: grp.groupby("symbol")["weight"].sum()
        for day, grp in work.groupby("day", sort=True)
    }
    out: Dict[str, Dict[str, float]] = {}
    ptr = -1
    for day in dates:
        while ptr + 1 < len(rebalance_days) and rebalance_days[ptr + 1] <= day:
            ptr += 1
        if ptr < 0:
            continue
        weights = by_day[rebalance_days[ptr]].copy()
        total = float(weights.sum())
        if total > 2.0:
            weights = weights / 100.0
        out[day] = {str(sym): float(val) for sym, val in weights.items() if float(val) > 0}
    return out or None


def _index_code_candidates(symbol: Optional[str]) -> List[str]:
    text = str(symbol or "").strip()
    if not text:
        return ["000300.SH"]
    names = [text]
    body = text[6:] if text.startswith("index_") else text
    qmt = to_qmt_code(body)
    if qmt and qmt not in names:
        names.append(qmt)
    if "000300" in text and "000300.SH" not in names:
        names.append("000300.SH")
    return names


def _brief(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(result.get("summary") or {})
    recon = result.get("reconciliation") or []
    failed = [row for row in recon if isinstance(row, dict) and not row.get("passed", True)]
    return {
        "status": result.get("status"),
        "summary": summary,
        "tables": [row.get("key") for row in (result.get("tables") or []) if isinstance(row, dict)],
        "warnings": len(result.get("quality_warnings") or []),
        "reconcile_failed": len(failed),
    }
