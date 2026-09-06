#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对 universe_index_weight 里每个配置代码：名录正式代码 vs index_weight 谁有成分。

    python tests/_probe_index_weight_codes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yg_quant_repo import ensure_repo_root

ensure_repo_root()

import pandas as pd
import tushare as ts

from DailyUpdates.data_fetcher.data_sources.tushare_data_source import (
    INDEX_WEIGHT_CODE_ALIAS,
)

INDEX_LIST = [
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "000016.SH",
    "399006.SZ",
    "000985.SH",
    "399101.SZ",
    "000688.SH",
]


def _token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("没有 TUSHARE_TOKEN")
    return token


def _safe(fn):
    try:
        frame = fn()
    except Exception as exc:
        print(f"  调用失败: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    return frame if frame is not None else pd.DataFrame()


def _stem(code: str) -> str:
    return str(code).split(".", 1)[0]


def _candidates(code: str) -> list[str]:
    stem = _stem(code)
    out = [code]
    alias = INDEX_WEIGHT_CODE_ALIAS.get(code)
    if alias:
        out.append(alias)
    out.append(f"{stem}.CSI")
    if code.endswith(".SH") and stem.startswith("000"):
        out.append(f"399{stem[3:]}.SZ")
    if code.endswith(".SZ") and stem.startswith("399"):
        out.append(f"000{stem[3:]}.SH")
        out.append(f"000{stem[3:]}.CSI")
    seen = []
    for item in out:
        if item not in seen:
            seen.append(item)
    return seen


def main() -> None:
    pro = ts.pro_api(_token())
    end = pd.Timestamp.today().normalize()
    month_end = end.replace(day=1) - pd.Timedelta(days=1)
    month_start = month_end.replace(day=1)
    start_s = month_start.strftime("%Y%m%d")
    end_s = month_end.strftime("%Y%m%d")
    print(f"探测窗口: {start_s} ~ {end_s}\n")

    basics = []
    for market in ("SSE", "SZSE", "CSI"):
        frame = _safe(lambda m=market: pro.index_basic(market=m))
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["_market"] = market
        basics.append(frame)
    catalog = pd.concat(basics, ignore_index=True) if basics else pd.DataFrame()

    rows = []
    for code in INDEX_LIST:
        stem = _stem(code)
        print(f"==== {code} ====")
        if not catalog.empty and "ts_code" in catalog.columns:
            hits = catalog[catalog["ts_code"].astype(str).str.startswith(stem)]
            if hits.empty:
                print("  index_basic: 无同号")
            else:
                cols = [c for c in ("ts_code", "name", "market", "publisher") if c in hits.columns]
                print(hits[cols].to_string(index=False))
        weight_hits = []
        for cand in _candidates(code):
            part = _safe(
                lambda c=cand: pro.index_weight(
                    index_code=c, start_date=start_s, end_date=end_s
                )
            )
            n = 0 if part is None or part.empty else len(part)
            returned = ""
            if n and "index_code" in part.columns:
                returned = ",".join(sorted(set(part["index_code"].astype(str))))
            mark = ""
            if cand == code:
                mark = " [配置]"
            elif cand == INDEX_WEIGHT_CODE_ALIAS.get(code):
                mark = " [已有别名]"
            print(f"  index_weight {cand:>14} -> {n:6d} 行{mark}  返回={returned or '-'}")
            weight_hits.append((cand, n))
        config_n = next(n for c, n in weight_hits if c == code)
        best = max(weight_hits, key=lambda item: item[1])
        need = config_n == 0 and best[1] > 0
        rows.append(
            {
                "config": code,
                "config_rows": config_n,
                "best_request": best[0],
                "best_rows": best[1],
                "need_alias": need,
                "current_alias": INDEX_WEIGHT_CODE_ALIAS.get(code, ""),
            }
        )
        print()

    summary = pd.DataFrame(rows)
    print("==== 汇总 ====")
    print(summary.to_string(index=False))
    missing = summary[summary["need_alias"]]
    if missing.empty:
        print("\n配置代码或已有别名都能拉到成分，无需再加镜像。")
    else:
        print("\n这些配置代码本身无成分，应加别名：")
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
