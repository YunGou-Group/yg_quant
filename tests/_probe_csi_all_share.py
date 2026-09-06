#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""现场探测中证全指在 Tushare 里的正式代码，以及 index_weight 哪个名字有成分。

    python tests/_probe_csi_all_share.py
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


SEED_CODES = (
    "000985.SH",
    "000985.CSI",
    "399985.SZ",
)
NAME_NEEDLES = ("中证全指", "中证全指指数", "CSI All Share")


def _token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("没有 TUSHARE_TOKEN，请写在仓库根 .env 或环境变量里")
    return token


def _safe(label: str, fn):
    try:
        frame = fn()
    except Exception as exc:
        print(f"[{label}] 调用失败: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    if frame is None:
        return pd.DataFrame()
    return frame


def _filter_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    keep = pd.Series(False, index=frame.index)
    if "ts_code" in frame.columns:
        text = frame["ts_code"].astype(str)
        keep = keep | text.str.contains("000985|399985", case=False, regex=True)
    for col in ("name", "indx_name", "indx_csname"):
        if col not in frame.columns:
            continue
        text = frame[col].astype(str)
        for needle in NAME_NEEDLES:
            keep = keep | text.str.contains(needle, case=False, na=False)
    return frame.loc[keep].copy()


def _print_frame(title: str, frame: pd.DataFrame, cols=None) -> None:
    print(f"\n=== {title}  ({0 if frame is None else len(frame)} 行) ===")
    if frame is None or frame.empty:
        print("(空)")
        return
    show = frame
    if cols:
        show = frame.loc[:, [c for c in cols if c in frame.columns]]
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print(show.to_string(index=False))


def main() -> None:
    pro = ts.pro_api(_token())
    end = pd.Timestamp.today().normalize()
    last_month_end = end.replace(day=1) - pd.Timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    start_s = last_month_start.strftime("%Y%m%d")
    end_s = last_month_end.strftime("%Y%m%d")
    print(f"探测窗口（上一个完整自然月）: {start_s} ~ {end_s}")

    catalogs = []
    for market in ("SSE", "SZSE", "CSI"):
        frame = _safe(
            f"index_basic market={market}",
            lambda m=market: pro.index_basic(market=m),
        )
        hit = _filter_catalog(frame)
        _print_frame(
            f"index_basic[{market}] 匹配 000985/中证全指",
            hit,
            ["ts_code", "name", "market", "publisher", "category", "list_date", "exp_date"],
        )
        catalogs.append(hit)

    etf = _safe("etf_index", lambda: pro.etf_index())
    etf_hit = _filter_catalog(etf)
    _print_frame(
        "etf_index 匹配 000985/中证全指",
        etf_hit,
        ["ts_code", "indx_name", "indx_csname", "pub_party_name", "pub_date", "base_date"],
    )
    catalogs.append(etf_hit)

    found = []
    for frame in catalogs:
        if frame is None or frame.empty or "ts_code" not in frame.columns:
            continue
        found.extend(str(v).strip() for v in frame["ts_code"].tolist() if str(v).strip())
    candidates = list(dict.fromkeys([*SEED_CODES, *found]))
    print(f"\n=== 将探测 index_weight 的代码 ({len(candidates)}) ===")
    print("\n".join(candidates) or "(无)")

    rows = []
    for code in candidates:
        part = _safe(
            f"index_weight {code}",
            lambda c=code: pro.index_weight(
                index_code=c, start_date=start_s, end_date=end_s
            ),
        )
        n = 0 if part is None or part.empty else len(part)
        returned = ""
        if n and "index_code" in part.columns:
            returned = ",".join(sorted(set(part["index_code"].astype(str))))
        print(f"index_weight {code:>14} -> {n:6d} 行  返回index_code={returned or '-'}")
        rows.append({"query": code, "rows": n, "returned_index_code": returned})

    summary = pd.DataFrame(rows).sort_values("rows", ascending=False)
    _print_frame("index_weight 汇总（有行的才是能用的请求代码）", summary)


if __name__ == "__main__":
    main()
