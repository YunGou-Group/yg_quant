#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照库内缺失涨跌停与 Tushare 官方 stk_limit，并核对规则补齐。

    python tests/_probe_missing_limits.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yg_quant_repo import default_db_path, ensure_repo_root

ensure_repo_root()

import pandas as pd
import tushare as ts

from DailyUpdates.data_fetcher.preprocessing.market_bars import sanitize_market_bars
from DailyUpdates.storage.sqlite_storage import SQLiteStorage

# 审计里空洞最重的年份 + 板块规则切换日。
TARGET_DATES = [
    "2019-06-03",  # 创业板仍 10%
    "2020-08-24",  # 创业板改 20% 首日
    "2021-07-15",  # 2021 年空洞高峰附近，脚本会按库内实际缺口覆盖
    "2022-01-10",  # 北交所已开
]


def _token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("没有 TUSHARE_TOKEN，请写在仓库根 .env 或环境变量里")
    return token


def _to_ts_code(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if text.startswith("index_"):
        return ""
    if text.startswith(("SH", "SZ", "BJ")) and len(text) >= 8:
        return f"{text[2:]}.{text[:2]}"
    return text


def _ymd8(value: str) -> str:
    return str(value).replace("-", "")[:8]


def _pick_hole_dates(connection: sqlite3.Connection, extra: list[str]) -> list[str]:
    holes = pd.read_sql_query(
        """
        SELECT trade_date, COUNT(*) AS missing
        FROM market_data
        WHERE close IS NOT NULL
          AND (up_limit IS NULL OR down_limit IS NULL)
          AND symbol NOT LIKE 'index\\_%' ESCAPE '\\'
          AND trade_date BETWEEN '2015-01-01' AND '2022-12-31'
        GROUP BY trade_date
        ORDER BY missing DESC
        LIMIT 8
        """,
        connection,
    )
    print("库内 2015–2022 缺涨跌停最多的交易日：")
    print(holes.to_string(index=False))
    dates = list(holes["trade_date"].astype(str))
    for day in extra:
        if day not in dates:
            dates.append(day)
    return dates[:12]


def _fetch_official(pro, day: str) -> pd.DataFrame:
    ymd = _ymd8(day)
    frame = pro.stk_limit(
        trade_date=ymd, fields="ts_code,trade_date,up_limit,down_limit"
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
    return frame


def _compare_one(storage: SQLiteStorage, official: pd.DataFrame, day: str) -> dict:
    bars = storage.read_market_data(
        fields=["pre_close", "close", "up_limit", "down_limit"],
        start_date=day,
        end_date=day,
        include_indexes=False,
        ordered=False,
        adjust="none",
    )
    if bars.empty:
        return {"day": day, "db_rows": 0}

    bars = bars.rename(columns={"ts_code": "symbol"})
    db_missing = bars[
        bars["close"].notna()
        & (bars["up_limit"].isna() | bars["down_limit"].isna())
    ]
    official = official.copy()
    official["symbol"] = official["ts_code"].astype(str).map(SQLiteStorage._normalize_symbol)
    official["up_limit"] = pd.to_numeric(official["up_limit"], errors="coerce")
    official["down_limit"] = pd.to_numeric(official["down_limit"], errors="coerce")

    namechange = storage.read_stock_namechange()
    basic = storage.read_stock_basic()
    list_dates = None
    if basic is not None and not basic.empty and "list_date" in basic.columns:
        listed = pd.to_datetime(
            basic.drop_duplicates("symbol").set_index("symbol")["list_date"],
            errors="coerce",
        )
        list_dates = listed[listed.notna()]

    # 用官方有、库内缺的行测补齐；同时对「官方也有」的全日做规则对照。
    masked = bars.copy()
    masked["up_limit"] = pd.NA
    masked["down_limit"] = pd.NA
    try:
        calendar = storage.list_trade_dates()
    except Exception:
        calendar = None
    filled = sanitize_market_bars(
        masked, namechange=namechange, list_dates=list_dates, calendar=calendar
    )
    merged = filled.merge(
        official[["symbol", "up_limit", "down_limit"]].rename(
            columns={"up_limit": "off_up", "down_limit": "off_dn"}
        ),
        on="symbol",
        how="inner",
    )
    merged["fill_up"] = pd.to_numeric(merged["up_limit"], errors="coerce")
    merged["fill_dn"] = pd.to_numeric(merged["down_limit"], errors="coerce")
    comparable = merged[merged["off_up"].notna() & merged["off_dn"].notna()]
    up_ok = (comparable["fill_up"] - comparable["off_up"]).abs() <= 0.011
    dn_ok = (comparable["fill_dn"] - comparable["off_dn"]).abs() <= 0.011
    both_ok = up_ok & dn_ok
    mismatch = comparable.loc[~both_ok].copy()
    mismatch["du"] = (mismatch["fill_up"] - mismatch["off_up"]).round(4)
    mismatch["dd"] = (mismatch["fill_dn"] - mismatch["off_dn"]).round(4)

    official_codes = set(official["symbol"])
    missing_also_in_api = db_missing[~db_missing["symbol"].isin(official_codes)]
    missing_but_api_has = db_missing[db_missing["symbol"].isin(official_codes)]

    return {
        "day": day,
        "db_rows": int(len(bars)),
        "db_missing": int(len(db_missing)),
        "api_rows": int(len(official)),
        "db_missing_api_empty": int(len(missing_also_in_api)),
        "db_missing_api_has": int(len(missing_but_api_has)),
        "compared": int(len(comparable)),
        "match": int(both_ok.sum()),
        "mismatch": int((~both_ok).sum()),
        "mismatch_sample": mismatch.head(8),
    }


def main() -> None:
    token = _token()
    ts.set_token(token)
    pro = ts.pro_api()
    path = default_db_path()
    print(f"DB: {path}")
    storage = SQLiteStorage(str(path))
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    dates = _pick_hole_dates(connection, TARGET_DATES)
    connection.close()

    rows = []
    for day in dates:
        official = _fetch_official(pro, day)
        stats = _compare_one(storage, official, day)
        rows.append(stats)
        print(
            f"\n{day}  库行={stats.get('db_rows')}  库缺限价={stats.get('db_missing')}  "
            f"API={stats.get('api_rows')}  库缺且API也无={stats.get('db_missing_api_empty')}  "
            f"库缺但API有={stats.get('db_missing_api_has')}  "
            f"对照={stats.get('compared')}  一致={stats.get('match')}  不一致={stats.get('mismatch')}"
        )
        sample = stats.get("mismatch_sample")
        if sample is not None and len(sample):
            cols = ["symbol", "pre_close", "fill_up", "off_up", "du", "fill_dn", "off_dn", "dd"]
            print(sample[cols].to_string(index=False))

    summary = pd.DataFrame(
        [
            {k: v for k, v in row.items() if k != "mismatch_sample"}
            for row in rows
        ]
    )
    print("\n汇总：")
    print(summary.to_string(index=False))
    total_cmp = int(summary["compared"].fillna(0).sum())
    total_ok = int(summary["match"].fillna(0).sum())
    if total_cmp:
        print(f"\n官方有限价的行，规则补齐一致率：{total_ok}/{total_cmp} = {total_ok / total_cmp:.4%}")


if __name__ == "__main__":
    main()
