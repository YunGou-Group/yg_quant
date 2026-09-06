#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查 SZ000022 / SZ000043 / SZ300114 为何有行情却不在 stock_basic。"""

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

CODES = [
    ("SZ000022", "000022.SZ"),
    ("SZ000043", "000043.SZ"),
    ("SZ300114", "300114.SZ"),
]
SUCCESSORS = [
    ("SZ001872", "001872.SZ"),
    ("SZ302132", "302132.SZ"),
]


def main() -> None:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("没有 TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()
    path = default_db_path()
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    print("==== 库内 ====")
    for symbol, _ in CODES:
        market = connection.execute(
            """
            SELECT COUNT(*) AS n, MIN(trade_date) AS first, MAX(trade_date) AS last,
                   SUM(close IS NOT NULL) AS with_close
            FROM market_data WHERE symbol=?
            """,
            (symbol,),
        ).fetchone()
        basic = connection.execute(
            "SELECT symbol, name, list_status, list_date, delist_date "
            "FROM stock_basic WHERE symbol=?",
            (symbol,),
        ).fetchall()
        names = connection.execute(
            "SELECT name, start_date, end_date FROM stock_namechange "
            "WHERE symbol=? ORDER BY start_date",
            (symbol,),
        ).fetchall()
        print(
            symbol,
            dict(market),
            "basic=",
            [dict(row) for row in basic],
            "namechange=",
            [dict(row) for row in names],
        )

    print("\n==== 库内疑似后继代码 ====")
    for symbol, _ in SUCCESSORS:
        rows = connection.execute(
            "SELECT symbol, name, list_status, list_date, delist_date "
            "FROM stock_basic WHERE symbol=?",
            (symbol,),
        ).fetchall()
        print(symbol, [dict(row) for row in rows])

    print("\n==== Tushare stock_basic(ts_code) L/D/P ====")
    for _, ts_code in CODES + SUCCESSORS:
        frames = []
        for status in ("L", "D", "P"):
            part = pro.stock_basic(
                ts_code=ts_code,
                list_status=status,
                fields="ts_code,symbol,name,list_status,list_date,delist_date,market",
            )
            if part is not None and not part.empty:
                frames.append(part)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print(ts_code, "空" if out.empty else out.to_string(index=False))

    print("\n==== Tushare daily 近端 / 2018 ====")
    for _, ts_code in CODES:
        recent = pro.daily(ts_code=ts_code, start_date="20260901", end_date="20260904")
        old = pro.daily(ts_code=ts_code, start_date="20180102", end_date="20180110")
        print(
            ts_code,
            "recent",
            0 if recent is None or recent.empty else len(recent),
            "old",
            0 if old is None or old.empty else len(old),
        )
        if recent is not None and not recent.empty:
            print(recent[["ts_code", "trade_date", "close"]].to_string(index=False))

    print("\n==== 2026-09-04 日线是否仍含这三只 ====")
    day = pro.daily(trade_date="20260904", fields="ts_code,close")
    hit = day[day["ts_code"].isin([ts for _, ts in CODES])] if day is not None else pd.DataFrame()
    print("日线总行", 0 if day is None else len(day), "命中", hit.to_string(index=False))

    print("\n==== 最后有收盘日 / instruments ====")
    for symbol, _ in CODES:
        last_close = connection.execute(
            "SELECT MAX(trade_date) FROM market_data "
            "WHERE symbol=? AND close IS NOT NULL",
            (symbol,),
        ).fetchone()[0]
        inst = connection.execute(
            "SELECT symbol, start_date, end_date FROM instruments WHERE symbol=?",
            (symbol,),
        ).fetchone()
        print(symbol, "last_close", last_close, "instruments", dict(inst) if inst else None)

    print("\n==== 后继代码 namechange ====")
    for symbol, _ in SUCCESSORS:
        rows = connection.execute(
            "SELECT name, start_date, end_date FROM stock_namechange "
            "WHERE symbol=? ORDER BY start_date",
            (symbol,),
        ).fetchall()
        print(symbol, [dict(row) for row in rows])

    print("\n==== 库内名称含 赤湾/善达/电测/中航地产 ====")
    rows = connection.execute(
        """
        SELECT symbol, name, start_date, end_date
        FROM stock_namechange
        WHERE name LIKE '%赤湾%' OR name LIKE '%善达%'
           OR name LIKE '%电测%' OR name LIKE '%中航地产%'
        ORDER BY symbol, start_date
        """
    ).fetchall()
    for row in rows:
        print(dict(row))

    print("\n==== Tushare 000043 最后有日线的日期 ====")
    old043 = pro.daily(ts_code="000043.SZ", start_date="20191101", end_date="20200131")
    print("empty" if old043 is None or old043.empty else old043.tail(3).to_string(index=False))

    print("\n==== Tushare namechange ====")
    for _, ts_code in CODES + SUCCESSORS:
        frame = pro.namechange(ts_code=ts_code)
        print(ts_code)
        if frame is None or frame.empty:
            print("  空")
        else:
            cols = [c for c in ("ts_code", "name", "start_date", "end_date") if c in frame.columns]
            print(frame[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
