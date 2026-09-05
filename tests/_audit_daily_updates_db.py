#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读核对本地 DailyUpdates 库：日历、行情覆盖、sidecar、脏点。"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yg_quant_repo import default_db_path, default_factor_dir


EXPECTED_INDEXES = {
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "000016.SH",
    "399006.SZ",
    "000985.SH",
    "399101.SZ",
    "000688.SH",
}
EXPECTED_INDEX_BARS = {
    "index_SH000001",
    "index_SZ399001",
    "index_SZ399006",
    "index_SH000300",
}
MARKET_VALUE_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_factor",
    "pe_ttm",
    "pb",
    "total_mv",
    "up_limit",
    "down_limit",
)


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _one(connection: sqlite3.Connection, sql: str, params=()):
    row = connection.execute(sql, params).fetchone()
    return None if row is None else row[0]


def audit() -> dict:
    db_path = default_db_path()
    report: dict = {"db_path": str(db_path), "ok": True, "problems": []}
    if not db_path.is_file():
        report["ok"] = False
        report["problems"].append("数据库文件不存在")
        return report

    with _connect(db_path) as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        report["tables"] = sorted(tables)
        counts = {}
        for table in (
            "market_data",
            "trade_calendar",
            "instruments",
            "stock_basic",
            "stock_namechange",
            "industry_classify",
            "industry_member",
            "financial_indicator",
            "index_constituent",
            "dataset_registry",
        ):
            counts[table] = int(_one(connection, f"SELECT COUNT(*) FROM {table}") or 0)
        report["counts"] = counts

        cal_min = _one(connection, "SELECT MIN(trade_date) FROM trade_calendar")
        cal_max = _one(connection, "SELECT MAX(trade_date) FROM trade_calendar")
        mkt_min = _one(connection, "SELECT MIN(trade_date) FROM market_data")
        mkt_max = _one(connection, "SELECT MAX(trade_date) FROM market_data")
        report["calendar_range"] = [cal_min, cal_max]
        report["market_range"] = [mkt_min, mkt_max]

        cal_missing_market = int(
            _one(
                connection,
                """
                SELECT COUNT(*) FROM trade_calendar c
                WHERE NOT EXISTS (
                    SELECT 1 FROM market_data m WHERE m.trade_date = c.trade_date
                )
                """,
            )
            or 0
        )
        market_missing_cal = int(
            _one(
                connection,
                """
                SELECT COUNT(DISTINCT m.trade_date) FROM market_data m
                WHERE NOT EXISTS (
                    SELECT 1 FROM trade_calendar c WHERE c.trade_date = m.trade_date
                )
                """,
            )
            or 0
        )
        report["calendar_without_market"] = cal_missing_market
        report["market_without_calendar"] = market_missing_cal

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(market_data)")
        }
        report["market_columns"] = sorted(columns)
        missing_cols = [name for name in MARKET_VALUE_FIELDS if name not in columns]
        report["missing_market_columns"] = missing_cols

        stock_days = int(
            _one(
                connection,
                "SELECT COUNT(*) FROM market_data WHERE symbol NOT LIKE 'index_%'",
            )
            or 0
        )
        index_days = int(
            _one(
                connection,
                "SELECT COUNT(*) FROM market_data WHERE symbol LIKE 'index_%'",
            )
            or 0
        )
        report["stock_rows"] = stock_days
        report["index_rows"] = index_days

        coverage = {}
        for field in MARKET_VALUE_FIELDS:
            if field not in columns:
                coverage[field] = None
                continue
            extra = " AND close IS NOT NULL" if field == "adj_factor" else ""
            total = int(
                _one(
                    connection,
                    f"""
                    SELECT COUNT(*) FROM market_data
                    WHERE symbol NOT LIKE 'index_%'{extra}
                    """,
                )
                or 0
            )
            filled = int(
                _one(
                    connection,
                    f"""
                    SELECT COUNT("{field}") FROM market_data
                    WHERE symbol NOT LIKE 'index_%'{extra}
                    """,
                )
                or 0
            )
            coverage[field] = {
                "total": total,
                "filled": filled,
                "missing": total - filled,
                "pct": round(100.0 * filled / total, 3) if total else None,
            }
        report["field_coverage"] = coverage

        incomplete_days = {}
        for field in ("close", "adj_factor", "up_limit", "pe_ttm"):
            if field not in columns:
                continue
            extra = ""
            if field == "adj_factor":
                extra = " AND symbol NOT LIKE 'index_%' AND close IS NOT NULL"
            elif field != "close":
                extra = " AND symbol NOT LIKE 'index_%'"
            rows = connection.execute(
                f"""
                SELECT trade_date, COUNT(*) AS n, COUNT("{field}") AS filled
                FROM market_data
                WHERE 1=1 {extra}
                GROUP BY trade_date
                HAVING COUNT("{field}") < COUNT(*)
                ORDER BY trade_date
                """,
            ).fetchall()
            incomplete_days[field] = {
                "days": len(rows),
                "first": rows[0]["trade_date"] if rows else None,
                "last": rows[-1]["trade_date"] if rows else None,
            }
        report["incomplete_days"] = incomplete_days

        dirty = {
            "tiny_close": int(
                _one(
                    connection,
                    """
                    SELECT COUNT(*) FROM market_data
                    WHERE symbol NOT LIKE 'index_%'
                      AND close IS NOT NULL AND close <= 0.05
                    """,
                )
                or 0
            ),
            "nonpos_adj": int(
                _one(
                    connection,
                    """
                    SELECT COUNT(*) FROM market_data
                    WHERE symbol NOT LIKE 'index_%'
                      AND adj_factor IS NOT NULL AND adj_factor <= 0
                    """,
                )
                or 0
            ),
            "ohlc_cross": int(
                _one(
                    connection,
                    """
                    SELECT COUNT(*) FROM market_data
                    WHERE symbol NOT LIKE 'index_%'
                      AND high IS NOT NULL AND low IS NOT NULL
                      AND close IS NOT NULL
                      AND (high < close OR low > close)
                    """,
                )
                or 0
            ),
        }
        report["dirty_rows"] = dirty

        inst_stocks = int(
            _one(connection, "SELECT COUNT(*) FROM instruments WHERE is_index = 0")
            or 0
        )
        inst_index = int(
            _one(connection, "SELECT COUNT(*) FROM instruments WHERE is_index = 1")
            or 0
        )
        report["instruments"] = {"stocks": inst_stocks, "indexes": inst_index}
        index_symbols = [
            row["symbol"]
            for row in connection.execute(
                "SELECT symbol FROM instruments WHERE is_index = 1 ORDER BY symbol"
            )
        ]
        report["index_symbols"] = index_symbols
        missing_index_bars = sorted(EXPECTED_INDEX_BARS - set(index_symbols))
        report["missing_index_bars"] = missing_index_bars

        stale_symbols = int(
            _one(
                connection,
                """
                SELECT COUNT(*) FROM stock_basic
                WHERE symbol NOT LIKE 'SH%'
                  AND symbol NOT LIKE 'SZ%'
                  AND symbol NOT LIKE 'BJ%'
                """,
            )
            or 0
        )
        report["stock_basic_stale_symbols"] = stale_symbols
        report["stock_basic_status"] = {
            row["list_status"]: row["n"]
            for row in connection.execute(
                "SELECT list_status, COUNT(*) AS n FROM stock_basic GROUP BY list_status"
            )
        }

        report["industry"] = {
            "classify": int(
                _one(
                    connection,
                    "SELECT COUNT(*) FROM industry_classify WHERE src='SW2021'",
                )
                or 0
            ),
            "members": int(
                _one(
                    connection,
                    "SELECT COUNT(*) FROM industry_member WHERE src='SW2021'",
                )
                or 0
            ),
            "members_new": int(
                _one(
                    connection,
                    "SELECT COUNT(*) FROM industry_member WHERE src='SW2021' AND is_new='Y'",
                )
                or 0
            ),
        }

        report["financial"] = {
            "rows": counts["financial_indicator"],
            "latest_end": _one(
                connection,
                "SELECT MAX(end_date) FROM financial_indicator WHERE end_date != ''",
            ),
            "flags": {
                row["update_flag"]: row["n"]
                for row in connection.execute(
                    "SELECT update_flag, COUNT(*) AS n "
                    "FROM financial_indicator GROUP BY update_flag"
                )
            },
        }

        constituents = {
            row["index_code"]: {
                "rows": row["n"],
                "latest": row["latest"],
                "dates": row["dates"],
            }
            for row in connection.execute(
                """
                SELECT index_code, COUNT(*) AS n,
                       MAX(trade_date) AS latest,
                       COUNT(DISTINCT trade_date) AS dates
                FROM index_constituent
                GROUP BY index_code
                ORDER BY index_code
                """
            )
        }
        report["index_constituents"] = constituents
        report["missing_index_constituents"] = sorted(
            EXPECTED_INDEXES - set(constituents)
        )

        registry = [
            row["dataset_name"]
            for row in connection.execute(
                "SELECT dataset_name FROM dataset_registry ORDER BY dataset_name"
            )
        ]
        report["registry"] = registry

    snapshot_path = db_path.parent / "dataset_config.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        report["snapshot_datasets"] = sorted(snapshot)
        report["registry_vs_snapshot"] = {
            "only_registry": sorted(set(registry) - set(snapshot)),
            "only_snapshot": sorted(set(snapshot) - set(registry)),
        }

    factor_dir = default_factor_dir()
    calendar_file = factor_dir / "calendars" / "day.txt"
    if calendar_file.is_file():
        days = [
            line.strip()
            for line in calendar_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report["factor_calendar"] = {
            "days": len(days),
            "first": days[0] if days else None,
            "last": days[-1] if days else None,
        }

    problems = []
    if counts["market_data"] == 0:
        problems.append("market_data 为空")
    if cal_missing_market:
        problems.append(f"日历有 {cal_missing_market} 天没有行情行")
    if market_missing_cal:
        problems.append(f"行情有 {market_missing_cal} 天不在日历")
    if missing_cols:
        problems.append(f"缺少行情列: {missing_cols}")
    if missing_index_bars:
        problems.append(f"缺少指数行情: {missing_index_bars}")
    if stale_symbols:
        problems.append(f"stock_basic 仍有 {stale_symbols} 个非仓库代码")
    if report["missing_index_constituents"]:
        problems.append(
            f"缺少指数成分: {report['missing_index_constituents']}"
        )
    if counts["stock_namechange"] == 0:
        problems.append("namechange 为空，PIT ST 会失效")
    if counts["financial_indicator"] == 0:
        problems.append("financial_indicator 为空")
    adj = coverage.get("adj_factor") or {}
    if adj.get("total") and adj.get("missing", 0) / adj["total"] > 0.02:
        problems.append(
            f"adj_factor 缺失 {adj['missing']}/{adj['total']} "
            f"({100 - (adj['pct'] or 0):.2f}%)"
        )
    if dirty["tiny_close"]:
        problems.append(f"仍有 {dirty['tiny_close']} 行 close<=0.05")
    if dirty["nonpos_adj"]:
        problems.append(f"仍有 {dirty['nonpos_adj']} 行 adj_factor<=0")
    report["problems"] = problems
    report["ok"] = not problems
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
