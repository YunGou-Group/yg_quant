#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充只读查询：行业 is_new、复权缺口、涨跌停年份。"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yg_quant_repo import default_db_path


def main() -> None:
    path = default_db_path()
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    out = {
        "industry_is_new": list(
            connection.execute(
                "SELECT is_new, COUNT(*) FROM industry_member GROUP BY is_new"
            )
        ),
        "industry_empty_out": connection.execute(
            "SELECT COUNT(*) FROM industry_member "
            "WHERE out_date IS NULL OR out_date = ''"
        ).fetchone()[0],
        "industry_l1": connection.execute(
            "SELECT COUNT(DISTINCT l1_code) FROM industry_member"
        ).fetchone()[0],
        "stock_basic_status": list(
            connection.execute(
                "SELECT list_status, COUNT(*) FROM stock_basic GROUP BY list_status"
            )
        ),
        "inst_not_in_basic_sample": list(
            connection.execute(
                """
                SELECT i.symbol, i.start_date, i.end_date
                FROM instruments i
                WHERE i.is_index = 0
                  AND NOT EXISTS (
                    SELECT 1 FROM stock_basic b WHERE b.symbol = i.symbol
                  )
                ORDER BY i.start_date
                LIMIT 10
                """
            )
        ),
        "adj_missing_top": list(
            connection.execute(
                """
                SELECT symbol, COUNT(*) AS n, MIN(trade_date), MAX(trade_date)
                FROM market_data
                WHERE symbol NOT LIKE 'index_%'
                  AND close IS NOT NULL
                  AND adj_factor IS NULL
                GROUP BY symbol
                ORDER BY n DESC
                LIMIT 10
                """
            )
        ),
        "up_limit_range": connection.execute(
            "SELECT MIN(trade_date), MAX(trade_date) "
            "FROM market_data WHERE up_limit IS NOT NULL"
        ).fetchone(),
        "up_limit_null_by_year": list(
            connection.execute(
                """
                SELECT substr(trade_date, 1, 4) AS y, COUNT(*) AS n
                FROM market_data
                WHERE symbol NOT LIKE 'index_%'
                  AND close IS NOT NULL
                  AND up_limit IS NULL
                GROUP BY y
                ORDER BY y
                """
            )
        ),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
