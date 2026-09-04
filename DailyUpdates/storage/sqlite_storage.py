#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite-backed storage for market data and sidecar tables."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .financial_schema import FINANCIAL_KEY_COLUMNS, FINANCIAL_VALUE_FIELDS


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SYSTEM_COLUMNS = {"symbol", "date", "ts_code", "trade_date"}

ADJ_FACTOR_COLUMN = "adj_factor"
# 后复权口径（Tushare 官方公式）：后复权价 = 原始价 × adj_factor。
# 成交量按 1/adj_factor 缩放，使 vwap = amount / vol 同样是后复权价。
ADJ_PRICE_FIELDS = frozenset(
    {"open", "high", "low", "close", "pre_close", "up_limit", "down_limit"}
)
ADJ_VOLUME_FIELDS = frozenset({"vol"})
# amount / pct_chg / 市值 / 估值字段本身已是除权口径或与复权无关，保持原值。
ADJUSTABLE_FIELDS = ADJ_PRICE_FIELDS | ADJ_VOLUME_FIELDS
ADJUST_MODES = ("none", "hfq")

logger = logging.getLogger("SQLiteStorage")


class _ClosingConnection(sqlite3.Connection):
    """Commit or roll back, then always release the Windows file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class SQLiteStorage:
    """Single-file SQLite storage used by all data and factor pipelines."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path, timeout=30, factory=_ClosingConnection
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            # 因子已迁至 Parquet，清理遗留 SQLite 表
            connection.execute("DROP TABLE IF EXISTS factor_values")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    PRIMARY KEY (symbol, trade_date)
                );
                CREATE INDEX IF NOT EXISTS idx_market_data_date
                    ON market_data (trade_date);

                CREATE TABLE IF NOT EXISTS instruments (
                    symbol TEXT PRIMARY KEY,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    is_index INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS trade_calendar (
                    trade_date TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS dataset_registry (
                    dataset_name TEXT PRIMARY KEY,
                    data_source TEXT NOT NULL DEFAULT '',
                    data_type TEXT NOT NULL DEFAULT 'daily',
                    api_name TEXT NOT NULL DEFAULT '',
                    fields_json TEXT NOT NULL DEFAULT '[]',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_action TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS stock_namechange (
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    start_date TEXT NOT NULL DEFAULT '',
                    end_date TEXT NOT NULL DEFAULT '',
                    ann_date TEXT NOT NULL DEFAULT '',
                    change_reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (symbol, start_date, name)
                );
                CREATE INDEX IF NOT EXISTS idx_stock_namechange_symbol
                    ON stock_namechange (symbol, start_date);

                CREATE TABLE IF NOT EXISTS industry_classify (
                    src TEXT NOT NULL,
                    index_code TEXT NOT NULL,
                    industry_name TEXT,
                    parent_code TEXT,
                    level TEXT,
                    industry_code TEXT,
                    is_pub TEXT,
                    PRIMARY KEY (src, index_code)
                );
                CREATE INDEX IF NOT EXISTS idx_industry_classify_level
                    ON industry_classify (src, level);

                CREATE TABLE IF NOT EXISTS industry_member (
                    src TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    l1_code TEXT,
                    l1_name TEXT,
                    l2_code TEXT,
                    l2_name TEXT,
                    l3_code TEXT NOT NULL,
                    l3_name TEXT,
                    name TEXT,
                    in_date TEXT NOT NULL DEFAULT '',
                    out_date TEXT,
                    is_new TEXT,
                    PRIMARY KEY (src, symbol, l3_code, in_date)
                );
                CREATE INDEX IF NOT EXISTS idx_industry_member_symbol
                    ON industry_member (symbol);
                CREATE INDEX IF NOT EXISTS idx_industry_member_l1
                    ON industry_member (src, l1_code);
                CREATE INDEX IF NOT EXISTS idx_industry_member_l3
                    ON industry_member (src, l3_code);

                CREATE TABLE IF NOT EXISTS stock_basic (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    area TEXT,
                    industry TEXT,
                    market TEXT,
                    list_date TEXT,
                    delist_date TEXT,
                    list_status TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_stock_basic_list_date
                    ON stock_basic (list_date);

                CREATE TABLE IF NOT EXISTS index_constituent (
                    index_code TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    weight REAL,
                    PRIMARY KEY (index_code, symbol, trade_date)
                );
                CREATE INDEX IF NOT EXISTS idx_index_constituent_date
                    ON index_constituent (trade_date);
                CREATE INDEX IF NOT EXISTS idx_index_constituent_symbol
                    ON index_constituent (symbol);
                """
            )
            self._migrate_financial_indicator(connection)
            self._ensure_financial_columns(connection)
            self._ensure_stock_basic_columns(connection)
            self._migrate_stock_basic_symbols(connection)

    @staticmethod
    def _financial_create_sql() -> str:
        value_sql = ",\n                    ".join(
            f"{field} REAL" for field in FINANCIAL_VALUE_FIELDS
        )
        return f"""
                CREATE TABLE IF NOT EXISTS financial_indicator (
                    symbol TEXT NOT NULL,
                    ann_date TEXT,
                    end_date TEXT NOT NULL,
                    update_flag TEXT NOT NULL DEFAULT '',
                    {value_sql},
                    PRIMARY KEY (symbol, end_date, ann_date, update_flag)
                );
                CREATE INDEX IF NOT EXISTS idx_financial_indicator_ann
                    ON financial_indicator (ann_date);
                CREATE INDEX IF NOT EXISTS idx_financial_indicator_end
                    ON financial_indicator (end_date);
                """

    def _ensure_financial_columns(self, connection: sqlite3.Connection) -> None:
        """给已有 financial_indicator 补 Barra10 数值列（SQLite ADD COLUMN）。"""
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(financial_indicator)")
        }
        if not existing:
            connection.executescript(self._financial_create_sql())
            return
        for field in FINANCIAL_VALUE_FIELDS:
            if field in existing:
                continue
            field = self._validate_identifier(field)
            connection.execute(
                f'ALTER TABLE financial_indicator ADD COLUMN "{field}" REAL'
            )

    def _ensure_stock_basic_columns(self, connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(stock_basic)")
        }
        if not existing or "delist_date" in existing:
            return
        connection.execute("ALTER TABLE stock_basic ADD COLUMN delist_date TEXT")

    def _migrate_financial_indicator(self, connection: sqlite3.Connection) -> None:
        """旧表主键无 update_flag 时重建，避免同键 keep=last 只能留一行。"""
        row = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='financial_indicator'"
        ).fetchone()
        if row is None:
            return
        cols = {
            r["name"] for r in connection.execute("PRAGMA table_info(financial_indicator)")
        }
        if "update_flag" in cols:
            return
        connection.execute("ALTER TABLE financial_indicator RENAME TO financial_indicator_old")
        connection.execute(
            """
            CREATE TABLE financial_indicator (
                symbol TEXT NOT NULL,
                ann_date TEXT,
                end_date TEXT NOT NULL,
                roe REAL,
                roa REAL,
                update_flag TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (symbol, end_date, ann_date, update_flag)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_financial_indicator_ann "
            "ON financial_indicator (ann_date)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_financial_indicator_end "
            "ON financial_indicator (end_date)"
        )
        # 迁移旧行（无 flag 时记为空串）；随后由调用方清空并重拉更稳妥
        connection.execute(
            """
            INSERT OR IGNORE INTO financial_indicator
                (symbol, ann_date, end_date, roe, roa, update_flag)
            SELECT symbol, ann_date, end_date, roe, roa, ''
            FROM financial_indicator_old
            """
        )
        connection.execute("DROP TABLE financial_indicator_old")

    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(identifier):
            raise ValueError(f"不安全的字段名: {identifier!r}")
        return identifier

    @staticmethod
    def _date_string(value) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    @staticmethod
    def _database_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _ensure_market_columns(
        self, connection: sqlite3.Connection, fields: Iterable[str]
    ) -> None:
        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(market_data)")
        }
        for field in sorted(set(fields) - _SYSTEM_COLUMNS):
            field = self._validate_identifier(field)
            if field not in existing:
                connection.execute(
                    f'ALTER TABLE market_data ADD COLUMN "{field}" REAL'
                )

    def get_market_fields(self) -> List[str]:
        with self._connect() as connection:
            return [
                row["name"]
                for row in connection.execute("PRAGMA table_info(market_data)")
                if row["name"] not in {"symbol", "trade_date"}
            ]

    def upsert_market_data(self, data: pd.DataFrame) -> int:
        """Insert or update a normalized DataFrame containing symbol/date columns."""
        if data is None or data.empty:
            return 0

        frame = data.copy()
        rename_map = {}
        if "symbol" not in frame.columns and "ts_code" in frame.columns:
            rename_map["ts_code"] = "symbol"
        if "date" not in frame.columns and "trade_date" in frame.columns:
            rename_map["trade_date"] = "date"
        frame = frame.rename(columns=rename_map)
        if not {"symbol", "date"}.issubset(frame.columns):
            raise ValueError("行情数据必须包含 symbol/date 或 ts_code/trade_date 列")

        frame = frame.dropna(subset=["symbol", "date"]).copy()
        frame["symbol"] = frame["symbol"].astype(str)
        frame["trade_date"] = frame["date"].map(self._date_string)
        frame = frame.drop(columns=["date"])
        frame = frame.drop_duplicates(["symbol", "trade_date"], keep="last")

        fields = [
            self._validate_identifier(column)
            for column in frame.columns
            if column not in {"symbol", "trade_date"}
        ]
        columns = ["symbol", "trade_date", *fields]
        placeholders = ", ".join("?" for _ in columns)
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        if fields:
            # COALESCE：局部补拉（如只回填 adj_factor）不会把该行其它字段抹成 NULL。
            # 新插入的停牌 NaN 行走 INSERT 分支，不受影响。
            updates = ", ".join(
                f'"{field}" = COALESCE(excluded."{field}", "{field}")'
                for field in fields
            )
            conflict_sql = f"DO UPDATE SET {updates}"
        else:
            conflict_sql = "DO NOTHING"
        sql = (
            f"INSERT INTO market_data ({quoted_columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(symbol, trade_date) {conflict_sql}"
        )
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]

        instrument_rows = []
        for symbol, group in frame.groupby("symbol"):
            instrument_rows.append(
                (
                    symbol,
                    group["trade_date"].min(),
                    group["trade_date"].max(),
                    int(str(symbol).startswith("index_")),
                )
            )
        dates = [(date,) for date in sorted(frame["trade_date"].unique())]

        with self._connect() as connection:
            self._ensure_market_columns(connection, fields)
            connection.executemany(sql, rows)
            connection.executemany(
                """
                INSERT INTO instruments (symbol, start_date, end_date, is_index)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    start_date = MIN(instruments.start_date, excluded.start_date),
                    end_date = MAX(instruments.end_date, excluded.end_date),
                    is_index = excluded.is_index
                """,
                instrument_rows,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO trade_calendar (trade_date) VALUES (?)", dates
            )
        return len(rows)

    def read_market_data(
        self,
        fields: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
        include_indexes: bool = False,
        ordered: bool = True,
        adjust: str = "none",
    ) -> pd.DataFrame:
        """读取行情宽表。adjust='hfq' 时按 adj_factor 输出后复权价。

        指数行没有 adj_factor，复权因子按 1.0 处理。
        """
        mode = str(adjust or "none").lower()
        if mode not in ADJUST_MODES:
            raise ValueError(f"不支持的复权模式: {adjust!r}，可选 {ADJUST_MODES}")
        available = set(self.get_market_fields())
        requested = list(fields) if fields is not None else sorted(available)
        missing = set(requested) - available
        if missing:
            raise ValueError(f"数据库中不存在字段: {sorted(missing)}")
        requested = [self._validate_identifier(field) for field in requested]

        selected = list(requested)
        needs_factor = mode == "hfq" and any(
            field in ADJUSTABLE_FIELDS for field in requested
        )
        if needs_factor and ADJ_FACTOR_COLUMN not in selected:
            if ADJ_FACTOR_COLUMN not in available:
                raise ValueError(
                    "请求了复权价但库中没有 adj_factor 列，请先回填 adj_factor 数据集"
                )
            selected.append(ADJ_FACTOR_COLUMN)

        select_columns = [
            "symbol AS ts_code",
            "trade_date",
            *(f'"{field}"' for field in selected),
        ]
        clauses: List[str] = []
        params: List[str] = []
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(self._date_string(start_date))
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(self._date_string(end_date))
        if not include_indexes:
            clauses.append("symbol NOT LIKE 'index\\_%' ESCAPE '\\'")
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(str(symbol) for symbol in symbols)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        # ORDER BY on ~20M rows is expensive; callers that re-sort can pass ordered=False.
        order_sql = " ORDER BY symbol, trade_date" if ordered else ""
        sql = (
            f"SELECT {', '.join(select_columns)} FROM market_data{where}"
            f"{order_sql}"
        )
        with self._connect() as connection:
            frame = pd.read_sql_query(sql, connection, params=params)
        if not needs_factor:
            return frame
        return self._apply_adjustment(frame, requested)

    @staticmethod
    def _apply_adjustment(
        frame: pd.DataFrame, requested: Sequence[str]
    ) -> pd.DataFrame:
        """按 adj_factor 就地缩放价格 / 成交量列，再收敛回调用方请求的列。"""
        if frame.empty:
            return frame.loc[:, ["ts_code", "trade_date", *requested]]
        factor = pd.to_numeric(frame[ADJ_FACTOR_COLUMN], errors="coerce")
        usable = factor > 0
        stock_rows = ~frame["ts_code"].astype(str).str.startswith("index_")
        gap = int((stock_rows & ~usable).sum())
        if gap:
            logger.warning(
                "adj_factor 缺失 %s/%s 行个股数据，这些行后复权价置为 NaN，禁止与原始价混用",
                gap,
                int(stock_rows.sum()),
            )
        scaled = factor.copy()
        scaled.loc[~stock_rows] = 1.0
        scaled.loc[stock_rows & ~usable] = np.nan
        factor_arr = scaled.to_numpy(dtype="float64")
        for field in requested:
            if field in ADJ_PRICE_FIELDS:
                frame[field] = pd.to_numeric(frame[field], errors="coerce") * factor_arr
            elif field in ADJ_VOLUME_FIELDS:
                frame[field] = pd.to_numeric(frame[field], errors="coerce") / factor_arr
        return frame.loc[:, ["ts_code", "trade_date", *requested]]

    def get_instruments(self, include_indexes: bool = False) -> Set[str]:
        sql = "SELECT symbol FROM instruments"
        if not include_indexes:
            sql += " WHERE is_index = 0"
        with self._connect() as connection:
            return {row["symbol"] for row in connection.execute(sql)}

    def get_latest_market_date(self) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(trade_date) AS latest FROM trade_calendar"
            ).fetchone()
            return row["latest"] if row else None

    def list_trade_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        clauses: List[str] = []
        params: List[str] = []
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(self._date_string(start_date))
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(self._date_string(end_date))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT trade_date FROM trade_calendar{where} ORDER BY trade_date",
                params,
            ).fetchall()
        return [row["trade_date"] for row in rows]

    def get_calendar_range(self) -> Tuple[Optional[str], Optional[str]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MIN(trade_date) AS first, MAX(trade_date) AS last "
                "FROM trade_calendar"
            ).fetchone()
            return (row["first"], row["last"]) if row else (None, None)

    def get_registry(self) -> Dict[str, Dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM dataset_registry ORDER BY dataset_name"
            ).fetchall()
        return {
            row["dataset_name"]: {
                "data_source": row["data_source"],
                "data_type": row["data_type"],
                "api_name": row["api_name"],
                "fields": json.loads(row["fields_json"]),
                "config": json.loads(row["config_json"]),
                "created_time": row["created_at"],
                "last_updated": row["updated_at"],
                "last_action": row["last_action"],
            }
            for row in rows
        }

    def register_dataset(
        self, dataset_name: str, config: Dict, action: str
    ) -> None:
        fields = sorted(
            set(config.get("fields", [])) - {"ts_code", "trade_date", "symbol", "date"}
        )
        safe_config = {
            key: value for key, value in config.items() if key.lower() != "token"
        }
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT fields_json FROM dataset_registry WHERE dataset_name = ?",
                (dataset_name,),
            ).fetchone()
            if existing:
                fields = sorted(set(json.loads(existing["fields_json"])) | set(fields))
            connection.execute(
                """
                INSERT INTO dataset_registry (
                    dataset_name, data_source, data_type, api_name, fields_json,
                    config_json, last_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_name) DO UPDATE SET
                    data_source = excluded.data_source,
                    data_type = excluded.data_type,
                    api_name = excluded.api_name,
                    fields_json = excluded.fields_json,
                    config_json = excluded.config_json,
                    updated_at = CURRENT_TIMESTAMP,
                    last_action = excluded.last_action
                """,
                (
                    dataset_name,
                    config.get("data_source", ""),
                    config.get("data_type", "daily"),
                    config.get("api_name", ""),
                    json.dumps(fields, ensure_ascii=False),
                    json.dumps(safe_config, ensure_ascii=False),
                    action,
                ),
            )

    def clear_registry(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM dataset_registry")

    def unregister_dataset(self, dataset_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM dataset_registry WHERE dataset_name = ?",
                (dataset_name,),
            )

    def delete_symbols(self, symbols: Sequence[str]) -> int:
        symbols = [str(symbol) for symbol in symbols]
        if not symbols:
            return 0
        placeholders = ", ".join("?" for _ in symbols)
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM market_data WHERE symbol IN ({placeholders})",
                symbols,
            )
            connection.execute(
                f"DELETE FROM instruments WHERE symbol IN ({placeholders})",
                symbols,
            )
        return len(symbols)

    def drop_market_fields(self, fields: Iterable[str]) -> List[str]:
        """Drop unused market_data columns. Shared system columns are ignored."""
        dropped: List[str] = []
        existing = set(self.get_market_fields())
        with self._connect() as connection:
            for field in sorted(set(fields) - _SYSTEM_COLUMNS):
                field = self._validate_identifier(field)
                if field not in existing:
                    continue
                connection.execute(
                    f'ALTER TABLE market_data DROP COLUMN "{field}"'
                )
                dropped.append(field)
        return dropped

    def replace_registry(self, dataset_config: Dict[str, Dict]) -> None:
        """Rewrite dataset_registry to match the current config snapshot."""
        with self._connect() as connection:
            connection.execute("DELETE FROM dataset_registry")
        for name, config in dataset_config.items():
            self.register_dataset(name, config, action="snapshot")

    def replace_industry_classify(self, data: pd.DataFrame, src: str) -> int:
        """Replace industry classification rows for one source version."""
        if data is None or data.empty:
            return 0
        frame = data.copy()
        frame["src"] = src
        columns = [
            "src",
            "index_code",
            "industry_name",
            "parent_code",
            "level",
            "industry_code",
            "is_pub",
        ]
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.dropna(subset=["index_code"]).copy()
        frame["index_code"] = frame["index_code"].astype(str)
        frame = frame.drop_duplicates(["src", "index_code"], keep="last")
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM industry_classify WHERE src = ?", (src,)
            )
            connection.executemany(
                f"INSERT INTO industry_classify ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                rows,
            )
        return len(rows)

    def replace_industry_members(self, data: pd.DataFrame, src: str) -> int:
        """Replace industry membership rows for one source version."""
        if data is None or data.empty:
            return 0
        frame = data.copy()
        if "symbol" not in frame.columns and "ts_code" in frame.columns:
            frame["symbol"] = frame["ts_code"].astype(str).map(self._normalize_symbol)
        frame["src"] = src
        columns = [
            "src",
            "symbol",
            "l1_code",
            "l1_name",
            "l2_code",
            "l2_name",
            "l3_code",
            "l3_name",
            "name",
            "in_date",
            "out_date",
            "is_new",
        ]
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.dropna(subset=["symbol", "l3_code"]).copy()
        frame["symbol"] = frame["symbol"].astype(str)
        frame["l3_code"] = frame["l3_code"].astype(str)
        frame["in_date"] = frame["in_date"].fillna("").astype(str)
        frame = frame.drop_duplicates(
            ["src", "symbol", "l3_code", "in_date"], keep="last"
        )
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM industry_member WHERE src = ?", (src,)
            )
            connection.executemany(
                f"INSERT INTO industry_member ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                rows,
            )
        return len(rows)

    def replace_stock_namechange(self, data: pd.DataFrame) -> int:
        """Replace the historical name table (PIT ST detection)."""
        frame = self._prepare_namechange(data)
        if frame is None:
            return 0
        columns = ["symbol", "name", "start_date", "end_date", "ann_date", "change_reason"]
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        with self._connect() as connection:
            connection.execute("DELETE FROM stock_namechange")
            connection.executemany(
                f"INSERT INTO stock_namechange ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                rows,
            )
        return len(rows)

    def upsert_stock_namechange(self, data: pd.DataFrame) -> int:
        frame = self._prepare_namechange(data)
        if frame is None:
            return 0
        columns = ["symbol", "name", "start_date", "end_date", "ann_date", "change_reason"]
        updates = ", ".join(
            f'"{field}" = excluded."{field}"'
            for field in ("end_date", "ann_date", "change_reason")
        )
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        with self._connect() as connection:
            connection.executemany(
                f"INSERT INTO stock_namechange ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)}) "
                f"ON CONFLICT(symbol, start_date, name) DO UPDATE SET {updates}",
                rows,
            )
        return len(rows)

    def _prepare_namechange(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if data is None or data.empty:
            return None
        frame = data.copy()
        if "symbol" not in frame.columns and "ts_code" in frame.columns:
            frame["symbol"] = frame["ts_code"].astype(str).map(self._to_repo_symbol)
        if "symbol" not in frame.columns:
            raise ValueError("namechange 数据缺少 ts_code/symbol 列")
        for column in ("name", "start_date", "end_date", "ann_date", "change_reason"):
            if column not in frame.columns:
                frame[column] = ""
            frame[column] = frame[column].fillna("").astype(str).replace({"nan": "", "None": ""})
        frame = frame.dropna(subset=["symbol"]).copy()
        frame["symbol"] = frame["symbol"].astype(str)
        for column in ("start_date", "end_date", "ann_date"):
            frame[column] = frame[column].map(self._iso_or_blank)
        frame = frame.drop_duplicates(["symbol", "start_date", "name"], keep="last")
        return frame

    @staticmethod
    def _iso_or_blank(value) -> str:
        digits = "".join(ch for ch in str(value or "") if ch.isdigit())
        if len(digits) < 8:
            return ""
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"

    def read_stock_namechange(self) -> pd.DataFrame:
        sql = (
            "SELECT symbol, name, start_date, end_date, ann_date, change_reason "
            "FROM stock_namechange ORDER BY symbol, start_date"
        )
        with self._connect() as connection:
            return pd.read_sql_query(sql, connection)

    def count_stock_namechange(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS c FROM stock_namechange"
            ).fetchone()
            return int(row["c"] if row else 0)

    @staticmethod
    def _normalize_symbol(ts_code: str) -> str:
        text = str(ts_code).strip().upper()
        if "." in text:
            code, market = text.split(".", 1)
            return f"{market}{code}"
        return text

    @classmethod
    def _to_repo_symbol(cls, value: str) -> str:
        """Tushare ts_code / 六位代码 / 已是仓库代码 → SZ000001。"""
        text = str(value).strip().upper()
        if not text:
            return text
        if "." in text:
            return cls._normalize_symbol(text)
        if text.startswith(("SH", "SZ", "BJ")) and len(text) >= 8:
            return text
        digits = "".join(ch for ch in text if ch.isdigit())
        code = digits[-6:].zfill(6) if digits else ""
        if not code:
            return text
        if code.startswith("688"):
            return f"SH{code}"
        if code.startswith(("5", "6", "9")):
            return f"SH{code}"
        if code.startswith(("4", "8")):
            return f"BJ{code}"
        return f"SZ{code}"

    @staticmethod
    def _ymd8(value) -> str:
        if pd.isna(value) or value is None:
            return ""
        text = str(value).replace("-", "").replace(".", "")[:8]
        if not text or text.lower() in {"nan", "none", "nat"}:
            return ""
        return text

    @staticmethod
    def _code6(symbol: str) -> str:
        digits = "".join(ch for ch in str(symbol) if ch.isdigit())
        return digits[-6:].zfill(6) if digits else ""

    def _migrate_stock_basic_symbols(self, connection: sqlite3.Connection) -> int:
        """把已入库的六位 / ts_code 主键改成与行情表一致的仓库代码。"""
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_basic'"
        ).fetchone()
        if exists is None:
            return 0
        stale = connection.execute(
            "SELECT 1 FROM stock_basic "
            "WHERE symbol NOT LIKE 'SH%' AND symbol NOT LIKE 'SZ%' "
            "AND symbol NOT LIKE 'BJ%' LIMIT 1"
        ).fetchone()
        if stale is None:
            return 0
        inst_map: Dict[str, str] = {}
        for row in connection.execute(
            "SELECT symbol FROM instruments "
            "WHERE is_index = 0 AND symbol NOT LIKE 'index\\_%' ESCAPE '\\'"
        ):
            code = self._code6(row["symbol"])
            if code and code not in inst_map:
                inst_map[code] = str(row["symbol"])
        by_qmt: Dict[str, tuple] = {}
        for row in connection.execute(
            "SELECT symbol, name, area, industry, market, list_date, delist_date, list_status "
            "FROM stock_basic"
        ):
            raw = str(row["symbol"])
            qmt = inst_map.get(self._code6(raw)) or self._to_repo_symbol(raw)
            by_qmt[qmt] = (
                qmt,
                row["name"],
                row["area"],
                row["industry"],
                row["market"],
                row["list_date"],
                row["delist_date"],
                row["list_status"],
            )
        connection.execute("DELETE FROM stock_basic")
        connection.executemany(
            "INSERT INTO stock_basic "
            "(symbol, name, area, industry, market, list_date, delist_date, list_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            list(by_qmt.values()),
        )
        return len(by_qmt)

    def clear_industry_data(
        self, src: Optional[str] = None, api_name: Optional[str] = None
    ) -> None:
        """Clear industry tables. Optionally scope by src / api_name."""
        with self._connect() as connection:
            if api_name in (None, "index_classify"):
                if src:
                    connection.execute(
                        "DELETE FROM industry_classify WHERE src = ?", (src,)
                    )
                else:
                    connection.execute("DELETE FROM industry_classify")
            if api_name in (None, "index_member_all"):
                if src:
                    connection.execute(
                        "DELETE FROM industry_member WHERE src = ?", (src,)
                    )
                else:
                    connection.execute("DELETE FROM industry_member")

    def prune_industry_sources(self, active_srcs: Iterable[str]) -> None:
        """Drop industry rows whose src is no longer configured."""
        active = {str(src) for src in active_srcs if src}
        with self._connect() as connection:
            existing = {
                row["src"]
                for row in connection.execute(
                    "SELECT DISTINCT src FROM industry_classify"
                )
            } | {
                row["src"]
                for row in connection.execute(
                    "SELECT DISTINCT src FROM industry_member"
                )
            }
            for src in sorted(existing - active):
                connection.execute(
                    "DELETE FROM industry_classify WHERE src = ?", (src,)
                )
                connection.execute(
                    "DELETE FROM industry_member WHERE src = ?", (src,)
                )

    def read_industry_members(
        self,
        src: Optional[str] = None,
        level: str = "l1",
        is_new: Optional[str] = "Y",
    ) -> pd.DataFrame:
        """Read membership rows; level selects which industry code column to keep."""
        level_key = level.lower()
        code_col = {
            "l1": "l1_code",
            "l2": "l2_code",
            "l3": "l3_code",
            "sector": "l1_code",
            "industry": "l2_code",
            "subindustry": "l3_code",
        }.get(level_key, "l1_code")
        name_col = code_col.replace("_code", "_name")
        clauses = []
        params: List[str] = []
        if src:
            clauses.append("src = ?")
            params.append(src)
        if is_new is not None:
            clauses.append("is_new = ?")
            params.append(is_new)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT symbol, {code_col} AS industry_code, "
            f"{name_col} AS industry_name, l1_code, l1_name, l2_code, l2_name, "
            f"l3_code, l3_name, in_date, out_date, is_new, src "
            f"FROM industry_member{where} ORDER BY symbol, {code_col}"
        )
        with self._connect() as connection:
            return pd.read_sql_query(sql, connection, params=params)

    def read_industry_classify(
        self, src: Optional[str] = None, level: Optional[str] = None
    ) -> pd.DataFrame:
        clauses = []
        params: List[str] = []
        if src:
            clauses.append("src = ?")
            params.append(src)
        if level:
            clauses.append("level = ?")
            params.append(level)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT src, index_code, industry_name, parent_code, level, "
            f"industry_code, is_pub FROM industry_classify{where} "
            "ORDER BY level, index_code"
        )
        with self._connect() as connection:
            return pd.read_sql_query(sql, connection, params=params)

    def replace_stock_basic(self, data: pd.DataFrame) -> int:
        """Replace stock_basic dimension table."""
        prepared = self._prepare_stock_basic(data)
        if prepared is None:
            return 0
        columns, rows = prepared
        with self._connect() as connection:
            connection.execute("DELETE FROM stock_basic")
            connection.executemany(
                f"INSERT INTO stock_basic ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                rows,
            )
        return len(rows)

    def clear_stock_basic(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM stock_basic")

    def read_stock_basic(self, symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        clauses = []
        params: List[str] = []
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(str(symbol) for symbol in symbols)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT symbol, name, area, industry, market, list_date, delist_date, list_status "
            f"FROM stock_basic{where} ORDER BY symbol"
        )
        with self._connect() as connection:
            return pd.read_sql_query(sql, connection, params=params)

    def replace_financial_indicator(self, data: pd.DataFrame) -> int:
        """Replace financial_indicator rows (full refresh)."""
        prepared = self._prepare_financial_indicator(data)
        if prepared is None:
            return 0
        columns, rows = prepared
        quoted = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as connection:
            self._ensure_financial_columns(connection)
            connection.execute("DELETE FROM financial_indicator")
            connection.executemany(
                f"INSERT INTO financial_indicator ({quoted}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)

    def upsert_financial_indicator(self, data: pd.DataFrame) -> int:
        """Insert/update financial_indicator rows without wiping the table."""
        prepared = self._prepare_financial_indicator(data)
        if prepared is None:
            return 0
        columns, rows = prepared
        quoted = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f'"{field}" = excluded."{field}"' for field in FINANCIAL_VALUE_FIELDS
        )
        sql = (
            f"INSERT INTO financial_indicator ({quoted}) VALUES ({placeholders}) "
            f"ON CONFLICT(symbol, end_date, ann_date, update_flag) DO UPDATE SET {updates}"
        )
        with self._connect() as connection:
            self._ensure_financial_columns(connection)
            connection.executemany(sql, rows)
        return len(rows)

    def _prepare_financial_indicator(self, data: pd.DataFrame):
        if data is None or data.empty:
            return None
        frame = data.copy()
        if "symbol" not in frame.columns and "ts_code" in frame.columns:
            frame["symbol"] = frame["ts_code"].astype(str).map(self._normalize_symbol)
        columns = list(FINANCIAL_KEY_COLUMNS) + list(FINANCIAL_VALUE_FIELDS)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.dropna(subset=["symbol", "end_date"]).copy()
        frame["symbol"] = frame["symbol"].astype(str)
        for date_col in ("ann_date", "end_date"):
            frame[date_col] = frame[date_col].map(
                lambda value: (
                    ""
                    if pd.isna(value) or value is None
                    else self._date_string(value)
                )
            )
        frame["ann_date"] = frame["ann_date"].fillna("")
        frame["update_flag"] = (
            frame["update_flag"].fillna("").astype(str).replace({"nan": "", "None": ""})
        )
        frame = frame.drop_duplicates(
            ["symbol", "end_date", "ann_date", "update_flag"], keep="first"
        )
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        return columns, rows

    def get_financial_latest_end_date(self) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(end_date) AS latest FROM financial_indicator "
                "WHERE end_date IS NOT NULL AND end_date != ''"
            ).fetchone()
            return row["latest"] if row and row["latest"] else None

    def clear_financial_indicator(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM financial_indicator")

    def read_financial_indicator(
        self,
        symbols: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        clauses = []
        params: List[str] = []
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(str(symbol) for symbol in symbols)
        if start_date:
            clauses.append("end_date >= ?")
            params.append(self._date_string(start_date))
        if end_date:
            clauses.append("end_date <= ?")
            params.append(self._date_string(end_date))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            self._ensure_financial_columns(connection)
            existing = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(financial_indicator)")
            }
            select_cols = [
                name
                for name in list(FINANCIAL_KEY_COLUMNS) + list(FINANCIAL_VALUE_FIELDS)
                if name in existing
            ]
            quoted = ", ".join(f'"{name}"' for name in select_cols)
            sql = (
                f"SELECT {quoted} FROM financial_indicator{where} "
                "ORDER BY symbol, end_date, ann_date, update_flag"
            )
            return pd.read_sql_query(sql, connection, params=params)

    def replace_index_constituents(
        self, data: pd.DataFrame, index_codes: Optional[Sequence[str]] = None
    ) -> int:
        """Replace index constituent rows; optionally scope delete by index_codes."""
        prepared = self._prepare_index_constituents(data)
        if prepared is None:
            return 0
        columns, rows, frame_codes = prepared
        codes = [
            str(code)
            for code in (index_codes or frame_codes)
        ]
        with self._connect() as connection:
            if codes:
                placeholders = ", ".join("?" for _ in codes)
                connection.execute(
                    f"DELETE FROM index_constituent WHERE index_code IN ({placeholders})",
                    codes,
                )
            else:
                connection.execute("DELETE FROM index_constituent")
            connection.executemany(
                f"INSERT INTO index_constituent ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                rows,
            )
        return len(rows)

    def upsert_index_constituents(self, data: pd.DataFrame) -> int:
        """Insert/update index constituent rows without wiping older months."""
        prepared = self._prepare_index_constituents(data)
        if prepared is None:
            return 0
        columns, rows, _ = prepared
        sql = (
            f"INSERT INTO index_constituent ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)}) "
            "ON CONFLICT(index_code, symbol, trade_date) DO UPDATE SET "
            "weight = excluded.weight"
        )
        with self._connect() as connection:
            connection.executemany(sql, rows)
        return len(rows)

    def _prepare_index_constituents(self, data: pd.DataFrame):
        if data is None or data.empty:
            return None
        frame = data.copy()
        if "symbol" not in frame.columns and "con_code" in frame.columns:
            frame["symbol"] = frame["con_code"].astype(str).map(self._normalize_symbol)
        elif "symbol" not in frame.columns and "ts_code" in frame.columns:
            frame["symbol"] = frame["ts_code"].astype(str).map(self._normalize_symbol)
        columns = ["index_code", "symbol", "trade_date", "weight"]
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.dropna(subset=["index_code", "symbol", "trade_date"]).copy()
        frame["index_code"] = frame["index_code"].astype(str)
        frame["symbol"] = frame["symbol"].astype(str)
        frame["trade_date"] = frame["trade_date"].map(self._date_string)
        frame = frame.drop_duplicates(
            ["index_code", "symbol", "trade_date"], keep="last"
        )
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        return columns, rows, sorted(frame["index_code"].unique())

    def get_index_constituent_latest_date(
        self, index_codes: Optional[Sequence[str]] = None
    ) -> Optional[str]:
        clauses = []
        params: List[str] = []
        if index_codes:
            placeholders = ", ".join("?" for _ in index_codes)
            clauses.append(f"index_code IN ({placeholders})")
            params.extend(str(code) for code in index_codes)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT MAX(trade_date) AS latest FROM index_constituent{where}",
                params,
            ).fetchone()
            return row["latest"] if row and row["latest"] else None

    def clear_index_constituents(
        self, index_codes: Optional[Sequence[str]] = None
    ) -> None:
        with self._connect() as connection:
            if index_codes:
                placeholders = ", ".join("?" for _ in index_codes)
                connection.execute(
                    f"DELETE FROM index_constituent WHERE index_code IN ({placeholders})",
                    [str(code) for code in index_codes],
                )
            else:
                connection.execute("DELETE FROM index_constituent")

    def prune_index_constituents(self, active_index_codes: Iterable[str]) -> None:
        active = {str(code) for code in active_index_codes if code}
        with self._connect() as connection:
            existing = {
                row["index_code"]
                for row in connection.execute(
                    "SELECT DISTINCT index_code FROM index_constituent"
                )
            }
            stale = sorted(existing - active)
            if stale:
                placeholders = ", ".join("?" for _ in stale)
                connection.execute(
                    f"DELETE FROM index_constituent WHERE index_code IN ({placeholders})",
                    stale,
                )

    def read_index_constituents(
        self,
        index_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        clauses = []
        params: List[str] = []
        if index_code:
            clauses.append("index_code = ?")
            params.append(str(index_code))
        if trade_date:
            clauses.append("trade_date = ?")
            params.append(self._date_string(trade_date))
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(self._date_string(start_date))
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(self._date_string(end_date))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT index_code, symbol, trade_date, weight "
            f"FROM index_constituent{where} "
            "ORDER BY index_code, trade_date, symbol"
        )
        with self._connect() as connection:
            return pd.read_sql_query(sql, connection, params=params)

    def count_stock_basic(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS c FROM stock_basic"
            ).fetchone()
            return int(row["c"] if row else 0)

    def count_industry_classify(self, src: Optional[str] = None) -> int:
        with self._connect() as connection:
            if src:
                row = connection.execute(
                    "SELECT COUNT(*) AS c FROM industry_classify WHERE src = ?",
                    (src,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS c FROM industry_classify"
                ).fetchone()
            return int(row["c"] if row else 0)

    def upsert_stock_basic(self, data: pd.DataFrame) -> int:
        """Insert/update stock_basic without wiping the whole table."""
        prepared = self._prepare_stock_basic(data)
        if prepared is None:
            return 0
        columns, rows = prepared
        sql = (
            f"INSERT INTO stock_basic ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)}) "
            "ON CONFLICT(symbol) DO UPDATE SET "
            "name = excluded.name, "
            "area = excluded.area, "
            "industry = excluded.industry, "
            "market = excluded.market, "
            "list_date = excluded.list_date, "
            "delist_date = excluded.delist_date, "
            "list_status = excluded.list_status"
        )
        with self._connect() as connection:
            connection.executemany(sql, rows)
        return len(rows)

    def _prepare_stock_basic(self, data: pd.DataFrame):
        if data is None or data.empty:
            return None
        frame = data.copy()
        # Tushare stock_basic 同时有 ts_code=000001.SZ 与 symbol=000001；
        # 主键必须与 market_data 一样用 ts_code 转成仓库代码。
        if "ts_code" in frame.columns:
            frame["symbol"] = frame["ts_code"].astype(str).map(self._to_repo_symbol)
        elif "symbol" in frame.columns:
            frame["symbol"] = frame["symbol"].astype(str).map(self._to_repo_symbol)
        columns = [
            "symbol",
            "name",
            "area",
            "industry",
            "market",
            "list_date",
            "delist_date",
            "list_status",
        ]
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.dropna(subset=["symbol"]).copy()
        frame["symbol"] = frame["symbol"].astype(str)
        for date_col in ("list_date", "delist_date"):
            if date_col in frame.columns:
                frame[date_col] = frame[date_col].map(self._ymd8)
        frame = frame.drop_duplicates(["symbol"], keep="last")
        rows = [
            tuple(self._database_value(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        return columns, rows
