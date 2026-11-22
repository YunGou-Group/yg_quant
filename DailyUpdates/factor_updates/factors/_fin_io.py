#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""财务表原始读取缓存。

FinEngine / StyleEngine 各自做口径规范化，这里只避免重复打 SQLite。
自定义因子不经过本模块。
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import pandas as pd

from yg_quant_repo import default_db_path

_LOCK = threading.RLock()
_DB_KEY: Optional[str] = None
_INDICATOR: Optional[pd.DataFrame] = None
_STATEMENTS: Dict[str, pd.DataFrame] = {}


def _db_key() -> str:
    return str(default_db_path())


def read_financial_indicator_raw() -> pd.DataFrame:
    path = _db_key()
    with _LOCK:
        global _INDICATOR, _DB_KEY
        if _INDICATOR is not None and _DB_KEY == path:
            return _INDICATOR
    from DailyUpdates.storage.sqlite_storage import SQLiteStorage

    frame = SQLiteStorage(path).read_financial_indicator()
    with _LOCK:
        _INDICATOR = frame
        _DB_KEY = path
        return _INDICATOR


def read_financial_statement_raw(table: str) -> pd.DataFrame:
    path = _db_key()
    with _LOCK:
        global _DB_KEY
        if _DB_KEY == path and table in _STATEMENTS:
            return _STATEMENTS[table]
    from DailyUpdates.storage.sqlite_storage import SQLiteStorage

    frame = SQLiteStorage(path).read_financial_statement(table)
    with _LOCK:
        if _DB_KEY != path:
            _STATEMENTS.clear()
            _DB_KEY = path
        _STATEMENTS[table] = frame
        return frame


def clear_financial_io_cache() -> None:
    global _DB_KEY, _INDICATOR
    with _LOCK:
        _DB_KEY = None
        _INDICATOR = None
        _STATEMENTS.clear()
