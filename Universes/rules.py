#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股票池底座规则：PIT ST、次新、退市、当日开盘、指数 asof、板块前缀。"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd

logger = logging.getLogger("Universes")

_EMPTY_NAMECHANGE_WARNED = False

BOARD_MAIN = "main"
BOARD_GEM = "gem"
BOARD_STAR = "star"
BOARD_BSE = "bse"


def iso_day(value) -> str:
    """Tushare YYYYMMDD / ISO → YYYY-MM-DD；空则 ''。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 8:
        return ""
    digits = digits[:8]
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def is_restricted_name(name: str) -> bool:
    """asof 简称是否 ST / *ST / 退市整理。"""
    text = str(name or "")
    if not text:
        return False
    upper = text.upper()
    return "ST" in upper or "*" in text or "退" in text


def board_of_symbol(symbol: str) -> str:
    """仓库代码（SH600000）→ 板块。代码本身不随时间变，天然 PIT。"""
    text = str(symbol).strip().upper()
    if text.startswith("BJ"):
        return BOARD_BSE
    digits = "".join(ch for ch in text if ch.isdigit())[:6]
    if digits.startswith("688"):
        return BOARD_STAR
    if digits.startswith("300") or digits.startswith("301"):
        return BOARD_GEM
    if digits.startswith("8") or digits.startswith("4"):
        return BOARD_BSE
    if digits.startswith("60") or digits.startswith("00"):
        return BOARD_MAIN
    return "other"


def apply_base_rules(
    mask: pd.DataFrame,
    open_panel: pd.DataFrame,
    storage,
    *,
    min_list_days: int = 60,
    require_next_open: bool = False,
) -> pd.DataFrame:
    """ST / 次新 / 退市 / 开盘。在调用方 AND 指数或板块之前执行。"""
    dates = pd.Index(mask.index.astype(str))
    symbols = pd.Index([str(s) for s in mask.columns])
    apply_pit_st(mask, storage)
    basic = _load_basic(storage)
    if not basic.empty:
        list_dates = pd.to_datetime(
            basic.drop_duplicates("symbol").set_index("symbol")["list_date"],
            errors="coerce",
        )
        mask_new_listings(mask, symbols, list_dates, storage, min_list_days)
        mask_delisted(mask, symbols, storage)
    aligned_open = open_panel.reindex(index=dates, columns=symbols)
    entry = aligned_open.shift(-1) if require_next_open else aligned_open
    return mask & entry.notna() & (entry != 0)


def apply_pit_st(mask: pd.DataFrame, storage) -> None:
    from .names import NameHistory

    history = NameHistory.from_storage(storage)
    if history.empty:
        global _EMPTY_NAMECHANGE_WARNED
        if not _EMPTY_NAMECHANGE_WARNED:
            logger.warning(
                "namechange 表为空，跳过 PIT ST 过滤（不会用当前简称回溯历史）"
            )
            _EMPTY_NAMECHANGE_WARNED = True
        return
    dates = np.asarray(mask.index.astype(str), dtype="U10")
    for symbol in mask.columns:
        spans = history.spans.get(str(symbol)) or ()
        if not spans:
            continue
        bad = np.zeros(len(dates), dtype=bool)
        for span in spans:
            if not is_restricted_name(span.value):
                continue
            start = span.start or "0000-01-01"
            end = span.end or "9999-12-31"
            bad |= (dates >= start) & (dates <= end)
        if bad.any():
            mask.loc[:, symbol] = np.asarray(mask[symbol], dtype=bool) & ~bad


def mask_new_listings(
    mask: pd.DataFrame,
    symbols: pd.Index,
    list_dates: pd.Series,
    storage,
    min_list_days: int,
) -> None:
    """按交易所日历计上市满 min_list_days 个交易日，不是回测窗口下标。"""
    listed = pd.to_datetime(list_dates.reindex(symbols), errors="coerce")
    valid = listed.notna().to_numpy()
    if not valid.any():
        return
    calendar = _trade_calendar(storage, mask.index)
    if not calendar:
        return
    cal = np.asarray(calendar, dtype="U10")
    listed_s = np.full(len(symbols), "9999-12-31", dtype="U10")
    listed_s[valid] = listed.dt.strftime("%Y-%m-%d").to_numpy(dtype="U10")[valid]
    ipo_i = np.searchsorted(cal, listed_s)
    never = valid & (ipo_i >= len(cal))
    cutoff_i = ipo_i + int(min_list_days)
    date_i = np.searchsorted(cal, np.asarray(mask.index.astype(str), dtype="U10"))
    too_new = date_i[:, None] < cutoff_i[None, :]
    too_new[:, ~valid] = False
    too_new[:, never] = True
    if too_new.any():
        mask.loc[:, :] = np.asarray(mask, dtype=bool) & ~too_new


def delist_on(symbols: Sequence[str], storage) -> np.ndarray:
    """与 symbols 对齐的退市日 YYYY-MM-DD；没有则为空串。"""
    n = len(symbols)
    out = np.full(n, "", dtype="U10")
    basic = _load_basic(storage)
    if basic.empty or "delist_date" not in basic.columns:
        return out
    series = (
        basic.drop_duplicates("symbol")
        .set_index("symbol")["delist_date"]
        .map(iso_day)
    )
    mapped = series.reindex([str(s) for s in symbols])
    text = mapped.fillna("").astype(str).to_numpy()
    out[:] = np.where(np.asarray([len(x) == 10 for x in text]), text, "")
    return out


def mask_delisted(mask: pd.DataFrame, symbols: pd.Index, storage) -> None:
    """决策日或次日开盘成交日已到退市日则不可买。退市日是已公告信息。"""
    delist = delist_on(symbols, storage)
    has = delist != ""
    if not has.any():
        return
    dates = np.asarray(mask.index.astype(str), dtype="U10")
    nxt = np.empty(len(dates), dtype="U10")
    if len(dates) > 1:
        nxt[:-1] = dates[1:]
    nxt[-1] = "9999-12-31"
    too_late = has[None, :] & (
        (dates[:, None] >= delist[None, :]) | (nxt[:, None] >= delist[None, :])
    )
    if too_late.any():
        mask.loc[:, :] = np.asarray(mask, dtype=bool) & ~too_late


def index_member_mask(
    dates: pd.Index,
    symbols: pd.Index,
    storage,
    index_codes: Sequence[str],
) -> pd.DataFrame:
    """最近一期完整成分快照 asof。缺任一指数表则报错，禁止空池装成功。"""
    codes = [str(c) for c in index_codes if str(c).strip()]
    if not codes:
        return pd.DataFrame(True, index=dates, columns=symbols)
    missing: List[str] = []
    snapshots: List[Dict[str, Set[str]]] = []
    for code in codes:
        frame = _load_constituents(storage, code)
        if frame is None or frame.empty:
            missing.append(code)
            continue
        by_date: Dict[str, Set[str]] = {}
        work = frame.copy()
        work["day"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        work = work.dropna(subset=["day"])
        work["symbol"] = work["symbol"].astype(str)
        for day, group in work.groupby("day"):
            by_date[str(day)] = set(group["symbol"])
        if not by_date:
            missing.append(code)
            continue
        snapshots.append(by_date)
    if missing:
        raise ValueError(
            "index_constituent 缺少 "
            + ", ".join(missing)
            + "。请在 DailyUpdates.data_fetcher.main_scheduler 的 "
            "universe_index_weight.index_list 中加入这些指数。"
        )
    member = pd.DataFrame(False, index=dates, columns=symbols)
    date_arr = np.asarray(dates.astype(str), dtype="U10")
    symbol_pos = {str(s): i for i, s in enumerate(symbols)}
    arr = np.zeros((len(dates), len(symbols)), dtype=bool)
    for by_date in snapshots:
        snap_days = np.asarray(sorted(by_date), dtype="U10")
        idx = np.searchsorted(snap_days, date_arr, side="right") - 1
        local = np.zeros((len(dates), len(symbols)), dtype=bool)
        for snap_i, day in enumerate(snap_days):
            rows = np.where(idx == snap_i)[0]
            if rows.size == 0:
                continue
            cols = [symbol_pos[s] for s in by_date[str(day)] if s in symbol_pos]
            if cols:
                local[np.ix_(rows, np.asarray(cols, dtype=int))] = True
        arr |= local
    member.loc[:, :] = arr
    return member


def board_member_mask(
    dates: pd.Index,
    symbols: pd.Index,
    boards: Sequence[str],
) -> pd.DataFrame:
    wanted = {str(b).strip().lower() for b in boards if str(b).strip()}
    if not wanted:
        return pd.DataFrame(True, index=dates, columns=symbols)
    flags = [board_of_symbol(symbol) in wanted for symbol in symbols]
    row = np.asarray(flags, dtype=bool)
    return pd.DataFrame(np.tile(row, (len(dates), 1)), index=dates, columns=symbols)


def _load_basic(storage) -> pd.DataFrame:
    try:
        frame = storage.read_stock_basic()
    except Exception:
        return pd.DataFrame()
    return frame if frame is not None else pd.DataFrame()


def _load_constituents(storage, index_code: str) -> pd.DataFrame:
    try:
        return storage.read_index_constituents(index_code=index_code)
    except Exception:
        return pd.DataFrame()


def _trade_calendar(storage, panel_dates: pd.Index) -> List[str]:
    try:
        calendar = [str(d) for d in storage.list_trade_dates()]
    except Exception:
        calendar = []
    if not calendar:
        calendar = [str(d) for d in panel_dates]
    return calendar
