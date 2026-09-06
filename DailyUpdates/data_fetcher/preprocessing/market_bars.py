#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""个股行情入库清洗：占位价、OHLC、成交量、复权因子跳变。指数行原样通过。"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MarketPreprocess")

MIN_RAW_PRICE = 0.05
MAX_ADJ_STEP = 8.0
# 创业板注册制：2020-08-24 起 20%，新股前 5 个交易日不设限；ST 也是 20%。
GEM_LIMIT_20_START = pd.Timestamp("2020-08-24")
# 主板注册制：2023-04-10 起新股前 5 个交易日不设限。
MAIN_REG_START = pd.Timestamp("2023-04-10")
# 2014-01-01 起核准制新股首日申报价不得高于发行价 144%、不低于 64%。
IPO_FIRST_DAY_44_START = pd.Timestamp("2014-01-01")
IPO_FIRST_DAY_UP = 0.44
IPO_FIRST_DAY_DOWN = 0.36
IPO_NOLIMIT_SESSIONS = 5
# 无交易日历时，用上市后 7 个自然日近似前 5 个交易日。
IPO_NOLIMIT_CAL_DAYS = 7
_PRICE_COLS = ("open", "high", "low", "close", "pre_close")
_OHLC_COLS = ("open", "high", "low", "close")
_CODE_COLS = ("ts_code", "symbol")
_DATE_COLS = ("trade_date", "date")


def _first_col(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _stock_mask(codes: pd.Series) -> pd.Series:
    return ~codes.astype(str).str.startswith("index_")


def sanitize_market_bars(
    frame: pd.DataFrame,
    *,
    prev_adj: Optional[pd.Series] = None,
    namechange: Optional[pd.DataFrame] = None,
    list_dates: Optional[pd.Series] = None,
    calendar: Optional[Iterable] = None,
) -> pd.DataFrame:
    """把脏点置空后再写入。指数行不改价格/因子。

    新交易日 INSERT 会落下 NaN。同一天重拉时 storage upsert 用 COALESCE，
    NaN 清不掉库里旧的 0.01，旧脏点需要全量重建。

    官方 up_limit / down_limit 为空时，按当时规则 × pre_close 补齐，已有官方
    限价不覆盖。细则见 ``_limit_ratio_pair``。
    """
    if frame is None or frame.empty:
        return frame
    code_col = _first_col(frame, _CODE_COLS)
    date_col = _first_col(frame, _DATE_COLS)
    if code_col is None or date_col is None:
        return frame

    out = frame.copy()
    stock = _stock_mask(out[code_col])
    n_tiny = n_ohlc = n_vol = n_adj_sign = n_jump = 0

    numeric_cols = [
        col
        for col in out.columns
        if col not in {code_col, date_col} and pd.api.types.is_numeric_dtype(out[col])
    ]
    if numeric_cols and stock.any():
        work = out.loc[stock, numeric_cols]
        coerced = work.apply(lambda col: pd.to_numeric(col, errors="coerce"))
        bad = ~np.isfinite(coerced.to_numpy(dtype="float64")) & work.notna().to_numpy()
        if bad.any():
            cleaned = coerced.mask(~np.isfinite(coerced))
            out.loc[stock, numeric_cols] = cleaned

    out = out.drop_duplicates([code_col, date_col], keep="last")
    stock = _stock_mask(out[code_col])

    price_cols = [c for c in _PRICE_COLS if c in out.columns]
    if price_cols and stock.any():
        tiny = pd.Series(False, index=out.index)
        for col in price_cols:
            px = pd.to_numeric(out[col], errors="coerce")
            tiny = tiny | (px.notna() & (px <= MIN_RAW_PRICE))
        tiny &= stock
        n_tiny = int(tiny.sum())
        if n_tiny:
            for col in price_cols:
                out.loc[tiny, col] = pd.NA

    ohlc = [c for c in _OHLC_COLS if c in out.columns]
    if {"high", "low"}.issubset(ohlc) and ("open" in ohlc or "close" in ohlc) and stock.any():
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        open_ = pd.to_numeric(out["open"], errors="coerce") if "open" in ohlc else None
        close = pd.to_numeric(out["close"], errors="coerce") if "close" in ohlc else None
        if open_ is not None and close is not None:
            body_hi = np.maximum(open_, close)
            body_lo = np.minimum(open_, close)
        elif open_ is not None:
            body_hi = body_lo = open_
        else:
            body_hi = body_lo = close
        broken = stock & (
            (high.notna() & body_hi.notna() & (high < body_hi))
            | (low.notna() & body_lo.notna() & (low > body_lo))
        )
        n_ohlc = int(broken.sum())
        if n_ohlc:
            for col in ohlc:
                out.loc[broken, col] = pd.NA

    for col in ("vol", "amount"):
        if col not in out.columns:
            continue
        val = pd.to_numeric(out[col], errors="coerce")
        bad = stock & val.notna() & (val <= 0)
        n_vol += int(bad.sum())
        if bad.any():
            out.loc[bad, col] = pd.NA

    if "adj_factor" in out.columns:
        adj = pd.to_numeric(out["adj_factor"], errors="coerce")
        nonpos = stock & adj.notna() & (adj <= 0)
        n_adj_sign = int(nonpos.sum())
        if n_adj_sign:
            out.loc[nonpos, "adj_factor"] = pd.NA
        ranked = out.sort_values([code_col, date_col], kind="mergesort")
        adj = pd.to_numeric(ranked["adj_factor"], errors="coerce")
        codes = ranked[code_col].astype(str)
        prev = adj.groupby(codes, sort=False).shift(1)
        if prev_adj is not None and len(prev_adj):
            mapped = codes.map(pd.to_numeric(prev_adj, errors="coerce"))
            prev = prev.where(prev.notna(), mapped)
        ratio = adj / prev
        jump = (
            ~codes.str.startswith("index_")
            & prev.notna()
            & (prev > 0)
            & adj.notna()
            & (adj > 0)
            & ((ratio >= MAX_ADJ_STEP) | (ratio <= 1.0 / MAX_ADJ_STEP))
        )
        n_jump = int(jump.sum())
        if n_jump:
            ranked.loc[jump, "adj_factor"] = pd.NA
        out["adj_factor"] = ranked["adj_factor"].reindex(out.index)

    n_limit = _fill_missing_limits(
        out,
        code_col,
        date_col,
        stock,
        namechange=namechange,
        list_dates=list_dates,
        calendar=calendar,
    )

    if n_tiny or n_ohlc or n_vol or n_adj_sign or n_jump:
        logger.warning(
            "行情预处理：占位价 %s 行，OHLC %s 行，成交 %s 行，"
            "adj≤0 %s 行，adj 跳变 %s 行（价≤%s 或因子变幅≥×%s）",
            n_tiny,
            n_ohlc,
            n_vol,
            n_adj_sign,
            n_jump,
            MIN_RAW_PRICE,
            int(MAX_ADJ_STEP),
        )
    if n_limit:
        logger.info("行情预处理：按板块规则补齐 %s 行缺失涨跌停", n_limit)
    return out


def _digits6(symbol: str) -> str:
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def _is_restricted_name(name: str) -> bool:
    """风险警示或未股改。退市整理用板块幅度，不算 5%。"""
    text = str(name or "").strip()
    if not text:
        return False
    upper = text.upper()
    if "ST" in upper:
        return True
    # 未股改：S佳通。SST / S*ST 已由 ST 命中。
    if upper.startswith("S") and len(text) > 1:
        return (not text[1].isascii()) or text[1] in "*＊"
    return False


def _round_limit(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").round(2)


def _asof_st_mask(
    codes: pd.Series, dates: pd.Series, namechange: Optional[pd.DataFrame]
) -> pd.Series:
    """namechange 区间覆盖当日且简称为 ST / *ST / 未股改 S 股。"""
    empty = pd.Series(False, index=codes.index)
    if namechange is None or namechange.empty or "name" not in namechange.columns:
        return empty
    symbol_col = "symbol" if "symbol" in namechange.columns else None
    if symbol_col is None:
        return empty
    work = namechange.loc[:, [symbol_col, "name"]].copy()
    work["start"] = (
        pd.to_datetime(namechange["start_date"], errors="coerce")
        if "start_date" in namechange.columns
        else pd.NaT
    )
    work["end"] = (
        pd.to_datetime(namechange["end_date"], errors="coerce")
        if "end_date" in namechange.columns
        else pd.NaT
    )
    work = work.rename(columns={symbol_col: "symbol"})
    work["symbol"] = work["symbol"].astype(str)
    work = work.loc[work["name"].map(_is_restricted_name)]
    if work.empty:
        return empty
    day = pd.to_datetime(dates, errors="coerce")
    uniq = pd.Index(day.dropna().unique())
    if len(uniq) <= 16:
        out = empty.copy()
        code_s = codes.astype(str)
        for one in uniq:
            cover = (work["start"].isna() | (work["start"] <= one)) & (
                work["end"].isna() | (work["end"] >= one)
            )
            st_set = set(work.loc[cover, "symbol"])
            if st_set:
                out |= (day == one) & code_s.isin(st_set)
        return out
    bars = pd.DataFrame(
        {"symbol": codes.astype(str), "date": day, "_ix": np.asarray(codes.index)}
    )
    joined = bars.merge(work[["symbol", "start", "end"]], on="symbol", how="inner")
    cover = (joined["start"].isna() | (joined["start"] <= joined["date"])) & (
        joined["end"].isna() | (joined["end"] >= joined["date"])
    )
    hit = joined.loc[cover, "_ix"]
    out = empty.copy()
    if len(hit):
        out.loc[out.index.isin(hit)] = True
    return out


def _as_calendar(calendar: Optional[Iterable]) -> Optional[pd.DatetimeIndex]:
    if calendar is None:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(calendar), errors="coerce")).normalize()
    idx = idx.dropna().unique().sort_values()
    return idx if len(idx) else None


def _ipo_session_number(
    listed: pd.Series, day: pd.Series, calendar: Optional[Iterable] = None
) -> pd.Series:
    """上市后第几个交易日（含上市日=1）。无法判断时为 NaN。"""
    listed_ts = pd.to_datetime(listed, errors="coerce").dt.normalize()
    day_ts = pd.to_datetime(day, errors="coerce").dt.normalize()
    out = pd.Series(np.nan, index=listed.index, dtype="float64")
    valid = listed_ts.notna() & day_ts.notna() & (day_ts >= listed_ts)
    if not valid.any():
        return out
    cal = _as_calendar(calendar)
    if cal is not None:
        start = np.searchsorted(cal.values, listed_ts[valid].values, side="left")
        pos = np.searchsorted(cal.values, day_ts[valid].values, side="left")
        n_cal = len(cal)
        landed = (pos < n_cal) & (cal.values[np.minimum(pos, n_cal - 1)] == day_ts[valid].values)
        sess = np.where(landed & (start < n_cal), pos - start + 1, np.nan)
        out.loc[valid] = sess
        return out
    delta = (day_ts - listed_ts).dt.days
    same = valid & (delta == 0)
    early = valid & (delta > 0) & (delta < IPO_NOLIMIT_CAL_DAYS)
    out.loc[same] = 1
    out.loc[early] = 2
    return out


def _limit_ratio_pair(
    codes: pd.Series,
    dates: pd.Series,
    *,
    namechange: Optional[pd.DataFrame] = None,
    list_dates: Optional[pd.Series] = None,
    calendar: Optional[Iterable] = None,
) -> tuple[pd.Series, pd.Series]:
    """当时适用的涨/跌幅。不设限的行是 NaN。首日核准制为 +44% / -36%。

    日常幅度：主板 10%，创业板 2020-08-24 起 20%（含 300/301/302），
    科创板 20%，北交所 30%。主板与注册制前创业板的 ST / 未股改 S 股 5%；
    注册制后创业板 / 科创板 / 北交所 ST 跟板块幅度。退市整理用板块幅度。

    上市初期：科创板、北交所、注册制创业板、2023-04-10 后主板新股前 5 个
    交易日不设限。2014-01-01 起核准制主板/创业板上市首日 +44% / -36%。
    """
    text = codes.astype(str)
    code6 = text.map(_digits6)
    day = pd.to_datetime(dates, errors="coerce")
    star = code6.str.startswith(("688", "689"))
    gem = code6.str.startswith(("300", "301", "302"))
    bse = text.str.upper().str.startswith("BJ") | code6.str.startswith(("8", "4"))
    bse = bse & ~star
    main = ~(star | gem | bse)

    up = pd.Series(0.10, index=codes.index, dtype="float64")
    up.loc[star | (gem & (day >= GEM_LIMIT_20_START))] = 0.20
    up.loc[bse] = 0.30
    down = up.copy()

    st = _asof_st_mask(codes, dates, namechange)
    st_five = st & (main | (gem & (day < GEM_LIMIT_20_START)))
    up.loc[st_five] = 0.05
    down.loc[st_five] = 0.05

    if list_dates is None or not len(list_dates):
        return up, down

    listed = pd.to_datetime(text.map(list_dates), errors="coerce")
    session = _ipo_session_number(listed, day, calendar)
    early = session.notna() & (session >= 1) & (session <= IPO_NOLIMIT_SESSIONS)
    first = session == 1
    gem_reg_ipo = gem & (listed >= GEM_LIMIT_20_START)
    main_reg_ipo = main & (listed >= MAIN_REG_START)
    no_limit = early & (star | bse | gem_reg_ipo | main_reg_ipo)
    first_44 = (
        first
        & ~no_limit
        & listed.notna()
        & (listed >= IPO_FIRST_DAY_44_START)
        & (main | gem)
    )
    up.loc[first_44] = IPO_FIRST_DAY_UP
    down.loc[first_44] = IPO_FIRST_DAY_DOWN
    up.loc[no_limit] = np.nan
    down.loc[no_limit] = np.nan
    return up, down


def _fill_missing_limits(
    out: pd.DataFrame,
    code_col: str,
    date_col: str,
    stock: pd.Series,
    *,
    namechange: Optional[pd.DataFrame] = None,
    list_dates: Optional[pd.Series] = None,
    calendar: Optional[Iterable] = None,
) -> int:
    """只补官方限价为空且 pre_close 可用的个股行。指数不补。"""
    if "pre_close" not in out.columns or not stock.any():
        return 0
    pre = pd.to_numeric(out["pre_close"], errors="coerce")
    usable = stock & pre.notna() & (pre > 0)
    if not usable.any():
        return 0
    up = (
        pd.to_numeric(out["up_limit"], errors="coerce")
        if "up_limit" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    down = (
        pd.to_numeric(out["down_limit"], errors="coerce")
        if "down_limit" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    need_up = usable & ~np.isfinite(up)
    need_dn = usable & ~np.isfinite(down)
    if not (need_up.any() or need_dn.any()):
        return 0
    up_ratio, down_ratio = _limit_ratio_pair(
        out[code_col],
        out[date_col],
        namechange=namechange,
        list_dates=list_dates,
        calendar=calendar,
    )
    if "up_limit" not in out.columns:
        out["up_limit"] = np.nan
    if "down_limit" not in out.columns:
        out["down_limit"] = np.nan
    fill_up = need_up & up_ratio.notna()
    fill_dn = need_dn & down_ratio.notna()
    if fill_up.any():
        out.loc[fill_up, "up_limit"] = _round_limit(pre[fill_up] * (1.0 + up_ratio[fill_up]))
    if fill_dn.any():
        out.loc[fill_dn, "down_limit"] = _round_limit(
            pre[fill_dn] * (1.0 - down_ratio[fill_dn])
        )
    return int(fill_up.sum() + fill_dn.sum())


def load_prev_adj(storage: Any, before_date: str) -> Optional[pd.Series]:
    """库内 ``before_date`` 之前最近一个交易日的 adj_factor，index 为代码。"""
    if storage is None or not before_date:
        return None
    try:
        available = set(storage.get_market_fields())
        if "adj_factor" not in available:
            return None
        day = str(before_date).replace("-", "")
        if len(day) >= 8:
            iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        else:
            iso = str(before_date)
        calendar = storage.list_trade_dates(end_date=iso)
        prior = [d for d in calendar if str(d) < iso]
        if not prior:
            return None
        prev_day = max(prior)
        frame = storage.read_market_data(
            fields=["adj_factor"],
            start_date=prev_day,
            end_date=prev_day,
            adjust="none",
        )
    except Exception:
        logger.debug("无法加载昨日 adj_factor", exc_info=True)
        return None
    if frame is None or frame.empty or "adj_factor" not in frame.columns:
        return None
    code_col = _first_col(frame, _CODE_COLS)
    if code_col is None:
        return None
    values = pd.to_numeric(frame["adj_factor"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame[code_col].astype(str), name="adj_factor")
    series = series[~series.index.str.startswith("index_")]
    series = series[series.notna() & (series > 0)]
    return series if not series.empty else None
