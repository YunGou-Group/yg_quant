#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Akshare 数据源：与 Tushare 使用同一套 data_type / api_name，输出相同列名。

不需要 token。个股日线 / 复权直接走新浪（东财易断连）。多数接口按股票循环，
全市场历史比 Tushare 慢一个数量级。涨跌停价由昨收和板块涨跌幅推算。
指数成分权重是中证官网最新快照，不是逐日调样。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd
from tqdm import tqdm

from DailyUpdates.data_fetcher.data_source_base import DataSourceBase, FetchSlice
from DailyUpdates.storage.financial_schema import FINANCIAL_VALUE_FIELDS

_HIST_MAP = {
    "日期": "trade_date",
    "股票代码": "symbol_raw",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "vol",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}

_EM_FIN_MAP = {
    "roe": ("ROEJQ", "ROE_YEARLY", "ROEKCJQ"),
    "roa": ("ROA", "ZZCJLL", "TOTAL_ASSETS_YIELD"),
    "eps": ("EPSJB", "BASIC_EPS", "EPSXS"),
    "ocfps": ("OCFPS", "MGJYXJJE"),
    "bps": ("BPS", "MGZB"),
    "netprofit_yoy": ("PARENTNETPROFITTZ", "PARENT_NETPROFIT_YOY"),
    "dt_netprofit_yoy": ("KCFJCXSYJLRTZ", "DEDUCTEDPROFIT_YOY"),
    "or_yoy": ("TOTALOPERATEREVETZ", "OPERATE_INCOME_YOY"),
    "equity_yoy": ("EQUITY_YOY", "NETASSET_YOY"),
    "debt_to_assets": ("ZCFZL", "DEBTASSET_RATIO", "ASSET_LIAB_RATIO"),
    "debt_to_eqt": ("CQBL", "DEBT_TO_EQT"),
    "assets_to_eqt": ("QYCS", "EQUITY_MULTIPLIER"),
    "longdeb_to_debt": ("LONGDEB_TO_DEBT", "CQFZBL"),
}


def _as_field_list(fields) -> Optional[List[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [item.strip() for item in fields.split(",") if item.strip()]
    return list(fields)


def _yyyymmdd(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        text = str(value).replace("-", "")[:8]
        return text if text.isdigit() else ""
    return ts.strftime("%Y%m%d")


def _em_symbol(code: str) -> str:
    text = str(code).strip().upper()
    if "." in text:
        text = text.split(".", 1)[0]
    if text.startswith(("SH", "SZ", "BJ")):
        text = text[2:]
    return text.zfill(6)


def _to_ts_code(code: str) -> str:
    raw = _em_symbol(code)
    if raw.startswith("6") or raw.startswith("9"):
        return f"{raw}.SH"
    if raw.startswith(("4", "8")):
        return f"{raw}.BJ"
    return f"{raw}.SZ"


def _si_code(code: str) -> str:
    text = str(code).strip().upper().replace(".SI", "")
    return f"{text}.SI" if text else ""


def _pause(config: Dict, default: float = 0.2) -> None:
    wait = float(config.get("pause_seconds", default) or 0)
    if wait > 0:
        time.sleep(wait)


def _select(frame: pd.DataFrame, fields) -> pd.DataFrame:
    wanted = _as_field_list(fields)
    if not wanted or frame.empty:
        return frame
    keep = [c for c in wanted if c in frame.columns]
    extra = [c for c in frame.columns if c not in keep]
    if not keep:
        return frame
    return frame[keep + [c for c in extra if c in {"ts_code", "trade_date"}]].copy()


def _limit_ratio(ts_code: str, name: str = "") -> float:
    text = str(name or "").upper()
    if "ST" in text or "退" in str(name or ""):
        return 0.05
    code = _em_symbol(ts_code)
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def _round_price(value: float) -> float:
    return float(f"{value:.2f}")


def _sina_symbol(code: str) -> str:
    text = str(code).strip().upper()
    if "." in text:
        number, market = text.split(".", 1)
        number = number.replace("SH", "").replace("SZ", "").replace("BJ", "")
        return f"{market.lower()}{number.zfill(6) if number.isdigit() else number}"
    ts_code = _to_ts_code(code)
    number, market = ts_code.split(".", 1)
    return f"{market.lower()}{number}"


def _retry_call(
    fn,
    attempts: int = 3,
    pause: float = 0.8,
    timeout: Optional[float] = 45.0,
):
    """限流/断连重试；timeout 秒内无返回则放弃本轮（避免无限卡住）。"""
    last = None
    for index in range(attempts):
        try:
            if timeout and timeout > 0:
                # wait=False：超时后不堵在僵死的 HTTP 线程上
                pool = ThreadPoolExecutor(max_workers=1)
                fut = pool.submit(fn)
                try:
                    return fut.result(timeout=timeout)
                except FuturesTimeoutError as exc:
                    raise TimeoutError(f"请求超时 ({timeout:.0f}s)") from exc
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
            return fn()
        except Exception as exc:
            last = exc
            if index + 1 < attempts:
                time.sleep(pause * (index + 1))
    raise last


def _map_value_em_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in frame.columns:
        text = str(col)
        if "PE(TTM)" in text:
            rename[col] = "pe_ttm"
        elif "市净" in text:
            rename[col] = "pb"
        elif text.startswith("PE") and "TTM" not in text:
            rename[col] = "pe"
        elif "流通市值" in text:
            rename[col] = "circ_mv"
        elif "总市值" in text:
            rename[col] = "total_mv"
        elif "市销" in text:
            rename[col] = "ps_ttm"
        elif "换手" in text:
            rename[col] = "turnover_rate"
    out = frame.rename(columns=rename).copy()
    date_col = next(
        (
            col
            for col in out.columns
            if col
            not in {
                "pe_ttm",
                "pb",
                "pe",
                "circ_mv",
                "total_mv",
                "ps_ttm",
                "turnover_rate",
            }
            and ("日期" in str(col) or "date" in str(col).lower())
        ),
        out.columns[0],
    )
    out["trade_date"] = out[date_col].map(_yyyymmdd)
    return out


class AkshareDataSource(DataSourceBase):
    """把东财 / 新浪 / 中证 / 申万公开接口映射成调度器认的表结构。"""

    fetch_slice = FetchSlice.BY_SYMBOL
    requires_token = False

    def fetch_data(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        data_type = str(config.get("data_type") or "daily")
        api_name = str(config.get("api_name") or data_type)
        fields = config.get("fields")
        if data_type == "daily":
            if api_name == "daily":
                return self._fetch_daily(config, fields, start_date, end_date)
            if api_name == "daily_basic":
                return self._fetch_daily_basic(config, fields, start_date, end_date)
            if api_name == "adj_factor":
                return self._fetch_adj_factor(config, fields, start_date, end_date)
            if api_name == "stk_limit":
                return self._fetch_stk_limit(config, fields, start_date, end_date)
            raise ValueError(f"不支持的日频 API: {api_name}")
        if data_type == "index":
            if api_name == "index_daily":
                return self._fetch_index_daily(config, fields, start_date, end_date)
            raise ValueError(f"不支持的指数 API: {api_name}")
        if data_type == "industry":
            if api_name == "index_classify":
                return self._fetch_index_classify(config, fields)
            if api_name == "index_member_all":
                return self._fetch_index_member_all(config, fields)
            raise ValueError(f"不支持的行业 API: {api_name}")
        if data_type == "stock_info":
            if api_name == "stock_basic":
                return self._fetch_stock_basic(config, fields)
            if api_name == "namechange":
                return self._fetch_namechange(config, fields)
            raise ValueError(f"不支持的股票信息 API: {api_name}")
        if data_type == "financial":
            if api_name in ("fina_indicator", "fina_indicator_vip"):
                return self._fetch_fina_indicator(config, fields, start_date, end_date)
            raise ValueError(f"不支持的财务 API: {api_name}")
        if data_type == "index_constituent":
            if api_name == "index_weight":
                return self._fetch_index_weight(config, fields, start_date, end_date)
            raise ValueError(f"不支持的指数成分 API: {api_name}")
        raise ValueError(f"不支持的数据类型: {data_type}")

    def _codes(self, config: Dict) -> List[str]:
        extra = config.get("symbols") or config.get("stock_list") or []
        if extra:
            return [_em_symbol(item) for item in extra]
        listed = ak.stock_info_a_code_name()
        if listed is None or listed.empty:
            raise RuntimeError("Akshare 未能获取 A 股列表")
        col = "code" if "code" in listed.columns else listed.columns[0]
        return listed[col].astype(str).map(_em_symbol).dropna().unique().tolist()

    def _hist_one(self, symbol: str, start: str, end: str, adjust: str = "") -> pd.DataFrame:
        return self._hist_from_sina(symbol, start, end, adjust)

    def _normalize_em_hist(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        frame = raw.rename(columns=_HIST_MAP).copy()
        if "trade_date" in frame.columns:
            frame["trade_date"] = frame["trade_date"].map(_yyyymmdd)
        frame["ts_code"] = _to_ts_code(symbol)
        if "amount" in frame.columns:
            # 东财成交额为元，Tushare 为千元。
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") / 1000.0
        if "close" in frame.columns and "change" in frame.columns:
            close = pd.to_numeric(frame["close"], errors="coerce")
            change = pd.to_numeric(frame["change"], errors="coerce")
            frame["pre_close"] = close - change
        return frame

    def _hist_from_sina(
        self, symbol: str, start: str, end: str, adjust: str = ""
    ) -> pd.DataFrame:
        raw = _retry_call(
            lambda: ak.stock_zh_a_daily(
                symbol=_sina_symbol(symbol),
                start_date=start,
                end_date=end,
                adjust=adjust or "",
            ),
            attempts=3,
            pause=1.0,
            timeout=45.0,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        frame = raw.rename(
            columns={
                "date": "trade_date",
                "volume": "vol",
                "turnover": "turnover_rate",
            }
        ).copy()
        frame["trade_date"] = frame["trade_date"].map(_yyyymmdd)
        frame["ts_code"] = _to_ts_code(symbol)
        if "vol" in frame.columns:
            # 新浪成交量为股，Tushare / 东财为手。
            frame["vol"] = pd.to_numeric(frame["vol"], errors="coerce") / 100.0
        if "amount" in frame.columns:
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") / 1000.0
        if "turnover_rate" in frame.columns:
            frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce") * 100.0
        close = pd.to_numeric(frame["close"], errors="coerce") if "close" in frame.columns else None
        if close is not None:
            frame["pre_close"] = close.shift(1)
            frame["change"] = close - frame["pre_close"]
            frame["pct_chg"] = frame["change"] / frame["pre_close"] * 100.0
        return frame

    def _fetch_daily(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        today = datetime.now().strftime("%Y%m%d")
        if start == end and start >= today:
            return self._daily_from_spot(fields, start)
        codes = self._codes(config)
        frames: List[pd.DataFrame] = []
        progress = tqdm(codes, desc="Akshare daily", unit="股")
        for symbol in progress:
            progress.set_postfix_str(symbol)
            try:
                part = self._hist_one(symbol, start, end, adjust="")
            except Exception as exc:
                tqdm.write(f"[akshare] daily {symbol} 失败: {exc}")
                part = pd.DataFrame()
            if part is not None and not part.empty:
                frames.append(part)
            _pause(config)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return _select(result, fields)

    def _daily_from_spot(self, fields, trade_date: str) -> pd.DataFrame:
        raw = ak.stock_zh_a_spot_em()
        if raw is None or raw.empty:
            return pd.DataFrame()
        frame = raw.rename(
            columns={
                "代码": "ts_code",
                "名称": "name",
                "开盘": "open",
                "最新价": "close",
                "最高": "high",
                "最低": "low",
                "昨收": "pre_close",
                "涨跌额": "change",
                "涨跌幅": "pct_chg",
                "成交量": "vol",
                "成交额": "amount",
                "换手率": "turnover_rate",
                "市盈率-动态": "pe",
                "市净率": "pb",
                "总市值": "total_mv",
                "流通市值": "circ_mv",
            }
        ).copy()
        frame["ts_code"] = frame["ts_code"].astype(str).map(_to_ts_code)
        frame["trade_date"] = trade_date
        if "amount" in frame.columns:
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") / 1000.0
        if "total_mv" in frame.columns:
            frame["total_mv"] = pd.to_numeric(frame["total_mv"], errors="coerce") / 10000.0
        if "circ_mv" in frame.columns:
            frame["circ_mv"] = pd.to_numeric(frame["circ_mv"], errors="coerce") / 10000.0
        return _select(frame, fields)

    def _fetch_daily_basic(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        today = datetime.now().strftime("%Y%m%d")
        if start == end and start >= today:
            frame = self._daily_from_spot(fields, start)
            return _select(frame, fields)
        codes = self._codes(config)
        frames: List[pd.DataFrame] = []
        progress = tqdm(codes, desc="Akshare daily_basic", unit="股")
        for symbol in progress:
            progress.set_postfix_str(symbol)
            try:
                raw = _retry_call(lambda: ak.stock_a_indicator_lg(symbol=symbol))
            except Exception as exc:
                tqdm.write(f"[akshare] daily_basic 乐咕乐股失败 {symbol}: {exc}，改用东财估值")
                try:
                    raw = ak.stock_value_em(symbol=symbol)
                    if raw is not None and not raw.empty:
                        raw = _map_value_em_columns(raw)
                except Exception as fallback_exc:
                    tqdm.write(f"[akshare] daily_basic {symbol} 失败: {fallback_exc}")
                    raw = pd.DataFrame()
            if raw is None or raw.empty:
                _pause(config)
                continue
            part = raw.copy()
            part.columns = [str(c).strip().lower() for c in part.columns]
            if "trade_date" in part.columns:
                part["trade_date"] = part["trade_date"].map(_yyyymmdd)
            else:
                date_col = next((c for c in part.columns if "date" in c), None)
                if date_col is None:
                    _pause(config)
                    continue
                part["trade_date"] = part[date_col].map(_yyyymmdd)
            part["ts_code"] = _to_ts_code(symbol)
            rename = {
                "pe_ttm": "pe_ttm",
                "pe": "pe",
                "pb": "pb",
                "ps_ttm": "ps_ttm",
                "ps": "ps",
                "dv_ttm": "dv_ttm",
                "dividend_yield": "dv_ttm",
                "total_mv": "total_mv",
                "circ_mv": "circ_mv",
                "turnover": "turnover_rate",
                "turnover_rate": "turnover_rate",
            }
            part = part.rename(columns={k: v for k, v in rename.items() if k in part.columns})
            for mv_col in ("total_mv", "circ_mv"):
                if mv_col in part.columns:
                    values = pd.to_numeric(part[mv_col], errors="coerce")
                    # 东财估值为元，Tushare 为万元。
                    if values.dropna().median() > 1e8:
                        part[mv_col] = values / 10000.0
            mask = (part["trade_date"] >= start) & (part["trade_date"] <= end)
            part = part.loc[mask]
            if not part.empty:
                frames.append(part)
            _pause(config)
        if not frames:
            return pd.DataFrame()
        return _select(pd.concat(frames, ignore_index=True), fields)

    def _fetch_adj_factor(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        codes = self._codes(config)
        frames: List[pd.DataFrame] = []
        progress = tqdm(codes, desc="Akshare adj_factor", unit="股")
        for symbol in progress:
            progress.set_postfix_str(symbol)
            try:
                raw = self._hist_one(symbol, start, end, adjust="")
                hfq = self._hist_one(symbol, start, end, adjust="hfq")
            except Exception as exc:
                tqdm.write(f"[akshare] adj_factor {symbol} 失败: {exc}")
                _pause(config)
                continue
            if raw.empty or hfq.empty:
                _pause(config)
                continue
            left = raw[["ts_code", "trade_date", "close"]].rename(columns={"close": "raw_close"})
            right = hfq[["trade_date", "close"]].rename(columns={"close": "hfq_close"})
            merged = left.merge(right, on="trade_date", how="inner")
            raw_close = pd.to_numeric(merged["raw_close"], errors="coerce")
            hfq_close = pd.to_numeric(merged["hfq_close"], errors="coerce")
            factor = hfq_close / raw_close
            merged["adj_factor"] = factor.replace([float("inf"), float("-inf")], pd.NA)
            merged.loc[raw_close.eq(0) | raw_close.isna(), "adj_factor"] = pd.NA
            frames.append(merged[["ts_code", "trade_date", "adj_factor"]])
            _pause(config)
        if not frames:
            return pd.DataFrame()
        return _select(pd.concat(frames, ignore_index=True), fields)

    def _fetch_stk_limit(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        daily = self._fetch_daily(config, ["ts_code", "trade_date", "pre_close"], start, end)
        if daily.empty or "pre_close" not in daily.columns:
            return pd.DataFrame()
        names = {}
        if "name" in daily.columns:
            names = dict(zip(daily["ts_code"].astype(str), daily["name"].astype(str)))
        rows = []
        for row in daily.itertuples(index=False):
            pre = pd.to_numeric(getattr(row, "pre_close", None), errors="coerce")
            if not pd.notna(pre) or float(pre) <= 0:
                continue
            ts_code = str(row.ts_code)
            ratio = _limit_ratio(ts_code, names.get(ts_code, ""))
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": row.trade_date,
                    "up_limit": _round_price(float(pre) * (1.0 + ratio)),
                    "down_limit": _round_price(float(pre) * (1.0 - ratio)),
                }
            )
        if not rows:
            return pd.DataFrame()
        return _select(pd.DataFrame(rows), fields)

    def _fetch_index_daily(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        index_list = config.get("index_list") or []
        if not index_list:
            raise ValueError("指数数据配置缺少 index_list")
        frames = []
        for ts_code in index_list:
            symbol = _em_symbol(ts_code)
            part = self._index_hist_one(ts_code, symbol, start, end)
            if part is None or part.empty:
                continue
            frames.append(part)
            _pause(config, 0.15)
        if not frames:
            return pd.DataFrame()
        return _select(pd.concat(frames, ignore_index=True), fields)

    def _index_hist_one(self, ts_code: str, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            raw = _retry_call(
                lambda: ak.index_zh_a_hist(
                    symbol=symbol, period="daily", start_date=start, end_date=end
                )
            )
            if raw is not None and not raw.empty:
                part = raw.rename(columns=_HIST_MAP).copy()
                part["trade_date"] = part["trade_date"].map(_yyyymmdd)
                part["ts_code"] = (
                    str(ts_code).upper() if "." in str(ts_code) else _to_ts_code(ts_code)
                )
                if "close" in part.columns and "change" in part.columns:
                    part["pre_close"] = pd.to_numeric(
                        part["close"], errors="coerce"
                    ) - pd.to_numeric(part["change"], errors="coerce")
                if "amount" in part.columns:
                    part["amount"] = pd.to_numeric(part["amount"], errors="coerce") / 1000.0
                return part
        except Exception as exc:
            print(f"[akshare] 东财指数失败 {ts_code}: {exc}，改用新浪", flush=True)
        try:
            raw = ak.stock_zh_index_daily(symbol=_sina_symbol(ts_code))
        except Exception as exc:
            print(f"[akshare] 新浪指数失败 {ts_code}: {exc}", flush=True)
            return pd.DataFrame()
        if raw is None or raw.empty:
            return pd.DataFrame()
        part = raw.rename(columns={"date": "trade_date", "volume": "vol"}).copy()
        part["trade_date"] = part["trade_date"].map(_yyyymmdd)
        part = part.loc[(part["trade_date"] >= start) & (part["trade_date"] <= end)]
        part["ts_code"] = str(ts_code).upper() if "." in str(ts_code) else _to_ts_code(ts_code)
        close = pd.to_numeric(part["close"], errors="coerce") if "close" in part.columns else None
        if close is not None:
            part["pre_close"] = close.shift(1)
            part["change"] = close - part["pre_close"]
        return part

    def _fetch_index_classify(self, config: Dict, fields) -> pd.DataFrame:
        src = config.get("src", "SW2021")
        levels = config.get("levels") or ["L1", "L2", "L3"]
        loaders = {
            "L1": ak.sw_index_first_info,
            "L2": ak.sw_index_second_info,
            "L3": ak.sw_index_third_info,
        }
        frames = []
        by_name: Dict[str, str] = {}
        for level in levels:
            getter = loaders.get(str(level).upper())
            if getter is None:
                continue
            raw = getter()
            if raw is None or raw.empty:
                continue
            part = raw.copy()
            part["industry_name"] = part["行业名称"].astype(str)
            part["industry_code"] = part["行业代码"].astype(str).str.replace(".SI", "", regex=False)
            part["index_code"] = part["industry_code"].map(_si_code)
            part["level"] = str(level).upper()
            part["src"] = src
            part["is_pub"] = None
            if "上级行业" in part.columns:
                part["parent_name"] = part["上级行业"].astype(str)
                part["parent_code"] = part["parent_name"].map(lambda n: by_name.get(n, ""))
            else:
                part["parent_code"] = ""
            for _, row in part.iterrows():
                by_name[str(row["industry_name"])] = str(row["index_code"])
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return _select(result, fields)

    def _fetch_index_member_all(self, config: Dict, fields) -> pd.DataFrame:
        src = config.get("src", "SW2021")
        classify = self._fetch_index_classify(
            {"src": src, "levels": ["L1", "L2", "L3"]},
            ["index_code", "industry_name", "industry_code", "level", "parent_code"],
        )
        raw = ak.stock_industry_clf_hist_sw()
        if raw is None or raw.empty:
            return pd.DataFrame()
        members = raw.copy()
        code_col = "symbol" if "symbol" in members.columns else members.columns[0]
        members["ts_code"] = members[code_col].map(_to_ts_code)
        members["l3_code"] = members["industry_code"].astype(str).map(_si_code)
        members["in_date"] = members["start_date"].map(_yyyymmdd)
        meta = {}
        if not classify.empty:
            for _, row in classify.iterrows():
                meta[str(row.get("index_code") or "")] = row
                meta[str(row.get("industry_code") or "")] = row

        def _level_row(code: str, want: str) -> Optional[pd.Series]:
            node = meta.get(code) or meta.get(_si_code(code))
            seen = 0
            while node is not None and seen < 4:
                if str(node.get("level") or "") == want:
                    return node
                parent = str(node.get("parent_code") or "")
                node = meta.get(parent) if parent else None
                seen += 1
            return None

        rows = []
        grouped = members.sort_values(["ts_code", "in_date"]).groupby("ts_code", sort=False)
        for ts_code, group in grouped:
            records = group.to_dict("records")
            for i, rec in enumerate(records):
                l3 = _level_row(str(rec.get("industry_code") or ""), "L3")
                l2 = _level_row(str(rec.get("industry_code") or ""), "L2")
                l1 = _level_row(str(rec.get("industry_code") or ""), "L1")
                if l3 is None and str(rec.get("industry_code") or ""):
                    # 历史表里的代码常常已经是三级。
                    l3_code = _si_code(rec["industry_code"])
                else:
                    l3_code = str(l3["index_code"]) if l3 is not None else _si_code(rec.get("industry_code"))
                out_date = ""
                if i + 1 < len(records):
                    nxt = pd.to_datetime(records[i + 1]["in_date"], errors="coerce")
                    if pd.notna(nxt):
                        out_date = (nxt - pd.Timedelta(days=1)).strftime("%Y%m%d")
                rows.append(
                    {
                        "src": src,
                        "ts_code": ts_code,
                        "name": "",
                        "l1_code": str(l1["index_code"]) if l1 is not None else "",
                        "l1_name": str(l1["industry_name"]) if l1 is not None else "",
                        "l2_code": str(l2["index_code"]) if l2 is not None else "",
                        "l2_name": str(l2["industry_name"]) if l2 is not None else "",
                        "l3_code": l3_code,
                        "l3_name": str(l3["industry_name"]) if l3 is not None else "",
                        "in_date": rec["in_date"],
                        "out_date": out_date,
                        "is_new": "Y" if not out_date else "N",
                    }
                )
        if not rows:
            return pd.DataFrame()
        return _select(pd.DataFrame(rows), fields)

    def _fetch_stock_basic(self, config: Dict, fields) -> pd.DataFrame:
        del config
        frames = []
        sh = ak.stock_info_sh_name_code(symbol="主板A股")
        kcb = ak.stock_info_sh_name_code(symbol="科创板")
        for raw, market in ((sh, "主板"), (kcb, "科创板")):
            if raw is None or raw.empty:
                continue
            part = raw.copy()
            part["symbol"] = part["证券代码"].astype(str).str.zfill(6)
            part["name"] = part["证券简称"].astype(str)
            part["ts_code"] = part["symbol"].map(_to_ts_code)
            part["list_date"] = part["上市日期"].map(_yyyymmdd)
            part["market"] = market
            part["area"] = None
            part["industry"] = None
            part["delist_date"] = None
            part["list_status"] = "L"
            frames.append(part)
        sz = ak.stock_info_sz_name_code(symbol="A股列表")
        if sz is not None and not sz.empty:
            part = sz.copy()
            part["symbol"] = part["A股代码"].astype(str).str.zfill(6)
            part["name"] = part["A股简称"].astype(str)
            part["ts_code"] = part["symbol"].map(_to_ts_code)
            part["list_date"] = part["A股上市日期"].map(_yyyymmdd)
            part["market"] = part["板块"].astype(str) if "板块" in part.columns else "深市A股"
            part["industry"] = part["所属行业"].astype(str) if "所属行业" in part.columns else None
            part["area"] = None
            part["delist_date"] = None
            part["list_status"] = "L"
            frames.append(part)
        bj = ak.stock_info_bj_name_code()
        if bj is not None and not bj.empty:
            part = bj.copy()
            part["symbol"] = part["证券代码"].astype(str).str.zfill(6)
            part["name"] = part["证券简称"].astype(str)
            part["ts_code"] = part["symbol"].map(_to_ts_code)
            part["list_date"] = part["上市日期"].map(_yyyymmdd) if "上市日期" in part.columns else None
            part["market"] = "北交所"
            part["area"] = part["地区"].astype(str) if "地区" in part.columns else None
            part["industry"] = part["所属行业"].astype(str) if "所属行业" in part.columns else None
            part["delist_date"] = None
            part["list_status"] = "L"
            frames.append(part)
        for getter, market in (
            (lambda: ak.stock_info_sh_delist(symbol="全部"), "上交所"),
            (lambda: ak.stock_info_sz_delist(symbol="终止上市公司"), "深交所"),
        ):
            try:
                raw = getter()
            except Exception:
                raw = pd.DataFrame()
            if raw is None or raw.empty:
                continue
            part = raw.copy()
            code_col = next((c for c in part.columns if "代码" in str(c)), None)
            name_col = next((c for c in part.columns if "简称" in str(c) or "名称" in str(c)), None)
            date_col = next((c for c in part.columns if "终止" in str(c) or "摘牌" in str(c)), None)
            if code_col is None:
                continue
            part["symbol"] = part[code_col].astype(str).str.zfill(6)
            part["ts_code"] = part["symbol"].map(_to_ts_code)
            part["name"] = part[name_col].astype(str) if name_col else ""
            part["delist_date"] = part[date_col].map(_yyyymmdd) if date_col else None
            part["list_date"] = None
            part["market"] = market
            part["area"] = None
            part["industry"] = None
            part["list_status"] = "D"
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["ts_code"], keep="first")
        return _select(result, fields)

    def _fetch_namechange(self, config: Dict, fields) -> pd.DataFrame:
        del config
        frames = []
        for kind in ("简称变更", "全称变更"):
            try:
                raw = ak.stock_info_sz_change_name(symbol=kind)
            except Exception as exc:
                print(f"[akshare] namechange {kind} 失败: {exc}", flush=True)
                continue
            if raw is None or raw.empty:
                continue
            part = raw.copy()
            code_col = next((c for c in part.columns if "代码" in str(c)), None)
            date_col = next((c for c in part.columns if "日期" in str(c)), None)
            name_col = next(
                (c for c in reversed(list(part.columns)) if "简称" in str(c) or "名称" in str(c)),
                None,
            )
            if code_col is None or date_col is None or name_col is None:
                continue
            out = pd.DataFrame(
                {
                    "ts_code": part[code_col].astype(str).map(_to_ts_code),
                    "name": part[name_col].astype(str),
                    "start_date": part[date_col].map(_yyyymmdd),
                    "end_date": None,
                    "ann_date": part[date_col].map(_yyyymmdd),
                    "change_reason": kind,
                }
            )
            frames.append(out)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["ts_code", "start_date", "name"], keep="last")
        print(
            "[akshare] namechange 目前只有深交所批量接口；沪/京历史更名需自行补 symbols",
            flush=True,
        )
        return _select(result, fields)

    def _fetch_fina_indicator(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        codes = self._codes(config)
        frames: List[pd.DataFrame] = []
        progress = tqdm(codes, desc="Akshare fina", unit="股")
        for symbol in progress:
            progress.set_postfix_str(symbol)
            ts_code = _to_ts_code(symbol)
            try:
                raw = ak.stock_financial_analysis_indicator_em(symbol=ts_code, indicator="按报告期")
            except Exception as exc:
                tqdm.write(f"[akshare] fina {ts_code} 失败: {exc}")
                raw = pd.DataFrame()
            if raw is None or raw.empty:
                _pause(config, 0.3)
                continue
            part = pd.DataFrame({"ts_code": ts_code}, index=raw.index)
            part["end_date"] = raw.get("REPORT_DATE", pd.Series(index=raw.index)).map(_yyyymmdd)
            part["ann_date"] = raw.get("NOTICE_DATE", raw.get("REPORT_DATE")).map(_yyyymmdd)
            part["update_flag"] = "0"
            for dest in FINANCIAL_VALUE_FIELDS:
                aliases = _EM_FIN_MAP.get(dest, (dest.upper(),))
                series = None
                for name in aliases:
                    if name in raw.columns:
                        series = pd.to_numeric(raw[name], errors="coerce")
                        break
                part[dest] = series
            if start:
                part = part.loc[part["end_date"] >= start[:8]]
            if end:
                part = part.loc[part["end_date"] <= end[:8]]
            if not part.empty:
                frames.append(part)
            _pause(config, 0.3)
        if not frames:
            return pd.DataFrame()
        return _select(pd.concat(frames, ignore_index=True), fields)

    def _fetch_index_weight(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        del start
        index_list = config.get("index_list") or []
        if not index_list:
            raise ValueError("指数成分配置缺少 index_list")
        stamp = end or datetime.now().strftime("%Y%m%d")
        frames = []
        for ts_code in index_list:
            symbol = _em_symbol(ts_code)
            try:
                raw = ak.index_stock_cons_weight_csindex(symbol=symbol)
            except Exception as exc:
                print(f"[akshare] index_weight {ts_code} 失败: {exc}", flush=True)
                continue
            if raw is None or raw.empty:
                continue
            part = pd.DataFrame(
                {
                    "index_code": str(ts_code).upper() if "." in str(ts_code) else _to_ts_code(ts_code),
                    "con_code": raw["成分券代码"].astype(str).map(_to_ts_code),
                    "trade_date": raw["日期"].map(_yyyymmdd) if "日期" in raw.columns else stamp,
                    "weight": pd.to_numeric(raw["权重"], errors="coerce"),
                }
            )
            part["trade_date"] = part["trade_date"].replace("", stamp)
            frames.append(part)
            _pause(config, 0.2)
        if not frames:
            return pd.DataFrame()
        print("[akshare] index_weight 为中证最新快照，不是逐日历史调样", flush=True)
        return _select(pd.concat(frames, ignore_index=True), fields)
