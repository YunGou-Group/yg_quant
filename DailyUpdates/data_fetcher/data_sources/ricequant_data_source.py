#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RiceQuant / rqdatac 数据源：与 Tushare 同一套 data_type / api_name，输出相同列名。

连接 URI 优先级：
1. 配置里的 ``uri`` / ``rqdatac_uri``，或以 ``tcp://`` 开头的 ``token``
2. 环境变量 ``RQDATAC_URI``
3. ``doc/config.py`` 里的 ``RQDATAC_URI``

按面板切片（``FetchSlice.BY_PANEL``）：一次交给 start/end，源内部按股票分批。
未安装 ``rqdatac`` 时，发现阶段仍可导入本模块；真正拉数再报错。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from DailyUpdates.data_fetcher.data_source_base import DataSourceBase, FetchSlice
from DailyUpdates.storage.etf_schema import ETF_EXTRA_TS_CODES
from DailyUpdates.storage.financial_schema import FINANCIAL_VALUE_FIELDS

_EXCHANGE_TO_RQ = {
    "SH": "XSHG",
    "SZ": "XSHE",
    "BJ": "XBEI",
    "CSI": "XSHG",
}
_RQ_TO_TS = {
    "XSHG": "SH",
    "XSHE": "SZ",
    "XBEI": "BJ",
    "XBSE": "BJ",
    "SH": "SH",
    "SZ": "SZ",
    "BJ": "BJ",
    "CSI": "SH",
}
_STATUS_MAP = {
    "ACTIVE": "L",
    "DELISTED": "D",
    "TEMPORARYSUSPENDED": "P",
    "SUSPENDED": "P",
}
_PRICE_FIELDS = ("open", "high", "low", "close", "volume", "total_turnover", "prev_close")
_LIMIT_FIELDS = ("limit_up", "limit_down")
_BASIC_FACTORS = {
    "pe": ("pe_ratio",),
    "pe_ttm": ("pe_ratio_ttm",),
    "pb": ("pb_ratio", "pb_ratio_ttm"),
    "ps": ("ps_ratio",),
    "ps_ttm": ("ps_ratio_ttm",),
    "dv_ttm": ("dividend_yield_ttm", "dividend_yield"),
    "turnover_rate": ("turnover_rate", "turn"),
    "turnover_rate_f": ("free_circulation_turnover_rate", "free_float_turnover_rate"),
    "total_mv": ("market_cap",),
    "circ_mv": ("a_share_market_val_in_circulation", "circulating_market_cap"),
    "total_share": ("total_shares", "total_a_shares"),
    "float_share": ("a_share_in_circulation", "circulating_a_shares"),
}
# RQ get_pit_financials_ex 字段 → Tushare fina_indicator 列。只映射线上存在的名字。
_FINA_FACTORS = {
    "roe": ("return_on_equity_weighted_average",),
    "eps": ("basic_earnings_per_share", "adjusted_earnings_per_share"),
    "ebit": ("ebit",),
    "ebitda": ("ebitda",),
    "fcff": ("fcff",),
    "fcfe": ("fcfe",),
    "fixed_assets": ("net_fixed_assets", "total_fixed_assets"),
    "undist_profit_ps": ("undistributed_profit",),
}
_MV_WAN_FIELDS = frozenset({"total_mv", "circ_mv", "total_share", "float_share"})
_PCT_FIELDS = frozenset({"dv_ttm", "turnover_rate", "turnover_rate_f"})


def _as_field_list(fields) -> Optional[List[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [item.strip() for item in fields.split(",") if item.strip()]
    return list(fields)


def _select(frame: pd.DataFrame, fields) -> pd.DataFrame:
    wanted = _as_field_list(fields)
    if not wanted or frame.empty:
        return frame
    keep = [c for c in wanted if c in frame.columns]
    extra = [c for c in ("ts_code", "trade_date") if c in frame.columns and c not in keep]
    if not keep:
        return frame
    return frame[keep + extra].copy()


def _yyyymmdd(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        text = str(value).replace("-", "")[:8]
        return text if text.isdigit() else ""
    return ts.strftime("%Y%m%d")


def _iso_date(value) -> str:
    digits = _yyyymmdd(value)
    if len(digits) != 8:
        return ""
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _to_quarter(value) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())[:8]
    if len(digits) < 6:
        raise ValueError(f"无法把 {value!r} 转成报告期")
    month = int(digits[4:6])
    return f"{digits[:4]}q{(month - 1) // 3 + 1}"


def _quarter_end(quarter: str) -> str:
    text = str(quarter).lower()
    if "q" not in text:
        return _yyyymmdd(quarter)
    year, part = text.split("q", 1)
    month = int(part) * 3
    last = pd.Timestamp(year=int(year), month=month, day=1) + pd.offsets.MonthEnd(0)
    return last.strftime("%Y%m%d")


def to_rq_id(code: str) -> str:
    """000001.SZ / sz000001 / 000001 → 000001.XSHE。"""
    text = str(code).strip().upper()
    if not text:
        return ""
    if text.endswith((".XSHG", ".XSHE", ".XBEI", ".XBSE")):
        return text
    if "." in text:
        number, market = text.split(".", 1)
        return f"{number}.{_EXCHANGE_TO_RQ.get(market, market)}"
    if text.startswith(("SH", "SZ", "BJ")) and len(text) >= 8:
        return to_rq_id(f"{text[2:]}.{text[:2]}")
    digits = "".join(ch for ch in text if ch.isdigit())
    raw = digits[-6:].zfill(6) if digits else text
    if raw.startswith(("6", "5", "9")):
        return f"{raw}.XSHG"
    if raw.startswith(("4", "8")) and not raw.startswith("399"):
        return f"{raw}.XBEI"
    return f"{raw}.XSHE"


def to_ts_code(order_book_id: str) -> str:
    """000001.XSHE → 000001.SZ。"""
    text = str(order_book_id).strip().upper()
    if not text:
        return ""
    if "." in text:
        number, market = text.split(".", 1)
        suffix = _RQ_TO_TS.get(market)
        if suffix:
            return f"{number}.{suffix}"
        return f"{number}.{market}"
    return text


def resolve_rqdatac_uri(config: Optional[Dict] = None) -> str:
    cfg = config or {}
    for key in ("uri", "rqdatac_uri"):
        value = str(cfg.get(key) or "").strip()
        if value.startswith("tcp://"):
            return value
    token = str(cfg.get("token") or "").strip()
    if token.startswith("tcp://"):
        return token
    env = os.getenv("RQDATAC_URI", "").strip()
    if env.startswith("tcp://"):
        return env
    return _uri_from_doc_config()


def _uri_from_doc_config() -> str:
    for parent in Path(__file__).resolve().parents:
        path = parent / "doc" / "config.py"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_yg_rqdata_doc_config", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        uri = str(getattr(module, "RQDATAC_URI", "") or getattr(module, "HARDCODED_RQDATAC_URI", "") or "")
        if uri.startswith("tcp://"):
            return uri
    return ""


def _import_rq():
    try:
        import rqdatac
    except ImportError as exc:
        raise RuntimeError("未安装 rqdatac，请先 pip install rqdatac") from exc
    return rqdatac


def _chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
    step = max(1, int(size))
    for i in range(0, len(items), step):
        yield list(items[i : i + step])


def _reset_price_frame(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.Series):
        frame = raw.to_frame(name=raw.name or "value")
    else:
        frame = raw
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [
            str(col[-1] if isinstance(col, tuple) else col) for col in work.columns
        ]
    if isinstance(work.index, pd.MultiIndex):
        work = work.reset_index()
    else:
        work = work.reset_index()
    rename = {}
    for col in work.columns:
        key = str(col).lower()
        if key in {"order_book_id", "order_book_ids"}:
            rename[col] = "order_book_id"
        elif key in {"datetime", "date"}:
            rename[col] = "datetime"
    return work.rename(columns=rename)


class RicequantDataSource(DataSourceBase):
    """把 rqdatac 面板接口映射成调度器认的表结构。"""

    fetch_slice = FetchSlice.BY_PANEL
    requires_token = True

    def __init__(self, client: Any = None):
        self._client = client

    def _rq(self, config: Optional[Dict] = None):
        if self._client is not None:
            return self._client
        uri = resolve_rqdatac_uri(config)
        if not uri:
            raise ValueError("Ricequant 缺少 URI：请设 RQDATAC_URI 或在配置里传 uri")
        rq = _import_rq()
        rq.init(uri=uri)
        self._client = rq
        return rq

    def fetch_trade_calendar(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        rq = self._rq(config)
        days = rq.get_trading_dates(start_date, end_date, market="cn")
        if days is None:
            return pd.DataFrame(columns=["trade_date"])
        iso = sorted({_iso_date(d) for d in days if _iso_date(d)})
        return pd.DataFrame({"trade_date": iso})

    def fetch_data(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        self._rq(config)
        data_type = str(config.get("data_type") or "daily")
        api_name = str(config.get("api_name") or data_type)
        fields = config.get("fields")
        if data_type == "event":
            raise ValueError(f"Ricequant 不提供事件接口 {api_name}")
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
                raise ValueError("Ricequant 不提供 namechange 历史")
            raise ValueError(f"不支持的股票信息 API: {api_name}")
        if data_type == "financial":
            if api_name in ("fina_indicator", "fina_indicator_vip"):
                return self._fetch_fina_indicator(config, fields, start_date, end_date)
            raise ValueError(f"不支持的财务 API: {api_name}")
        if data_type == "index_constituent":
            if api_name == "index_weight":
                return self._fetch_index_weight(config, fields, start_date, end_date)
            raise ValueError(f"不支持的指数成分 API: {api_name}")
        if data_type == "etf":
            if api_name == "fund_daily":
                return self._fetch_fund_daily(config, fields, start_date, end_date)
            raise ValueError(f"不支持的 ETF API: {api_name}")
        if data_type == "etf_info":
            if api_name == "fund_basic":
                return self._fetch_fund_basic(config, fields)
            raise ValueError(f"不支持的 ETF 信息 API: {api_name}")
        raise ValueError(f"不支持的数据类型: {data_type}")

    def _batch_size(self, config: Dict) -> int:
        return max(1, int(config.get("batch_size") or 250))

    def _stock_ids(self, config: Dict) -> List[str]:
        extra = config.get("symbols") or config.get("stock_list") or []
        if extra:
            return [to_rq_id(item) for item in extra if to_rq_id(item)]
        rq = self._rq(config)
        listed = rq.all_instruments(type="CS", market="cn")
        if listed is None or (isinstance(listed, pd.DataFrame) and listed.empty):
            raise RuntimeError("Ricequant 未能获取 A 股列表")
        if isinstance(listed, pd.DataFrame):
            col = "order_book_id" if "order_book_id" in listed.columns else listed.columns[0]
            return listed[col].astype(str).tolist()
        return [str(item) for item in listed]

    def _etf_ids(self, config: Dict) -> List[str]:
        extra = config.get("etf_list") or config.get("symbols") or []
        if extra:
            return [to_rq_id(item) for item in extra if to_rq_id(item)]
        rq = self._rq(config)
        listed = rq.all_instruments(type="ETF", market="cn")
        ids: List[str] = []
        if isinstance(listed, pd.DataFrame) and not listed.empty:
            col = "order_book_id" if "order_book_id" in listed.columns else listed.columns[0]
            ids.extend(listed[col].astype(str).tolist())
        for code in config.get("extra_ts_codes") or ETF_EXTRA_TS_CODES:
            rq_id = to_rq_id(code)
            if rq_id and rq_id not in ids:
                ids.append(rq_id)
        if not ids:
            raise RuntimeError("Ricequant 未能获取场内 ETF 列表")
        return ids

    def _get_price(
        self,
        ids: Sequence[str],
        start: str,
        end: str,
        fields: Sequence[str],
        *,
        adjust_type: str = "none",
        batch_size: int = 250,
    ) -> pd.DataFrame:
        rq = self._client
        frames: List[pd.DataFrame] = []
        for batch in _chunks(list(ids), batch_size):
            raw = rq.get_price(
                batch,
                start_date=start,
                end_date=end,
                frequency="1d",
                fields=list(fields),
                adjust_type=adjust_type,
                skip_suspended=False,
                expect_df=True,
                market="cn",
            )
            part = _reset_price_frame(raw)
            if not part.empty:
                frames.append(part)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _bars_from_price(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        frame = raw.copy()
        if "order_book_id" not in frame.columns:
            return pd.DataFrame()
        frame["ts_code"] = frame["order_book_id"].map(to_ts_code)
        date_col = "datetime" if "datetime" in frame.columns else None
        if date_col is None:
            for cand in ("date", "trade_date"):
                if cand in frame.columns:
                    date_col = cand
                    break
        if date_col is None:
            return pd.DataFrame()
        frame["trade_date"] = frame[date_col].map(_yyyymmdd)
        rename = {
            "volume": "vol",
            "total_turnover": "amount",
            "prev_close": "pre_close",
            "limit_up": "up_limit",
            "limit_down": "down_limit",
        }
        frame = frame.rename(columns=rename)
        if "vol" in frame.columns:
            frame["vol"] = pd.to_numeric(frame["vol"], errors="coerce") / 100.0
        if "amount" in frame.columns:
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") / 1000.0
        close = pd.to_numeric(frame["close"], errors="coerce") if "close" in frame.columns else None
        pre = pd.to_numeric(frame["pre_close"], errors="coerce") if "pre_close" in frame.columns else None
        if close is not None and pre is not None:
            frame["change"] = close - pre
            frame["pct_chg"] = (frame["change"] / pre.replace(0, pd.NA)) * 100.0
        return frame

    def _fetch_daily(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        ids = self._stock_ids(config)
        raw = self._get_price(
            ids, start, end, _PRICE_FIELDS, batch_size=self._batch_size(config)
        )
        return _select(self._bars_from_price(raw), fields)

    def _fetch_stk_limit(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        ids = self._stock_ids(config)
        raw = self._get_price(
            ids, start, end, _LIMIT_FIELDS, batch_size=self._batch_size(config)
        )
        bars = self._bars_from_price(raw)
        return _select(bars, fields)

    def _fetch_adj_factor(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        ids = self._stock_ids(config)
        batch = self._batch_size(config)
        none = self._get_price(ids, start, end, ("close",), adjust_type="none", batch_size=batch)
        post = self._get_price(ids, start, end, ("close",), adjust_type="post", batch_size=batch)
        if none.empty or post.empty:
            return pd.DataFrame()
        left = none.rename(columns={"close": "raw_close"})
        right = post.rename(columns={"close": "post_close"})
        keys = [c for c in ("order_book_id", "datetime") if c in left.columns and c in right.columns]
        merged = left.merge(right[keys + ["post_close"]], on=keys, how="inner")
        merged["ts_code"] = merged["order_book_id"].map(to_ts_code)
        date_col = "datetime" if "datetime" in merged.columns else "date"
        merged["trade_date"] = merged[date_col].map(_yyyymmdd)
        raw_close = pd.to_numeric(merged["raw_close"], errors="coerce")
        post_close = pd.to_numeric(merged["post_close"], errors="coerce")
        merged["adj_factor"] = post_close / raw_close.replace(0, pd.NA)
        return _select(merged, fields)

    def _fetch_daily_basic(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        ids = self._stock_ids(config)
        rq = self._client
        wanted = _as_field_list(fields) or list(_BASIC_FACTORS)
        factor_names: List[str] = []
        alias_to_out: Dict[str, str] = {}
        for out_name, aliases in _BASIC_FACTORS.items():
            if out_name not in wanted and out_name not in {"ts_code", "trade_date"}:
                continue
            for alias in aliases:
                factor_names.append(alias)
                alias_to_out.setdefault(alias, out_name)
        frames: List[pd.DataFrame] = []
        for batch in _chunks(ids, self._batch_size(config)):
            raw = None
            if hasattr(rq, "get_factor"):
                try:
                    raw = rq.get_factor(
                        batch,
                        factor_names,
                        start_date=start,
                        end_date=end,
                        expect_df=True,
                    )
                except Exception:
                    raw = None
            part = _reset_price_frame(raw)
            if part.empty:
                continue
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        work = pd.concat(frames, ignore_index=True)
        if "order_book_id" not in work.columns:
            return pd.DataFrame()
        date_col = "datetime" if "datetime" in work.columns else None
        if date_col is None:
            for cand in ("date", "trade_date"):
                if cand in work.columns:
                    date_col = cand
                    break
        out = pd.DataFrame(
            {
                "ts_code": work["order_book_id"].map(to_ts_code),
                "trade_date": work[date_col].map(_yyyymmdd) if date_col else "",
            }
        )
        for src, dest in alias_to_out.items():
            if src not in work.columns or dest in out.columns:
                continue
            series = pd.to_numeric(work[src], errors="coerce")
            if dest in _MV_WAN_FIELDS:
                series = series / 10000.0
            if dest in _PCT_FIELDS:
                sample = series.dropna()
                if not sample.empty and float(sample.abs().median()) < 1.0:
                    series = series * 100.0
            out[dest] = series
        return _select(out, fields)

    def _fetch_index_daily(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        index_list = config.get("index_list") or []
        if not index_list:
            raise ValueError("指数数据配置缺少 index_list")
        ids = [to_rq_id(code) for code in index_list if to_rq_id(code)]
        raw = self._get_price(ids, start, end, _PRICE_FIELDS, batch_size=len(ids) or 1)
        return _select(self._bars_from_price(raw), fields)

    def _month_ends(self, start: str, end: str) -> List[str]:
        rq = self._client
        days = rq.get_trading_dates(start, end, market="cn")
        if days is None or len(days) == 0:
            return []
        iso = [_iso_date(d) for d in days if _iso_date(d)]
        by_month: Dict[str, str] = {}
        for day in iso:
            by_month[day[:7]] = day
        return list(by_month.values())

    def _fetch_index_weight(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        index_list = config.get("index_list") or []
        if not index_list:
            raise ValueError("指数成分配置缺少 index_list")
        rq = self._client
        dates = self._month_ends(start, end) or [_iso_date(end) or _iso_date(pd.Timestamp.today())]
        rows: List[Dict[str, Any]] = []
        for ts_code in index_list:
            rq_id = to_rq_id(ts_code)
            for day in dates:
                try:
                    weights = rq.index_weights(rq_id, date=day)
                except Exception:
                    weights = None
                if weights is None:
                    continue
                if isinstance(weights, pd.Series):
                    items = weights.items()
                elif isinstance(weights, pd.DataFrame):
                    col = "weight" if "weight" in weights.columns else weights.columns[-1]
                    items = weights[col].items()
                else:
                    continue
                for con_id, weight in items:
                    rows.append(
                        {
                            "index_code": str(ts_code).upper() if "." in str(ts_code) else to_ts_code(ts_code),
                            "con_code": to_ts_code(con_id),
                            "trade_date": _yyyymmdd(day),
                            "weight": float(weight) if pd.notna(weight) else None,
                        }
                    )
        if not rows:
            return pd.DataFrame()
        return _select(pd.DataFrame(rows), fields)

    def _fetch_stock_basic(self, config: Dict, fields) -> pd.DataFrame:
        rq = self._client
        raw = rq.all_instruments(type="CS", market="cn")
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            return pd.DataFrame()
        frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        oid = "order_book_id" if "order_book_id" in frame.columns else frame.columns[0]
        if "abbrev_symbol" in frame.columns:
            names = frame["abbrev_symbol"]
        elif "symbol" in frame.columns:
            names = frame["symbol"]
        else:
            names = ""
        frame["ts_code"] = frame[oid].map(to_ts_code)
        frame["name"] = names.astype(str) if not isinstance(names, str) else names
        frame["symbol"] = frame["ts_code"].astype(str).str.split(".").str[0]
        list_col = next((c for c in ("listed_date", "list_date") if c in frame.columns), None)
        delist_col = next((c for c in ("de_listed_date", "delisted_date", "delist_date") if c in frame.columns), None)
        frame["list_date"] = frame[list_col].map(_yyyymmdd) if list_col else ""
        frame["delist_date"] = frame[delist_col].map(_yyyymmdd) if delist_col else ""
        status_col = "status" if "status" in frame.columns else None
        if status_col:
            frame["list_status"] = (
                frame[status_col].astype(str).str.upper().map(lambda s: _STATUS_MAP.get(s, "L"))
            )
        else:
            frame["list_status"] = "L"
        wanted = config.get("list_status")
        if wanted:
            flags = [wanted] if isinstance(wanted, str) else list(wanted)
            frame = frame.loc[frame["list_status"].isin(flags)]
        exchange = frame[oid].astype(str).str.split(".").str[-1]
        market_map = {"XSHG": "上交所", "XSHE": "深交所", "XBEI": "北交所", "XBSE": "北交所"}
        frame["market"] = exchange.map(market_map)
        frame["area"] = None
        frame["industry"] = frame["industry_name"] if "industry_name" in frame.columns else None
        return _select(frame, fields)

    def _fetch_index_classify(self, config: Dict, fields) -> pd.DataFrame:
        rq = self._client
        src = config.get("src", "SW2021")
        levels = {str(item).upper() for item in (config.get("levels") or ["L1", "L2", "L3"])}
        if not hasattr(rq, "get_industry_mapping"):
            raise ValueError("当前 rqdatac 没有 get_industry_mapping")
        raw = rq.get_industry_mapping(source="sws")
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            return pd.DataFrame()
        tree = raw.copy()
        rows: List[Dict[str, object]] = []
        specs = (
            ("L1", "index_code", "index_name", ""),
            ("L2", "second_index_code", "second_index_name", "index_code"),
            ("L3", "third_index_code", "third_index_name", "second_index_code"),
        )
        for level, code_col, name_col, parent_col in specs:
            if level not in levels or code_col not in tree.columns:
                continue
            part = tree[[c for c in (code_col, name_col, parent_col) if c and c in tree.columns]].drop_duplicates(subset=[code_col])
            for rec in part.to_dict("records"):
                code = str(rec.get(code_col) or "")
                if not code or code == "nan":
                    continue
                rows.append(
                    {
                        "index_code": code,
                        "industry_name": str(rec.get(name_col) or ""),
                        "parent_code": str(rec.get(parent_col) or "") if parent_col else "",
                        "level": level,
                        "industry_code": code,
                        "is_pub": 1,
                        "src": src,
                    }
                )
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows).drop_duplicates(subset=["index_code"], keep="last")
        return _select(result, fields)

    def _fetch_index_member_all(self, config: Dict, fields) -> pd.DataFrame:
        rq = self._client
        src = config.get("src", "SW2021")
        getter = getattr(rq, "get_instrument_industry", None)
        if getter is None:
            raise ValueError("当前 rqdatac 没有 get_instrument_industry")
        ids = self._stock_ids(config)
        frames: List[pd.DataFrame] = []
        for batch in _chunks(ids, self._batch_size(config)):
            raw = getter(batch, source="sws", level=0)
            part = raw.copy() if isinstance(raw, pd.DataFrame) else _reset_price_frame(raw)
            if part is None or part.empty:
                continue
            if "order_book_id" not in part.columns:
                part = part.reset_index()
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        work = pd.concat(frames, ignore_index=True)
        oid = "order_book_id" if "order_book_id" in work.columns else work.columns[0]
        out = pd.DataFrame(
            {
                "l1_code": work["first_industry_code"] if "first_industry_code" in work.columns else "",
                "l1_name": work["first_industry_name"] if "first_industry_name" in work.columns else "",
                "l2_code": work["second_industry_code"] if "second_industry_code" in work.columns else "",
                "l2_name": work["second_industry_name"] if "second_industry_name" in work.columns else "",
                "l3_code": work["third_industry_code"] if "third_industry_code" in work.columns else "",
                "l3_name": work["third_industry_name"] if "third_industry_name" in work.columns else "",
                "ts_code": work[oid].map(to_ts_code),
                "name": "",
                "in_date": "",
                "out_date": "",
                "is_new": "Y",
                "src": src,
            }
        )
        return _select(out, fields)

    def _fetch_fund_basic(self, config: Dict, fields) -> pd.DataFrame:
        rq = self._client
        raw = rq.all_instruments(type="ETF", market="cn")
        frames: List[pd.DataFrame] = []
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            part = raw.copy()
            oid = "order_book_id" if "order_book_id" in part.columns else part.columns[0]
            part["ts_code"] = part[oid].map(to_ts_code)
            name_col = "abbrev_symbol" if "abbrev_symbol" in part.columns else ("symbol" if "symbol" in part.columns else None)
            part["name"] = part[name_col].astype(str) if name_col else ""
            if "am_name" in part.columns:
                part["management"] = part["am_name"].astype(str)
            else:
                part["management"] = ""
            part["fund_type"] = "ETF"
            part["market"] = "E"
            if "status" in part.columns:
                part["status"] = part["status"].astype(str).str.upper().map(
                    lambda s: _STATUS_MAP.get(s, "L")
                )
            else:
                part["status"] = "L"
            list_col = next((c for c in ("listed_date", "list_date") if c in part.columns), None)
            delist_col = next((c for c in ("de_listed_date", "delist_date") if c in part.columns), None)
            part["list_date"] = part[list_col].map(_yyyymmdd) if list_col else ""
            part["delist_date"] = part[delist_col].map(_yyyymmdd) if delist_col else ""
            frames.append(part)
        have = set()
        if frames:
            have = set(frames[0]["ts_code"].astype(str).str.upper())
        extras = []
        for code in config.get("extra_ts_codes") or ETF_EXTRA_TS_CODES:
            ts_code = str(code).strip().upper()
            if not ts_code or ts_code in have:
                continue
            extras.append(
                {
                    "ts_code": ts_code if "." in ts_code else to_ts_code(to_rq_id(ts_code)),
                    "name": "",
                    "management": "",
                    "fund_type": "ETF",
                    "market": "E",
                    "status": "L",
                    "list_date": "",
                    "delist_date": "",
                }
            )
        if extras:
            frames.append(pd.DataFrame(extras))
        if not frames:
            return pd.DataFrame()
        return _select(pd.concat(frames, ignore_index=True), fields)

    def _fetch_fund_daily(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        ids = self._etf_ids(config)
        raw = self._get_price(ids, start, end, _PRICE_FIELDS, batch_size=self._batch_size(config))
        return _select(self._bars_from_price(raw), fields)

    def _fetch_fina_indicator(self, config: Dict, fields, start: str, end: str) -> pd.DataFrame:
        rq = self._client
        ids = self._stock_ids(config)
        wanted = [name for name in (_as_field_list(fields) or list(FINANCIAL_VALUE_FIELDS)) if name in _FINA_FACTORS]
        aliases: List[str] = []
        alias_to_out: Dict[str, str] = {}
        for out_name in wanted:
            for alias in _FINA_FACTORS[out_name]:
                aliases.append(alias)
                alias_to_out.setdefault(alias, out_name)
        if not aliases:
            raise ValueError("Ricequant 财务字段映射为空")
        getter = getattr(rq, "get_pit_financials_ex", None)
        if getter is None:
            raise ValueError("当前 rqdatac 没有 get_pit_financials_ex")
        start_q = _to_quarter(start)
        end_q = _to_quarter(end)
        frames: List[pd.DataFrame] = []
        for batch in _chunks(ids, self._batch_size(config)):
            raw = getter(batch, aliases, start_quarter=start_q, end_quarter=end_q)
            part = raw.copy() if isinstance(raw, pd.DataFrame) else _reset_price_frame(raw)
            if part is None or part.empty:
                continue
            if "order_book_id" not in part.columns or "quarter" not in part.columns:
                part = part.reset_index()
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        work = pd.concat(frames, ignore_index=True)
        oid = "order_book_id" if "order_book_id" in work.columns else work.columns[0]
        info_date = next((c for c in ("info_date", "announcement_date", "ann_date") if c in work.columns), None)
        quarter_col = "quarter" if "quarter" in work.columns else None
        flag = work["if_adjusted"].astype(str) if "if_adjusted" in work.columns else "0"
        out = pd.DataFrame(
            {
                "ts_code": work[oid].map(to_ts_code),
                "ann_date": work[info_date].map(_yyyymmdd) if info_date else "",
                "end_date": work[quarter_col].map(_quarter_end) if quarter_col else "",
                "update_flag": flag,
            }
        )
        for src, dest in alias_to_out.items():
            if src in work.columns and dest not in out.columns:
                out[dest] = pd.to_numeric(work[src], errors="coerce")
        return _select(out, fields)
