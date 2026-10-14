#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare数据源类，专门处理Tushare API调用
"""

import tushare as ts
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Sequence
import os
import time
from tqdm import tqdm
from DailyUpdates.data_fetcher.data_source_base import DataSourceBase, FetchSlice
from DailyUpdates.storage.etf_schema import (
    ETF_EXTRA_TS_CODES,
    ETF_TUSHARE_BASIC_FIELDS,
    ETF_TUSHARE_DAILY_FIELDS,
)
from DailyUpdates.storage.event_schema import event_spec
from DailyUpdates.storage.financial_schema import (
    FINANCIAL_TUSHARE_FIELDS,
    STATEMENT_TUSHARE_FIELDS,
    STATEMENT_VALUE_FIELDS,
    canonicalize_statement_fields,
    statement_table_for_api,
)

# index_weight 请求代码与配置代码可以不同，写入仍用配置里的代码。
# 000300.SH → 399300.SZ：https://github.com/waditu/tushare/issues/801
# 000985.SH → 000985.CSI：index_basic(market=CSI)；.SH / 399985.SZ 无成分。
INDEX_WEIGHT_CODE_ALIAS = {
    "000300.SH": "399300.SZ",
    "000985.SH": "000985.CSI",
}
# index_daily 同样：000985 在 Tushare 挂 CSI，.SH 日线为空。
INDEX_DAILY_CODE_ALIAS = {
    "000985.SH": "000985.CSI",
}
INDEX_WEIGHT_PROBE_CODES = ("000016.SH", "399300.SZ", "000905.SH")
# VIP 单页常见顶格；下一页为空且上一页正好落在这里，视为截断。
TUSHARE_PAGE_CAPS = frozenset({6000, 8000, 10000, 12000})


def _tushare_error_text(exc: BaseException) -> str:
    return str(exc)


def _is_non_retryable_tushare_error(exc: BaseException) -> bool:
    """参数/权限类错误重试没有意义。"""
    text = _tushare_error_text(exc)
    return any(
        token in text
        for token in ("必填参数", "参数错误", "没有接口访问权限", "权限不足")
    )


def _is_rate_limit_error(exc: BaseException) -> bool:
    return "频率" in _tushare_error_text(exc)


class _EventWindowTruncated(RuntimeError):
    """翻页回到同一页，或 offset 超过 10 万。日期区间需要拆开重拉。"""


# 服务端拒绝 offset > 100000，超出后只能把日期区间拆小。
_EVENT_OFFSET_LIMIT = 100000


def _event_rate_wait_seconds(text: str) -> float:
    """分钟额度等 65 秒，小时额度等 10 分钟，日额度等到次日 00:05。"""
    if "天" in text:
        now = pd.Timestamp.now()
        wake = now.normalize() + pd.Timedelta(days=1, minutes=5)
        return max((wake - now).total_seconds(), 60.0)
    if "小时" in text:
        return 600.0
    return 65.0


def _is_offset_ceiling(text: str) -> bool:
    lowered = text.lower()
    return "offset" in lowered and ("100000" in lowered or "缩小" in text)


def _is_unreleased_period_error(exc: BaseException) -> bool:
    """未到期报告期用 period 全市场拉取时，Tushare 会改口要求 ts_code。"""
    text = _tushare_error_text(exc)
    return "必填参数" in text and "ts_code" in text


class TushareDataSource(DataSourceBase):
    """Tushare数据源类，专门处理Tushare API调用"""

    fetch_slice = FetchSlice.BY_DATE
    requires_token = True
    
    def __init__(self):
        """初始化Tushare数据源"""
        # 不在初始化时设置token，在fetch_data时从config获取
        self.pro: Any = None
    
    def fetch_data(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        根据配置获取Tushare数据
        
        Parameters
        ----------
        config : Dict
            数据源配置
        start_date : str
            开始日期，格式：YYYYMMDD
        end_date : str
            结束日期，格式：YYYYMMDD
        """
        # 从config获取token并初始化API
        token = config.get('token')
        if not token:
            raise ValueError("Tushare配置中缺少token参数")
        
        # 设置token并创建API实例
        ts.set_token(token)
        self.pro = ts.pro_api()
        
        data_type = config.get('data_type', 'daily')
        api_name = config.get('api_name', data_type)  # 如果没有api_name，使用data_type作为默认值
        
        # 获取字段列表
        fields = config.get('fields', 'ts_code,trade_date,open,high,low,close,vol,amount')
        
        # 根据数据类型和API接口名处理
        if data_type == 'daily':
            # 日频数据：根据api_name调用不同接口
            if api_name == 'daily':
                return self._fetch_daily_by_day(
                    self.pro.daily, fields, start_date, end_date, "daily"
                )
            elif api_name == 'daily_basic':
                return self._fetch_daily_by_day(
                    self.pro.daily_basic, fields, start_date, end_date, "daily_basic"
                )
            elif api_name == 'stk_limit':
                return self._fetch_stk_limit(config, fields, start_date, end_date)
            elif api_name == 'adj_factor':
                return self._fetch_adj_factor(config, fields, start_date, end_date)
            else:
                raise ValueError(f"不支持的日频API接口: {api_name}")
        
        elif data_type == 'index':
            # 指数数据：根据api_name调用不同接口
            index_list = config.get('index_list', [])
            if not index_list:
                raise ValueError(f"指数数据配置缺少 index_list 字段")
            
            if api_name == 'index_daily':
                return self._fetch_index_daily_range(
                    self.pro.index_daily, index_list, fields, start_date, end_date
                )
            elif api_name == 'index_dailybasic':
                return self._fetch_index_daily_range(
                    self.pro.index_dailybasic, index_list, fields, start_date, end_date
                )
            else:
                raise ValueError(f"不支持的指数API接口: {api_name}")

        elif data_type == 'industry':
            if api_name == 'index_classify':
                return self._fetch_index_classify(config, fields)
            if api_name == 'index_member_all':
                return self._fetch_index_member_all(config, fields)
            raise ValueError(f"不支持的行业API接口: {api_name}")

        elif data_type == 'stock_info':
            if api_name == 'stock_basic':
                return self._fetch_stock_basic(config, fields)
            if api_name == 'namechange':
                return self._fetch_namechange(config, fields)
            raise ValueError(f"不支持的股票信息API接口: {api_name}")

        elif data_type == 'financial':
            if api_name in ('fina_indicator', 'fina_indicator_vip'):
                return self._fetch_fina_indicator(config, fields, start_date, end_date)
            if statement_table_for_api(api_name):
                return self._fetch_financial_statement(
                    config, fields, start_date, end_date
                )
            raise ValueError(f"不支持的财务API接口: {api_name}")

        elif data_type == 'event':
            return self._fetch_event(config, fields, start_date, end_date)

        elif data_type == 'index_constituent':
            if api_name == 'index_weight':
                return self._fetch_index_weight(config, fields, start_date, end_date)
            raise ValueError(f"不支持的指数成分API接口: {api_name}")

        elif data_type == 'etf':
            if api_name == 'fund_daily':
                return self._fetch_fund_daily(config, fields, start_date, end_date)
            raise ValueError(f"不支持的ETF API接口: {api_name}")

        elif data_type == 'etf_info':
            if api_name == 'fund_basic':
                return self._fetch_fund_basic(config, fields)
            raise ValueError(f"不支持的ETF信息API接口: {api_name}")
        
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    def fetch_trade_calendar(self, config: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        token = config.get("token")
        if not token:
            raise ValueError("Tushare 配置中缺少 token，无法拉取交易日历")
        ts.set_token(token)
        pro = ts.pro_api()
        digits_s = "".join(ch for ch in str(start_date) if ch.isdigit())[:8]
        digits_e = "".join(ch for ch in str(end_date) if ch.isdigit())[:8]
        frame = pro.trade_cal(
            exchange="SSE",
            start_date=digits_s,
            end_date=digits_e,
            is_open="1",
        )
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["trade_date"])
        col = "cal_date" if "cal_date" in frame.columns else frame.columns[0]
        days = sorted(pd.Timestamp(str(v)).strftime("%Y-%m-%d") for v in frame[col])
        return pd.DataFrame({"trade_date": days})

    def _fetch_index_classify(self, config: Dict, fields) -> pd.DataFrame:
        """申万行业分类：https://tushare.pro/document/2?doc_id=181"""
        src = config.get('src', 'SW2021')
        levels = config.get('levels') or ['L1', 'L2', 'L3']
        field_list = self._as_field_list(fields)
        frames = []
        print(f"[industry] 拉取申万分类 src={src}, levels={levels}", flush=True)
        for level in levels:
            paras = {'level': level, 'src': src}
            part = self._call_api(self.pro.index_classify, paras, field_list)
            rows = 0 if part is None or part.empty else len(part)
            print(f"[industry] index_classify level={level} -> {rows} 行", flush=True)
            if part is not None and not part.empty:
                if 'src' not in part.columns:
                    part = part.copy()
                    part['src'] = src
                frames.append(part)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=['index_code'], keep='last'
        )
        print(f"[industry] index_classify 合计 {len(result)} 行", flush=True)
        return result

    @staticmethod
    def _index_member_flags(is_new) -> List[str]:
        """Tushare index_member_all 默认 is_new=Y，不传拿不到调出历史。

        配置缺省时显式拉 Y+N；显式传入则只拉该档。
        """
        if is_new is None:
            return ["Y", "N"]
        return [str(is_new)]

    def _fetch_index_member_all(self, config: Dict, fields) -> pd.DataFrame:
        """申万行业成分：https://tushare.pro/document/2?doc_id=335"""
        src = config.get('src', 'SW2021')
        is_new = config.get('is_new', None)
        flags = self._index_member_flags(is_new)
        field_list = self._as_field_list(fields)
        # 全表一次拉取会被截断（实测约 3000 行）；按一级行业分页更完整且更快。
        print(
            f"[industry] 准备按 L1 拉取成分 src={src}, is_new={is_new}, flags={flags}",
            flush=True,
        )
        classify = self._fetch_index_classify(
            {
                'src': src,
                'levels': ['L1'],
                'fields': ['index_code', 'industry_name', 'level', 'src'],
            },
            ['index_code', 'industry_name', 'level', 'src'],
        )
        if classify.empty or 'index_code' not in classify.columns:
            raise RuntimeError("无法获取申万一级行业列表，industry member 拉取中止")

        codes = classify['index_code'].astype(str).tolist()
        name_map = {}
        if 'industry_name' in classify.columns:
            name_map = dict(
                zip(
                    classify['index_code'].astype(str),
                    classify['industry_name'].astype(str),
                )
            )
        print(
            f"[industry] 共 {len(codes)} 个一级行业，开始逐个请求 index_member_all",
            flush=True,
        )

        frames = []
        empty_count = 0
        row_count = 0
        progress = tqdm(codes, desc="拉取行业成分", unit="行业")
        for index_code in progress:
            industry_name = name_map.get(index_code, "")
            progress.set_postfix_str(
                f"{index_code} {industry_name[:8]} rows={row_count}"
            )
            for flag in flags:
                paras = {'l1_code': index_code, 'is_new': flag}
                part = self._call_api(self.pro.index_member_all, paras, field_list)
                if part is None or part.empty:
                    empty_count += 1
                    time.sleep(0.15)
                    continue
                frames.append(part)
                row_count += len(part)
                if len(part) >= 2000:
                    tqdm.write(
                        f"[industry] 警告: {index_code}({industry_name}) "
                        f"is_new={flag} 返回 {len(part)} 行，可能被 2000 上限截断"
                    )
                time.sleep(0.15)

        print(
            f"[industry] 成分拉取结束: 成功行业={len(frames)}, "
            f"空结果={empty_count}, 原始行数={row_count}",
            flush=True,
        )
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result['src'] = src
        if {'ts_code', 'l3_code', 'in_date'}.issubset(result.columns):
            result = result.drop_duplicates(
                subset=['ts_code', 'l3_code', 'in_date'], keep='last'
            )
        else:
            result = result.drop_duplicates(keep='last')
        print(f"[industry] 去重后成分 {len(result)} 行", flush=True)
        return result

    def _fetch_daily_by_day(
        self, getter: Callable, fields, start_date: str, end_date: str, label: str
    ) -> pd.DataFrame:
        """日频全市场接口：宽区间按自然日切片，避免一次返回被行数上限截断。"""
        field_list = self._as_field_list(fields)
        if not start_date or not end_date:
            raise ValueError(f"{label} 需要 start_date/end_date")
        if start_date == end_date:
            return self.getTushareInfo(
                getter, {"trade_date": start_date}, field_list
            )
        dates = pd.date_range(
            pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D"
        )
        frames = []
        print(
            f"[daily] 按日拉取 {label}: {start_date} -> {end_date} "
            f"共 {len(dates)} 个自然日",
            flush=True,
        )
        progress = tqdm(dates, desc=f"拉取{label}", unit="日")
        for day in progress:
            trade_date = day.strftime("%Y%m%d")
            progress.set_postfix_str(trade_date)
            part = self._call_api(getter, {"trade_date": trade_date}, field_list)
            if part is not None and not part.empty:
                frames.append(part)
            time.sleep(0.12)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        print(f"[daily] {label} 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_index_daily_range(
        self, getter: Callable, index_list: List[str], fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """按代码 + 年份拉取。index_daily 忽略 offset，不能走 getTushareInfo 翻页。"""
        field_list = self._as_field_list(fields)
        if not start_date or not end_date:
            raise ValueError("指数日线需要 start_date/end_date")
        start = pd.Timestamp(str(start_date))
        end = pd.Timestamp(str(end_date))
        parts = []
        for index_code in index_list:
            request_code = INDEX_DAILY_CODE_ALIAS.get(
                str(index_code).strip().upper(), index_code
            )
            year = int(start.year)
            while year <= int(end.year):
                chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
                chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
                part = self._call_api(
                    getter,
                    {
                        "ts_code": request_code,
                        "start_date": chunk_start.strftime("%Y%m%d"),
                        "end_date": chunk_end.strftime("%Y%m%d"),
                    },
                    field_list,
                )
                if part is not None and not part.empty:
                    if request_code != index_code and "ts_code" in part.columns:
                        part = part.copy()
                        part["ts_code"] = index_code
                    parts.append(part)
                year += 1
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def _fetch_stk_limit(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """涨跌停价格：按交易日拉取后合并。"""
        del config
        field_list = self._as_field_list(fields)
        if not start_date or not end_date:
            raise ValueError("stk_limit 需要 start_date/end_date")
        dates = pd.date_range(
            pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D"
        )
        frames = []
        print(
            f"[daily] 拉取 stk_limit: {start_date} -> {end_date} "
            f"共 {len(dates)} 个自然日",
            flush=True,
        )
        progress = tqdm(dates, desc="拉取涨跌停", unit="日")
        for day in progress:
            trade_date = day.strftime("%Y%m%d")
            progress.set_postfix_str(trade_date)
            part = self._call_api(
                self.pro.stk_limit, {"trade_date": trade_date}, field_list
            )
            if part is not None and not part.empty:
                frames.append(part)
            time.sleep(0.12)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        print(f"[daily] stk_limit 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_adj_factor(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """复权因子：https://tushare.pro/document/2?doc_id=28

        后复权价 = 原始价 × adj_factor，历史值不随后续分红变动。
        """
        field_list = self._as_field_list(fields) or [
            "ts_code",
            "trade_date",
            "adj_factor",
        ]
        if not start_date or not end_date:
            raise ValueError("adj_factor 需要 start_date/end_date")
        dates = pd.date_range(
            pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D"
        )
        frames = []
        print(
            f"[daily] 拉取 adj_factor: {start_date} -> {end_date} "
            f"共 {len(dates)} 个自然日",
            flush=True,
        )
        pause = float(config.get("pause_seconds", 0.12))
        progress = tqdm(dates, desc="拉取复权因子", unit="日")
        for day in progress:
            trade_date = day.strftime("%Y%m%d")
            progress.set_postfix_str(trade_date)
            part = self._call_api(
                self.pro.adj_factor, {"trade_date": trade_date}, field_list
            )
            if part is not None and not part.empty:
                frames.append(part)
            if pause > 0:
                time.sleep(pause)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        subset = [c for c in ["ts_code", "trade_date"] if c in result.columns]
        if subset:
            result = result.drop_duplicates(subset=subset, keep="last")
        print(f"[daily] adj_factor 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_stock_basic(self, config: Dict, fields) -> pd.DataFrame:
        """股票列表：https://tushare.pro/document/2?doc_id=25"""
        field_list = self._as_field_list(fields) or [
            "ts_code",
            "symbol",
            "name",
            "area",
            "industry",
            "market",
            "list_date",
            "delist_date",
            "list_status",
        ]
        statuses = config.get("list_status") or ["L", "D", "P"]
        if isinstance(statuses, str):
            statuses = [statuses]
        frames = []
        print(f"[stock_info] 拉取 stock_basic list_status={statuses}", flush=True)
        for status in statuses:
            part = self._call_api(
                self.pro.stock_basic,
                {"exchange": "", "list_status": status},
                field_list,
            )
            rows = 0 if part is None or part.empty else len(part)
            print(f"[stock_info] stock_basic status={status} -> {rows} 行", flush=True)
            if part is None or part.empty:
                continue
            part = part.copy()
            if "list_status" not in part.columns:
                part["list_status"] = status
            frames.append(part)
            time.sleep(0.2)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        if "ts_code" in result.columns:
            result = result.drop_duplicates(subset=["ts_code"], keep="last")
        print(f"[stock_info] stock_basic 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_namechange(self, config: Dict, fields) -> pd.DataFrame:
        """历史名称变更：https://tushare.pro/document/2?doc_id=100

        用于按 asof 判定当时是否 ST，而不是拿今天的名字回溯历史。
        """
        del config
        field_list = self._as_field_list(fields) or [
            "ts_code",
            "name",
            "start_date",
            "end_date",
            "ann_date",
            "change_reason",
        ]
        print("[stock_info] 拉取 namechange 全量历史名称", flush=True)
        result = self.getTushareInfo(self.pro.namechange, {}, field_list)
        if result is None or result.empty:
            return pd.DataFrame()
        subset = [c for c in ["ts_code", "start_date", "name"] if c in result.columns]
        if subset:
            result = result.drop_duplicates(subset=subset, keep="last")
        print(f"[stock_info] namechange 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_event(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """预告/快报按报告期。增减持按日期区间翻页，单日只用于增量预告。"""
        spec = event_spec(str(config.get("api_name") or ""))
        field_list = self._as_field_list(fields) or list(spec.tushare_fields())
        getter = getattr(self.pro, spec.api_name, None)
        if getter is None:
            raise RuntimeError(f"{spec.api_name} 不可用")
        mode = str(config.get("fetch_mode") or spec.fetch).lower()
        pause = float(config.get("pause_seconds", spec.pause))
        if mode == "range":
            start = start_date
            if spec.earliest and (not start or start < spec.earliest):
                start = spec.earliest
            return self._fetch_event_ranges(
                getter,
                field_list,
                start,
                end_date,
                spec.chunk_days,
                spec.min_interval,
                spec.page_rows,
                spec.table,
            )
        if mode == "day":
            start = start_date
            if spec.earliest and (not start or start < spec.earliest):
                start = spec.earliest
            return self._fetch_event_days(
                getter,
                spec.day_param,
                field_list,
                start,
                end_date,
                pause,
                spec.page_rows,
                spec.table,
            )
        if mode != "period":
            raise ValueError(f"{spec.api_name} 不支持的 fetch_mode: {mode}")
        as_of = min(
            pd.Timestamp(end_date) if end_date else pd.Timestamp.today(),
            pd.Timestamp.today(),
        ).normalize()
        periods = config.get("periods") or self._quarter_periods(
            start_date, end_date, as_of=as_of.strftime("%Y-%m-%d")
        )
        if not periods:
            raise ValueError(f"{spec.api_name} 未生成任何报告期，请检查日期范围")
        print(
            f"[event] {spec.table} 按报告期拉取 periods={len(periods)}",
            flush=True,
        )
        frames = []
        progress = tqdm(periods, desc=f"拉取{spec.table}", unit="期")
        for period in progress:
            progress.set_postfix_str(str(period))
            part = self._fetch_event_pages(
                getter, {"period": period}, field_list, pause, spec.page_rows
            )
            if part is not None and not part.empty:
                frames.append(part)
                tqdm.write(f"[event] {spec.table} period={period} -> {len(part)} 行")
            else:
                tqdm.write(f"[event] {spec.table} period={period} -> 空")
            if pause > 0:
                time.sleep(pause)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        print(f"[event] {spec.table} 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_event_page(
        self, getter: Callable, paras: Dict, fields: Optional[List[str]]
    ) -> pd.DataFrame:
        """单页拉取。分钟、小时、日额度都等待后重试，不中断这次回补。"""
        last_error = None
        ordinary_attempts = 0
        rate_waits = 0
        while ordinary_attempts < 6:
            try:
                result = getter(**paras, fields=fields)
                return result if result is not None else pd.DataFrame()
            except _EventWindowTruncated:
                raise
            except Exception as exc:
                last_error = exc
                text = _tushare_error_text(exc)
                if _is_offset_ceiling(text):
                    raise _EventWindowTruncated(
                        f"{paras} offset 超过 {_EVENT_OFFSET_LIMIT}，需要缩小日期区间"
                    ) from exc
                if _is_rate_limit_error(exc):
                    if rate_waits >= 40:
                        raise RuntimeError(
                            f"{paras} 频率超限，已等待 {rate_waits} 次: {text}"
                        ) from exc
                    seconds = _event_rate_wait_seconds(text)
                    print(
                        f"[event] 频率超限，等待 {seconds / 60:.0f} 分钟后重试 "
                        f"paras={paras}",
                        flush=True,
                    )
                    time.sleep(seconds)
                    rate_waits += 1
                    continue
                if _is_non_retryable_tushare_error(exc):
                    raise
                ordinary_attempts += 1
                time.sleep(1)
        raise RuntimeError(
            f"Tushare 事件接口失败 paras={paras}: {last_error}"
        ) from last_error

    def _fetch_event_pages(
        self,
        getter: Callable,
        paras: Dict,
        fields: Optional[List[str]],
        pause: float,
        page_rows: int,
    ) -> pd.DataFrame:
        """只有顶满单次上限才翻页。下一页为空表示这个区间已经取完。"""
        frames = []
        offset = 0
        seen = set()
        while True:
            current = dict(paras)
            if offset:
                current["offset"] = offset
            part = self._fetch_event_page(getter, current, fields)
            if part is None or part.empty:
                break
            marker = tuple(part.iloc[0].astype(str).tolist())
            if marker in seen:
                raise _EventWindowTruncated(f"{paras} 翻页返回了重复记录")
            seen.add(marker)
            frames.append(part)
            if len(part) < page_rows:
                break
            offset += len(part)
            if offset > _EVENT_OFFSET_LIMIT:
                raise _EventWindowTruncated(
                    f"{paras} 下一页 offset={offset} 超过 {_EVENT_OFFSET_LIMIT}"
                )
            if pause > 0:
                time.sleep(pause)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _fetch_event_ranges(
        self,
        getter: Callable,
        field_list: List[str],
        start_date: str,
        end_date: str,
        chunk_days: int,
        min_interval: float,
        page_rows: int,
        label: str,
    ) -> pd.DataFrame:
        """按日期区间取，区间内用 offset 翻满页。超过 offset 上限时把区间拆开。"""
        if not start_date or not end_date:
            raise ValueError(f"{label} 按区间拉取需要 start_date/end_date")
        today = pd.Timestamp.today().normalize()
        start = pd.Timestamp(start_date).normalize()
        end = min(pd.Timestamp(end_date).normalize(), today)
        if start > end:
            return pd.DataFrame()
        width = max(int(chunk_days), 1)
        print(
            f"[event] {label} 按区间拉取 {start.strftime('%Y%m%d')}->{end.strftime('%Y%m%d')} "
            f"chunk={width} 页宽={page_rows} 间隔={min_interval:.0f}s",
            flush=True,
        )
        frames = []
        cursor = start
        first = True

        def _pull(span_start: pd.Timestamp, span_end: pd.Timestamp) -> pd.DataFrame:
            nonlocal first
            if not first and min_interval > 0:
                time.sleep(min_interval)
            first = False
            paras = {
                "start_date": span_start.strftime("%Y%m%d"),
                "end_date": span_end.strftime("%Y%m%d"),
            }
            try:
                part = self._fetch_event_pages(
                    getter, paras, field_list, min_interval, page_rows
                )
            except _EventWindowTruncated:
                if span_start >= span_end:
                    raise RuntimeError(
                        f"{label} {paras['start_date']} 单日仍被截断，拒绝写入残缺区间"
                    ) from None
                mid = span_start + (span_end - span_start) / 2
                mid = pd.Timestamp(mid).normalize()
                if mid >= span_end:
                    mid = span_end - pd.Timedelta(days=1)
                print(
                    f"[event] {label} {paras['start_date']}->{paras['end_date']} "
                    f"翻页到顶，拆成 {span_start.strftime('%Y%m%d')}->{mid.strftime('%Y%m%d')}",
                    flush=True,
                )
                left = _pull(span_start, mid)
                right = _pull(mid + pd.Timedelta(days=1), span_end)
                pieces = [frame for frame in (left, right) if frame is not None and not frame.empty]
                if not pieces:
                    return pd.DataFrame()
                return pd.concat(pieces, ignore_index=True)
            rows = 0 if part is None or part.empty else len(part)
            print(
                f"[event] {label} {paras['start_date']}->{paras['end_date']} -> {rows} 行",
                flush=True,
            )
            return part if rows else pd.DataFrame()

        while cursor <= end:
            chunk_end = min(cursor + pd.Timedelta(days=width - 1), end)
            part = _pull(cursor, chunk_end)
            if part is not None and not part.empty:
                frames.append(part)
            cursor = chunk_end + pd.Timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        print(f"[event] {label} 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_event_days(
        self,
        getter: Callable,
        param_name: str,
        field_list: List[str],
        start_date: str,
        end_date: str,
        pause: float,
        page_rows: int,
        label: str,
    ) -> pd.DataFrame:
        if not start_date or not end_date:
            raise ValueError(f"{label} 按日拉取需要 start_date/end_date")
        today = pd.Timestamp.today().normalize()
        start = pd.Timestamp(start_date).normalize()
        end = min(pd.Timestamp(end_date).normalize(), today)
        if start > end:
            return pd.DataFrame()
        days = pd.date_range(start, end, freq="D")
        print(
            f"[event] {label} 按日拉取 days={len(days)} "
            f"{start.strftime('%Y%m%d')}->{end.strftime('%Y%m%d')}",
            flush=True,
        )
        frames = []
        progress = tqdm(days, desc=f"拉取{label}", unit="日")
        for day in progress:
            stamp = day.strftime("%Y%m%d")
            progress.set_postfix_str(stamp)
            part = self._fetch_event_pages(
                getter, {param_name: stamp}, field_list, pause, page_rows
            )
            if part is not None and not part.empty:
                frames.append(part)
                tqdm.write(f"[event] {label} {param_name}={stamp} -> {len(part)} 行")
            if pause > 0:
                time.sleep(pause)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        print(f"[event] {label} 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_fina_indicator(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """财务指标：全量按报告期，日常按公告日。"""
        field_list = self._as_field_list(fields) or list(FINANCIAL_TUSHARE_FIELDS)
        # 与原版缓存一致：保留接口返回顺序中的首次命中，勿 keep=last 覆盖
        if "update_flag" not in field_list:
            field_list = list(field_list) + ["update_flag"]
        use_vip = config.get("use_vip", True)
        getter = None
        if use_vip and hasattr(self.pro, "fina_indicator_vip"):
            getter = self.pro.fina_indicator_vip
        if str(config.get("fetch_mode") or "period").lower() == "ann_date":
            return self._finalize_fina_frame(
                self._fetch_fina_by_ann_dates(
                    getter, field_list, start_date, end_date, config
                )
            )
        as_of = min(
            pd.Timestamp(end_date) if end_date else pd.Timestamp.today(),
            pd.Timestamp.today(),
        ).normalize()
        periods = config.get("periods") or self._quarter_periods(
            start_date, end_date, as_of=as_of.strftime("%Y-%m-%d")
        )
        if not periods:
            raise ValueError("fina_indicator 未生成任何报告期，请检查日期范围")
        print(
            f"[financial] 按报告期拉取 periods={len(periods)} use_vip={getter is not None}",
            flush=True,
        )
        frames = []
        progress = tqdm(periods, desc="拉取财务指标", unit="期")
        for period in progress:
            progress.set_postfix_str(str(period))
            part = self._fetch_fina_period(getter, period, field_list, as_of)
            if part is not None and not part.empty:
                frames.append(part)
                tqdm.write(f"[financial] period={period} -> {len(part)} 行")
            else:
                tqdm.write(f"[financial] period={period} -> 空")
            time.sleep(float(config.get("pause_seconds", 1.0)))
        if not frames:
            return pd.DataFrame()
        return self._finalize_fina_frame(pd.concat(frames, ignore_index=True))

    @staticmethod
    def _finalize_fina_frame(result: pd.DataFrame) -> pd.DataFrame:
        if result is None or result.empty:
            return pd.DataFrame()
        if "update_flag" not in result.columns:
            result["update_flag"] = ""
        else:
            result["update_flag"] = (
                result["update_flag"].fillna("").astype(str).replace({"nan": ""})
            )
        subset = [
            c
            for c in ["ts_code", "end_date", "ann_date", "update_flag"]
            if c in result.columns
        ]
        if subset:
            result = result.drop_duplicates(subset=subset, keep="first")
        print(f"[financial] 财务指标合计 {len(result)} 行", flush=True)
        return result

    def _fetch_fina_by_ann_dates(
        self,
        getter: Optional[Callable],
        field_list: List[str],
        start_date: str,
        end_date: str,
        config: Dict,
    ) -> pd.DataFrame:
        """按公告日逐日拉取，覆盖提前/延期披露和更正。"""
        if getter is None:
            raise RuntimeError("fina_indicator_vip 不可用，无法按公告日拉全市场")
        today = pd.Timestamp.today().normalize()
        start = pd.Timestamp(start_date).normalize()
        end = min(pd.Timestamp(end_date).normalize(), today)
        if start > end:
            return pd.DataFrame()
        days = pd.date_range(start, end, freq="D")
        print(
            f"[financial] 按公告日拉取 days={len(days)} "
            f"{start.strftime('%Y%m%d')}->{end.strftime('%Y%m%d')}",
            flush=True,
        )
        frames = []
        progress = tqdm(days, desc="拉取财务公告", unit="日")
        for day in progress:
            stamp = day.strftime("%Y%m%d")
            progress.set_postfix_str(stamp)
            try:
                part = self.getTushareInfo(getter, {"ann_date": stamp}, field_list)
            except Exception as exc:
                if not _is_unreleased_period_error(exc):
                    raise
                tqdm.write(f"[financial] ann_date={stamp} -> 无公告")
                part = pd.DataFrame()
            if part is not None and not part.empty:
                frames.append(part)
                tqdm.write(f"[financial] ann_date={stamp} -> {len(part)} 行")
            time.sleep(float(config.get("pause_seconds", 1.0)))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _fetch_fina_period(
        self,
        getter: Optional[Callable],
        period: str,
        field_list: List[str],
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """按报告期翻页拉取。普通接口必须 ts_code，空结果不能降级。"""
        if getter is None:
            raise RuntimeError("fina_indicator_vip 不可用，无法按报告期拉全市场")
        deadline = self._official_publish_date(pd.Timestamp(period))
        past_deadline = deadline <= as_of
        try:
            part = self.getTushareInfo(getter, {"period": period}, field_list)
        except Exception as exc:
            if _is_unreleased_period_error(exc):
                if past_deadline:
                    raise RuntimeError(
                        f"period={period} 已过法定披露截止日 {deadline.date()}，"
                        "但 VIP 仍要求 ts_code，拒绝写成空期"
                    ) from exc
                tqdm.write(f"[financial] period={period} -> 披露窗口内尚无全市场截面")
                return pd.DataFrame()
            raise
        if part is None or part.empty:
            if past_deadline:
                raise RuntimeError(
                    f"period={period} 已过法定披露截止日 {deadline.date()}，"
                    "但 VIP 返回空表，拒绝整表替换掉该期"
                )
            return pd.DataFrame()
        return part

    def _fetch_financial_statement(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """资产负债表 / 现金流量表瘦身切片。默认 VIP 按报告期，限股票走普通接口。"""
        api_name = str(config.get("api_name") or "")
        table = statement_table_for_api(api_name)
        if table is None:
            raise ValueError(f"不支持的财务报表接口: {api_name}")
        value_fields = STATEMENT_VALUE_FIELDS[table]
        field_list = list(
            canonicalize_statement_fields(
                self._as_field_list(fields) or STATEMENT_TUSHARE_FIELDS[table]
            )
        )
        if "update_flag" not in field_list:
            field_list.append("update_flag")
        codes = self._codes_from_config(config)
        use_vip = bool(config.get("use_vip", not codes))
        getter = self._statement_getter(table, use_vip=use_vip and not codes)
        label = table
        if codes:
            result = self._fetch_statement_by_codes(
                getter, codes, field_list, start_date, end_date, config
            )
        elif str(config.get("fetch_mode") or "period").lower() == "ann_date":
            result = self._fetch_fina_by_ann_dates(
                getter, field_list, start_date, end_date, config
            )
        else:
            as_of = min(
                pd.Timestamp(end_date) if end_date else pd.Timestamp.today(),
                pd.Timestamp.today(),
            ).normalize()
            periods = config.get("periods") or self._quarter_periods(
                start_date, end_date, as_of=as_of.strftime("%Y-%m-%d")
            )
            if not periods:
                raise ValueError(f"{api_name} 未生成任何报告期，请检查日期范围")
            print(
                f"[financial] {label} 按报告期拉取 periods={len(periods)} "
                f"use_vip={use_vip}",
                flush=True,
            )
            frames = []
            progress = tqdm(periods, desc=f"拉取{label}", unit="期")
            for period in progress:
                progress.set_postfix_str(str(period))
                part = self._fetch_fina_period(getter, period, field_list, as_of)
                if part is not None and not part.empty:
                    frames.append(part)
                    tqdm.write(f"[financial] {label} period={period} -> {len(part)} 行")
                else:
                    tqdm.write(f"[financial] {label} period={period} -> 空")
                time.sleep(float(config.get("pause_seconds", 1.0)))
            result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        finalized = self._finalize_fina_frame(result)
        return self._require_statement_columns(finalized, value_fields, api_name)

    def _statement_getter(self, table: str, use_vip: bool) -> Callable:
        vip_name = f"{table}_vip"
        if use_vip:
            getter = getattr(self.pro, vip_name, None)
            if getter is None:
                raise RuntimeError(f"{vip_name} 不可用，无法按报告期拉全市场")
            return getter
        getter = getattr(self.pro, table, None)
        if getter is None:
            raise RuntimeError(f"当前 tushare 没有 {table} 接口")
        return getter

    def _fetch_statement_by_codes(
        self,
        getter: Callable,
        codes: List[str],
        field_list: List[str],
        start_date: str,
        end_date: str,
        config: Dict,
    ) -> pd.DataFrame:
        periods = config.get("periods")
        frames = []
        pause = float(config.get("pause_seconds", 0.35))
        if periods:
            jobs = [(code, str(period)) for code in codes for period in periods]
            progress = tqdm(jobs, desc="拉取报表(股票)", unit="次")
            for code, period in progress:
                progress.set_postfix_str(f"{code} {period}")
                part = self.getTushareInfo(
                    getter, {"ts_code": code, "period": period}, field_list
                )
                if part is not None and not part.empty:
                    frames.append(part)
                time.sleep(pause)
        else:
            start = "".join(ch for ch in str(start_date) if ch.isdigit())[:8]
            end = "".join(ch for ch in str(end_date) if ch.isdigit())[:8]
            progress = tqdm(codes, desc="拉取报表(股票)", unit="股")
            for code in progress:
                progress.set_postfix_str(code)
                part = self.getTushareInfo(
                    getter,
                    {"ts_code": code, "start_date": start, "end_date": end},
                    field_list,
                )
                if part is not None and not part.empty:
                    frames.append(part)
                time.sleep(pause)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _codes_from_config(config: Dict) -> List[str]:
        raw = config.get("symbols") or config.get("ts_codes") or config.get("ts_code")
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            items = [raw]
        else:
            items = list(raw)
        return [str(item).strip().upper() for item in items if str(item).strip()]

    @staticmethod
    def _require_statement_columns(
        frame: pd.DataFrame, value_fields: Sequence[str], api_name: str
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame() if frame is None else frame
        missing = [name for name in value_fields if name not in frame.columns]
        if missing:
            raise RuntimeError(
                f"{api_name} 返回表缺少字段 {missing}，实际列={list(frame.columns)}。"
                "多半是字段名和 Tushare 文档不一致（筹资净额为 n_cash_flows_fnc_act）。"
            )
        return frame

    def _fetch_index_weight(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """指数成分权重：https://tushare.pro/document/2?doc_id=96"""
        index_list = config.get("index_list") or []
        if not index_list:
            raise ValueError("index_weight 配置缺少 index_list")
        if not start_date or not end_date:
            raise ValueError("index_weight 需要 start_date/end_date")
        field_list = self._as_field_list(fields) or [
            "index_code",
            "con_code",
            "trade_date",
            "weight",
        ]
        self._assert_index_weight_available(field_list, end_date)
        probe_start, probe_end = self._index_weight_probe_window(end_date)
        optional = {
            str(code).strip()
            for code in (config.get("optional_index_codes") or [])
        }
        request_by_config: Dict[str, str] = {}
        unavailable: List[str] = []
        for index_code in index_list:
            resolved = self._resolve_index_weight_request_code(
                str(index_code), field_list, probe_start, probe_end
            )
            if resolved is None:
                unavailable.append(str(index_code))
            else:
                request_by_config[str(index_code)] = resolved
        required_empty = [code for code in unavailable if code not in optional]
        if required_empty:
            raise RuntimeError(
                f"index_weight 以下指数全程 0 行: {required_empty}。"
                "接口无权限或代码无效时通常不报错、只给空表；"
                "其它指数已有数据时不能把缺表当成成功。"
            )
        for code in unavailable:
            print(
                f"[index_constituent] {code} 跳过（optional，探测 {probe_start}~{probe_end} 无成分）",
                flush=True,
            )
        # 按月切片，避免单次过大
        month_starts = pd.date_range(
            pd.Timestamp(start_date).replace(day=1),
            pd.Timestamp(end_date),
            freq="MS",
        )
        frames = []
        hits: Dict[str, int] = {str(code): 0 for code in request_by_config}
        tasks = [(code, month) for code in request_by_config for month in month_starts]
        print(
            f"[index_constituent] 拉取 index_weight: "
            f"{len(request_by_config)} 指数 x {len(month_starts)} 月 = {len(tasks)} 次",
            flush=True,
        )
        pause = float(config.get("pause_seconds", 0.2))
        progress = tqdm(tasks, desc="拉取指数成分", unit="月")
        for index_code, month in progress:
            month_start = month.strftime("%Y%m%d")
            month_end = (month + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")
            if month_start < start_date:
                month_start = start_date
            if month_end > end_date:
                month_end = end_date
            request_code = request_by_config[index_code]
            progress.set_postfix_str(f"{index_code} {month_start}")
            part = self._call_api(
                self.pro.index_weight,
                {
                    "index_code": request_code,
                    "start_date": month_start,
                    "end_date": month_end,
                },
                field_list,
            )
            if part is not None and not part.empty:
                part = part.copy()
                if request_code != index_code and "index_code" in part.columns:
                    part["index_code"] = index_code
                frames.append(part)
                hits[str(index_code)] += len(part)
            time.sleep(pause)
        for code, rows in hits.items():
            req = request_by_config[code]
            alias = "" if req == code else f"（请求 {req}）"
            print(f"[index_constituent] {code}{alias} -> {rows} 行", flush=True)
        if not frames:
            # 月初增量经常还没有本月快照；探测已证明接口可用，交给安装器当「无新数据」。
            print(
                f"[index_constituent] {start_date}~{end_date} 无新快照（成分权重按月发布，不是每个交易日）",
                flush=True,
            )
            return pd.DataFrame()
        empty_codes = [code for code, rows in hits.items() if rows == 0]
        if empty_codes:
            raise RuntimeError(
                f"index_weight 以下指数全程 0 行: {empty_codes}。"
                "接口无权限或代码无效时通常不报错、只给空表；"
                "其它指数已有数据时不能把缺表当成成功。"
            )
        result = pd.concat(frames, ignore_index=True)
        subset = [
            c for c in ["index_code", "con_code", "trade_date"] if c in result.columns
        ]
        if subset:
            result = result.drop_duplicates(subset=subset, keep="last")
        print(f"[index_constituent] 合计 {len(result)} 行", flush=True)
        return result

    def _assert_index_weight_available(self, field_list: List[str], end_date: str) -> None:
        """打上一个完整自然月。本月前几天还没有新快照，不能用来判断权限。"""
        start, end_s = self._index_weight_probe_window(end_date)
        for code in INDEX_WEIGHT_PROBE_CODES:
            part = self._call_api(
                self.pro.index_weight,
                {"index_code": code, "start_date": start, "end_date": end_s},
                field_list,
            )
            rows = 0 if part is None or part.empty else len(part)
            print(
                f"[index_constituent] 探测 {code} {start}~{end_s} -> {rows} 行",
                flush=True,
            )
            if rows:
                return
        raise RuntimeError(self._index_weight_empty_message(start, end_s))

    @staticmethod
    def _index_weight_probe_window(end_date: str) -> tuple[str, str]:
        end = pd.Timestamp(end_date)
        last_month_end = end.replace(day=1) - pd.Timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start.strftime("%Y%m%d"), last_month_end.strftime("%Y%m%d")

    @classmethod
    def _index_weight_candidates(cls, code: str) -> List[str]:
        key = str(code).strip().upper()
        raw = str(code).strip()
        seen: List[str] = []
        alias = INDEX_WEIGHT_CODE_ALIAS.get(key)
        if alias:
            seen.append(alias)
        if raw not in seen:
            seen.append(raw)
        return seen

    @staticmethod
    def _index_weight_request_code(code: str) -> str:
        return TushareDataSource._index_weight_candidates(code)[0]

    def _resolve_index_weight_request_code(
        self,
        index_code: str,
        field_list: List[str],
        probe_start: str,
        probe_end: str,
    ) -> Optional[str]:
        """用上一个完整自然月探测镜像代码；全空则返回 None，避免空指数再扫 200 个月。"""
        for candidate in self._index_weight_candidates(index_code):
            part = self._call_api(
                self.pro.index_weight,
                {
                    "index_code": candidate,
                    "start_date": probe_start,
                    "end_date": probe_end,
                },
                field_list,
            )
            rows = 0 if part is None or part.empty else len(part)
            print(
                f"[index_constituent] 解析 {index_code} -> {candidate} "
                f"{probe_start}~{probe_end} -> {rows} 行",
                flush=True,
            )
            if rows:
                return candidate
        return None

    @staticmethod
    def _index_weight_empty_message(start: str, end: str) -> str:
        return (
            f"Tushare index_weight 在 {start}~{end} 返回空表。"
            "接口无权限时通常不报错、只给空 DataFrame。"
            "请确认积分 >= 2000 且开通了指数成分权限；"
            "若 HTTP 走了 127.0.0.1:7890 一类本地代理，超时后也可能拿到空响应，"
            "可对 tushare.pro 直连后再跑。"
        )

    @staticmethod
    def _official_publish_date(period: pd.Timestamp) -> pd.Timestamp:
        """A 股定期报告法定披露截止日，不是报告期本身。

        年报：次年 4 月 30 日；一季报：当年 4 月 30 日；
        半年报：当年 8 月 31 日；三季报：当年 10 月 31 日。
        """
        stamp = pd.Timestamp(period)
        month = int(stamp.month)
        year = int(stamp.year)
        if month == 3:
            return pd.Timestamp(year=year, month=4, day=30)
        if month == 6:
            return pd.Timestamp(year=year, month=8, day=31)
        if month == 9:
            return pd.Timestamp(year=year, month=10, day=31)
        if month == 12:
            return pd.Timestamp(year=year + 1, month=4, day=30)
        raise ValueError(f"不是标准季报/年报期末: {stamp.date()}")

    @staticmethod
    def _quarter_periods(
        start_date: str,
        end_date: str,
        as_of: Optional[str] = None,
    ) -> List[str]:
        """窗口内已结束的报告期。

        及时性看报告期期末是否已过，不把法定截止日当开拉条件。
        历史窗口下界仍用法定披露日：2005 年起的任务要包含 2004 年报
        （2005-04-30 才正式发布），不能拿行情 start_date 去比期末。
        """
        if not start_date or not end_date:
            end = pd.Timestamp.today()
            start = end - pd.DateOffset(years=5)
        else:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
        today = (
            pd.Timestamp(as_of).normalize()
            if as_of is not None
            else pd.Timestamp.today().normalize()
        )
        as_of_date = min(end, today)
        periods = []
        year = int(start.year) - 1
        last_year = int(as_of_date.year) + 1
        while year <= last_year:
            for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
                period = pd.Timestamp(year=year, month=month, day=day)
                published = TushareDataSource._official_publish_date(period)
                if period <= as_of_date and published >= start:
                    periods.append(period.strftime("%Y%m%d"))
            year += 1
        return periods

    def _fetch_fund_basic(self, config: Dict, fields) -> pd.DataFrame:
        """场内基金列表：https://tushare.pro/document/2?doc_id=19"""
        field_list = self._as_field_list(fields) or list(ETF_TUSHARE_BASIC_FIELDS)
        market = config.get("market") or "E"
        statuses = config.get("status") or ["L", "D"]
        if isinstance(statuses, str):
            statuses = [statuses]
        frames = []
        print(f"[etf_info] 拉取 fund_basic market={market} status={statuses}", flush=True)
        for status in statuses:
            part = self._call_api(
                self.pro.fund_basic,
                {"market": market, "status": status},
                field_list,
            )
            rows = 0 if part is None or part.empty else len(part)
            print(f"[etf_info] fund_basic status={status} -> {rows} 行", flush=True)
            if part is None or part.empty:
                continue
            part = part.copy()
            if "status" not in part.columns:
                part["status"] = status
            if "market" not in part.columns:
                part["market"] = market
            frames.append(part)
            time.sleep(0.2)
        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=field_list)
        )
        if "ts_code" in result.columns and not result.empty:
            result["ts_code"] = result["ts_code"].astype(str).str.upper()
            result = result.drop_duplicates(subset=["ts_code"], keep="last")
        extras = [
            str(code).strip().upper()
            for code in (config.get("extra_ts_codes") or ETF_EXTRA_TS_CODES)
            if str(code).strip()
        ]
        have = set(result["ts_code"].astype(str)) if "ts_code" in result.columns else set()
        missing = [code for code in extras if code not in have]
        extra_frames = []
        for code in missing:
            part = self._call_api(self.pro.fund_basic, {"ts_code": code}, field_list)
            rows = 0 if part is None or part.empty else len(part)
            print(f"[etf_info] fund_basic extra {code} -> {rows} 行", flush=True)
            if part is None or part.empty:
                extra_frames.append(
                    pd.DataFrame(
                        {
                            "ts_code": [code],
                            "name": [""],
                            "management": [""],
                            "fund_type": [""],
                            "market": [market],
                            "status": ["L"],
                            "list_date": [""],
                            "delist_date": [""],
                        }
                    )
                )
            else:
                extra_frames.append(part)
            time.sleep(0.2)
        if extra_frames:
            result = pd.concat([result, *extra_frames], ignore_index=True)
            if "ts_code" in result.columns:
                result["ts_code"] = result["ts_code"].astype(str).str.upper()
                result = result.drop_duplicates(subset=["ts_code"], keep="first")
        result = self._filter_fund_codes(result, config)
        print(f"[etf_info] fund_basic 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_fund_daily(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """场内基金日线：https://tushare.pro/document/2?doc_id=127"""
        field_list = self._as_field_list(fields) or list(ETF_TUSHARE_DAILY_FIELDS)
        result = self._fetch_daily_by_day(
            self.pro.fund_daily, field_list, start_date, end_date, "fund_daily"
        )
        return self._filter_fund_codes(result, config)

    @staticmethod
    def _filter_fund_codes(frame: pd.DataFrame, config: Dict) -> pd.DataFrame:
        if frame is None or frame.empty or "ts_code" not in frame.columns:
            return frame
        wanted = config.get("etf_list") or config.get("symbols")
        if not wanted:
            return frame
        codes = {str(code).strip().upper() for code in wanted if str(code).strip()}
        out = frame.copy()
        out["ts_code"] = out["ts_code"].astype(str).str.upper()
        return out.loc[out["ts_code"].isin(codes)].copy()

    @staticmethod
    def _ended_quarter_periods(as_of: pd.Timestamp, count: int = 4) -> List[str]:
        """最近 count 个已结束报告期，增量时回补晚披露和更正。"""
        as_of_date = pd.Timestamp(as_of).normalize()
        ended: List[pd.Timestamp] = []
        year = int(as_of_date.year) - 2
        while year <= int(as_of_date.year):
            for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
                period = pd.Timestamp(year=year, month=month, day=day)
                if period <= as_of_date:
                    ended.append(period)
            year += 1
        ended.sort()
        return [stamp.strftime("%Y%m%d") for stamp in ended[-int(count) :]]

    def _fetch_fund_basic(self, config: Dict, fields) -> pd.DataFrame:
        """场内基金列表：https://tushare.pro/document/2?doc_id=19"""
        field_list = self._as_field_list(fields) or list(ETF_TUSHARE_BASIC_FIELDS)
        market = config.get("market") or "E"
        statuses = config.get("status") or ["L", "D"]
        if isinstance(statuses, str):
            statuses = [statuses]
        frames = []
        print(f"[etf_info] 拉取 fund_basic market={market} status={statuses}", flush=True)
        for status in statuses:
            part = self._call_api(
                self.pro.fund_basic,
                {"market": market, "status": status},
                field_list,
            )
            rows = 0 if part is None or part.empty else len(part)
            print(f"[etf_info] fund_basic status={status} -> {rows} 行", flush=True)
            if part is None or part.empty:
                continue
            part = part.copy()
            if "status" not in part.columns:
                part["status"] = status
            if "market" not in part.columns:
                part["market"] = market
            frames.append(part)
            time.sleep(0.2)
        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=field_list)
        )
        if "ts_code" in result.columns and not result.empty:
            result["ts_code"] = result["ts_code"].astype(str).str.upper()
            result = result.drop_duplicates(subset=["ts_code"], keep="last")
        extras = [
            str(code).strip().upper()
            for code in (config.get("extra_ts_codes") or ETF_EXTRA_TS_CODES)
            if str(code).strip()
        ]
        have = set(result["ts_code"].astype(str)) if "ts_code" in result.columns else set()
        missing = [code for code in extras if code not in have]
        extra_frames = []
        for code in missing:
            part = self._call_api(self.pro.fund_basic, {"ts_code": code}, field_list)
            rows = 0 if part is None or part.empty else len(part)
            print(f"[etf_info] fund_basic extra {code} -> {rows} 行", flush=True)
            if part is None or part.empty:
                extra_frames.append(
                    pd.DataFrame(
                        {
                            "ts_code": [code],
                            "name": [""],
                            "management": [""],
                            "fund_type": [""],
                            "market": [market],
                            "status": ["L"],
                            "list_date": [""],
                            "delist_date": [""],
                        }
                    )
                )
            else:
                extra_frames.append(part)
            time.sleep(0.2)
        if extra_frames:
            result = pd.concat([result, *extra_frames], ignore_index=True)
            if "ts_code" in result.columns:
                result["ts_code"] = result["ts_code"].astype(str).str.upper()
                result = result.drop_duplicates(subset=["ts_code"], keep="first")
        result = self._filter_fund_codes(result, config)
        print(f"[etf_info] fund_basic 合计 {len(result)} 行", flush=True)
        return result

    def _fetch_fund_daily(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """场内基金日线：https://tushare.pro/document/2?doc_id=127"""
        field_list = self._as_field_list(fields) or list(ETF_TUSHARE_DAILY_FIELDS)
        result = self._fetch_daily_by_day(
            self.pro.fund_daily, field_list, start_date, end_date, "fund_daily"
        )
        return self._filter_fund_codes(result, config)

    @staticmethod
    def _filter_fund_codes(frame: pd.DataFrame, config: Dict) -> pd.DataFrame:
        if frame is None or frame.empty or "ts_code" not in frame.columns:
            return frame
        wanted = config.get("etf_list") or config.get("symbols")
        if not wanted:
            return frame
        codes = {str(code).strip().upper() for code in wanted if str(code).strip()}
        out = frame.copy()
        out["ts_code"] = out["ts_code"].astype(str).str.upper()
        return out.loc[out["ts_code"].isin(codes)].copy()

    @staticmethod
    def _as_field_list(fields) -> Optional[List[str]]:
        if fields is None:
            return None
        if isinstance(fields, str):
            return [item.strip() for item in fields.split(',') if item.strip()]
        return list(fields)

    def _call_api(
        self, getter: Callable, paras: Dict, fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """无 offset 的单次/重试调用，适合行业维表接口。

        重试耗尽后抛错，绝不返回空表 —— 否则调用方无法区分
        「接口挂了」和「本来就没有数据」，会把残缺结果当成功写入库。
        """
        last_error = None
        for attempt in range(61):
            try:
                if fields:
                    result = getter(**paras, fields=fields)
                else:
                    result = getter(**paras)
                return result if result is not None else pd.DataFrame()
            except Exception as exc:
                last_error = exc
                if _is_non_retryable_tushare_error(exc):
                    raise
                if attempt in (0, 5, 15, 30, 60):
                    print(
                        f"[api] 重试 {attempt + 1}/61 paras={paras} err={exc}",
                        flush=True,
                    )
                time.sleep(1)
        raise RuntimeError(
            f"Tushare 接口连续 61 次失败 paras={paras}: {last_error}"
        ) from last_error

    def getTushareInfo(
        self, getter: Callable, paras: Dict, fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        从tushare获取数据（按 offset 翻页）
        :param getter: tushare的接口函数
        :param paras: 接口函数的参数
        :param fields: 需要获取的字段
        :return->pd.DataFrame: 获取到的数据
        """
        df = pd.DataFrame()
        last_page_len = 0
        while True:
            tmp = None
            last_error = None
            for _ in range(61):
                try:
                    # 创建参数字典的副本，避免修改原始参数
                    current_paras = paras.copy()
                    current_paras["offset"] = len(df)
                    tmp = getter(**current_paras, fields=fields)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if _is_non_retryable_tushare_error(exc):
                        raise
                    time.sleep(1)
            if last_error is not None:
                # 整页始终拿不到：抛错而不是把已取到的部分当完整结果返回
                raise RuntimeError(
                    f"Tushare 接口连续 61 次失败 paras={paras} offset={len(df)}: "
                    f"{last_error}"
                ) from last_error
            if tmp is None or len(tmp) == 0:
                if last_page_len in TUSHARE_PAGE_CAPS:
                    raise RuntimeError(
                        f"Tushare 分页疑似截断: 上一页 {last_page_len} 行"
                        f"（常见上限 {sorted(TUSHARE_PAGE_CAPS)}）且下一页为空 "
                        f"paras={paras} offset={len(df)}"
                    )
                return df
            last_page_len = len(tmp)
            df = tmp if len(df) == 0 else pd.concat([df, tmp], ignore_index=True)
