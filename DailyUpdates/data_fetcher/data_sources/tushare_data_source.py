#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare数据源类，专门处理Tushare API调用
"""

import tushare as ts
import pandas as pd
from typing import Any, Callable, Dict, List, Optional
import os
import time
from tqdm import tqdm
from DailyUpdates.data_fetcher.data_source_base import DataSourceBase, FetchSlice
from DailyUpdates.storage.financial_schema import FINANCIAL_TUSHARE_FIELDS

# index_weight 请求代码与配置代码可以不同，写入仍用配置里的代码。
# 000300.SH → 399300.SZ：https://github.com/waditu/tushare/issues/801
# 000985.SH → 000985.CSI：index_basic(market=CSI)；.SH / 399985.SZ 无成分。
INDEX_WEIGHT_CODE_ALIAS = {
    "000300.SH": "399300.SZ",
    "000985.SH": "000985.CSI",
}
INDEX_WEIGHT_PROBE_CODES = ("000016.SH", "399300.SZ", "000905.SH")


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
                # 获取所有指定指数的数据
                all_data = []
                for index_code in index_list:
                    paras = {'ts_code': index_code, 'start_date': start_date, 'end_date': end_date}
                    index_data = self.getTushareInfo(self.pro.index_daily, paras, fields)
                    if not index_data.empty:
                        all_data.append(index_data)
                
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
                else:
                    return pd.DataFrame()
            elif api_name == 'index_dailybasic':
                # 获取所有指定指数的数据
                all_data = []
                for index_code in index_list:
                    paras = {'ts_code': index_code, 'start_date': start_date, 'end_date': end_date}
                    index_data = self.getTushareInfo(self.pro.index_dailybasic, paras, fields)
                    if not index_data.empty:
                        all_data.append(index_data)
                
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
                else:
                    return pd.DataFrame()
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
            raise ValueError(f"不支持的财务API接口: {api_name}")

        elif data_type == 'index_constituent':
            if api_name == 'index_weight':
                return self._fetch_index_weight(config, fields, start_date, end_date)
            raise ValueError(f"不支持的指数成分API接口: {api_name}")
        
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

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

    def _fetch_fina_indicator(
        self, config: Dict, fields, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """财务指标：优先按报告期调用 fina_indicator_vip。"""
        field_list = self._as_field_list(fields) or list(FINANCIAL_TUSHARE_FIELDS)
        # 与原版缓存一致：保留接口返回顺序中的首次命中，勿 keep=last 覆盖
        if "update_flag" not in field_list:
            field_list = list(field_list) + ["update_flag"]
        use_vip = config.get("use_vip", True)
        periods = config.get("periods") or self._quarter_periods(start_date, end_date)
        if not periods:
            raise ValueError("fina_indicator 未生成任何报告期，请检查日期范围")
        print(
            f"[financial] 拉取财务指标 periods={len(periods)} use_vip={use_vip}",
            flush=True,
        )
        frames = []
        progress = tqdm(periods, desc="拉取财务指标", unit="期")
        for period in progress:
            progress.set_postfix_str(str(period))
            part = pd.DataFrame()
            if use_vip and hasattr(self.pro, "fina_indicator_vip"):
                part = self._call_api(
                    self.pro.fina_indicator_vip, {"period": period}, field_list
                )
            if part is None or part.empty:
                # 降级：按 period 调普通接口（部分权限可用）
                part = self._call_api(
                    self.pro.fina_indicator, {"period": period}, field_list
                )
            if part is not None and not part.empty:
                frames.append(part)
                tqdm.write(f"[financial] period={period} -> {len(part)} 行")
            else:
                tqdm.write(f"[financial] period={period} -> 空")
            time.sleep(float(config.get("pause_seconds", 1.0)))
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        if "update_flag" not in result.columns:
            result["update_flag"] = ""
        else:
            result["update_flag"] = (
                result["update_flag"].fillna("").astype(str).replace({"nan": ""})
            )
        # 仅去掉完全相同主键的重复；同公告日不同 update_flag 都保留（对齐原版缓存可存多行）
        # keep=first：与原版 to_dict 后按接口顺序首次命中一致，禁止 keep=last 乱盖
        subset = [
            c
            for c in ["ts_code", "end_date", "ann_date", "update_flag"]
            if c in result.columns
        ]
        if subset:
            result = result.drop_duplicates(subset=subset, keep="first")
        print(f"[financial] 财务指标合计 {len(result)} 行", flush=True)
        return result

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
    def _quarter_periods(start_date: str, end_date: str) -> List[str]:
        if not start_date or not end_date:
            # 默认近 5 年季度
            end = pd.Timestamp.today()
            start = end - pd.DateOffset(years=5)
        else:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
        periods = []
        year = int(start.year)
        while year <= int(end.year):
            for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
                period = pd.Timestamp(year=year, month=month, day=day)
                if start <= period <= end:
                    periods.append(period.strftime("%Y%m%d"))
            year += 1
        return periods

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
        page_caps = {6000, 8000}
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
                    time.sleep(1)
            if last_error is not None:
                # 整页始终拿不到：抛错而不是把已取到的部分当完整结果返回
                raise RuntimeError(
                    f"Tushare 接口连续 61 次失败 paras={paras} offset={len(df)}: "
                    f"{last_error}"
                ) from last_error
            if tmp is None or len(tmp) == 0:
                if last_page_len in page_caps or last_page_len >= 8000:
                    raise RuntimeError(
                        f"Tushare 分页疑似截断: 上一页 {last_page_len} 行"
                        f"（常见上限 6000/8000）且下一页为空 "
                        f"paras={paras} offset={len(df)}"
                    )
                return df
            last_page_len = len(tmp)
            df = tmp if len(df) == 0 else pd.concat([df, tmp], ignore_index=True)
