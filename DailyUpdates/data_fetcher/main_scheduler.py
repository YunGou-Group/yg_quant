#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero-argument entrypoint for the SQLite data scheduler."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "yg_quant_repo.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
else:
    raise RuntimeError("找不到 yg_quant 仓库根目录（缺少 yg_quant_repo.py）")
from yg_quant_repo import default_db_path, ensure_repo_root

ensure_repo_root()

from DailyUpdates.data_fetcher.unified_scheduler import UnifiedScheduler
from DailyUpdates.factor_updates.factor_updater import FactorUpdater
from DailyUpdates.storage.financial_schema import DAILY_BASIC_BARRA_FIELDS, FINANCIAL_TUSHARE_FIELDS


def main():
    parser = argparse.ArgumentParser(description="行情调度 + 因子更新")
    parser.add_argument(
        "--start",
        default=os.getenv("YG_QUANT_START_DATE") or None,
        help="可选起点 YYYYMMDD；空库默认仍为 20050101",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    db_path = str(default_db_path())

    tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()

    if not tushare_token:
        raise RuntimeError("请先设置 TUSHARE_TOKEN 环境变量")
        
    end_date = os.getenv("YG_QUANT_END_DATE") or datetime.now().strftime("%Y%m%d")
    dataset_config = {
        'daily': {
            'data_source': 'Tushare',  # 数据源：Tushare
            'data_type': 'daily',      # 数据类型：日频行情
            'api_name': 'daily',       # API接口名：daily
            'token': tushare_token,    # Tushare token
            'fields': ['ts_code', 'trade_date','open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'],
            'primary_key': ['ts_code', 'trade_date']
        },
        'daily_basic': {
            'data_source': 'Tushare',  # 数据源：Tushare
            'data_type': 'daily',      # 数据类型：日频数据
            'api_name': 'daily_basic', # API接口名：daily_basic
            'token': tushare_token,    # Tushare token
            'fields': ['ts_code', 'trade_date', *DAILY_BASIC_BARRA_FIELDS],
            'primary_key': ['ts_code', 'trade_date'],
            'pause_seconds': 0.15,
        },
        'index': {
            'data_source': 'Tushare',
            'data_type': 'index',
            'api_name': 'index_daily',
            'token': tushare_token,
            # 上证指数 / 深证成指 / 创业板指 / 沪深300
            'index_list': ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH'],
            'fields': [
                'ts_code', 'trade_date',
                'open', 'high', 'low', 'close',
                'pre_close', 'change', 'pct_chg', 'vol', 'amount',
            ],
            'primary_key': ['ts_code', 'trade_date'],
        },
        # 申万行业分类：https://tushare.pro/document/2?doc_id=181
        'sw_industry_classify': {
            'data_source': 'Tushare',
            'data_type': 'industry',
            'api_name': 'index_classify',
            'token': tushare_token,
            'src': 'SW2021',
            'levels': ['L1', 'L2', 'L3'],
            'fields': [
                'index_code', 'industry_name', 'parent_code',
                'level', 'industry_code', 'is_pub', 'src',
            ],
            'primary_key': ['src', 'index_code'],
        },
        # 申万行业成分：https://tushare.pro/document/2?doc_id=335
        # is_new 不传 = 拉全量历史（含调入/调出）；写入时按 src 整表替换，无需手删库。
        'sw_industry_member': {
            'data_source': 'Tushare',
            'data_type': 'industry',
            'api_name': 'index_member_all',
            'token': tushare_token,
            'src': 'SW2021',
            'fields': [
                'l1_code', 'l1_name', 'l2_code', 'l2_name',
                'l3_code', 'l3_name', 'ts_code', 'name',
                'in_date', 'out_date', 'is_new',
            ],
            'primary_key': ['src', 'ts_code', 'l3_code', 'in_date'],
        },
        # 股票基础信息（ST / 上市日 / 退市日）：https://tushare.pro/document/2?doc_id=25
        # 入库主键用 ts_code → 仓库代码（SZ000001），不用 Tushare 六位 symbol 字段
        'stock_basic': {
            'data_source': 'Tushare',
            'data_type': 'stock_info',
            'api_name': 'stock_basic',
            'token': tushare_token,
            'list_status': ['L', 'D', 'P'],
            'fields': [
                'ts_code', 'symbol', 'name', 'area',
                'industry', 'market', 'list_date', 'delist_date', 'list_status',
            ],
            'primary_key': ['ts_code'],
        },
        # 历史名称变更：https://tushare.pro/document/2?doc_id=100
        # 用于按 asof 判断当时是否 ST，而不是拿今天的名字回溯历史。
        'namechange': {
            'data_source': 'Tushare',
            'data_type': 'stock_info',
            'api_name': 'namechange',
            'token': tushare_token,
            'fields': [
                'ts_code', 'name', 'start_date', 'end_date',
                'ann_date', 'change_reason',
            ],
            'primary_key': ['ts_code', 'start_date', 'name'],
        },
        # 财务指标（按季）：Barra10 Growth/Leverage/EY 描述子；PIT 用 ann_date
        'fina_indicator': {
            'data_source': 'Tushare',
            'data_type': 'financial',
            'api_name': 'fina_indicator_vip',
            'token': tushare_token,
            'use_vip': True,
            'pause_seconds': 1.0,
            'fields': list(FINANCIAL_TUSHARE_FIELDS),
            # 同报告期可并存首发/更正两行（update_flag 区分），禁止 keep=last 覆盖
            'primary_key': ['ts_code', 'end_date', 'ann_date', 'update_flag'],
        },
        # 股票池指数成分（asof，禁止用最新快照回溯）
        'universe_index_weight': {
            'data_source': 'Tushare',
            'data_type': 'index_constituent',
            'api_name': 'index_weight',
            'token': tushare_token,
            'index_list': [
                '000300.SH',
                '000905.SH',
                '000852.SH',
                '000016.SH',
                '399006.SZ',
                '000985.SH',
                '399101.SZ',
                '000688.SH',
            ],
            'pause_seconds': 0.2,
            'fields': ['index_code', 'con_code', 'trade_date', 'weight'],
            'primary_key': ['index_code', 'con_code', 'trade_date'],
        },
        # 复权因子（写入 market_data）：后复权价 = 原始价 × adj_factor
        # 历史值不随后续分红变化，因此增量入库与 Bin 因子轴都安全。
        'adj_factor': {
            'data_source': 'Tushare',
            'data_type': 'daily',
            'api_name': 'adj_factor',
            'token': tushare_token,
            'pause_seconds': 0.12,
            'fields': ['ts_code', 'trade_date', 'adj_factor'],
            'primary_key': ['ts_code', 'trade_date'],
        },
        # 涨跌停价（写入 market_data）
        'stk_limit': {
            'data_source': 'Tushare',
            'data_type': 'daily',
            'api_name': 'stk_limit',
            'token': tushare_token,
            'fields': ['ts_code', 'trade_date', 'up_limit', 'down_limit'],
            'primary_key': ['ts_code', 'trade_date'],
        },
    }

    logger = logging.getLogger("main_scheduler")
    scheduler = UnifiedScheduler(db_path=db_path)

    result = scheduler.execute_schedule(
        dataset_config=dataset_config,
        end_date=end_date,
        tushare_token=tushare_token,
        start_date=args.start,
    )
    # 行情残缺时继续更新因子，会得到「看起来成功」但底层不完整的因子序列。
    if not result.get("success"):
        for line in result.get("execution_log") or []:
            logger.error("调度日志: %s", line)
        logger.error("行情调度失败，跳过因子更新: %s", result.get("error_message"))
        sys.exit(1)

    # 行情更新完成后：清理已删除的因子类对应数据，再增量/全量更新因子。
    # 首次全量默认写满库内可用历史；若只想从某日起算可设 YG_QUANT_FACTOR_START_DATE。
    factor_updater = FactorUpdater(
        db_path=db_path,
        max_workers=int(os.getenv("YG_QUANT_FACTOR_WORKERS", "8")),
    )
    factor_results = factor_updater.update_factors()
    logger.info("因子更新完成: %s 个任务", len(factor_results))
    if not factor_results:
        logger.error("因子阶段 0 个任务，视为失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
