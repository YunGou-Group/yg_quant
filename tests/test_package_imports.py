#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开入口与重命名后的模块能按类名导入。"""

from __future__ import annotations


def test_storage_and_updater_imports():
    from DailyUpdates.storage import BinStorage, SQLiteStorage
    from DailyUpdates.factor_updates.base_factor import BaseFactor
    from DailyUpdates.factor_updates.factor_updater import FactorUpdater
    from DailyUpdates.data_fetcher.unified_scheduler import UnifiedScheduler
    from DailyUpdates.data_fetcher.data_sources.akshare_data_source import AkshareDataSource
    from DailyUpdates.data_fetcher.data_sources.tushare_data_source import TushareDataSource

    assert SQLiteStorage.__name__ == "SQLiteStorage"
    assert BinStorage.__name__ == "BinStorage"
    assert issubclass(FactorUpdater, object)
    assert issubclass(BaseFactor, object)
    assert UnifiedScheduler.__name__ == "UnifiedScheduler"
    assert TushareDataSource.__name__ == "TushareDataSource"
    assert AkshareDataSource.__name__ == "AkshareDataSource"


def test_eval_and_app_imports():
    from FactorEvaluates import FactorEvalService, MetricDiscoverer
    from FactorEvaluates.batch.batch_eval_runner import BatchEvalRunner
    from FactorEvaluates.batch.daily_matrix_engine import DailyMatrixEngine
    from FactorEvaluates.metrics.quantile_metric import QuantileMetric
    from App.server import create_app
    from yg_quant_repo import default_db_path, repo_root

    assert MetricDiscoverer().__class__.__name__ == "MetricDiscoverer"
    assert FactorEvalService.__name__ == "FactorEvalService"
    assert BatchEvalRunner.__name__ == "BatchEvalRunner"
    assert DailyMatrixEngine.__name__ == "DailyMatrixEngine"
    assert QuantileMetric().get_name()
    assert create_app is not None
    assert repo_root().name
    assert default_db_path().name in {"yg_quant.db", "qmt.db"}


def test_strategy_and_universe_imports():
    from StrategyEngine.attribution.engine import run_attribution
    from StrategyEngine.backtest.engine import Engine
    from StrategyEngine.live import to_qmt_code
    from StyleCrowding.config import default_settings
    from Universes.catalog import names
    from Strategies.small_cap import SmallCapStrategy

    assert Engine.__name__ == "Engine"
    assert callable(run_attribution)
    assert to_qmt_code("SH600000") == "600000.SH"
    assert default_settings().output_root
    assert "all" in names()
    assert SmallCapStrategy.name
