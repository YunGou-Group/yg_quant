"""数据源按 FetchSlice 分类，引擎不按厂商名硬编码。"""

import pandas as pd

from DailyUpdates.data_fetcher.data_fetcher import DataFetcher, market_uses_range_window
from DailyUpdates.data_fetcher.data_source_base import DataSourceBase, FetchSlice
from DailyUpdates.data_fetcher.dataset_installer import DatasetInstaller
from DailyUpdates.data_fetcher.data_sources.akshare_data_source import AkshareDataSource
from DailyUpdates.data_fetcher.data_sources.test_data_source import TestDataSource
from DailyUpdates.data_fetcher.data_sources.tushare_data_source import TushareDataSource
from DailyUpdates.data_fetcher.unified_scheduler import UnifiedScheduler


class _PanelSource(DataSourceBase):
    fetch_slice = FetchSlice.BY_PANEL
    requires_token = False

    def fetch_data(self, config, start_date, end_date):
        del config, start_date, end_date
        return None


def test_source_classes_declare_slice_type():
    assert TushareDataSource.fetch_slice is FetchSlice.BY_DATE
    assert TushareDataSource.uses_range_window() is False
    assert TushareDataSource.requires_token is True
    assert AkshareDataSource.fetch_slice is FetchSlice.BY_SYMBOL
    assert AkshareDataSource.uses_range_window() is True
    assert TestDataSource.fetch_slice is FetchSlice.BY_SYMBOL
    assert _PanelSource.uses_range_window() is True


def test_engine_follows_slice_type_not_vendor_name():
    classes = {
        "Tushare": TushareDataSource,
        "Akshare": AkshareDataSource,
        "Ricequant": _PanelSource,
    }
    tushare = [{"data_source": "Tushare"}]
    assert market_uses_range_window(tushare, classes) is False
    akshare = [{"data_source": "Akshare"}]
    assert market_uses_range_window(akshare, classes) is True
    panel = [{"data_source": "Ricequant"}]
    assert market_uses_range_window(panel, classes) is True
    mixed = [{"data_source": "Tushare"}, {"data_source": "Akshare"}]
    assert market_uses_range_window(mixed, classes) is False


def test_installer_day_slice_only_for_by_date_sources():
    class _Proc:
        data_source_classes = {
            "Tushare": TushareDataSource,
            "Akshare": AkshareDataSource,
        }

    installer = DatasetInstaller.__new__(DatasetInstaller)
    installer.data_fetcher = _Proc()
    by_date = {"data_source": "Tushare", "api_name": "adj_factor"}
    by_symbol = {"data_source": "Akshare", "api_name": "adj_factor"}
    daily = {"data_source": "Tushare", "api_name": "daily"}
    assert installer._should_install_by_trade_calendar(by_date) is True
    assert installer._should_install_by_trade_calendar(by_symbol) is False
    assert installer._should_install_by_trade_calendar(daily) is False


def test_token_injected_by_requires_token_flag():
    configs = {
        "daily": {"data_source": "Tushare", "api_name": "daily"},
        "ak": {"data_source": "Akshare", "api_name": "daily"},
    }
    out = UnifiedScheduler._with_token(configs, "secret-token")
    assert out["daily"]["token"] == "secret-token"
    assert "token" not in out["ak"]


def test_optional_empty_dataset_does_not_abort_day():
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.data_source_classes = {"Tushare": object}

    def fake_fetch(config, start_date, end_date):
        del start_date, end_date
        if config.get("api_name") == "stk_limit":
            return pd.DataFrame()
        return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [1.0]})

    fetcher._create_data_source_and_fetch = fake_fetch
    frames = fetcher.fetch_market_datasets(
        {
            "daily": {
                "data_source": "Tushare",
                "data_type": "daily",
                "api_name": "daily",
            },
            "stk_limit": {
                "data_source": "Tushare",
                "data_type": "daily",
                "api_name": "stk_limit",
                "optional": True,
            },
        },
        "20050104",
        "20050104",
    )
    assert "daily" in frames
    assert "stk_limit" not in frames
