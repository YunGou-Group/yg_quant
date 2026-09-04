"""数据源按 FetchSlice 分类，引擎不按厂商名硬编码。"""

from DailyUpdates.data_fetcher.data_processor import market_uses_range_window
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
    installer.data_processor = _Proc()
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


def test_local_calendar_does_not_block_days_after_last_stored():
    from DailyUpdates.data_fetcher.data_processor import DataProcessor

    class Store:
        def list_trade_dates(self, start_date=None, end_date=None):
            return ["2026-08-28", "2026-08-31"]

    proc = DataProcessor(storage=Store())
    assert proc._is_trading_day("20260831") is True
    assert proc._is_trading_day("20260829") is False
    assert proc._is_trading_day("20260903") is True
