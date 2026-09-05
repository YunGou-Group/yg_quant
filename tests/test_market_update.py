"""增量行情：交易所日历为准，预期交易日 0 行则失败。"""

import pandas as pd
import pytest

from DailyUpdates.data_fetcher.stock_data_updater import StockDataUpdater


class _Store:
    def __init__(self, latest="2026-09-02", local_days=None):
        self.latest = latest
        self.local_days = list(local_days or ["2026-09-02"])

    def get_latest_market_date(self):
        return self.latest

    def list_trade_dates(self, start_date=None, end_date=None):
        out = list(self.local_days)
        if start_date:
            out = [d for d in out if d >= str(start_date)]
        if end_date:
            out = [d for d in out if d <= str(end_date)]
        return out

    def get_calendar_range(self):
        if not self.local_days:
            return None, None
        return min(self.local_days), max(self.local_days)


class _Pro:
    def __init__(self, days):
        self.days = list(days)

    def trade_cal(self, exchange="SSE", start_date=None, end_date=None, is_open="1"):
        del exchange, is_open
        start = str(start_date or "")
        end = str(end_date or "99999999")
        kept = [d for d in self.days if start <= str(d) <= end]
        return pd.DataFrame({"cal_date": kept})


def _updater(*, pro=None, latest="2026-09-02", local_days=None):
    updater = StockDataUpdater.__new__(StockDataUpdater)
    updater.storage = _Store(latest, local_days)
    updater.pro = pro
    updater.dataset_config = {
        "daily": {"data_source": "Tushare", "data_type": "daily", "fields": ["close"]}
    }
    updater.data_fetcher = type("P", (), {"data_source_classes": {}})()
    updater.data_processor = type("P", (), {})()
    updater.first_date_str = "20050101"
    return updater


def test_trade_days_uses_remote_calendar_after_local_end():
    updater = _updater(pro=_Pro(["20260902", "20260903"]))
    assert updater._trade_days("20260903", "20260903") == ["2026-09-03"]


def test_trade_days_sorts_newest_first_calendar_ascending():
    updater = _updater(pro=_Pro(["20260904", "20260903", "20260902"]))
    assert updater._trade_days("20260902", "20260904") == [
        "2026-09-02",
        "2026-09-03",
        "2026-09-04",
    ]


def test_trade_days_empty_remote_window_is_ok():
    updater = _updater(pro=_Pro([]))
    assert updater._trade_days("20260905", "20260906") == []


def test_trade_days_without_pro_refuses_truncated_local():
    updater = _updater(pro=None, local_days=["2026-09-03"])
    with pytest.raises(RuntimeError, match="覆盖不到"):
        updater._trade_days("20260903", "20260904")


def test_update_fails_when_expected_session_writes_zero():
    updater = _updater(pro=_Pro(["20260903"]))
    updater.update_all_stock_by_trade_day = lambda ymd: 0
    assert updater.update_market_data(end_date="20260903") is False


def test_update_succeeds_when_already_current():
    updater = _updater(
        pro=_Pro(["20260903"]),
        latest="2026-09-03",
        local_days=["2026-09-03"],
    )
    assert updater.update_market_data(end_date="20260903") is True


def test_range_window_zero_rows_fails_if_sessions_exist():
    updater = _updater(pro=_Pro(["20260903"]))
    updater._market_uses_range_window = lambda: True
    updater._upsert_range = lambda start, end: 0
    assert updater.update_market_data(end_date="20260903") is False


def test_update_backfills_calendar_holes_before_latest():
    updater = _updater(
        pro=_Pro(["20260901", "20260902", "20260903", "20260904"]),
        latest="2026-09-04",
        local_days=["2026-08-31", "2026-09-03", "2026-09-04"],
    )
    written = []
    updater.update_all_stock_by_trade_day = lambda ymd: written.append(ymd) or 1
    assert updater.update_market_data(end_date="20260904") is True
    assert written == ["20260901", "20260902"]


def test_rebuild_fails_when_expected_session_writes_zero():
    updater = _updater(pro=_Pro(["20260903"]), latest=None)
    updater.first_date_str = "20260903"
    updater.update_all_stock_by_trade_day = lambda ymd: 0
    assert updater.rebuild_market_data(end_date="20260903") is False


def test_rebuild_writes_trade_days_ascending():
    updater = _updater(
        pro=_Pro(["20260904", "20260903", "20260902"]),
        latest=None,
    )
    updater.first_date_str = "20260902"
    written = []
    updater.update_all_stock_by_trade_day = lambda ymd: written.append(ymd) or 1
    assert updater.rebuild_market_data(end_date="20260904") is True
    assert written == ["20260902", "20260903", "20260904"]
