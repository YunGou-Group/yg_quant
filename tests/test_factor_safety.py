"""因子导入失败不删历史；全 NaN 水位在 BinStorage 测试里覆盖。"""

import logging

import pytest

from DailyUpdates.factor_updates.factor_updater import FactorUpdater


def test_import_errors_abort_without_purge():
    updater = FactorUpdater.__new__(FactorUpdater)
    updater.logger = logging.getLogger("test")
    updater.factors = []
    updater._import_errors = ["DailyUpdates.factor_updates.factors.broken"]
    deleted = []

    class _Store:
        def list_stored_factor_names(self):
            return ["alpha001"]

        def delete_factor_data(self, name):
            deleted.append(name)
            return 1

    updater.factor_storage = _Store()
    with pytest.raises(RuntimeError, match="导入失败"):
        updater.update_factors()
    assert deleted == []
