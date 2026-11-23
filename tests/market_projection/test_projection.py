from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from DataStore import (
    MarketPanelLoader,
    MarketProjectionReader,
    ProjectionSchemaError,
    ProjectionStaleError,
)
from DailyUpdates.storage import SQLiteStorage
from DailyUpdates.storage.market_projection import (
    MarketProjectionWriter,
    load_manifest,
    load_pointer,
)
from DailyUpdates.factor_updates.base_factor import BaseFactor
from DailyUpdates.factor_updates.factor_updater import FactorUpdater
from StrategyEngine.backtest.panel import PanelStore


def _stock_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ"],
            "name": ["A", "B"],
            "list_date": ["20200101", "20240103"],
            "delist_date": [None, "20240104"],
            "list_status": ["L", "D"],
        }
    )


def _market(days=("2024-01-02", "2024-01-03")) -> pd.DataFrame:
    rows = []
    for i, day in enumerate(days):
        rows.append(
            {
                "ts_code": "SH600000",
                "trade_date": day,
                "open": 10.0 + i,
                "close": 10.5 + i,
                "vol": 100.0 + i,
                "amount": 1000.0 + i,
                "adj_factor": 2.0,
            }
        )
        if day == "2024-01-03":
            rows.append(
                {
                    "ts_code": "SZ000001",
                    "trade_date": day,
                    "open": 20.0,
                    "close": 21.0,
                    "vol": 200.0,
                    "amount": 2000.0,
                    "adj_factor": 1.5,
                }
            )
    rows.append(
        {
            "ts_code": "index_SH000300",
            "trade_date": days[-1],
            "open": 3000.0,
            "close": 3010.0,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture()
def projected(tmp_path):
    db = SQLiteStorage(str(tmp_path / "market.db"))
    db.upsert_stock_basic(_stock_basic())
    db.upsert_market_data(_market())
    root = tmp_path / "projection"
    writer = MarketProjectionWriter(db, str(root))
    manifest = writer.rebuild(
        fields=["open", "close", "vol", "amount", "adj_factor"],
        chunk_days=1,
    )
    return db, root, writer, manifest


def test_rebuild_and_read_raw_hfq(projected):
    db, root, writer, manifest = projected
    assert manifest["dtype"] == "<f8"
    assert manifest["fields"] == ["open", "close", "vol", "amount", "adj_factor"]
    reader = MarketProjectionReader(str(root), sidecar_storage=db)
    raw = reader.read_panel("open", adjust="none")
    hfq = reader.read_panel("open", adjust="hfq")
    volume = reader.read_panel("vol", adjust="hfq")
    assert list(raw.index) == ["2024-01-02", "2024-01-03"]
    assert list(raw.columns) == ["SH600000", "SZ000001"]
    assert raw.loc["2024-01-02", "SH600000"] == 10.0
    assert np.isnan(raw.loc["2024-01-02", "SZ000001"])
    assert hfq.loc["2024-01-03", "SZ000001"] == 30.0
    assert volume.loc["2024-01-03", "SZ000001"] == pytest.approx(200 / 1.5)
    assert writer.validate()["commit"] == manifest["commit"]


def test_long_reader_matches_sqlite(projected):
    db, root, _writer, _manifest = projected
    reader = MarketProjectionReader(str(root), sidecar_storage=db)
    fields = ["open", "close", "vol", "amount", "adj_factor"]
    got = reader.read_market_data(fields, ordered=True)
    want = db.read_market_data(fields=fields, ordered=True, adjust="none")
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        want.reset_index(drop=True),
        check_dtype=False,
    )


def test_projection_excludes_index_but_loader_keeps_index_path(projected):
    db, root, _writer, _manifest = projected
    loader = MarketPanelLoader(
        str(db.db_path), projection_dir=str(root), strict_projection=True
    )
    assert all(not name.startswith("index_") for name in loader.load_open().columns)
    index = loader.load_index_close("index_SH000300")
    assert index.loc["2024-01-03"] == 3010.0


def test_tail_append_keeps_epoch_and_switches_commit(projected):
    db, root, writer, before = projected
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-04",
                    "open": 12.0,
                    "close": 12.5,
                    "vol": 102.0,
                    "amount": 1002.0,
                    "adj_factor": 2.0,
                }
            ]
        )
    )
    after = writer.publish(chunk_days=1)
    assert after["epoch"] == before["epoch"]
    assert after["commit"] != before["commit"]
    reader = MarketProjectionReader(str(root), sidecar_storage=db)
    assert reader.read_panel("close").loc["2024-01-04", "SH600000"] == 12.5


def test_empty_publish_keeps_generation(projected):
    _db, _root, writer, before = projected
    after = writer.publish()
    assert (after["epoch"], after["commit"]) == (
        before["epoch"],
        before["commit"],
    )


def test_reader_date_and_symbol_slice(projected):
    db, root, _writer, _manifest = projected
    panel = MarketProjectionReader(str(root), sidecar_storage=db).read_panel(
        "open",
        start_date="2024-01-03",
        end_date="2024-01-03",
        symbols=["SZ000001"],
    )
    assert list(panel.index) == ["2024-01-03"]
    assert list(panel.columns) == ["SZ000001"]
    assert panel.iloc[0, 0] == 20.0


def test_adj_factor_correction_rebuilds_and_updates_hfq(projected):
    db, root, writer, before = projected
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-02",
                    "adj_factor": 3.0,
                }
            ]
        )
    )
    after = writer.publish(chunk_days=1)
    assert after["epoch"] != before["epoch"]
    reader = MarketProjectionReader(str(root), sidecar_storage=db)
    assert reader.read_panel("open", adjust="none").iloc[0, 0] == 10.0
    assert reader.read_panel("open", adjust="hfq").iloc[0, 0] == 30.0


def test_orphan_payload_tail_is_removed_before_append(projected):
    db, root, writer, before = projected
    epoch = root / "epochs" / before["epoch"]
    payload = epoch / "fields" / "open.f64"
    with payload.open("ab") as stream:
        stream.write(np.asarray([999.0], dtype="<f8").tobytes())
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-04",
                    "open": 12.0,
                    "close": 12.5,
                    "vol": 102.0,
                    "amount": 1002.0,
                    "adj_factor": 2.0,
                }
            ]
        )
    )
    after = writer.publish(chunk_days=1)
    assert payload.stat().st_size == after["field_meta"]["open"]["payload_bytes"]
    assert (
        MarketProjectionReader(str(root), sidecar_storage=db)
        .read_panel("open")
        .loc["2024-01-04", "SH600000"]
        == 12.0
    )


def test_historical_correction_builds_new_epoch(projected):
    db, root, writer, before = projected
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-02",
                    "close": 88.0,
                }
            ]
        )
    )
    after = writer.publish(chunk_days=1)
    assert after["epoch"] != before["epoch"]
    assert (
        MarketProjectionReader(str(root), sidecar_storage=db)
        .read_panel("close")
        .loc["2024-01-02", "SH600000"]
        == 88.0
    )


def test_reader_rejects_unpublished_sqlite_change(projected):
    db, root, _writer, _manifest = projected
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-04",
                    "open": 12.0,
                }
            ]
        )
    )
    with pytest.raises(ProjectionStaleError):
        MarketProjectionReader(str(root), sidecar_storage=db)


def test_missing_payload_is_schema_error(projected):
    db, root, _writer, manifest = projected
    payload = root / "epochs" / manifest["epoch"] / "fields" / "open.f64"
    payload.unlink()
    reader = MarketProjectionReader(str(root), sidecar_storage=db, strict=False)
    with pytest.raises(ProjectionSchemaError):
        reader.read_panel("open")


def test_current_is_stable_when_rebuild_fails(projected, monkeypatch):
    _db, root, writer, manifest = projected
    current_before = (root / "CURRENT").read_bytes()

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(writer, "_write_payloads", fail)
    with pytest.raises(RuntimeError, match="injected"):
        writer.rebuild(fields=manifest["fields"])
    assert (root / "CURRENT").read_bytes() == current_before
    assert load_manifest(root, load_pointer(root))["commit"] == manifest["commit"]


def test_float64_precision_is_preserved(tmp_path):
    db = SQLiteStorage(str(tmp_path / "market.db"))
    db.upsert_stock_basic(_stock_basic().iloc[:1])
    value = 123456789.12345679
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-02",
                    "open": value,
                }
            ]
        )
    )
    root = tmp_path / "projection"
    MarketProjectionWriter(db, str(root)).rebuild(fields=["open"])
    got = MarketProjectionReader(str(root), sidecar_storage=db).read_panel("open")
    assert got.iloc[0, 0] == value


def test_actual_market_range_is_not_clipped_by_official_listing_dates(tmp_path):
    db = SQLiteStorage(str(tmp_path / "market.db"))
    db.upsert_stock_basic(
        pd.DataFrame(
            {
                "ts_code": ["600018.SH"],
                "name": ["A"],
                "list_date": ["20240103"],
                "delist_date": ["20240103"],
                "list_status": ["D"],
            }
        )
    )
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600018",
                    "trade_date": "2024-01-02",
                    "open": 10.0,
                },
                {
                    "ts_code": "SH600018",
                    "trade_date": "2024-01-03",
                    "open": 11.0,
                },
            ]
        )
    )
    root = tmp_path / "projection"
    MarketProjectionWriter(db, str(root)).rebuild(fields=["open"])
    got = MarketProjectionReader(str(root), sidecar_storage=db).read_panel("open")
    assert got["SH600018"].tolist() == [10.0, 11.0]


def test_factor_updater_reads_projection(projected, monkeypatch):
    db, root, _writer, _manifest = projected

    class _CloseFactor(BaseFactor):
        name = "_test_close"
        dependencies = ["close"]

        def calculate(self, data, start_date=None):
            del start_date
            return data.rename(columns={"close": "factor_value"})[
                ["ts_code", "trade_date", "factor_value"]
            ]

    monkeypatch.setenv("YG_QUANT_MARKET_PROJECTION_DIR", str(root))
    monkeypatch.setenv("YG_QUANT_DB_PATH", str(db.db_path))
    updater = FactorUpdater(str(db.db_path), max_workers=1)
    updater.factors = [_CloseFactor()]
    loaded = updater._load_market_data(None, factors=updater.factors)
    want = db.read_market_data(fields=["close"], ordered=True, adjust="hfq")
    got = loaded.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_panel_store_loads_stock_market_from_projection(projected):
    db, root, _writer, _manifest = projected
    loader = MarketPanelLoader(
        str(db.db_path), projection_dir=str(root), strict_projection=True
    )
    store = PanelStore.load(
        start="2024-01-02",
        end="2024-01-03",
        market_fields=("close",),
        market_loader=loader,
        adjust="none",
    )
    assert store.dates == ["2024-01-02", "2024-01-03"]
    a = store.symbols.index("SH600000")
    assert store.market["open"][0, a] == 10.0
    assert store.market["close"][1, a] == 11.5


def test_manifest_captures_sqlite_change_sequence(projected):
    db, _root, _writer, manifest = projected
    assert manifest["source_change_seq"] == db.get_projection_change_seq()
    assert db.list_projection_changes(after_seq=manifest["source_change_seq"]) == []


def test_new_projected_field_triggers_schema_rebuild(projected):
    db, root, writer, before = projected
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-02",
                    "pb": 1.25,
                }
            ]
        )
    )
    after = writer.publish(chunk_days=1)
    assert after["epoch"] != before["epoch"]
    assert "pb" in after["fields"]
    assert (
        MarketProjectionReader(str(root), sidecar_storage=db)
        .read_panel("pb")
        .loc["2024-01-02", "SH600000"]
        == 1.25
    )


def test_stock_axis_change_triggers_new_epoch(projected):
    db, root, writer, before = projected
    db.upsert_stock_basic(
        pd.DataFrame(
            {
                "ts_code": ["300001.SZ"],
                "name": ["C"],
                "list_date": ["20240104"],
                "list_status": ["L"],
            }
        )
    )
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SZ300001",
                    "trade_date": "2024-01-04",
                    "open": 30.0,
                    "close": 31.0,
                    "vol": 300.0,
                    "amount": 3000.0,
                    "adj_factor": 1.0,
                }
            ]
        )
    )
    after = writer.publish(chunk_days=1)
    assert after["epoch"] != before["epoch"]
    reader = MarketProjectionReader(str(root), sidecar_storage=db)
    assert "SZ300001" in reader.read_panel("open").columns


def test_replace_stock_basic_marks_axis_dirty_and_rebuilds(projected):
    db, root, writer, before = projected
    db.replace_stock_basic(
        pd.DataFrame(
            {
                "ts_code": ["600000.SH"],
                "name": ["A"],
                "list_date": ["20240103"],
                "list_status": ["L"],
            }
        )
    )
    with pytest.raises(ProjectionStaleError):
        MarketProjectionReader(str(root), sidecar_storage=db)
    after = writer.publish(chunk_days=1)
    assert after["epoch"] != before["epoch"]
    # 实际 market_data 边界优先于官方上市日，不能裁掉底账行情。
    panel = MarketProjectionReader(str(root), sidecar_storage=db).read_panel("open")
    assert panel.loc["2024-01-02", "SH600000"] == 10.0
    assert panel.loc["2024-01-03", "SH600000"] == 11.0


def test_clear_stock_basic_marks_axis_dirty(projected):
    db, root, writer, before = projected
    db.clear_stock_basic()
    with pytest.raises(ProjectionStaleError):
        MarketProjectionReader(str(root), sidecar_storage=db)
    after = writer.publish(chunk_days=1)
    assert after["epoch"] != before["epoch"]
    assert after["source_change_seq"] == db.get_projection_change_seq()


def test_loader_uses_env_projection_dir_for_default_db(projected, monkeypatch, tmp_path):
    db, root, _writer, _manifest = projected
    monkeypatch.setenv("YG_QUANT_DB_PATH", str(db.db_path))
    monkeypatch.setenv("YG_QUANT_MARKET_PROJECTION_DIR", str(root))
    other = tmp_path / "ignored_projection"
    other.mkdir()
    loader = MarketPanelLoader(str(db.db_path), strict_projection=True)
    assert loader._projection_root == Path(root).resolve()
    assert loader.load_open(adjust="none").loc["2024-01-02", "SH600000"] == 10.0


def test_failure_after_epoch_build_does_not_switch_current(projected, monkeypatch):
    _db, root, writer, manifest = projected
    current_before = (root / "CURRENT").read_bytes()

    def fail(*_args, **_kwargs):
        raise RuntimeError("commit-failed")

    monkeypatch.setattr(writer, "_publish_commit", fail)
    with pytest.raises(RuntimeError, match="commit-failed"):
        writer.rebuild(fields=manifest["fields"])
    assert (root / "CURRENT").read_bytes() == current_before


def test_stale_writer_lock_is_recovered(projected):
    _db, _root, writer, manifest = projected
    writer.lock_path.write_text("2147483647", encoding="ascii")
    after = writer.rebuild(fields=manifest["fields"], chunk_days=1)
    assert after["commit"] != manifest["commit"]
    assert not writer.lock_path.exists()


def test_writer_lock_cleans_abandoned_temp_directories(projected):
    _db, root, writer, manifest = projected
    epoch_tmp = root / "epochs" / ".epoch-abandoned.tmp"
    commit_tmp = (
        root
        / "epochs"
        / manifest["epoch"]
        / "commits"
        / ".commit-abandoned.tmp"
    )
    epoch_tmp.mkdir()
    commit_tmp.mkdir()
    writer.publish()
    assert not epoch_tmp.exists()
    assert not commit_tmp.exists()


def test_change_while_waiting_for_writer_lock_forces_safe_rebuild(projected):
    db, root, writer, before = projected
    db.upsert_market_data(
        pd.DataFrame(
            [
                {
                    "ts_code": "SH600000",
                    "trade_date": "2024-01-04",
                    "open": 12.0,
                }
            ]
        )
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        with writer._writer_lock():
            future = pool.submit(writer.publish, chunk_days=1)
            db.upsert_market_data(
                pd.DataFrame(
                    [
                        {
                            "ts_code": "SH600000",
                            "trade_date": "2024-01-02",
                            "open": 99.0,
                        }
                    ]
                )
            )
        after = future.result(timeout=10)
    assert after["epoch"] != before["epoch"]
    assert (
        MarketProjectionReader(str(root), sidecar_storage=db)
        .read_panel("open")
        .loc["2024-01-02", "SH600000"]
        == 99.0
    )
