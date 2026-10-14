#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from DailyUpdates.storage.bin_storage import BinStorage, listed_mask, normalize_symbol


def _frame(rows):
    return pd.DataFrame(rows, columns=["ts_code", "trade_date", "factor_value"])


def _store(tmp_path: Path) -> BinStorage:
    return BinStorage(str(tmp_path / "factors"))


def test_normalize_symbol():
    assert normalize_symbol("000001.SZ") == "SZ000001"
    assert normalize_symbol("sz000001") == "SZ000001"
    assert normalize_symbol("SH600000") == "SH600000"


def test_roundtrip_wide_panel(tmp_path):
    store = _store(tmp_path)
    data = _frame(
        [
            ("SZ000001", "2020-01-02", 1.5),
            ("SH600000", "2020-01-02", 2.5),
            ("SZ000001", "2020-01-03", 1.75),
            ("SH600000", "2020-01-03", 2.75),
        ]
    )
    written = store.upsert_factor_data("alpha001", data)
    assert written == 4
    panel = store.read_factor_panel("alpha001")
    assert list(panel.index) == ["2020-01-02", "2020-01-03"]
    assert list(panel.columns) == ["SZ000001", "SH600000"]
    assert panel.loc["2020-01-02", "SZ000001"] == pytest.approx(1.5)
    assert panel.loc["2020-01-03", "SH600000"] == pytest.approx(2.75)


def test_incremental_append_leaves_old_bytes(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "mom",
        _frame([("SZ000001", "2020-01-02", 1.0), ("SH600000", "2020-01-02", 2.0)]),
    )
    f32 = store.features_dir / "mom.f32"
    before = f32.read_bytes()
    store.upsert_factor_data(
        "mom",
        _frame([("SZ000001", "2020-01-03", 3.0), ("SH600000", "2020-01-03", 4.0)]),
    )
    after = f32.read_bytes()
    assert after[: len(before)] == before
    panel = store.read_factor_panel("mom")
    assert panel.loc["2020-01-02", "SZ000001"] == pytest.approx(1.0)
    assert panel.loc["2020-01-03", "SH600000"] == pytest.approx(4.0)


def test_new_stock_only_widens_new_row(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data("f", _frame([("SZ000001", "2020-01-02", 1.0)]))
    f32 = store.features_dir / "f.f32"
    old = f32.read_bytes()
    store.upsert_factor_data(
        "f",
        _frame(
            [
                ("SZ000001", "2020-01-03", 2.0),
                ("SH600000", "2020-01-03", 9.0),
            ]
        ),
    )
    assert f32.read_bytes()[: len(old)] == old
    layout = store._read_layout("f")
    assert int(layout.indptr[1] - layout.indptr[0]) == 1
    assert int(layout.indptr[2] - layout.indptr[1]) == 2
    panel = store.read_factor_panel("f")
    assert np.isnan(panel.loc["2020-01-02", "SH600000"])
    assert panel.loc["2020-01-03", "SH600000"] == pytest.approx(9.0)


def test_delist_shortens_later_row(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "f",
        _frame(
            [
                ("SZ000001", "2020-01-02", 1.0),
                ("SH600000", "2020-01-02", 2.0),
                ("SZ000001", "2020-01-03", 3.0),
                ("SH600000", "2020-01-03", 4.0),
            ]
        ),
    )
    store.register_delist("SZ000001", "2020-01-06")
    store.upsert_factor_data(
        "f",
        _frame([("SH600000", "2020-01-06", 5.0)]),
    )
    layout = store._read_layout("f")
    assert int(layout.indptr[1] - layout.indptr[0]) == 2
    assert int(layout.indptr[2] - layout.indptr[1]) == 2
    assert int(layout.indptr[3] - layout.indptr[2]) == 1
    panel = store.read_factor_panel("f")
    assert np.isnan(panel.loc["2020-01-06", "SZ000001"])
    assert panel.loc["2020-01-06", "SH600000"] == pytest.approx(5.0)
    long = store.read_factor_data("f", start_date="2020-01-06")
    assert "SZ000001" not in set(long["ts_code"])


def test_suspend_keeps_nan(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "f",
        _frame(
            [
                ("SZ000001", "2020-01-02", 1.0),
                ("SH600000", "2020-01-02", 2.0),
            ]
        ),
    )
    store.upsert_factor_data("f", _frame([("SH600000", "2020-01-03", 3.0)]))
    panel = store.read_factor_panel("f")
    assert np.isnan(panel.loc["2020-01-03", "SZ000001"])
    assert panel.loc["2020-01-03", "SH600000"] == pytest.approx(3.0)
    layout = store._read_layout("f")
    assert int(layout.indptr[2] - layout.indptr[1]) == 2


def test_missing_listed_filled_with_nan(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "f",
        _frame(
            [
                ("SZ000001", "2020-01-02", 1.0),
                ("SH600000", "2020-01-02", np.nan),
            ]
        ),
    )
    # first write drops all-NaN rows per stock after dropna on keys only; SH600000 stays
    panel = store.read_factor_panel("f")
    assert panel.loc["2020-01-02", "SZ000001"] == pytest.approx(1.0)
    assert np.isnan(panel.loc["2020-01-02", "SH600000"])
    layout = store._read_layout("f")
    assert int(layout.indptr[1] - layout.indptr[0]) == 2


def test_hash_mismatch_rejects_read(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data("f", _frame([("SZ000001", "2020-01-02", 1.0)]))
    lines = store._columns_path.read_text(encoding="utf-8").splitlines()
    parts = lines[0].split("\t")
    parts[2] = "2019-01-01"
    store._columns_path.write_text("\t".join(parts) + "\n", encoding="utf-8")
    store._columns_cache = None
    with pytest.raises(ValueError, match="拒绝读取|哈希"):
        store.read_factor_panel("f")


def test_short_payload_rejects_read(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data("f", _frame([("SZ000001", "2020-01-02", 1.0)]))
    f32 = store.features_dir / "f.f32"
    f32.write_bytes(f32.read_bytes()[:2])
    with pytest.raises(ValueError, match="拒绝读取|短于"):
        store.read_factor_panel("f")


def test_width_mismatch_rejects_read(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "f",
        _frame(
            [
                ("SZ000001", "2020-01-02", 1.0),
                ("SH600000", "2020-01-02", 2.0),
            ]
        ),
    )
    layout = store._read_layout("f")
    layout.indptr[1] = 1
    layout.n_floats = 1
    _, layout_path = store._factor_paths("f")
    layout_path.write_bytes(layout.dumps())
    (store.features_dir / "f.f32").write_bytes(np.float32(1.0).tobytes())
    with pytest.raises(ValueError, match="拒绝读取|行宽"):
        store.read_factor_panel("f")


def test_calendar_tail_append_no_remap(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data("f", _frame([("SZ000001", "2020-01-02", 1.0)]))
    f32 = store.features_dir / "f.f32"
    before = f32.read_bytes()
    store.ensure_calendar(["2020-01-02", "2020-01-03"])
    assert f32.read_bytes() == before
    assert store.read_calendar()[-1] == "2020-01-03"
    panel = store.read_factor_panel("f")
    assert panel.loc["2020-01-02", "SZ000001"] == pytest.approx(1.0)


def test_calendar_mid_insert_remap(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "f",
        _frame(
            [
                ("SZ000001", "2020-01-02", 1.0),
                ("SZ000001", "2020-01-06", 2.0),
            ]
        ),
    )
    store.ensure_calendar(["2020-01-02", "2020-01-03", "2020-01-06"])
    assert store.read_calendar() == ["2020-01-02", "2020-01-03", "2020-01-06"]
    panel = store.read_factor_panel("f")
    assert panel.loc["2020-01-02", "SZ000001"] == pytest.approx(1.0)
    assert np.isnan(panel.loc["2020-01-03", "SZ000001"])
    assert panel.loc["2020-01-06", "SZ000001"] == pytest.approx(2.0)


def test_replace_and_delete(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data("f", _frame([("SZ000001", "2020-01-02", 1.0)]))
    store.upsert_factor_data(
        "f",
        _frame([("SZ000001", "2020-01-02", 9.0)]),
        replace=True,
    )
    assert store.read_factor_panel("f").loc["2020-01-02", "SZ000001"] == pytest.approx(9.0)
    store.delete_factors(["f"])
    assert "f" not in store.list_stored_factor_names()
    assert not (store.features_dir / "f.f32").exists()
    assert not (store.features_dir / "f.layout").exists()


def test_date_and_symbol_slice(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "f",
        _frame(
            [
                ("SZ000001", "2020-01-02", 1.0),
                ("SH600000", "2020-01-02", 2.0),
                ("SZ000001", "2020-01-03", 3.0),
                ("SH600000", "2020-01-03", 4.0),
            ]
        ),
    )
    one = store.read_factor_panel("f", start_date="2020-01-03", end_date="2020-01-03")
    assert list(one.index) == ["2020-01-03"]
    subset = store.read_factor_panel("f", symbols=["SH600000"])
    assert list(subset.columns) == ["SH600000"]
    assert subset.loc["2020-01-02", "SH600000"] == pytest.approx(2.0)


def test_all_nan_rejected(tmp_path):
    store = _store(tmp_path)
    written = store.upsert_factor_data(
        "f",
        _frame([("SZ000001", "2020-01-02", np.nan)]),
    )
    assert written == 0
    assert store._read_layout("f") is None


def test_append_without_layout_stays_on_old_publish(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data(
        "f",
        _frame([("SZ000001", "2020-01-02", 1.0)]),
    )
    layout_before = (store.features_dir / "f.layout").read_bytes()
    extra = np.asarray([9.0], dtype=np.float32).tobytes()
    with (store.features_dir / "f.f32").open("ab") as handle:
        handle.write(extra)
    panel = store.read_factor_panel("f")
    assert panel.loc["2020-01-02", "SZ000001"] == pytest.approx(1.0)
    assert (store.features_dir / "f.layout").read_bytes() == layout_before
    assert (store.features_dir / "f.f32").stat().st_size > 4


def test_listed_mask_rules():
    from DailyUpdates.storage.bin_storage import Column

    columns = [
        Column(0, "A", "2020-01-02"),
        Column(1, "B", "2020-01-03"),
    ]
    delist = {0: "2020-01-06"}
    assert listed_mask("2020-01-02", columns, delist).tolist() == [True, False]
    assert listed_mask("2020-01-03", columns, delist).tolist() == [True, True]
    assert listed_mask("2020-01-06", columns, delist).tolist() == [False, True]


def test_instruments_append_only(tmp_path):
    store = _store(tmp_path)
    store.write_instruments({"SZ000001": ("2020-01-02", "")})
    store.write_instruments(
        {"SZ000001": ("2010-01-01", ""), "SH600000": ("2020-01-03", "")}
    )
    columns = store.read_columns()
    assert columns[0].symbol == "SZ000001"
    assert columns[0].list_date == "2020-01-02"
    assert columns[1].symbol == "SH600000"
    mapping = store.read_instruments()
    assert mapping["SZ000001"] == ("2020-01-02", "")


def test_factor_latest_date_ignores_future_calendar(tmp_path):
    store = _store(tmp_path)
    store.upsert_factor_data("f", _frame([("SZ000001", "2026-09-21", 1.0)]))
    store.ensure_calendar(["2026-09-21", "2027-10-27"])
    assert store.read_calendar()[-1] == "2027-10-27"
    assert store.get_factor_latest_date("f") == "2026-09-21"
