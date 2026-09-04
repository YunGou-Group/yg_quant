#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alpha101：全 NaN 窗口与 close 掩码，避免幽灵有限值。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from DailyUpdates.factor_updates.factors._alpha101_engine import (
    _ts_argmax_2d,
    _ts_argmin_2d,
    _wide_to_long,
    indneutralize,
)


def test_argmax_all_nan_window_is_nan():
    data = np.full((8, 2), np.nan)
    data[:, 1] = np.arange(8, dtype=np.float64)
    out = _ts_argmax_2d(data, 5)
    assert np.isnan(out[-1, 0])
    assert np.isfinite(out[-1, 1])
    assert out[-1, 1] == 5.0


def test_argmin_all_nan_window_is_nan():
    data = np.full((6, 1), np.nan)
    out = _ts_argmin_2d(data, 3)
    assert np.isnan(out).all()


def test_wide_to_long_masks_missing_close():
    template = pd.DataFrame(
        [[10.0, np.nan], [11.0, np.nan]],
        index=["2020-01-02", "2020-01-03"],
        columns=["000001.SZ", "SIDECAR.SZ"],
    )
    values = pd.DataFrame(
        [[0.5, 9.9], [0.6, 8.8]],
        index=template.index,
        columns=template.columns,
    )
    long = _wide_to_long(values, template, "alpha001")
    sidecar = long.loc[long["ts_code"] == "SIDECAR.SZ", "alpha001"]
    listed = long.loc[long["ts_code"] == "000001.SZ", "alpha001"]
    assert sidecar.isna().all()
    assert listed.notna().all()
    assert not np.isfinite(sidecar.to_numpy(dtype="float64")).any()


def test_wide_to_long_start_date_matches_full_then_filter():
    dates = [f"2020-01-{day:02d}" for day in range(2, 12)]
    template = pd.DataFrame(
        np.arange(len(dates) * 2, dtype=float).reshape(len(dates), 2) + 10.0,
        index=dates,
        columns=["000001.SZ", "000002.SZ"],
    )
    values = template / 10.0
    full = _wide_to_long(values, template, "alpha001")
    sliced = _wide_to_long(values, template, "alpha001", start_date="2020-01-08")
    expect = full.loc[full["trade_date"] >= "2020-01-08"].reset_index(drop=True)
    pd.testing.assert_frame_equal(sliced.reset_index(drop=True), expect)


def test_indneutralize_uses_asof_industry_not_latest():
    dates = ["2026-01-15", "2026-02-15"]
    cols = ["A", "B"]
    values = pd.DataFrame([[2.0, 4.0], [10.0, 20.0]], index=dates, columns=cols)
    labels = pd.DataFrame([["X", "X"], ["X", "Y"]], index=dates, columns=cols)
    out = indneutralize(values, labels)
    assert abs(out.loc["2026-01-15", "A"] + 1.0) < 1e-9
    assert abs(out.loc["2026-01-15", "B"] - 1.0) < 1e-9
    assert abs(out.loc["2026-02-15", "A"]) < 1e-9
    assert abs(out.loc["2026-02-15", "B"]) < 1e-9
    leaked = pd.DataFrame([["Y", "X"], ["X", "Y"]], index=dates, columns=cols)
    leaked_out = indneutralize(values, leaked)
    assert abs(leaked_out.loc["2026-01-15", "A"]) < 1e-9
    assert abs(out.loc["2026-01-15", "A"] - leaked_out.loc["2026-01-15", "A"]) > 0.5
