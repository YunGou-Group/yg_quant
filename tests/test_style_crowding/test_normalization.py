#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from FactorEvaluates.matrix_utils import causal_percentile, prior_rolling_zscore
from StyleCrowding.catalog import CORE_INDICATORS, INDICATORS, indicator_key
from StyleCrowding.config import StyleCrowdingSettings
from StyleCrowding.query.bootstrap import crowding_temperature
from StyleCrowding.universe import GroupSnapshot, build_group_snapshot, reverse_group_snapshots


def test_indicator_keys_unique():
    keys = [indicator_key(i.file) for i in INDICATORS]
    assert len(keys) == len(set(keys))
    assert len(INDICATORS) == 29
    mom_cores = [k for k in CORE_INDICATORS if "momentum" in k or k.startswith("06_")]
    assert mom_cores == ["06_factor_cumulative_return_momentum_21d"]


def test_prior_z_uses_only_past():
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    s = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx.strftime("%Y-%m-%d"))
    z = prior_rolling_zscore(s, window=20, min_history=10)
    # 第一个有效 Z 不应因“未来”数据而改变：截断后重算应一致
    z_full = prior_rolling_zscore(s, window=20, min_history=10)
    z_trunc = prior_rolling_zscore(s.iloc[:40], window=20, min_history=10)
    overlap = z_full.iloc[20:40]
    assert np.allclose(overlap.dropna(), z_trunc.iloc[20:40].dropna(), equal_nan=True)


def test_causal_percentile_no_future():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    p = causal_percentile(s, min_prior=2)
    assert np.isnan(p.iloc[0])
    assert np.isnan(p.iloc[1])
    assert p.iloc[2] == pytest.approx(1.0)
    assert p.iloc[3] == pytest.approx(1.0)


def test_reverse_groups_not_inverse_percentile():
    snap = GroupSnapshot(
        "style_momentum",
        "2020-01-01",
        100,
        100,
        1.0,
        ("A", "B"),
        ("Y", "Z"),
        True,
        "OK",
    )
    rev = reverse_group_snapshots({"style_momentum": {"2020-01-01": snap}})["style_momentum"]["2020-01-01"]
    assert rev.high == ("Z", "Y")
    assert rev.low == ("B", "A")


def test_build_group_snapshot_sizes():
    settings = StyleCrowdingSettings(min_group_size=1)
    exp = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0, "g": 7.0, "h": 8.0, "i": 9.0, "j": 10.0})
    snap = build_group_snapshot("2020-01-01", exp, "style_momentum", settings, group_count=10)
    assert snap.valid
    assert len(snap.high) == 1
    assert snap.high[0] == "j"
    assert snap.low[0] == "a"


def test_crowding_temperature_directions():
    assert crowding_temperature(0.9, "high") == pytest.approx(0.9)
    assert crowding_temperature(0.1, "low") == pytest.approx(0.9)
    assert crowding_temperature(0.2, "both") == pytest.approx(0.8)


def test_manifest_contract_fields():
    spec = INDICATORS[0]
    meta = spec.event_metadata()
    assert "key" in meta
    assert meta["crowding_direction"] in {"high", "low", "both"}
