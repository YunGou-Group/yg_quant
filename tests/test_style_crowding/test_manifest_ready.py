#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StyleCrowding：manifest 按实际文件、就绪状态、JSON 不含 NaN、最后有限值。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from StyleCrowding.config import StyleCrowdingSettings
from StyleCrowding.query.bootstrap import last_finite
from StyleCrowding.writer import (
    READY_MIN_OBS,
    publish_manifest,
    series_path,
    write_json,
    write_series,
)


def test_last_finite_skips_trailing_nan():
    series = pd.Series(
        [1.0, np.nan, 3.0, np.nan],
        index=["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
    )
    value, as_of = last_finite(series)
    assert value == pytest.approx(3.0)
    assert as_of == "2020-01-06"
    empty, empty_as = last_finite(pd.Series([np.nan, np.inf], index=["a", "b"]))
    assert empty is None and empty_as is None


def test_write_json_rejects_nan_payload(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, {"x": float("nan"), "y": 1.0, "z": float("inf")})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["x"] is None
    assert payload["y"] == 1.0
    assert payload["z"] is None


def test_manifest_from_written_files_and_readiness(tmp_path):
    root = Path(tmp_path) / "published"
    dates = pd.bdate_range("2020-01-02", periods=10).strftime("%Y-%m-%d")
    short = pd.Series(np.arange(10, dtype=float), index=dates)
    write_series(
        series_path(root, "style_size", 10, "positive", "01_valuation_bp_z"),
        short,
    )
    settings = StyleCrowdingSettings(
        output_root=tmp_path,
        group_counts=(10, 5),
        orientations=("positive", "reverse"),
        skip_indicators=("pairwise",),
    )
    publish_manifest(root, settings)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["group_counts"] == [10]
    assert manifest["orientations"] == ["positive"]
    assert manifest["indicators"] == ["01_valuation_bp_z"]
    assert manifest["skip"] == ["pairwise"]
    assert manifest["ready"] is False
    key = "style_size/group_10/positive/01_valuation_bp_z"
    info = manifest["series"][key]
    assert info["valid_count"] == 10
    assert info["first_valid_date"] == str(dates[0])
    assert info["ready"] is False

    long_dates = pd.bdate_range("2018-01-02", periods=READY_MIN_OBS).strftime("%Y-%m-%d")
    long = pd.Series(1.0, index=long_dates)
    write_series(
        series_path(root, "style_size", 10, "positive", "01_valuation_bp_z"),
        long,
    )
    publish_manifest(root, settings)
    ready_man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert ready_man["ready"] is True
    assert ready_man["series"][key]["valid_count"] == READY_MIN_OBS
    assert ready_man["series"][key]["ready"] is True
