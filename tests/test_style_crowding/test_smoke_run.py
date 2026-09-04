#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from StyleCrowding.config import StyleCrowdingSettings
from StyleCrowding.context import StyleCrowdingContext
from StyleCrowding.indicators.compute import compute_indicators
from StyleCrowding.runner import run_style_crowding
from StyleCrowding.universe import GroupSnapshot
from StyleCrowding.writer import publish_manifest, series_path, write_series


def _minimal_ctx(tmp_path: Path) -> StyleCrowdingContext:
    dates = ("2020-01-02", "2020-01-03", "2020-01-06")
    symbols = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")
    idx = pd.Index(dates)
    cols = pd.Index(symbols)
    close = pd.DataFrame(10.0, index=idx, columns=cols)
    pre = close * 0.99
    settings = StyleCrowdingSettings(
        output_root=tmp_path,
        start_date=dates[0],
        group_counts=(10,),
        orientations=("positive",),
        skip_indicators=("07_pairwise_correlation_63d", "pairwise"),
        min_group_size=1,
    )
    style_panel = pd.DataFrame(
        {s: float(i) for i, s in enumerate(symbols, start=1)},
        index=idx,
    ).T
    style_panel = style_panel.T  # wrong shape fix
    style_panel = pd.DataFrame(
        np.tile(np.arange(1, 11, dtype=float), (len(dates), 1)),
        index=idx,
        columns=cols,
    )
    groups = {}
    for d in dates:
        groups[d] = GroupSnapshot(
            "style_momentum",
            d,
            10,
            10,
            1.0,
            tuple(symbols[-1:]),
            tuple(symbols[:1]),
            True,
            "OK",
        )
    dispersion_row = {
        "d_top": 0.01,
        "d_rel": 0.5,
        "market_std": 0.02,
        "d_top_50": 0.01,
        "d_top_100": 0.01,
        "d_top_200": 0.01,
        "d_bottom_50": 0.01,
    }
    dispersion = pd.DataFrame([dispersion_row] * len(dates), index=idx)
    close_ret = close / pre - 1.0
    return StyleCrowdingContext(
        settings=settings,
        dates=dates,
        output_dates=dates,
        factor_ids=("style_momentum",),
        symbols=symbols,
        close=close,
        pre_close=pre,
        pb=pd.DataFrame(2.0, index=idx, columns=cols),
        pe=pd.DataFrame(20.0, index=idx, columns=cols),
        ps=pd.DataFrame(5.0, index=idx, columns=cols),
        circ_mv=pd.DataFrame(1e9, index=idx, columns=cols),
        mask=pd.DataFrame(True, index=idx, columns=cols),
        style_panels={"style_momentum": style_panel},
        exposure_btop=style_panel * 0.1,
        close_returns=close_ret,
        groups={10: {"style_momentum": groups}},
        dispersion={10: {"style_momentum": dispersion}},
        factor_returns={10: pd.DataFrame({"style_momentum": [0.01, 0.02, -0.01]}, index=idx)},
        canonical_factor_returns=pd.DataFrame({"style_momentum": [0.01, 0.02, -0.01]}, index=idx),
        risk_factor_returns={
            "style_momentum": pd.DataFrame(
                {"target_return": [0.01, 0.02, -0.01], "market_return": [0.005, 0.004, 0.003]},
                index=idx,
            )
        },
        specific_returns=np.zeros((len(dates), len(symbols)), dtype=np.float32),
        symbol_index={s: i for i, s in enumerate(symbols)},
    )


def test_smoke_compute_and_publish(tmp_path):
    ctx = _minimal_ctx(tmp_path)
    out = compute_indicators(ctx, "style_momentum", 10, "positive")
    assert len(out) >= 20
    root = ctx.settings.published_root()
    root.mkdir(parents=True, exist_ok=True)
    for key, series in out.items():
        write_series(series_path(root, "style_momentum", 10, "positive", key), series)
    publish_manifest(root, ctx.settings, extra={"n_dates": len(ctx.dates)})
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "qmt-style-crowding-v1"
    assert "01_valuation_bp_z" in manifest["indicators"]


def _db_has_adj_factor() -> bool:
    """复权口径要求库里已有 adj_factor 列，否则集成 smoke 无意义。"""
    path = Path(__import__("yg_quant_repo").default_db_path())
    if not path.is_file():
        return False
    from DailyUpdates.storage import SQLiteStorage

    try:
        return "adj_factor" in set(SQLiteStorage(str(path)).get_market_fields())
    except Exception:
        return False


@pytest.mark.skipif(
    not _db_has_adj_factor(),
    reason="需要本地 yg_quant.db 且已回填 adj_factor 做集成 smoke",
)
def test_integration_run_short():
    settings = StyleCrowdingSettings(
        start_date="2024-01-01",
        end_date="2024-03-01",
        group_counts=(10,),
        orientations=("positive",),
        skip_indicators=("07_pairwise_correlation_63d", "pairwise"),
    )
    result = run_style_crowding(settings)
    assert result["n_dates"] > 10


def test_pairwise_reused_across_reverse(tmp_path, monkeypatch):
    ctx = _minimal_ctx(tmp_path)
    ctx.settings.skip_indicators = ()
    ctx.settings.pairwise_window = 1
    ctx.settings.pairwise_min_pair_obs = 1
    ctx.settings.pairwise_min_member_ratio = 0.0
    calls = {"n": 0}

    def fake_pair(*_a, **_k):
        calls["n"] += 1
        return 0.2

    monkeypatch.setattr(
        "StyleCrowding.indicators.compute._pairwise_corr_day", fake_pair
    )
    compute_indicators(ctx, "style_momentum", 10, "positive")
    first = calls["n"]
    assert first == len(ctx.dates)
    compute_indicators(ctx, "style_momentum", 10, "reverse")
    assert calls["n"] == first
    assert ("style_momentum", 10) in ctx.pairwise_cache
