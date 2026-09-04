#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计修复：反向 dispersion、coverage 门槛、警报解除状态机、增量拼接。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from StyleCrowding.config import StyleCrowdingSettings
from StyleCrowding.context import StyleCrowdingContext, _build_dispersion_for_groups
from StyleCrowding.event_analysis import (
    CONFIRM_HITS,
    CONFIRM_WINDOW,
    RELEASE_DAYS,
    THRESHOLD,
    _alarm_path,
)
from StyleCrowding.runner import _splice, warmup_trading_days
from StyleCrowding.universe import (
    COVERAGE_REASON,
    build_group_snapshot,
    snapshot_usable,
)


def _settings(**kw) -> StyleCrowdingSettings:
    return StyleCrowdingSettings(**kw)


def test_coverage_below_threshold_marks_invalid():
    settings = _settings(min_group_size=2, min_group_coverage=0.80)
    # 池子 100 只，因子只覆盖 20 只
    values = [float(i) for i in range(20)] + [np.nan] * 80
    exposure = pd.Series(values, index=[f"S{i:03d}" for i in range(100)])
    snap = build_group_snapshot(
        "2020-01-02", exposure, "style_momentum", settings, group_count=5
    )
    assert not snap.valid
    assert snap.reason == COVERAGE_REASON
    assert pytest.approx(snap.factor_coverage, abs=1e-9) == 0.20
    # 分组仍然保留，供更松的 article 口径使用
    assert snap.high and snap.low


def test_full_coverage_stays_valid():
    settings = _settings(min_group_size=2, min_group_coverage=0.80)
    exposure = pd.Series(
        np.arange(100, dtype=float), index=[f"S{i:03d}" for i in range(100)]
    )
    snap = build_group_snapshot(
        "2020-01-02", exposure, "style_momentum", settings, group_count=5
    )
    assert snap.valid
    assert snap.reason == "OK"


def test_article_threshold_is_looser():
    settings = _settings(min_group_size=2, min_group_coverage=0.80)
    values = [float(i) for i in range(70)] + [np.nan] * 30
    exposure = pd.Series(values, index=[f"S{i:03d}" for i in range(100)])
    snap = build_group_snapshot(
        "2020-01-02", exposure, "style_momentum", settings, group_count=5
    )
    assert not snapshot_usable(snap)
    assert snapshot_usable(snap, min_coverage=0.60)
    assert not snapshot_usable(snap, min_coverage=0.90)


def _dispersion_ctx() -> StyleCrowdingContext:
    dates = ("2020-01-02", "2020-01-03")
    symbols = [f"S{i:02d}" for i in range(10)]
    idx = pd.Index(dates)
    # 低组波动大、高组波动小，正反两个方向的 d_top 必须不一样
    rows = []
    for _ in dates:
        rows.append([0.10, -0.10, 0.09, -0.09, 0.08] + [0.001] * 5)
    close_returns = pd.DataFrame(rows, index=idx, columns=symbols)
    settings = _settings(min_group_size=1, dispersion_top_counts=(5,))
    from StyleCrowding.universe import GroupSnapshot

    snaps = {
        d: GroupSnapshot(
            "style_momentum",
            d,
            10,
            10,
            1.0,
            tuple(symbols[5:]),
            tuple(symbols[:5]),
            True,
            "OK",
        )
        for d in dates
    }
    empty = pd.DataFrame(index=idx, columns=symbols, dtype=float)
    return StyleCrowdingContext(
        settings=settings,
        dates=dates,
        output_dates=dates,
        factor_ids=("style_momentum",),
        symbols=tuple(symbols),
        close=empty,
        pre_close=empty,
        pb=empty,
        pe=empty,
        ps=empty,
        circ_mv=empty,
        mask=pd.DataFrame(True, index=idx, columns=symbols),
        style_panels={},
        exposure_btop=empty,
        close_returns=close_returns,
        groups={10: {"style_momentum": snaps}},
        dispersion={
            10: {
                "style_momentum": _build_dispersion_for_groups(
                    dates, close_returns, snaps, settings
                )
            }
        },
    )


def test_reverse_dispersion_uses_flipped_groups():
    ctx = _dispersion_ctx()
    positive = ctx.dispersion_for(10, "style_momentum", "positive")
    reverse = ctx.dispersion_for(10, "style_momentum", "reverse")
    # 正向高组是后 5 只（近乎无波动），反向高组是前 5 只（波动大）
    assert float(positive["d_top"].iloc[0]) < float(reverse["d_top"].iloc[0])
    # 缓存命中后仍然是同一个结果
    assert reverse.equals(ctx.dispersion_for(10, "style_momentum", "reverse"))


def _pct(values) -> pd.Series:
    return pd.Series(values, index=range(len(values)), dtype="float64")


def test_alarm_stays_on_until_release_days_of_calm():
    hot = THRESHOLD + 0.05
    cold = 0.5
    # 先连续触发，然后彻底冷下来
    values = [hot] * CONFIRM_HITS + [cold] * (RELEASE_DAYS * 2)
    active = _alarm_path(_pct(values), direction="high")
    assert bool(active[CONFIRM_HITS - 1])
    # 解除前的每一天都还在警报里
    for i in range(CONFIRM_HITS, CONFIRM_HITS + RELEASE_DAYS - 1):
        assert bool(active[i]), f"第 {i} 天不该解除"
    assert not bool(active[CONFIRM_HITS + RELEASE_DAYS - 1])


def test_alarm_does_not_flicker_on_brief_dips():
    hot = THRESHOLD + 0.05
    cold = 0.5
    # 触发后每隔几天回到尾部一次，永远攒不满 RELEASE_DAYS
    values = [hot] * CONFIRM_HITS
    for _ in range(4):
        values += [cold] * (RELEASE_DAYS - 2) + [hot]
    active = _alarm_path(_pct(values), direction="high")
    assert active[CONFIRM_HITS - 1 :].all()


def test_alarm_needs_confirmation_before_firing():
    hot = THRESHOLD + 0.05
    cold = 0.5
    # 窗口内命中次数不够，不该触发
    values = ([hot] + [cold] * CONFIRM_WINDOW) * 4
    active = _alarm_path(_pct(values), direction="high")
    assert not active.any()


def test_warmup_covers_stacked_windows():
    settings = _settings()
    warm = warmup_trading_days(settings)
    assert warm >= settings.risk_covariance_window + settings.prior_z_window


def test_splice_keeps_published_head(tmp_path):
    from StyleCrowding.writer import write_series

    published = tmp_path / "s.parquet"
    old = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
    )
    write_series(published, old)

    fresh = pd.Series(
        [30.0, 40.0, 50.0],
        index=["2020-01-06", "2020-01-07", "2020-01-08"],
    )
    merged = _splice(published, fresh, "2020-01-07")
    assert list(merged.index) == [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
    ]
    # 拼接点之前保留旧值，之后用新值
    np.testing.assert_allclose(merged.to_numpy(), [1.0, 2.0, 3.0, 40.0, 50.0])


def test_splice_without_published_returns_fresh(tmp_path):
    fresh = pd.Series([1.0], index=["2020-01-02"])
    assert _splice(tmp_path / "missing.parquet", fresh, "2020-01-02").equals(fresh)
