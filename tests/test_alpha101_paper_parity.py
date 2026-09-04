"""Compare rewritten Alpha101 formulas against pre-rewrite golden values (all 101)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from DailyUpdates.factor_updates.factors import _alpha101_engine as eng

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from alpha101_parity_panel import (  # noqa: E402
    PARITY_ALPHA_IDS,
    make_parity_market,
    patch_industry_loader,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "alpha101_parity.npz"
RTOL = 1e-5
ATOL = 1e-6


def _wide_from_engine(engine, template: pd.DataFrame, alpha_id: int) -> np.ndarray:
    values = getattr(engine, f"alpha{alpha_id:03d}")()
    wide = values.to_frame() if isinstance(values, pd.Series) else values
    wide = wide.reindex(index=template.index, columns=template.columns)
    if wide.dtypes.apply(lambda t: np.issubdtype(t, np.bool_)).any():
        wide = wide.astype(np.float64)
    close_ok = np.isfinite(template.to_numpy(dtype=np.float64, copy=False))
    wide = wide.where(pd.DataFrame(close_ok, index=template.index, columns=template.columns))
    return wide.to_numpy(dtype=np.float64)


@pytest.fixture(scope="module")
def golden():
    assert GOLDEN_PATH.is_file(), f"missing golden fixture: {GOLDEN_PATH}"
    packed = np.load(GOLDEN_PATH)
    return packed


@pytest.fixture(scope="module")
def computed():
    orig = patch_industry_loader(eng)
    eng.clear_prepared_engine()
    try:
        engine, template = eng.prepare_alpha_engine(make_parity_market())
        stack = np.stack(
            [_wide_from_engine(engine, template, aid) for aid in PARITY_ALPHA_IDS],
            axis=0,
        )
        return {
            "dates": template.index.astype(str).to_numpy(),
            "symbols": template.columns.astype(str).to_numpy(),
            "values": stack,
        }
    finally:
        eng._load_industry_panel = orig
        eng.clear_prepared_engine()


def test_golden_covers_representative_ids(golden):
    assert tuple(int(x) for x in golden["alpha_ids"]) == PARITY_ALPHA_IDS


@pytest.mark.parametrize("alpha_id", PARITY_ALPHA_IDS)
def test_paper_rewrite_matches_frozen_golden(alpha_id, golden, computed):
    ids = [int(x) for x in golden["alpha_ids"]]
    idx = ids.index(alpha_id)
    expect = golden["values"][idx]
    got = computed["values"][idx]
    assert list(computed["dates"]) == list(golden["dates"])
    assert list(computed["symbols"]) == list(golden["symbols"])
    if expect.shape != got.shape:
        raise AssertionError(f"alpha{alpha_id:03d} shape {got.shape} != {expect.shape}")
    if np.allclose(got, expect, rtol=RTOL, atol=ATOL, equal_nan=True):
        return
    both_nan = np.isnan(got) & np.isnan(expect)
    diff = np.abs(got - expect)
    diff[both_nan] = 0.0
    n_bad = int(np.sum(~both_nan & ~np.isclose(got, expect, rtol=RTOL, atol=ATOL, equal_nan=True)))
    raise AssertionError(
        f"alpha{alpha_id:03d} mismatch: max_abs={np.nanmax(diff):.6g} "
        f"n_cells={n_bad}/{diff.size}"
    )
