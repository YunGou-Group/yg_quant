#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估口径：RankIC 交集排名、两条路径对拍、暴露回归缺失值、跨批相关块。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from FactorEvaluates.cross_section_ic_calculator import CrossSectionICCalculator
from FactorEvaluates.exposure_engine import ExposureEngine, ExposureMatrix, _align
from FactorEvaluates.matrix_utils import (
    spearman_corr_matrix,
    spearman_cross_corr,
    spearman_pairwise,
)


class SpearmanIntersectionTests(unittest.TestCase):
    def test_matches_scipy_on_intersection(self):
        rng = np.random.default_rng(7)
        n = 200
        factor = rng.normal(size=n)
        returns = rng.normal(size=n)
        factor[:40] = np.nan
        returns[-40:] = np.nan

        got = spearman_pairwise(factor[:, None], returns, min_obs=10)[0]
        both = np.isfinite(factor) & np.isfinite(returns)
        want = stats.spearmanr(factor[both], returns[both]).statistic
        self.assertAlmostEqual(got, want, places=10)

    def test_rank_ic_matches_doc_rank_then_pairwise(self):
        rng = np.random.default_rng(11)
        n_days, n_stocks = 25, 120
        factor = rng.normal(size=(n_days, n_stocks))
        fwd = 0.3 * factor + rng.normal(size=(n_days, n_stocks))
        factor[rng.random(factor.shape) < 0.2] = np.nan
        fwd[rng.random(fwd.shape) < 0.2] = np.nan

        calc = CrossSectionICCalculator()
        single = calc.daily_rank_ic(
            pd.DataFrame(factor), pd.DataFrame(fwd), min_obs=10
        ).to_numpy()

        from FactorEvaluates.matrix_utils import rank_ic_pairwise

        batch = np.array(
            [rank_ic_pairwise(factor[t], fwd[t], min_obs=10)[0] for t in range(n_days)]
        )
        np.testing.assert_allclose(single, batch, rtol=1e-9, atol=1e-12)

    def test_rank_ic_differs_from_intersection_spearman(self):
        rng = np.random.default_rng(3)
        n = 150
        factor = rng.normal(size=n)
        returns = rng.normal(size=n)
        returns[:50] = np.nan

        from FactorEvaluates.matrix_utils import rank_ic_pairwise

        doc = rank_ic_pairwise(factor[:, None], returns, min_obs=10)[0]
        intersection = spearman_pairwise(factor[:, None], returns, min_obs=10)[0]
        self.assertNotAlmostEqual(doc, intersection, places=6)


class CrossCorrTests(unittest.TestCase):
    def test_cross_block_matches_full_matrix(self):
        rng = np.random.default_rng(5)
        values = rng.normal(size=(300, 6))
        values[rng.random(values.shape) < 0.1] = np.nan

        full = spearman_corr_matrix(values, min_obs=20)
        block = spearman_cross_corr(values[:, :2], values[:, 2:], min_obs=20)
        np.testing.assert_allclose(full[:2, 2:], block, rtol=1e-9, atol=1e-12)


class ExposureRegressionTests(unittest.TestCase):
    def test_missing_rows_stay_nan(self):
        rng = np.random.default_rng(13)
        n, k = 200, 3
        x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
        y = np.column_stack([x @ np.array([0.5, 1.0, -0.4]) + 0.1 * rng.normal(size=n)])
        y[:30, 0] = np.nan

        beta, resid = ExposureEngine.regress_matrix_day(y, x, min_obs=20)
        self.assertTrue(np.all(np.isnan(resid[:30, 0])))
        self.assertTrue(np.all(np.isfinite(resid[30:, 0])))
        # 缺失行不参与回归 => 残差在有效行上几乎为零均值且方差很小
        self.assertLess(abs(float(np.nanmean(resid[:, 0]))), 1e-9)
        self.assertTrue(np.all(np.isfinite(beta[:, 0])))

    def test_columns_with_different_masks_are_independent(self):
        rng = np.random.default_rng(17)
        n = 240
        x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        truth = x @ np.array([0.0, 2.0, -1.0])
        y = np.column_stack([truth.copy(), truth.copy()])
        y += 0.01 * rng.normal(size=y.shape)
        y[:60, 0] = np.nan
        y[-60:, 1] = np.nan

        beta, resid = ExposureEngine.regress_matrix_day(y, x, min_obs=20)
        self.assertTrue(np.all(np.isnan(resid[:60, 0])))
        self.assertTrue(np.all(np.isnan(resid[-60:, 1])))
        # 两列的 slope 方向一致（同一真值），说明各自用了自己的有效行
        self.assertGreater(beta[1, 0], 0)
        self.assertGreater(beta[1, 1], 0)
        self.assertLess(float(np.nanmax(np.abs(resid))), 0.5)

    def test_all_missing_column_returns_nan(self):
        n = 100
        x = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        y = np.full((n, 1), np.nan)
        beta, resid = ExposureEngine.regress_matrix_day(y, x, min_obs=20)
        self.assertTrue(np.all(np.isnan(beta)))
        self.assertTrue(np.all(np.isnan(resid)))


class ExposureAlignTests(unittest.TestCase):
    def test_align_returns_same_frame_when_already_matched(self):
        frame = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.Index(["2020-01-02", "2020-01-03"]),
            columns=pd.Index(["AAA", "BBB"]),
        )
        out = _align(frame, frame.index, frame.columns)
        self.assertIs(out, frame)

    def test_align_reindexes_without_requiring_a_prior_copy(self):
        frame = pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.Index([pd.Timestamp("2020-01-02")]),
            columns=pd.Index(["AAA", "BBB"]),
        )
        out = _align(
            frame,
            pd.Index(["2020-01-02", "2020-01-03"]),
            pd.Index(["AAA", "CCC"]),
        )
        self.assertEqual(list(out.index), ["2020-01-02", "2020-01-03"])
        self.assertEqual(list(out.columns), ["AAA", "CCC"])
        self.assertEqual(out.loc["2020-01-02", "AAA"], 1.0)
        self.assertTrue(np.isnan(out.loc["2020-01-02", "CCC"]))
        self.assertTrue(np.isnan(out.loc["2020-01-03", "AAA"]))

    def test_matrix_align_returns_self_when_axes_match(self):
        panel = pd.DataFrame(
            np.arange(6.0).reshape(2, 3),
            index=pd.Index(["2020-01-02", "2020-01-03"]),
            columns=pd.Index(["A", "B", "C"]),
        )
        matrix = ExposureMatrix(
            panels={"style_size": panel, "nlsize": panel},
            names=("style_size", "nlsize"),
            weighted=False,
            winsor_p=0.05,
            weights=panel,
        )
        aligned = matrix.align(panel.index, panel.columns)
        self.assertIs(aligned, matrix)


if __name__ == "__main__":
    unittest.main()
