#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pack_weights：归一化后仍要满足逐项上限。"""

from __future__ import annotations

import unittest

import numpy as np

from StrategyEngine.allocators.common import pack_weights


class PackWeightsTests(unittest.TestCase):
    def _check(self, weights: np.ndarray, cap: float) -> None:
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertLessEqual(float(weights.max()), cap + 1e-12)
        self.assertGreaterEqual(float(weights.min()), -1e-15)

    def test_uncapped_input_normalizes(self):
        out = pack_weights([1.0, 2.0, 3.0, 4.0], cap=1.0)
        self._check(out, 1.0)
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4])

    def test_cap_survives_normalization(self):
        """旧实现先 clip 再除以总和，会把上限重新放大。"""
        raw = [0.9, 0.05, 0.03, 0.02] + [0.01] * 6
        out = pack_weights(raw, cap=0.2)
        self._check(out, 0.2)
        self.assertAlmostEqual(float(out[0]), 0.2, places=10)

    def test_cascading_overflow(self):
        """钉住第一项后，第二项也可能越界，必须继续迭代。"""
        out = pack_weights([100.0, 50.0, 1.0, 1.0, 1.0], cap=0.3)
        self._check(out, 0.3)
        self.assertAlmostEqual(float(out[0]), 0.3, places=10)
        self.assertAlmostEqual(float(out[1]), 0.3, places=10)

    def test_equal_weight_at_binding_cap(self):
        out = pack_weights([5.0, 1.0, 1.0, 1.0, 1.0], cap=0.2)
        self._check(out, 0.2)
        np.testing.assert_allclose(out, np.full(5, 0.2))

    def test_negative_input_clipped_to_zero(self):
        out = pack_weights([-1.0, 2.0, 2.0], cap=0.6)
        self._check(out, 0.6)
        self.assertAlmostEqual(float(out[0]), 0.0)

    def test_infeasible_cap_falls_back_to_equal_weight(self):
        """cap × n < 1 时满仓与上限无法同时成立，取离可行域最近的等权。"""
        out = pack_weights([1.0, 1.0, 1.0], cap=0.1)
        np.testing.assert_allclose(out, np.full(3, 1.0 / 3.0))

    def test_all_zero_raises(self):
        with self.assertRaises(RuntimeError):
            pack_weights([0.0, 0.0, 0.0], cap=0.5)

    def test_random_inputs_always_feasible(self):
        rng = np.random.default_rng(21)
        for _ in range(200):
            n = int(rng.integers(2, 40))
            raw = rng.random(n) ** 3
            cap = float(rng.uniform(1.0 / n, 1.0))
            self._check(pack_weights(raw, cap=cap), cap)


if __name__ == "__main__":
    unittest.main()
