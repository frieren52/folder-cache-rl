from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import small_config  # noqa: E402
from src.data import build_history_inputs  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_half_open_history_and_duplicate_counts(self) -> None:
        config = small_config()
        snapshot = 100
        times = np.asarray([80, 96, 96, 99, 100, 101], dtype=np.int64)
        scales, state, auxiliary = build_history_inputs(times, snapshot, config)
        self.assertEqual(scales["second_counts"].shape, (4,))
        np.testing.assert_allclose(
            np.expm1(scales["second_counts"]), np.asarray([2, 0, 0, 1]), atol=1e-6
        )
        self.assertEqual(float(state[2]), 0.0)
        self.assertEqual(float(state[3]), 0.0)
        self.assertAlmostEqual(float(auxiliary["raw_continuous"][0]), np.log1p(1))
        self.assertAlmostEqual(float(auxiliary["raw_continuous"][1]), np.log1p(3))

    def test_no_history_flags_and_standardization(self) -> None:
        config = small_config()
        stats = {"mean": [1.0, 2.0], "used_std": [2.0, 4.0]}
        _, state, _ = build_history_inputs(
            np.empty(0, dtype=np.int64), 100, config, feature_stats=stats
        )
        self.assertEqual(state[2:].tolist(), [1.0, 1.0])
        self.assertTrue(np.all(np.isfinite(state)))


if __name__ == "__main__":
    unittest.main()
