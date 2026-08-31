from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from src.sampling import OrderStatisticIndex, RollingHistoryPools  # noqa: E402


class OrderStatisticTests(unittest.TestCase):
    def test_rank_order_and_remove(self) -> None:
        index = OrderStatisticIndex()
        for key, entity in (((5, 2), 2), ((1, 9), 9), ((5, 1), 1)):
            index.add(key, entity)
        self.assertEqual(index.entities(), [9, 1, 2])
        self.assertEqual(index.remove((5, 1)), 1)
        self.assertEqual(index.entities(), [9, 2])

    def test_rolling_pool_sampling_is_deterministic(self) -> None:
        pools = RollingHistoryPools(range(8))
        pools.apply_deltas(
            {
                1: (3, 1),
                2: (2, 2),
                3: (5, 3),
                4: (9, 4),
                5: (1, 1),
            }
        )
        config = {
            "seed": 2026,
            "history_low_quantile": 0.5,
            "history_high_quantile": 0.9,
            "high_history_per_snapshot": 1,
            "medium_history_per_snapshot": 1,
            "low_history_per_snapshot": 1,
            "single_history_per_snapshot": 1,
            "no_history_per_snapshot": 1,
        }
        first, sizes = pools.sample(100, config)
        second, _ = pools.sample(100, config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(sizes["no_history"], 3)
        pools.apply_deltas({4: (-9, -4)})
        pools.assert_consistent()
        self.assertEqual(len(pools.no_history), 4)


if __name__ == "__main__":
    unittest.main()
