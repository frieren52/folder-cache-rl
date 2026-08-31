from __future__ import annotations

import unittest

import numpy as np

from folder_cache_actor.sampling import sample_pairs, sample_positives
from folder_cache_actor.sample_builder import split_for_cutoff
from folder_cache_actor.state import HistoryVectorIndex, NegativeTierState, RollingAccessState


class StateTests(unittest.TestCase):
    def test_context_selection_and_expiry(self) -> None:
        state = RollingAccessState()
        state.ingest_group(100, {3: 4, 1: 2})
        state.ingest_group(110, {2: 1})
        context = state.select_context(max_objects=4, recent_objects=2)
        self.assertEqual(context["path_indices"][:3].tolist(), [2, 1, 3])
        self.assertEqual(state.advance(100 + 86400), [1, 3])

    def test_compact_warm_restore(self) -> None:
        state = RollingAccessState()
        state.restore(10_000, {1: 9_900, 2: 1_000}, [(9_500, {1: 2}), (9_900, {1: 1})])
        self.assertEqual(state.recent_ids(2), [1, 2])
        self.assertEqual(state.hot_ids(2), [1])
        tiers = NegativeTierState([1, 2, 3])
        tiers.restore({1: 9_900, 2: 7_000}, 10_000)
        self.assertIn(1, tiers.pools()[0])
        self.assertIn(2, tiers.pools()[1])
        self.assertIn(3, tiers.pools()[3])

    def test_negative_tier_transitions(self) -> None:
        tiers = NegativeTierState([1, 2])
        tiers.ingest(100, [1])
        tiers.advance(399)
        self.assertIn(1, tiers.pools()[0])
        tiers.advance(400)
        self.assertIn(1, tiers.pools()[1])
        tiers.advance(3700)
        self.assertIn(1, tiers.pools()[2])
        tiers.advance(86500)
        self.assertIn(1, tiers.pools()[3])

    def test_history_revision_and_crossing_schedule(self) -> None:
        index = HistoryVectorIndex()
        revision = index.publish(100, [7], np.ones((1, 128), dtype=np.float32), {7: 90})
        self.assertEqual(revision, 1)
        self.assertEqual(index.entries[7].next_refresh_time, 400)
        refresh, removed = index.select_changes(400, {7: 90}, [], [])
        self.assertEqual(refresh, [7])
        self.assertEqual(removed, [])


class SamplingTests(unittest.TestCase):
    def test_cutoff_split_has_train_test_and_state_only_boundaries(self) -> None:
        horizon = 3600
        test_start = 10_000
        test_end = 20_000
        self.assertEqual(split_for_cutoff(test_start - horizon, horizon, test_start, test_end), "train")
        self.assertEqual(split_for_cutoff(test_start - 1, horizon, test_start, test_end), "transition")
        self.assertEqual(split_for_cutoff(test_start, horizon, test_start, test_end), "test")
        self.assertEqual(split_for_cutoff(test_end - horizon + 1, horizon, test_start, test_end), "tail")

    def test_sampling_is_deterministic_and_excludes_future_positive(self) -> None:
        first = {value: value % 3600 for value in range(200)}
        counts = {value: 200 - value for value in first}
        one = sample_positives(first, counts, seed=42)
        two = sample_positives(first, counts, seed=42)
        self.assertEqual(len(one.sampled_ids), 96)
        self.assertTrue(np.array_equal(one.sampled_ids, two.sampled_ids))
        pools = [range(150, 260), range(260, 360), range(360, 460), range(460, 560)]
        pairs = sample_pairs(one, pools, [0.45, 0.27, 0.18, 0.10], 3, seed=43)
        self.assertTrue(set(pairs.negative_ids.tolist()).isdisjoint(one.all_ids.tolist()))
        self.assertLessEqual(len(pairs.negative_ids), 96 * 3)


if __name__ == "__main__":
    unittest.main()
