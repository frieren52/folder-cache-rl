from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from common import MODULE_ROOT  # noqa: F401
from src.access import load_access_requests
from src.cache import ByteLRUCache
from src.environment import AccessRequest, FolderCacheEnvironment
from src.rewards import RewardScales, compare_macro_results


class CacheTests(unittest.TestCase):
    def test_byte_lru_eviction_and_touch(self) -> None:
        cache = ByteLRUCache(10)
        cache.insert(1, 4)
        cache.insert(2, 4)
        cache.touch(1)
        result = cache.insert(3, 4)
        self.assertEqual([item.path_index for item in result.evicted], [2])
        self.assertEqual(cache.lru_items(), ((1, 4), (3, 4)))

    def test_oversize_not_cached(self) -> None:
        cache = ByteLRUCache(10)
        self.assertFalse(cache.insert(1, 11).inserted)
        self.assertEqual(cache.occupied_bytes, 0)


class EnvironmentTests(unittest.TestCase):
    def test_compact_trace_preserves_order_and_summarizes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "access_20260619.txt"
            path.write_text(
                "2 2026-06-19 00:00:00\n1 2026-06-19 00:00:00\n2 2026-06-19 00:00:01\n",
                encoding="utf-8",
            )
            trace = load_access_requests(path.parent, {1: 10, 2: 20})
            self.assertEqual([trace[index].path_index for index in range(len(trace))], [2, 1, 2])
            start = int(trace[0].access_time)
            self.assertEqual(trace.summarize(start, start + 1), (2, 30))
            self.assertEqual(trace.access_times_for(2).tolist(), [start, start + 1])

    def test_completion_precedes_request_at_same_time(self) -> None:
        requests = [
            AccessRequest(0, 1, 0.0, 10, 0),
            AccessRequest(1, 1, 10.0, 10, 0),
        ]
        env = FolderCacheEnvironment(requests, {1: 10}, 20, 1, 1.0, start_time=0.0)
        env.advance_to(10.0)
        self.assertFalse(env.results[0].hit)
        self.assertNotIn(1, env.results)
        env.advance_to(10.1)
        self.assertTrue(env.results[1].hit)

    def test_request_merges_into_prefetch_without_hit(self) -> None:
        requests = [AccessRequest(0, 1, 1.0, 10, 0)]
        env = FolderCacheEnvironment(requests, {1: 10}, 20, 1, 1.0, start_time=0.0)
        env.submit_prefetch(1)
        env.advance_to(2.0)
        self.assertFalse(env.results[0].hit)
        self.assertTrue(env.results[0].merged_into_prefetch)
        self.assertIsNone(env.results[0].ready_time)
        env.advance_to(10.0)
        self.assertEqual(env.results[0].ready_time, 10.0)

    def test_snapshot_clone_is_independent(self) -> None:
        env = FolderCacheEnvironment([], {1: 5}, 10, 1, 5.0, start_time=0.0)
        clone = env.clone()
        clone.submit_prefetch(1)
        self.assertFalse(env.is_in_flight(1))
        self.assertTrue(clone.is_in_flight(1))

    def test_reward_compares_request_ids(self) -> None:
        requests = [AccessRequest(0, 1, 0.0, 10, 0), AccessRequest(1, 1, 2.0, 10, 0)]
        baseline = FolderCacheEnvironment(requests, {1: 10}, 20, 1, 10.0, start_time=0.0)
        policy = baseline.clone()
        policy.submit_prefetch(1)
        baseline.advance_to(3.0)
        policy.advance_to(3.0)
        reward = compare_macro_results(policy, baseline, 0.0, 3.0, RewardScales(1.0, 10.0))
        self.assertEqual(reward.request_count, 2)
        self.assertTrue(isinstance(reward.reward, float))


if __name__ == "__main__":
    unittest.main()
