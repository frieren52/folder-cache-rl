from __future__ import annotations

import unittest

import numpy as np
from folder_cache_actor.vector_store import StaticVectorStore

from common import MODULE_ROOT  # noqa: F401
from src.actor_rl import build_advantage_pairs
from src.candidate import CandidateEngine, CurrentHistoryBatch
from src.environment import AccessRequest, FolderCacheEnvironment
from src.errors import ArtifactCompatibilityError
from src.evaluation import PolicyMetrics, policy_metrics_from_environment, select_frozen_policy
from src.policy_router import PolicyRouter, ScoreResult


class _HistoryEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, cutoff_time: int, path_indices: list[int]) -> CurrentHistoryBatch:
        ids = np.asarray(path_indices, dtype=np.int64)
        self.calls.append(tuple(ids.tolist()))
        vectors = np.zeros((len(ids), 128), dtype=np.float32)
        probabilities = np.zeros((len(ids), 10), dtype=np.float32)
        for row, path_index in enumerate(ids):
            vectors[row, int(path_index) % 128] = 1.0
            probabilities[row, 0] = 0.2
            probabilities[row, 6] = 0.3
            probabilities[row, 9] = 0.5
        return CurrentHistoryBatch(ids, vectors, probabilities, np.full(len(ids), 10.0, dtype=np.float32), cutoff_time)


def _engine_and_batch() -> tuple[CandidateEngine, FolderCacheEnvironment, object, _HistoryEncoder]:
    ids = np.asarray([1, 2, 3], dtype=np.int64)
    vectors = np.zeros((3, 128), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    vectors[2, 2] = 1.0
    store = StaticVectorStore(ids, vectors, {1: 0, 2: 1, 3: 2}, {})
    history = _HistoryEncoder()
    engine = CandidateEngine(store, {1: 10, 2: 20, 3: 30}, history, 3, 3)
    environment = FolderCacheEnvironment([], {1: 10, 2: 20, 3: 30}, 100, 1, 10.0, start_time=0.0)
    environment.cache.insert(1, 10)
    batch = engine.build(
        environment,
        0,
        3600,
        vectors[0],
        vectors[1],
        np.asarray([0.5, 0.5], dtype=np.float32),
        ids,
        vectors,
        7,
    )
    return engine, environment, batch, history


class CandidatePolicyTests(unittest.TestCase):
    def test_cheap_mask_skips_history_and_count_uses_five_minutes(self) -> None:
        _, _, batch, history = _engine_and_batch()
        self.assertEqual(history.calls, [(2, 3)])
        by_id = {item.path_index: item for item in batch.records}
        self.assertEqual(by_id[1].mask_reason, "cached")
        self.assertIsNone(by_id[1].candidate_as_of_time)
        self.assertAlmostEqual(by_id[2].expected_access_count_5m, 4.0, places=5)

    def test_critic_recomputes_after_selection_and_stop_wins_tie(self) -> None:
        engine, environment, batch, _ = _engine_and_batch()
        environment.cache = environment.cache.__class__(100)
        batch = engine.refresh_dynamic(batch.records, environment, 0, 3600, np.zeros(3, dtype=np.bool_))
        calls = 0

        def scorer(current: object) -> ScoreResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ScoreResult(np.asarray([0.0, 2.0, 1.0]), np.asarray([0.0, 2.0, 1.0]), 0.0, 0.0)
            return ScoreResult(np.ones(3), np.ones(3), 1.0, 1.0)

        decision = PolicyRouter(2, 1.0).decide(
            "critic", batch, environment, engine, 3600, "actor", "critic", scorer
        )
        self.assertEqual(decision.selected_path_indices, (2,))
        self.assertEqual(calls, 2)

    def test_critic_route_requires_checkpoint(self) -> None:
        engine, environment, batch, _ = _engine_and_batch()
        with self.assertRaises(ArtifactCompatibilityError):
            PolicyRouter(2, 1.0).decide("critic", batch, environment, engine, 3600, "actor")


class AdvantageEvaluationTests(unittest.TestCase):
    def test_advantage_pairs_are_bounded_and_deterministic(self) -> None:
        values = np.asarray([2.0, 1.5, -1.0, -0.5, 0.2], dtype=np.float32)
        legal = np.ones(5, dtype=np.bool_)
        first = build_advantage_pairs(values, legal, 1.0, 3, np.random.default_rng(2026))
        second = build_advantage_pairs(values, legal, 1.0, 3, np.random.default_rng(2026))
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 3)
        self.assertTrue(all(item.positive_position in {0, 1} for item in first))

    def test_policy_selection_applies_protection_then_metric_order(self) -> None:
        baseline = PolicyMetrics("no_prefetch", 0.10, 0.20, 100.0, 2.0, 4.0, 1000, 0, 0)
        greedy = PolicyMetrics("simple_greedy", 0.12, 0.21, 100.0, 2.0, 4.0, 1050, 10, 100)
        critic = PolicyMetrics("critic", 0.20, 0.22, 200.0, 4.0, 8.0, 1050, 10, 100)
        protection = {
            "max_count_hit_rate_drop_points": 1.0,
            "max_wait_seconds_increase_ratio": 0.05,
            "max_p95_p99_wait_increase_ratio": 0.10,
            "max_physical_read_bytes_increase_ratio": 0.10,
            "max_unused_prefetch_bytes_ratio": 0.30,
        }
        selected, failures = select_frozen_policy(
            {item.policy_name: item for item in (baseline, greedy, critic)}, protection, True
        )
        self.assertEqual(selected, "simple_greedy")
        self.assertIn("wait_seconds", failures["critic"])

    def test_metrics_settle_without_reading_future_requests(self) -> None:
        requests = [
            AccessRequest(0, 1, 0.5, 10, 0),
            AccessRequest(1, 1, 100.0, 10, 0),
        ]
        environment = FolderCacheEnvironment(
            requests, {1: 10}, 20, 1, 10.0, start_time=0.0, track_wait_samples=True
        )
        environment.submit_prefetch(1)
        environment.advance_to(2.0)
        metrics, diagnostics = policy_metrics_from_environment("simple_greedy", environment, 0, 2)
        self.assertEqual(metrics.count_hit_rate, 0.0)
        self.assertAlmostEqual(metrics.wait_seconds, 0.5)
        self.assertEqual(metrics.unused_prefetch_bytes, 0)
        self.assertEqual(diagnostics["merged_into_prefetch_count"], 1)
        self.assertNotIn(1, environment.results)


if __name__ == "__main__":
    unittest.main()
