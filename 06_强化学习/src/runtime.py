from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from folder_cache_actor.data import build_context_features, time_features
from folder_cache_actor.inference import SupervisedActor
from folder_cache_actor.state import HistoryIndexEntry, HistoryVectorIndex, RollingAccessState
from folder_cache_actor.upstream import HistoryEncoderAdapter
from folder_cache_actor.vector_store import StaticVectorStore

from .access import load_access_requests, load_object_catalog
from .candidate import CandidateBatch, CandidateEngine, CurrentHistoryBatch, known_only_shadow
from .environment import AccessRequest, FolderCacheEnvironment
from .errors import DataIntegrityError, IllegalActionError
from .features import (
    CANDIDATE_LOG1P_INDICES,
    RESOURCE_LOG1P_INDICES,
    FeatureNormalizer,
    RunningFeatureStats,
)
from .policy_router import PolicyRouter, RoutedDecision, ScoreResult, initial_exploration_route
from .replay import MacroStepRecord, StateArray
from .rewards import RewardScales, compare_macro_results


@dataclass(frozen=True)
class ActorDecisionState:
    cutoff_time: int
    object_features: np.ndarray
    object_valid_mask: np.ndarray
    time_features: np.ndarray
    static_query: np.ndarray
    history_query: np.ndarray
    fusion_weights: np.ndarray
    history_ids: np.ndarray
    history_vectors: np.ndarray
    history_revision: int


def restore_history_index(path: Path, recent_refresh_seconds: int = 300, long_refresh_seconds: int = 3600) -> HistoryVectorIndex:
    values = np.load(path, allow_pickle=False)
    index = HistoryVectorIndex(
        recent_window_seconds=3600,
        active_window_seconds=86400,
        recent_refresh_seconds=recent_refresh_seconds,
        long_refresh_seconds=long_refresh_seconds,
    )
    ids = values["path_indices"].astype(np.int64)
    vectors = values["history_vectors"].astype(np.float32)
    last = values["last_event_times"].astype(np.int64)
    as_of = values["vector_as_of_times"].astype(np.int64)
    due = values["next_refresh_times"].astype(np.int64)
    for path_index, vector, last_time, vector_time, next_time in zip(ids, vectors, last, as_of, due):
        key = int(path_index)
        index.entries[key] = HistoryIndexEntry(vector, int(last_time), int(vector_time), int(next_time))
        heapq.heappush(index._due_heap, (int(next_time), key))
    index.revision = int(values["history_revision"][0])
    return index


class RuntimeStateProvider:
    """按10秒截点维护05上下文与持久历史索引。"""

    def __init__(
        self,
        requests: Sequence[AccessRequest],
        train_start: int,
        static_store: StaticVectorStore,
        actor: SupervisedActor,
        history_encoder: HistoryEncoderAdapter,
        initial_history_index: Path,
        decision_interval_seconds: int,
        history_encode_batch_size: int = 512,
    ) -> None:
        self.requests = requests
        self.train_start = int(train_start)
        self.static_store = static_store
        self.actor = actor
        self.history_encoder = history_encoder
        self.history_encode_batch_size = int(history_encode_batch_size)
        self.decision_interval = int(decision_interval_seconds)
        if self.decision_interval <= 0:
            raise ValueError("决策周期必须大于0")
        self._compact_access_times = getattr(requests, "access_times_for", None)
        self.access_times: dict[int, np.ndarray] = {}
        if not callable(self._compact_access_times):
            grouped_by_path: dict[int, list[int]] = {}
            for request in requests:
                grouped_by_path.setdefault(request.path_index, []).append(int(request.access_time))
            self.access_times = {key: np.asarray(value, dtype=np.int64) for key, value in grouped_by_path.items()}
        self.access_state = RollingAccessState(3600, 86400)
        warm_last: dict[int, int] = {}
        warm_groups: list[tuple[int, dict[int, int]]] = []
        cursor = 0
        while cursor < len(requests) and requests[cursor].access_time < self.train_start:
            event_time = int(requests[cursor].access_time)
            counts: dict[int, int] = {}
            while cursor < len(requests) and int(requests[cursor].access_time) == event_time:
                path_index = requests[cursor].path_index
                counts[path_index] = counts.get(path_index, 0) + 1
                warm_last[path_index] = event_time
                cursor += 1
            if event_time >= self.train_start - 3600:
                warm_groups.append((event_time, counts))
        self.access_state.restore(self.train_start, warm_last, warm_groups)
        self.cursor = cursor
        self.index = restore_history_index(initial_history_index)
        self.last_cutoff = self.train_start - self.decision_interval

    def access_times_for(self, path_index: int) -> np.ndarray:
        if callable(self._compact_access_times):
            return np.asarray(self._compact_access_times(int(path_index)), dtype=np.int64)
        return self.access_times.get(int(path_index), np.empty(0, dtype=np.int64))

    def encode_current(self, cutoff_time: int, path_indices: Sequence[int]) -> CurrentHistoryBatch:
        encoded = self.history_encoder.encode_as_of(
            int(cutoff_time),
            path_indices,
            self.access_times_for,
            batch_size=self.history_encode_batch_size,
        )
        return CurrentHistoryBatch(
            encoded["path_indices"],
            encoded["vectors"],
            encoded["next_access_probs"],
            encoded["expected_access_counts"],
            int(cutoff_time),
        )

    def _ingest_until(self, cutoff_time: int) -> None:
        while self.cursor < len(self.requests) and self.requests[self.cursor].access_time < cutoff_time:
            event_time = int(self.requests[self.cursor].access_time)
            counts: dict[int, int] = {}
            while self.cursor < len(self.requests) and int(self.requests[self.cursor].access_time) == event_time:
                path_index = self.requests[self.cursor].path_index
                counts[path_index] = counts.get(path_index, 0) + 1
                self.cursor += 1
            self.access_state.ingest_group(event_time, counts)

    def state_at(self, cutoff_time: int) -> ActorDecisionState:
        cutoff = int(cutoff_time)
        if cutoff != self.last_cutoff + self.decision_interval:
            raise DataIntegrityError(
                f"RuntimeStateProvider必须逐决策周期推进：interval={self.decision_interval}, "
                f"last={self.last_cutoff}, next={cutoff}"
            )
        self._ingest_until(cutoff)
        removed = self.access_state.advance(cutoff)
        context = self.access_state.select_context(256, 128)
        dirty = self.access_state.consume_dirty()
        refresh, expired = self.index.select_changes(
            cutoff,
            self.access_state.last_access,
            dirty,
            context["path_indices"][context["valid_mask"]],
        )
        expired = sorted(set(expired) | set(removed))
        if refresh:
            current = self.encode_current(cutoff, refresh)
            refresh_vectors = current.vectors
        else:
            refresh_vectors = np.empty((0, 128), dtype=np.float32)
        revision = self.index.publish(cutoff, refresh, refresh_vectors, self.access_state.last_access, expired)
        persistent = {key: entry.vector for key, entry in self.index.entries.items()}
        features = build_context_features(
            self.static_store,
            context["path_indices"],
            context["valid_mask"],
            context["recent_mask"],
            context["hot_mask"],
            persistent,
        )
        times = time_features(cutoff)
        outputs = self.actor.predict(features[None], context["valid_mask"][None], times[None])
        history_ids, history_vectors = self.index.arrays()
        self.last_cutoff = cutoff
        return ActorDecisionState(
            cutoff,
            features,
            context["valid_mask"].astype(np.bool_),
            times,
            outputs["static_query"][0],
            outputs["history_query"][0],
            outputs["fusion_weights"][0],
            history_ids,
            history_vectors,
            revision,
        )


def state_array(actor_state: ActorDecisionState, candidates: CandidateBatch) -> StateArray:
    return StateArray(
        actor_state.object_features,
        actor_state.object_valid_mask,
        actor_state.time_features,
        candidates.resource_features,
        candidates.path_indices,
        candidates.features,
        candidates.valid_mask,
        candidates.action_mask,
        candidates.selected_mask,
    )


class TorchCriticScorer:
    def __init__(
        self,
        model: object,
        actor_state: ActorDecisionState,
        device: torch.device,
        candidate_normalizer: FeatureNormalizer,
        resource_normalizer: FeatureNormalizer,
    ) -> None:
        self.model = model
        self.actor_state = actor_state
        self.device = device
        self.candidate_normalizer = candidate_normalizer
        self.resource_normalizer = resource_normalizer

    @torch.no_grad()
    def __call__(self, batch: CandidateBatch) -> ScoreResult:
        candidate_features = self.candidate_normalizer.transform(batch.features)
        resource_features = self.resource_normalizer.transform(batch.resource_features)
        arguments = (
            torch.as_tensor(self.actor_state.object_features[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actor_state.object_valid_mask[None], dtype=torch.bool, device=self.device),
            torch.as_tensor(self.actor_state.time_features[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(resource_features[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(candidate_features[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(batch.valid_mask[None], dtype=torch.bool, device=self.device),
            torch.as_tensor(batch.selected_mask[None], dtype=torch.bool, device=self.device),
        )
        first, second = self.model.online(*arguments)
        return ScoreResult(
            first.candidate_q_values[0].float().cpu().numpy(),
            second.candidate_q_values[0].float().cpu().numpy(),
            float(first.stop_q_value[0, 0].cpu()),
            float(second.stop_q_value[0, 0].cpu()),
        )


@dataclass(frozen=True)
class SimulationResult:
    record: MacroStepRecord
    routed: RoutedDecision


class SimulationRunner:
    def __init__(
        self,
        policy_environment: FolderCacheEnvironment,
        baseline_environment: FolderCacheEnvironment,
        state_provider: RuntimeStateProvider,
        candidate_engine: CandidateEngine,
        router: PolicyRouter,
        reward_scales: RewardScales,
        actor_sha256: str,
        run_id: str,
        scoring_end_time: int,
        action_end_time: int,
        rng: np.random.Generator,
        decision_interval_seconds: int,
        initial_exploration: Mapping[str, float] | None = None,
        epsilon_exploration: Mapping[str, float] | None = None,
    ) -> None:
        self.policy = policy_environment
        self.baseline = baseline_environment
        self.state_provider = state_provider
        self.candidate_engine = candidate_engine
        self.router = router
        self.scales = reward_scales
        self.actor_sha256 = actor_sha256
        self.run_id = run_id
        self.scoring_end_time = int(scoring_end_time)
        self.action_end_time = int(action_end_time)
        self.rng = rng
        self.decision_interval = int(decision_interval_seconds)
        if self.decision_interval <= 0:
            raise ValueError("决策周期必须大于0")
        self.initial_exploration = dict(initial_exploration or {})
        self.epsilon_exploration = dict(epsilon_exploration or {})

    def macro_step(
        self,
        macro_step_id: int,
        cutoff_time: int,
        split: str,
        behavior: str,
        scorer_factory: Callable[[ActorDecisionState], object] | None = None,
        critic_sha256: str | None = None,
        epsilon: float = 0.0,
    ) -> SimulationResult:
        cutoff = int(cutoff_time)
        end = cutoff + self.decision_interval
        self.policy.advance_to(cutoff)
        self.baseline.advance_to(cutoff)
        actor_state = self.state_provider.state_at(cutoff)
        scorer = None if scorer_factory is None else scorer_factory(actor_state)
        shadow = known_only_shadow(self.policy, cutoff + self.router.decision_latency)
        candidates = self.candidate_engine.build(
            shadow,
            cutoff,
            self.scoring_end_time,
            actor_state.static_query,
            actor_state.history_query,
            actor_state.fusion_weights,
            actor_state.history_ids,
            actor_state.history_vectors,
            actor_state.history_revision,
        )
        if cutoff >= self.action_end_time:
            routed = self.router.decide_with_trace(
                "no_prefetch", candidates, shadow, self.candidate_engine, self.scoring_end_time, self.actor_sha256
            )
        elif behavior == "initial":
            route = initial_exploration_route(
                self.rng,
                float(self.initial_exploration.get("simple_greedy_probability", 0.50)),
                float(self.initial_exploration.get("direct_stop_probability", 0.25)),
                float(self.initial_exploration.get("random_probability", 0.25)),
            )
            if route == "random":
                routed = self.router.random_with_trace(
                    candidates,
                    shadow,
                    self.candidate_engine,
                    self.scoring_end_time,
                    self.actor_sha256,
                    self.rng,
                    top_k_probability=float(self.initial_exploration.get("random_top32_probability", 0.80)),
                    top_k=int(self.initial_exploration.get("random_top_k", 32)),
                )
            else:
                routed = self.router.decide_with_trace(
                    route, candidates, shadow, self.candidate_engine, self.scoring_end_time, self.actor_sha256
                )
        elif behavior == "epsilon":
            if scorer is None or not critic_sha256:
                raise DataIntegrityError("epsilon行为需要Critic scorer和摘要")
            if float(self.rng.random()) < float(epsilon):
                if float(self.rng.random()) < float(
                    self.epsilon_exploration.get("exploration_stop_probability", 0.50)
                ):
                    routed = self.router.decide_with_trace(
                        "no_prefetch", candidates, shadow, self.candidate_engine, self.scoring_end_time, self.actor_sha256
                    )
                else:
                    routed = self.router.random_with_trace(
                        candidates,
                        shadow,
                        self.candidate_engine,
                        self.scoring_end_time,
                        self.actor_sha256,
                        self.rng,
                        top_k_probability=1.0,
                        top_k=int(self.epsilon_exploration.get("exploration_top_k", 32)),
                    )
            else:
                routed = self.router.decide_with_trace(
                    "critic",
                    candidates,
                    shadow,
                    self.candidate_engine,
                    self.scoring_end_time,
                    self.actor_sha256,
                    critic_sha256,
                    scorer,
                )
        elif behavior in PolicyRouter.ROUTES:
            routed = self.router.decide_with_trace(
                behavior,
                candidates,
                shadow,
                self.candidate_engine,
                self.scoring_end_time,
                self.actor_sha256,
                critic_sha256,
                scorer,
            )
        else:
            raise ValueError(f"未知Replay行为：{behavior}")
        submit_time = cutoff + self.router.decision_latency
        self.policy.advance_to(submit_time)
        for path_index in routed.decision.selected_path_indices:
            try:
                self.policy.submit_prefetch(path_index)
            except IllegalActionError:
                pass
        self.policy.advance_to(end)
        self.baseline.advance_to(end)
        reward = compare_macro_results(self.policy, self.baseline, cutoff, end, self.scales)
        record = MacroStepRecord(
            self.run_id,
            int(macro_step_id),
            cutoff,
            split,
            routed.decision.policy_name,
            tuple(state_array(actor_state, item) for item in routed.states),
            np.asarray(routed.action_positions, dtype=np.int64),
            reward.reward,
            reward.delta_count,
            reward.delta_bytes,
            end >= self.scoring_end_time,
            self.actor_sha256,
            critic_sha256,
        )
        return SimulationResult(record, routed)


def make_feature_accumulators() -> tuple[RunningFeatureStats, RunningFeatureStats]:
    return (
        RunningFeatureStats(274, CANDIDATE_LOG1P_INDICES),
        RunningFeatureStats(10, RESOURCE_LOG1P_INDICES),
    )


def update_feature_accumulators(
    candidate_stats: RunningFeatureStats,
    resource_stats: RunningFeatureStats,
    record: MacroStepRecord,
) -> None:
    for state in record.states:
        legal = state.candidate_valid_mask & state.candidate_action_mask
        if np.any(legal):
            candidate_stats.update(state.candidate_features[legal])
        resource_stats.update(state.resource_features[None])


def load_runtime_inputs(module_root: Path, config: Mapping[str, object]) -> tuple[dict[int, int], Sequence[AccessRequest]]:
    paths = config["paths"]  # type: ignore[index]
    from .utils import resolve_path

    sizes, _ = load_object_catalog(resolve_path(module_root, str(paths["path_catalog"])))
    requests = load_access_requests(resolve_path(module_root, str(paths["access_dir"])), sizes)
    return sizes, requests


def summarize_requests(requests: Sequence[AccessRequest], start_time: int, end_time: int) -> tuple[int, int]:
    compact_summary = getattr(requests, "summarize", None)
    if callable(compact_summary):
        return compact_summary(int(start_time), int(end_time))
    count = 0
    total_bytes = 0
    for request in requests:
        if request.access_time < start_time:
            continue
        if request.access_time >= end_time:
            break
        count += 1
        total_bytes += request.total_size_bytes
    return count, total_bytes
