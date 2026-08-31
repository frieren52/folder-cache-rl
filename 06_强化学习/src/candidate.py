from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from folder_cache_actor.retrieval import ExactDualRetriever
from folder_cache_actor.vector_store import StaticVectorStore

from .environment import FolderCacheEnvironment
from .errors import DataIntegrityError, IllegalActionError
from .features import (
    CANDIDATE_DIM,
    CandidateFeatureInput,
    build_candidate_feature,
    completion_probabilities,
    expected_access_count_5m,
    selected_max_similarity,
)


@dataclass(frozen=True)
class CurrentHistoryBatch:
    path_indices: np.ndarray
    vectors: np.ndarray
    next_access_probabilities: np.ndarray
    expected_access_counts: np.ndarray
    as_of_time: int


class CurrentHistoryEncoder(Protocol):
    def __call__(self, cutoff_time: int, path_indices: Sequence[int]) -> CurrentHistoryBatch: ...


@dataclass(frozen=True)
class CandidateRecord:
    path_index: int
    total_size_bytes: int
    static_score: float
    retrieval_history_score: float
    retrieval_fusion_score: float
    current_history_score: float
    current_fusion_score: float
    history_revision: int
    candidate_as_of_time: int | None
    mask_reason: str | None
    static_vector: np.ndarray
    current_history_vector: np.ndarray
    next_access_probabilities: np.ndarray
    expected_access_count_5m: float
    estimated_ready_time: float
    probability_after_completion: float


@dataclass(frozen=True)
class CandidateBatch:
    cutoff_time: int
    path_indices: np.ndarray
    records: tuple[CandidateRecord, ...]
    features: np.ndarray
    valid_mask: np.ndarray
    action_mask: np.ndarray
    selected_mask: np.ndarray
    resource_features: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.path_indices)
        if self.features.shape != (count, CANDIDATE_DIM):
            raise DataIntegrityError("候选特征形状错误")
        for value in (self.valid_mask, self.action_mask, self.selected_mask):
            if value.shape != (count,) or value.dtype != np.bool_:
                raise DataIntegrityError("候选Mask形状或类型错误")

    def legal_positions(self) -> np.ndarray:
        return np.flatnonzero(self.valid_mask & self.action_mask & ~self.selected_mask)


class CandidateEngine:
    """05双路召回、廉价Mask、04临时刷新和影子动态特征。"""

    def __init__(
        self,
        static_store: StaticVectorStore,
        object_sizes: Mapping[int, int],
        current_history_encoder: CurrentHistoryEncoder,
        static_top_k: int = 256,
        history_top_k: int = 256,
        max_completion_horizon_seconds: int = 3600,
        eviction_history: Callable[[Sequence[int], int], tuple[int, float]] | None = None,
    ) -> None:
        self.static_store = static_store
        self.object_sizes = {int(key): int(value) for key, value in object_sizes.items()}
        self.current_history_encoder = current_history_encoder
        self.max_horizon = int(max_completion_horizon_seconds)
        self.eviction_history = eviction_history
        self.retriever = ExactDualRetriever(
            static_store.path_indices,
            static_store.vectors,
            static_top_k=static_top_k,
            history_top_k=history_top_k,
            fusion_top_k=static_top_k,
        )

    def _cheap_mask(
        self,
        path_index: int,
        environment: FolderCacheEnvironment,
        cutoff_time: int,
        scoring_end_time: int,
    ) -> tuple[str | None, float]:
        size = self.object_sizes.get(int(path_index))
        if size is None or size <= 0:
            return "invalid_size", float("inf")
        if path_index in environment.cache:
            return "cached", float("inf")
        if environment.is_in_flight(path_index):
            return "in_flight", float("inf")
        if size > environment.cache.capacity_bytes:
            return "oversize", float("inf")
        ready = environment.estimate_prefetch_ready_time(path_index)
        if ready - cutoff_time >= self.max_horizon:
            return "completion_horizon", ready
        if ready > scoring_end_time:
            return "after_scoring_end", ready
        return None, ready

    def build(
        self,
        environment: FolderCacheEnvironment,
        cutoff_time: int,
        scoring_end_time: int,
        static_query: np.ndarray,
        history_query: np.ndarray,
        fusion_weights: np.ndarray,
        history_ids: np.ndarray,
        history_vectors: np.ndarray,
        history_revision: int,
    ) -> CandidateBatch:
        retrieval = self.retriever.retrieve(
            static_query,
            history_query,
            fusion_weights,
            history_ids,
            history_vectors,
        )
        path_indices = retrieval.union_ids
        if len(path_indices) > 512:
            raise DataIntegrityError("自然候选并集超过512")
        retrieval_fusion = (
            float(fusion_weights[0]) * retrieval.static_scores
            + float(fusion_weights[1]) * retrieval.history_scores
        )
        mask_reasons: list[str | None] = []
        ready_times: list[float] = []
        refresh_ids: list[int] = []
        for path_index in path_indices:
            reason, ready = self._cheap_mask(int(path_index), environment, cutoff_time, scoring_end_time)
            mask_reasons.append(reason)
            ready_times.append(ready)
            if reason is None:
                refresh_ids.append(int(path_index))
        current_map: dict[int, tuple[np.ndarray, np.ndarray, float, int]] = {}
        if refresh_ids:
            current = self.current_history_encoder(cutoff_time, refresh_ids)
            if not np.array_equal(current.path_indices, np.asarray(refresh_ids, dtype=np.int64)):
                raise DataIntegrityError("04临时刷新必须保持候选输入顺序")
            if (
                current.vectors.shape != (len(refresh_ids), 128)
                or current.next_access_probabilities.shape != (len(refresh_ids), 10)
                or current.expected_access_counts.shape != (len(refresh_ids),)
            ):
                raise DataIntegrityError("04候选临时刷新输出形状错误")
            for path_index, vector, probabilities, expected in zip(
                current.path_indices,
                current.vectors,
                current.next_access_probabilities,
                current.expected_access_counts,
            ):
                current_map[int(path_index)] = (vector, probabilities, float(expected), int(current.as_of_time))
        records: list[CandidateRecord] = []
        history_query = np.asarray(history_query, dtype=np.float32)
        for position, path_index_value in enumerate(path_indices):
            path_index = int(path_index_value)
            static_vector = self.static_store.get([path_index])[0]
            current_item = current_map.get(path_index)
            reason = mask_reasons[position]
            current_vector = np.zeros(128, dtype=np.float32)
            probabilities = np.zeros(10, dtype=np.float32)
            probabilities[9] = 1.0
            expected_5m = 0.0
            as_of: int | None = None
            current_history_score = 0.0
            current_fusion_score = float(fusion_weights[0]) * float(retrieval.static_scores[position])
            probability_after = 0.0
            if current_item is not None:
                raw_vector, raw_probabilities, expected_1h, raw_as_of = current_item
                valid_history_output = (
                    np.all(np.isfinite(raw_vector))
                    and np.all(np.isfinite(raw_probabilities))
                    and np.all(raw_probabilities >= -1e-6)
                    and np.isclose(raw_probabilities.sum(), 1.0, atol=1e-4)
                    and np.isfinite(expected_1h)
                    and expected_1h >= 0
                )
                if not valid_history_output:
                    reason = "invalid_history_output"
                else:
                    current_vector = raw_vector
                    probabilities = raw_probabilities
                    as_of = raw_as_of
                    current_history_score = float(history_query @ current_vector)
                    current_fusion_score = (
                        float(fusion_weights[0]) * float(retrieval.static_scores[position])
                        + float(fusion_weights[1]) * current_history_score
                    )
                    _, probability_after = completion_probabilities(probabilities, ready_times[position] - cutoff_time)
                    expected_5m = expected_access_count_5m(probabilities, expected_1h)
            records.append(
                CandidateRecord(
                    path_index,
                    self.object_sizes.get(path_index, 0),
                    float(retrieval.static_scores[position]),
                    float(retrieval.history_scores[position]),
                    float(retrieval_fusion[position]),
                    current_history_score,
                    current_fusion_score,
                    int(history_revision),
                    as_of,
                    reason,
                    static_vector,
                    np.asarray(current_vector, dtype=np.float32),
                    np.asarray(probabilities, dtype=np.float32),
                    expected_5m,
                    ready_times[position],
                    probability_after,
                )
            )
        selected = np.zeros(len(records), dtype=np.bool_)
        return self.refresh_dynamic(tuple(records), environment, cutoff_time, scoring_end_time, selected)

    def refresh_dynamic(
        self,
        records: tuple[CandidateRecord, ...],
        environment: FolderCacheEnvironment,
        cutoff_time: int,
        scoring_end_time: int,
        selected_mask: np.ndarray,
    ) -> CandidateBatch:
        static_vectors = np.stack([item.static_vector for item in records]) if records else np.empty((0, 128), dtype=np.float32)
        features: list[np.ndarray] = []
        updated: list[CandidateRecord] = []
        action_mask = np.zeros(len(records), dtype=np.bool_)
        valid_mask = np.ones(len(records), dtype=np.bool_)
        for position, record in enumerate(records):
            reason = record.mask_reason
            ready = record.estimated_ready_time
            if reason is None and not bool(selected_mask[position]):
                reason, ready = self._cheap_mask(record.path_index, environment, cutoff_time, scoring_end_time)
            if bool(selected_mask[position]):
                reason = "selected"
            finite_ready = bool(np.isfinite(ready))
            completion = max(0.0, ready - cutoff_time) if finite_ready else 0.0
            beyond = finite_ready and completion >= self.max_horizon
            before, after = completion_probabilities(record.next_access_probabilities, completion)
            evicted = (
                environment.cache.projected_evictions(record.total_size_bytes)
                if record.total_size_bytes > 0
                else []
            )
            evicted_ids = [item.path_index for item in evicted]
            recent_count, min_age = (0, 0.0)
            if evicted_ids and self.eviction_history is not None:
                recent_count, min_age = self.eviction_history(evicted_ids, cutoff_time)
            service = (
                environment.scheduler.fixed_setup_seconds
                + record.total_size_bytes / environment.scheduler.bandwidth
                if record.total_size_bytes > 0
                else 0.0
            )
            queue = max(0.0, completion - service) if finite_ready else 0.0
            feature = build_candidate_feature(
                CandidateFeatureInput(
                    static_vector=record.static_vector,
                    current_history_vector=record.current_history_vector,
                    history_valid=record.candidate_as_of_time is not None,
                    static_score=record.static_score,
                    current_history_score=record.current_history_score,
                    current_fusion_score=record.current_fusion_score,
                    size_ratio=record.total_size_bytes / environment.cache.capacity_bytes,
                    evicted_count=len(evicted),
                    evicted_bytes_ratio=sum(item.size_bytes for item in evicted) / environment.cache.capacity_bytes,
                    evicted_recent_access_count=recent_count,
                    evicted_min_age_seconds=min_age,
                    selected_static_similarity=selected_max_similarity(static_vectors, selected_mask, position),
                    triggers_eviction=bool(evicted),
                    queue_seconds=queue,
                    service_seconds=service,
                    completion_seconds=completion,
                    probability_before_completion=before,
                    probability_after_completion=after,
                    beyond_horizon=beyond,
                    expected_access_count_5m=record.expected_access_count_5m,
                )
            )
            final_reason = reason
            if final_reason is None and (not finite_ready or beyond or ready > scoring_end_time):
                final_reason = "dynamic_completion_horizon"
            action_mask[position] = final_reason is None
            valid_mask[position] = record.total_size_bytes > 0
            features.append(feature)
            updated.append(replace(record, mask_reason=final_reason, estimated_ready_time=ready, probability_after_completion=after))
        return CandidateBatch(
            cutoff_time,
            np.asarray([item.path_index for item in records], dtype=np.int64),
            tuple(updated),
            np.stack(features) if features else np.empty((0, CANDIDATE_DIM), dtype=np.float32),
            valid_mask,
            action_mask,
            np.asarray(selected_mask, dtype=np.bool_),
            environment.resource_features(np.asarray([item.path_index for index, item in enumerate(records) if selected_mask[index]], dtype=np.int64)),
        )

    def select_in_shadow(
        self,
        batch: CandidateBatch,
        environment: FolderCacheEnvironment,
        position: int,
        scoring_end_time: int,
    ) -> CandidateBatch:
        index = int(position)
        if index < 0 or index >= len(batch.records) or not batch.action_mask[index] or batch.selected_mask[index]:
            raise IllegalActionError(f"影子批次动作不可执行：position={index}")
        environment.submit_prefetch(batch.records[index].path_index)
        selected = batch.selected_mask.copy()
        selected[index] = True
        return self.refresh_dynamic(batch.records, environment, batch.cutoff_time, scoring_end_time, selected)


def known_only_shadow(environment: FolderCacheEnvironment, target_time: float) -> FolderCacheEnvironment:
    shadow = environment.clone(include_metrics=False)
    shadow.requests = ()
    shadow.request_cursor = 0
    shadow.advance_to(target_time)
    return shadow
