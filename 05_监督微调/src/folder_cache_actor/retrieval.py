from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, eps)


def stable_top_k(path_indices: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(path_indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float32)
    if ids.ndim != 1 or values.shape != ids.shape:
        raise ValueError("path_indices和scores必须是一维同形数组")
    if k <= 0:
        raise ValueError("k必须大于0")
    order = np.lexsort((ids, -values))[: min(k, len(ids))]
    return ids[order], values[order]


@dataclass(frozen=True)
class RetrievalResult:
    static_ids: np.ndarray
    history_ids: np.ndarray
    union_ids: np.ndarray
    static_scores: np.ndarray
    history_scores: np.ndarray
    fusion_scores: np.ndarray
    fusion_ids: np.ndarray


class ExactDualRetriever:
    def __init__(
        self,
        static_ids: np.ndarray,
        static_vectors: np.ndarray,
        static_top_k: int = 256,
        history_top_k: int = 256,
        fusion_top_k: int = 256,
    ) -> None:
        self.static_ids = np.asarray(static_ids, dtype=np.int64)
        self.static_vectors = normalize_rows(static_vectors)
        if self.static_vectors.shape != (len(self.static_ids), 128):
            raise ValueError("静态向量必须为[N,128]")
        self.static_top_k = int(static_top_k)
        self.history_top_k = int(history_top_k)
        self.fusion_top_k = int(fusion_top_k)
        self._static_position = {int(value): index for index, value in enumerate(self.static_ids)}

    def retrieve(
        self,
        static_query: np.ndarray,
        history_query: np.ndarray,
        fusion_weights: np.ndarray,
        history_ids: np.ndarray,
        history_vectors: np.ndarray,
    ) -> RetrievalResult:
        static_query = normalize_rows(np.asarray(static_query, dtype=np.float32)[None])[0]
        history_query = normalize_rows(np.asarray(history_query, dtype=np.float32)[None])[0]
        weights = np.asarray(fusion_weights, dtype=np.float32)
        if weights.shape != (2,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-5):
            raise ValueError("fusion_weights必须为和为1的两个非负权重")
        history_ids = np.asarray(history_ids, dtype=np.int64)
        history_vectors = normalize_rows(np.asarray(history_vectors, dtype=np.float32))
        static_all_scores = self.static_vectors @ static_query
        static_ids, _ = stable_top_k(self.static_ids, static_all_scores, self.static_top_k)
        if len(history_ids):
            history_all_scores = history_vectors @ history_query
            selected_history_ids, _ = stable_top_k(history_ids, history_all_scores, self.history_top_k)
            history_score_map = {int(key): float(value) for key, value in zip(history_ids, history_all_scores)}
        else:
            selected_history_ids = np.empty(0, dtype=np.int64)
            history_score_map = {}
        union_ids = np.asarray(sorted(set(static_ids.tolist()) | set(selected_history_ids.tolist())), dtype=np.int64)
        union_static = np.asarray(
            [static_all_scores[self._static_position[int(path_index)]] for path_index in union_ids],
            dtype=np.float32,
        )
        union_history = np.asarray([history_score_map.get(int(path_index), 0.0) for path_index in union_ids], dtype=np.float32)
        fusion = weights[0] * union_static + weights[1] * union_history
        fusion_ids, fusion_scores = stable_top_k(union_ids, fusion, self.fusion_top_k)
        return RetrievalResult(
            static_ids=static_ids,
            history_ids=selected_history_ids,
            union_ids=union_ids,
            static_scores=union_static,
            history_scores=union_history,
            fusion_scores=fusion_scores,
            fusion_ids=fusion_ids,
        )

