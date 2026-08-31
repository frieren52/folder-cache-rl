from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .errors import DataIntegrityError
from .utils import finite_or_raise


CANDIDATE_DIM = 274
RESOURCE_DIM = 10
TIME_BIN_EDGES = np.asarray([0, 5, 10, 30, 60, 120, 300, 600, 1800, 3600], dtype=np.float32)
CANDIDATE_LOG1P_INDICES = (261, 263, 264, 267, 268, 269, 273)
RESOURCE_LOG1P_INDICES = (2, 3, 4, 6, 7, 8, 9)


@dataclass(frozen=True)
class CandidateFeatureInput:
    static_vector: np.ndarray
    current_history_vector: np.ndarray
    history_valid: bool
    static_score: float
    current_history_score: float
    current_fusion_score: float
    size_ratio: float
    evicted_count: int
    evicted_bytes_ratio: float
    evicted_recent_access_count: int
    evicted_min_age_seconds: float
    selected_static_similarity: float
    triggers_eviction: bool
    queue_seconds: float
    service_seconds: float
    completion_seconds: float
    probability_before_completion: float
    probability_after_completion: float
    beyond_horizon: bool
    expected_access_count_5m: float


def build_candidate_feature(value: CandidateFeatureInput) -> np.ndarray:
    static = np.asarray(value.static_vector, dtype=np.float32)
    history = np.asarray(value.current_history_vector, dtype=np.float32)
    if static.shape != (128,) or history.shape != (128,):
        raise DataIntegrityError("候选静态和历史向量必须均为[128]")
    result = np.zeros(CANDIDATE_DIM, dtype=np.float32)
    result[:128] = static
    result[128:256] = history
    result[256:] = np.asarray(
        [
            float(value.history_valid),
            value.static_score,
            value.current_history_score,
            value.current_fusion_score,
            value.size_ratio,
            value.evicted_count,
            value.evicted_bytes_ratio,
            value.evicted_recent_access_count,
            value.evicted_min_age_seconds,
            value.selected_static_similarity,
            float(value.triggers_eviction),
            value.queue_seconds,
            value.service_seconds,
            value.completion_seconds,
            value.probability_before_completion,
            value.probability_after_completion,
            float(value.beyond_horizon),
            value.expected_access_count_5m,
        ],
        dtype=np.float32,
    )
    finite_or_raise("candidate_feature", result)
    return result


def completion_probabilities(probabilities: Sequence[float], completion_seconds: float) -> tuple[float, float]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.shape != (10,) or np.any(values < -1e-6) or not np.isclose(values.sum(), 1.0, atol=1e-4):
        raise DataIntegrityError("04首次访问概率必须为和为1的10维非负向量")
    access_probability = float(values[:9].sum())
    delta = max(0.0, float(completion_seconds))
    if delta >= float(TIME_BIN_EDGES[-1]):
        return access_probability, 0.0
    before = 0.0
    for index in range(9):
        lower = float(TIME_BIN_EDGES[index])
        upper = float(TIME_BIN_EDGES[index + 1])
        if delta >= upper:
            before += float(values[index])
            continue
        if delta > lower:
            before += float(values[index]) * (delta - lower) / (upper - lower)
        break
    return before, max(0.0, access_probability - before)


def expected_access_count_5m(probabilities: Sequence[float], expected_access_count_1h: float) -> float:
    """按首次访问概率在前5分钟的占比折算04的一小时次数点估计。"""
    values = np.asarray(probabilities, dtype=np.float32)
    if values.shape != (10,) or np.any(values < -1e-6) or not np.isclose(values.sum(), 1.0, atol=1e-4):
        raise DataIntegrityError("04首次访问概率必须为和为1的10维非负向量")
    expected = float(expected_access_count_1h)
    if not np.isfinite(expected) or expected < 0:
        raise DataIntegrityError("04预计访问次数必须为非负有限值")
    within_hour = float(values[:9].sum())
    if within_hour <= 1e-12:
        return 0.0
    within_five_minutes = float(values[:6].sum())
    return expected * min(1.0, max(0.0, within_five_minutes / within_hour))


@dataclass(frozen=True)
class NormalizationStats:
    dimension: int
    log1p_indices: tuple[int, ...]
    means: np.ndarray
    standard_deviations: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "log1p_indices": list(self.log1p_indices),
            "means": self.means.tolist(),
            "standard_deviations": self.standard_deviations.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "NormalizationStats":
        dimension = int(value["dimension"])
        indices = tuple(int(item) for item in value["log1p_indices"])  # type: ignore[arg-type]
        means = np.asarray(value["means"], dtype=np.float32)
        deviations = np.asarray(value["standard_deviations"], dtype=np.float32)
        if means.shape != (dimension,) or deviations.shape != (dimension,):
            raise DataIntegrityError("归一化统计维度错误")
        return cls(dimension, indices, means, deviations)


class FeatureNormalizer:
    def __init__(self, stats: NormalizationStats | None, dimension: int, log1p_indices: Iterable[int], epsilon: float = 1e-6) -> None:
        self.dimension = int(dimension)
        self.indices = tuple(sorted(set(int(value) for value in log1p_indices)))
        self.epsilon = float(epsilon)
        self.stats = stats

    def _log_transform(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).copy()
        if result.shape[-1] != self.dimension:
            raise DataIntegrityError(f"特征末维必须为{self.dimension}")
        if self.indices:
            selected = result[..., self.indices]
            if np.any(selected < 0):
                raise DataIntegrityError("log1p字段不能为负数")
            result[..., self.indices] = np.log1p(selected)
        return result

    def fit(self, values: np.ndarray) -> NormalizationStats:
        transformed = self._log_transform(values)
        finite_or_raise("normalizer_fit", transformed)
        means = np.zeros(self.dimension, dtype=np.float32)
        deviations = np.ones(self.dimension, dtype=np.float32)
        if self.indices:
            flat = transformed.reshape(-1, self.dimension)
            means[list(self.indices)] = flat[:, self.indices].mean(axis=0)
            deviations[list(self.indices)] = np.maximum(flat[:, self.indices].std(axis=0), self.epsilon)
        self.stats = NormalizationStats(self.dimension, self.indices, means, deviations)
        return self.stats

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.stats is None:
            raise DataIntegrityError("归一化统计尚未拟合")
        if self.stats.dimension != self.dimension or self.stats.log1p_indices != self.indices:
            raise DataIntegrityError("归一化统计与当前字段合同不一致")
        result = self._log_transform(values)
        if self.indices:
            result[..., self.indices] = (
                result[..., self.indices] - self.stats.means[list(self.indices)]
            ) / self.stats.standard_deviations[list(self.indices)]
        finite_or_raise("normalized_features", result)
        return result


class RunningFeatureStats:
    """只累计需要log1p+标准化的字段，避免常驻全部Replay特征。"""

    def __init__(self, dimension: int, log1p_indices: Iterable[int], epsilon: float = 1e-6) -> None:
        self.dimension = int(dimension)
        self.indices = tuple(sorted(set(int(value) for value in log1p_indices)))
        self.epsilon = float(epsilon)
        self.count = 0
        self.sum = np.zeros(len(self.indices), dtype=np.float64)
        self.sum_squares = np.zeros(len(self.indices), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.dimension:
            raise DataIntegrityError(f"运行统计特征末维必须为{self.dimension}")
        flat = array.reshape(-1, self.dimension)
        if not len(flat):
            return
        selected = flat[:, self.indices]
        if np.any(selected < 0) or not np.all(np.isfinite(selected)):
            raise DataIntegrityError("运行统计log1p字段非法")
        transformed = np.log1p(selected).astype(np.float64)
        self.count += len(transformed)
        self.sum += transformed.sum(axis=0)
        self.sum_squares += np.square(transformed).sum(axis=0)

    def finalize(self) -> NormalizationStats:
        if self.count <= 0:
            raise DataIntegrityError("没有可用于拟合归一化统计的训练特征")
        means_selected = self.sum / self.count
        variance = np.maximum(0.0, self.sum_squares / self.count - np.square(means_selected))
        means = np.zeros(self.dimension, dtype=np.float32)
        deviations = np.ones(self.dimension, dtype=np.float32)
        means[list(self.indices)] = means_selected.astype(np.float32)
        deviations[list(self.indices)] = np.maximum(np.sqrt(variance), self.epsilon).astype(np.float32)
        return NormalizationStats(self.dimension, self.indices, means, deviations)

def selected_max_similarity(static_vectors: np.ndarray, selected_mask: np.ndarray, position: int) -> float:
    selected = np.asarray(selected_mask, dtype=np.bool_)
    if not np.any(selected):
        return 0.0
    vectors = np.asarray(static_vectors, dtype=np.float32)
    return float(np.max(vectors[selected] @ vectors[int(position)]))
