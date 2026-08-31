from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .environment import FolderCacheEnvironment, RequestResult
from .errors import DataIntegrityError


@dataclass(frozen=True)
class RewardScales:
    count_scale: float
    bytes_scale: float

    def __post_init__(self) -> None:
        if self.count_scale <= 0 or self.bytes_scale <= 0:
            raise ValueError("奖励归一化尺度必须大于0")


@dataclass(frozen=True)
class MacroReward:
    delta_count: int
    delta_bytes: int
    normalized_count: float
    normalized_bytes: float
    reward: float
    request_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def compute_training_scales(results: Mapping[int, RequestResult], macro_step_count: int) -> RewardScales:
    if macro_step_count <= 0 or not results:
        raise ValueError("训练尺度要求非空请求和正宏步数")
    return RewardScales(
        count_scale=len(results) / macro_step_count,
        bytes_scale=sum(item.total_size_bytes for item in results.values()) / macro_step_count,
    )


def compare_macro_results(
    policy: FolderCacheEnvironment,
    baseline: FolderCacheEnvironment,
    start_time: float,
    end_time: float,
    scales: RewardScales,
    byte_weight: float = 0.8,
    count_weight: float = 0.2,
) -> MacroReward:
    policy_rows = {
        key: value for key, value in policy.results.items() if start_time <= value.access_time < end_time
    }
    baseline_rows = {
        key: value for key, value in baseline.results.items() if start_time <= value.access_time < end_time
    }
    if policy_rows.keys() != baseline_rows.keys():
        raise DataIntegrityError("策略和基线宏步请求编号不一致")
    delta_count = 0
    delta_bytes = 0
    for request_id in sorted(policy_rows):
        policy_item = policy_rows[request_id]
        baseline_item = baseline_rows[request_id]
        if policy_item.path_index != baseline_item.path_index or policy_item.total_size_bytes != baseline_item.total_size_bytes:
            raise DataIntegrityError(f"策略和基线请求内容不一致：request_id={request_id}")
        delta = int(policy_item.hit) - int(baseline_item.hit)
        delta_count += delta
        delta_bytes += policy_item.total_size_bytes * delta
    normalized_count = delta_count / scales.count_scale
    normalized_bytes = delta_bytes / scales.bytes_scale
    reward = float(byte_weight) * normalized_bytes + float(count_weight) * normalized_count
    if not np.isfinite(reward):
        raise FloatingPointError("宏步奖励不是有限值")
    return MacroReward(delta_count, delta_bytes, normalized_count, normalized_bytes, reward, len(policy_rows))


def reward_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max_abs": 0.0}
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("奖励序列包含非有限值")
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max_abs": float(np.max(np.abs(array))),
    }

