from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .environment import FolderCacheEnvironment


POLICY_COMPLEXITY = ("no_prefetch", "simple_greedy", "critic", "actor_critic")


@dataclass(frozen=True)
class PolicyMetrics:
    policy_name: str
    byte_hit_rate: float
    count_hit_rate: float
    wait_seconds: float
    p95_wait_seconds: float
    p99_wait_seconds: float
    physical_read_bytes: int
    unused_prefetch_bytes: int
    prefetch_read_bytes: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PolicyMetrics":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})  # type: ignore[arg-type]

    @property
    def unused_prefetch_ratio(self) -> float:
        return self.unused_prefetch_bytes / max(1, self.prefetch_read_bytes)


def protection_failures(
    metrics: PolicyMetrics,
    baseline: PolicyMetrics,
    protection: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    count_drop_points = (baseline.count_hit_rate - metrics.count_hit_rate) * 100.0
    if count_drop_points > float(protection["max_count_hit_rate_drop_points"]):
        failures.append("count_hit_rate")
    if metrics.wait_seconds > baseline.wait_seconds * (1.0 + float(protection["max_wait_seconds_increase_ratio"])):
        failures.append("wait_seconds")
    wait_ratio = float(protection["max_p95_p99_wait_increase_ratio"])
    if metrics.p95_wait_seconds > baseline.p95_wait_seconds * (1.0 + wait_ratio):
        failures.append("p95_wait_seconds")
    if metrics.p99_wait_seconds > baseline.p99_wait_seconds * (1.0 + wait_ratio):
        failures.append("p99_wait_seconds")
    if metrics.physical_read_bytes > baseline.physical_read_bytes * (1.0 + float(protection["max_physical_read_bytes_increase_ratio"])):
        failures.append("physical_read_bytes")
    if metrics.unused_prefetch_ratio > float(protection["max_unused_prefetch_bytes_ratio"]):
        failures.append("unused_prefetch_ratio")
    return failures


def select_frozen_policy(
    values: Mapping[str, PolicyMetrics],
    protection: Mapping[str, float],
    actor_recall_ok: bool,
) -> tuple[str, dict[str, list[str]]]:
    if "no_prefetch" not in values:
        raise ValueError("策略选择缺少no_prefetch")
    baseline = values["no_prefetch"]
    failures: dict[str, list[str]] = {"no_prefetch": []}
    eligible = [baseline]
    for name in POLICY_COMPLEXITY[1:]:
        metrics = values.get(name)
        if metrics is None:
            continue
        reasons = protection_failures(metrics, baseline, protection)
        if name == "actor_critic" and not actor_recall_ok:
            reasons.append("actor_natural_recall")
        failures[name] = reasons
        if not reasons:
            eligible.append(metrics)
    complexity = {name: index for index, name in enumerate(POLICY_COMPLEXITY)}
    selected = sorted(
        eligible,
        key=lambda item: (-item.byte_hit_rate, -item.count_hit_rate, complexity[item.policy_name]),
    )[0]
    return selected.policy_name, failures


def settle_known_tasks(environment: FolderCacheEnvironment) -> FolderCacheEnvironment:
    """在不读取后续访问的前提下结清评分终点的排队和服务中任务。"""
    settled = environment.clone()
    settled.requests = ()
    settled.request_cursor = 0
    while settled.scheduler.in_flight:
        ready_time = settled.scheduler.next_ready_time()
        if ready_time is None:
            raise RuntimeError("存在在途任务但没有可完成的服务中任务")
        settled.advance_to(float(ready_time))
    return settled


def policy_metrics_from_environment(
    policy_name: str,
    environment: FolderCacheEnvironment,
    start_time: int,
    end_time: int,
) -> tuple[PolicyMetrics, dict[str, float | int]]:
    settled = settle_known_tasks(environment)
    counters = settled.counters
    if counters.request_count <= 0:
        raise ValueError("评价时间段没有逻辑请求")
    waits = np.frombuffer(settled.wait_samples, dtype=np.float64)
    if len(waits) != counters.request_count or not np.all(np.isfinite(waits)):
        raise RuntimeError(
            f"评价等待样本与请求数不一致：waits={len(waits)}, requests={counters.request_count}"
        )
    live_unused = sum(
        settled.object_sizes[path_index]
        for path_index, used in settled.prefetch_cache_state.items()
        if used is False and path_index in settled.cache
    )
    unused_prefetch_bytes = settled.counters.unused_prefetch_bytes + live_unused
    metrics = PolicyMetrics(
        policy_name,
        counters.hit_bytes / max(1, counters.request_bytes),
        counters.hit_count / counters.request_count,
        float(waits.sum()),
        float(np.percentile(waits, 95)),
        float(np.percentile(waits, 99)),
        settled.counters.physical_read_bytes,
        unused_prefetch_bytes,
        settled.counters.prefetch_read_bytes,
    )
    diagnostics: dict[str, float | int] = {
        "request_count": counters.request_count,
        "request_bytes": counters.request_bytes,
        "hit_count": counters.hit_count,
        "hit_bytes": counters.hit_bytes,
        "mean_wait_seconds": float(waits.mean()),
        "max_wait_seconds": float(waits.max()),
        "merged_into_prefetch_count": counters.merged_into_prefetch_count,
        "demand_read_bytes": settled.counters.demand_read_bytes,
        "completed_prefetch_bytes": settled.counters.completed_prefetch_bytes,
        "useful_prefetch_bytes": settled.counters.useful_prefetch_bytes,
        "pollution_evicted_bytes": settled.counters.pollution_evicted_bytes,
        "illegal_action_count": settled.counters.illegal_action_count,
    }
    return metrics, diagnostics
