from __future__ import annotations

import copy
import math
from array import array
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .cache import ByteLRUCache
from .errors import DataIntegrityError, IllegalActionError
from .scheduler import FetchTask, FifoChannelScheduler, WaitingRequest


@dataclass(frozen=True)
class AccessRequest:
    request_id: int
    path_index: int
    access_time: float
    total_size_bytes: int
    sequence: int


@dataclass
class RequestResult:
    request_id: int
    path_index: int
    access_time: float
    total_size_bytes: int
    hit: bool
    ready_time: float | None
    merged_into_prefetch: bool = False

    @property
    def wait_seconds(self) -> float | None:
        return None if self.ready_time is None else max(0.0, self.ready_time - self.access_time)


@dataclass
class EnvironmentCounters:
    request_count: int = 0
    request_bytes: int = 0
    hit_count: int = 0
    hit_bytes: int = 0
    merged_into_prefetch_count: int = 0
    physical_read_bytes: int = 0
    demand_read_bytes: int = 0
    prefetch_read_bytes: int = 0
    completed_prefetch_bytes: int = 0
    useful_prefetch_bytes: int = 0
    unused_prefetch_bytes: int = 0
    pollution_evicted_bytes: int = 0
    illegal_action_count: int = 0


class FolderCacheEnvironment:
    """严格事件顺序的有限通道、FIFO、字节LRU环境。"""

    def __init__(
        self,
        requests: Sequence[AccessRequest],
        object_sizes: Mapping[int, int],
        cache_capacity_bytes: int,
        channel_count: int,
        bandwidth_bytes_per_second_per_channel: float,
        fixed_setup_seconds: float = 0.0,
        start_time: float | None = None,
        result_retention_seconds: float | None = None,
        track_wait_samples: bool = False,
    ) -> None:
        if getattr(requests, "is_time_ordered", False):
            self.requests = requests
        else:
            ordered = sorted(requests, key=lambda item: (item.access_time, item.sequence))
            if list(requests) != ordered:
                raise DataIntegrityError("访问请求必须按时间和原始顺序排列")
            self.requests = tuple(ordered)
        self.object_sizes = {int(key): int(value) for key, value in object_sizes.items()}
        if any(value <= 0 for value in self.object_sizes.values()):
            raise DataIntegrityError("对象大小必须全部为正数")
        self.cache = ByteLRUCache(cache_capacity_bytes)
        self.scheduler = FifoChannelScheduler(channel_count, bandwidth_bytes_per_second_per_channel, fixed_setup_seconds)
        inferred_start = float(self.requests[0].access_time) if self.requests else 0.0
        self.current_time = inferred_start if start_time is None else float(start_time)
        cursor_at = getattr(self.requests, "cursor_at", None)
        self.request_cursor = (
            int(cursor_at(self.current_time))
            if callable(cursor_at)
            else next(
                (index for index, item in enumerate(self.requests) if item.access_time >= self.current_time),
                len(self.requests),
            )
        )
        self.results: dict[int, RequestResult] = {}
        self.counters = EnvironmentCounters()
        self.prefetch_cache_state: dict[int, bool] = {}
        self.result_retention_seconds = (
            None if result_retention_seconds is None else float(result_retention_seconds)
        )
        self.track_wait_samples = bool(track_wait_samples)
        self.wait_samples = array("d")
        self.metrics_start_time = self.current_time
        self._requests_since_prune = 0

    def _size(self, path_index: int) -> int:
        try:
            return self.object_sizes[int(path_index)]
        except KeyError as exc:
            raise DataIntegrityError(f"对象主表缺少path_index={path_index}") from exc

    def is_in_flight(self, path_index: int) -> bool:
        return self.scheduler.task_for(int(path_index)) is not None

    def _finish_task(self, task: FetchTask, event_time: float) -> None:
        inserted = self.cache.insert(task.path_index, task.total_size_bytes)
        if task.source == "prefetch":
            self.counters.completed_prefetch_bytes += task.total_size_bytes
            if inserted.inserted:
                used_by_waiter = bool(task.waiting_requests)
                self.prefetch_cache_state[task.path_index] = used_by_waiter
                if used_by_waiter:
                    self.counters.useful_prefetch_bytes += task.total_size_bytes
            self.counters.pollution_evicted_bytes += sum(item.size_bytes for item in inserted.evicted)
        for evicted in inserted.evicted:
            used = self.prefetch_cache_state.pop(evicted.path_index, None)
            if used is False:
                self.counters.unused_prefetch_bytes += evicted.size_bytes
        for waiter in task.waiting_requests:
            result = self.results.get(waiter.request_id)
            if result is not None:
                result.ready_time = float(event_time)
                if task.source == "prefetch":
                    result.merged_into_prefetch = True
            if self.track_wait_samples and waiter.arrival_time >= self.metrics_start_time:
                self.wait_samples.append(max(0.0, float(event_time) - waiter.arrival_time))

    def _process_completions(self, event_time: float) -> None:
        while True:
            completed = self.scheduler.complete_at(event_time)
            if not completed:
                break
            for task in completed:
                self._finish_task(task, float(task.ready_time))
            self.scheduler.dispatch(event_time)
            next_ready = self.scheduler.next_ready_time()
            if next_ready is None or next_ready > event_time + 1e-9:
                break

    def _process_request(self, request: AccessRequest) -> None:
        if request.request_id in self.results:
            raise DataIntegrityError(f"请求被重复处理：request_id={request.request_id}")
        if request.path_index in self.cache:
            self.cache.touch(request.path_index)
            if self.prefetch_cache_state.get(request.path_index) is False:
                self.prefetch_cache_state[request.path_index] = True
                self.counters.useful_prefetch_bytes += request.total_size_bytes
            self.results[request.request_id] = RequestResult(
                request.request_id,
                request.path_index,
                request.access_time,
                request.total_size_bytes,
                True,
                request.access_time,
            )
            self.counters.request_count += 1
            self.counters.request_bytes += request.total_size_bytes
            self.counters.hit_count += 1
            self.counters.hit_bytes += request.total_size_bytes
            if self.track_wait_samples and request.access_time >= self.metrics_start_time:
                self.wait_samples.append(0.0)
            self._prune_results_if_due(request.access_time)
            return
        result = RequestResult(
            request.request_id,
            request.path_index,
            request.access_time,
            request.total_size_bytes,
            False,
            None,
        )
        self.results[request.request_id] = result
        self.counters.request_count += 1
        self.counters.request_bytes += request.total_size_bytes
        waiter = WaitingRequest(request.request_id, request.access_time)
        task, created = self.scheduler.submit(
            request.path_index,
            "demand",
            request.access_time,
            request.total_size_bytes,
            waiter,
        )
        if task.source == "prefetch":
            result.merged_into_prefetch = True
            self.counters.merged_into_prefetch_count += 1
        if created:
            self.counters.physical_read_bytes += request.total_size_bytes
            self.counters.demand_read_bytes += request.total_size_bytes
        self.scheduler.dispatch(request.access_time)
        self._prune_results_if_due(request.access_time)

    def _prune_results_if_due(self, event_time: float) -> None:
        if self.result_retention_seconds is None:
            return
        self._requests_since_prune += 1
        if self._requests_since_prune < 100_000:
            return
        cutoff = float(event_time) - self.result_retention_seconds
        self.results = {
            request_id: result
            for request_id, result in self.results.items()
            if result.access_time >= cutoff
        }
        self._requests_since_prune = 0

    def reset_metrics(self, start_time: float) -> None:
        self.counters = EnvironmentCounters()
        self.wait_samples = array("d")
        self.metrics_start_time = float(start_time)

    def advance_to(self, target_time: float) -> None:
        target = float(target_time)
        if target < self.current_time:
            raise ValueError("环境时间不能倒退")
        while True:
            next_request_time = math.inf
            if self.request_cursor < len(self.requests):
                candidate = self.requests[self.request_cursor]
                if candidate.access_time < target:
                    next_request_time = float(candidate.access_time)
            next_ready = self.scheduler.next_ready_time()
            completion_time = math.inf if next_ready is None or next_ready > target + 1e-9 else float(next_ready)
            if completion_time == math.inf and next_request_time == math.inf:
                break
            if completion_time <= next_request_time:
                self.current_time = completion_time
                self._process_completions(completion_time)
                continue
            self.current_time = next_request_time
            while self.request_cursor < len(self.requests):
                request = self.requests[self.request_cursor]
                if request.access_time != next_request_time or request.access_time >= target:
                    break
                self._process_request(request)
                self.request_cursor += 1
        self.current_time = target

    def submit_prefetch(self, path_index: int, submit_time: float | None = None) -> FetchTask:
        key = int(path_index)
        when = self.current_time if submit_time is None else float(submit_time)
        if when < self.current_time - 1e-9:
            raise IllegalActionError("预取提交时间早于环境当前时间")
        size = self._size(key)
        if key in self.cache or self.is_in_flight(key) or size > self.cache.capacity_bytes:
            self.counters.illegal_action_count += 1
            raise IllegalActionError(f"不可执行预取动作：path_index={key}")
        task, created = self.scheduler.submit(key, "prefetch", when, size)
        if not created:
            raise IllegalActionError(f"预取动作发生重复在途：path_index={key}")
        self.counters.physical_read_bytes += size
        self.counters.prefetch_read_bytes += size
        self.scheduler.dispatch(when)
        return task

    def estimate_prefetch_ready_time(self, path_index: int, now: float | None = None) -> float:
        return self.scheduler.estimate_appended_ready_time(
            self.current_time if now is None else float(now),
            self._size(path_index),
        )

    def resource_features(self, shadow_batch_ids: Iterable[int] = ()) -> np.ndarray:
        batch = [int(value) for value in shadow_batch_ids]
        batch_bytes = sum(self._size(value) for value in batch)
        service_bytes = sum(task.total_size_bytes for task in self.scheduler.channels.values())
        queue_bytes = sum(task.total_size_bytes for task in self.scheduler.queue)
        demand_tasks = sum(task.source == "demand" for task in self.scheduler.in_flight.values())
        prefetch_tasks = sum(task.source == "prefetch" for task in self.scheduler.in_flight.values())
        return np.asarray(
            [
                (self.scheduler.channel_count - len(self.scheduler.channels)) / self.scheduler.channel_count,
                len(self.scheduler.channels) / self.scheduler.channel_count,
                len(self.scheduler.queue),
                queue_bytes / self.cache.capacity_bytes,
                service_bytes / self.cache.capacity_bytes,
                self.cache.occupied_bytes / self.cache.capacity_bytes,
                demand_tasks,
                prefetch_tasks,
                len(batch),
                batch_bytes / self.cache.capacity_bytes,
            ],
            dtype=np.float32,
        )

    def snapshot(self, include_metrics: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "current_time": self.current_time,
            "request_cursor": self.request_cursor,
            "cache": self.cache.snapshot(),
            "scheduler": self.scheduler.snapshot(),
            "results": (
                {str(key): asdict(item) for key, item in self.results.items()}
                if include_metrics
                else {}
            ),
            "counters": asdict(self.counters),
            "prefetch_cache_state": {str(key): value for key, value in self.prefetch_cache_state.items()},
            "result_retention_seconds": self.result_retention_seconds,
            "track_wait_samples": self.track_wait_samples,
            "metrics_start_time": self.metrics_start_time,
        }
        if include_metrics:
            value["wait_samples"] = list(self.wait_samples)
        return value

    def restore(self, value: Mapping[str, object]) -> None:
        self.current_time = float(value["current_time"])
        self.request_cursor = int(value["request_cursor"])
        self.cache = ByteLRUCache.restore(value["cache"])  # type: ignore[arg-type]
        self.scheduler = FifoChannelScheduler.restore(value["scheduler"])  # type: ignore[arg-type]
        raw_results = value.get("results", {})
        self.results = {int(key): RequestResult(**item) for key, item in raw_results.items()}  # type: ignore[union-attr,arg-type]
        self.counters = EnvironmentCounters(**value.get("counters", {}))  # type: ignore[arg-type]
        self.prefetch_cache_state = {int(key): bool(item) for key, item in value.get("prefetch_cache_state", {}).items()}  # type: ignore[union-attr]
        raw_retention = value.get("result_retention_seconds")
        self.result_retention_seconds = None if raw_retention is None else float(raw_retention)
        self.track_wait_samples = bool(value.get("track_wait_samples", False))
        self.metrics_start_time = float(value.get("metrics_start_time", self.current_time))
        self.wait_samples = array("d", value.get("wait_samples", []))  # type: ignore[arg-type]
        self._requests_since_prune = 0

    def clone(self, include_metrics: bool = True) -> "FolderCacheEnvironment":
        cloned = FolderCacheEnvironment(
            self.requests,
            self.object_sizes,
            self.cache.capacity_bytes,
            self.scheduler.channel_count,
            self.scheduler.bandwidth,
            self.scheduler.fixed_setup_seconds,
            None,
            self.result_retention_seconds,
            self.track_wait_samples if include_metrics else False,
        )
        snapshot = copy.deepcopy(self.snapshot(include_metrics=include_metrics))
        if not include_metrics:
            snapshot["counters"] = asdict(EnvironmentCounters())
            snapshot["track_wait_samples"] = False
            snapshot["wait_samples"] = []
        cloned.restore(snapshot)
        return cloned


def build_requests_from_grouped_events(events: object, catalog: object, object_sizes: Mapping[int, int]) -> list[AccessRequest]:
    """把04/05共用的聚合事件结构展开为稳定的逻辑请求序列。"""
    requests: list[AccessRequest] = []
    request_id = 0
    group_times = getattr(events, "group_times")
    group_offsets = getattr(events, "group_offsets")
    group_positions = getattr(events, "group_positions")
    group_counts = getattr(events, "group_counts")
    path_indices = getattr(catalog, "path_indices")
    for group_index, raw_time in enumerate(group_times):
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
        sequence = 0
        for position in range(start, end):
            path_index = int(path_indices[int(group_positions[position])])
            for _ in range(int(group_counts[position])):
                requests.append(AccessRequest(request_id, path_index, float(raw_time), int(object_sizes[path_index]), sequence))
                request_id += 1
                sequence += 1
    return requests
