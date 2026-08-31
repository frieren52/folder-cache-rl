from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from .errors import DataIntegrityError


@dataclass
class WaitingRequest:
    request_id: int
    arrival_time: float


@dataclass
class FetchTask:
    path_index: int
    source: str
    submit_time: float
    start_sequence: int
    total_size_bytes: int
    service_start_time: float | None = None
    ready_time: float | None = None
    channel_id: int | None = None
    waiting_requests: list[WaitingRequest] = field(default_factory=list)
    demand_merged_into_prefetch: bool = False

    def service_seconds(self, fixed_setup_seconds: float, bandwidth: float) -> float:
        return float(fixed_setup_seconds) + self.total_size_bytes / float(bandwidth)


class FifoChannelScheduler:
    """需求和预取共享的中央FIFO与独占通道调度器。"""

    def __init__(
        self,
        channel_count: int,
        bandwidth_bytes_per_second_per_channel: float,
        fixed_setup_seconds: float = 0.0,
    ) -> None:
        if int(channel_count) <= 0 or float(bandwidth_bytes_per_second_per_channel) <= 0:
            raise ValueError("通道数和带宽必须大于0")
        if float(fixed_setup_seconds) < 0:
            raise ValueError("固定准备时间不能为负")
        self.channel_count = int(channel_count)
        self.bandwidth = float(bandwidth_bytes_per_second_per_channel)
        self.fixed_setup_seconds = float(fixed_setup_seconds)
        self.queue: deque[FetchTask] = deque()
        self.channels: dict[int, FetchTask] = {}
        self.in_flight: dict[int, FetchTask] = {}
        self._next_sequence = 0

    def __len__(self) -> int:
        return len(self.in_flight)

    def task_for(self, path_index: int) -> FetchTask | None:
        return self.in_flight.get(int(path_index))

    def submit(
        self,
        path_index: int,
        source: str,
        submit_time: float,
        total_size_bytes: int,
        waiting_request: WaitingRequest | None = None,
    ) -> tuple[FetchTask, bool]:
        key = int(path_index)
        if source not in {"demand", "prefetch"}:
            raise ValueError(f"未知任务来源：{source}")
        existing = self.in_flight.get(key)
        if existing is not None:
            if waiting_request is not None:
                existing.waiting_requests.append(waiting_request)
                if existing.source == "prefetch":
                    existing.demand_merged_into_prefetch = True
            return existing, False
        size = int(total_size_bytes)
        if size <= 0:
            raise DataIntegrityError(f"任务对象大小必须为正数：path_index={key}")
        task = FetchTask(
            path_index=key,
            source=source,
            submit_time=float(submit_time),
            start_sequence=self._next_sequence,
            total_size_bytes=size,
            waiting_requests=[] if waiting_request is None else [waiting_request],
        )
        self._next_sequence += 1
        self.queue.append(task)
        self.in_flight[key] = task
        return task, True

    def dispatch(self, now: float) -> list[FetchTask]:
        started: list[FetchTask] = []
        idle_channels = [channel for channel in range(self.channel_count) if channel not in self.channels]
        for channel_id in idle_channels:
            if not self.queue:
                break
            task = self.queue.popleft()
            task.channel_id = channel_id
            task.service_start_time = float(now)
            task.ready_time = float(now) + task.service_seconds(self.fixed_setup_seconds, self.bandwidth)
            self.channels[channel_id] = task
            started.append(task)
        return started

    def next_ready_time(self) -> float | None:
        values = [task.ready_time for task in self.channels.values() if task.ready_time is not None]
        return min(values) if values else None

    def complete_at(self, event_time: float) -> list[FetchTask]:
        completed = [
            task
            for task in self.channels.values()
            if task.ready_time is not None and task.ready_time <= float(event_time) + 1e-9
        ]
        completed.sort(key=lambda task: (float(task.ready_time), task.start_sequence))
        for task in completed:
            if task.channel_id is None:
                raise DataIntegrityError("服务中任务缺少channel_id")
            self.channels.pop(task.channel_id)
            self.in_flight.pop(task.path_index, None)
        return completed

    def estimate_appended_ready_time(self, now: float, total_size_bytes: int) -> float:
        availability: list[tuple[float, int]] = []
        for channel_id in range(self.channel_count):
            task = self.channels.get(channel_id)
            ready = float(now) if task is None else max(float(now), float(task.ready_time))
            heapq.heappush(availability, (ready, channel_id))
        queued_sizes = [task.total_size_bytes for task in self.queue]
        queued_sizes.append(int(total_size_bytes))
        candidate_ready = float(now)
        for size in queued_sizes:
            available, channel_id = heapq.heappop(availability)
            start = max(float(now), available)
            candidate_ready = start + self.fixed_setup_seconds + size / self.bandwidth
            heapq.heappush(availability, (candidate_ready, channel_id))
        return candidate_ready

    def snapshot(self) -> dict[str, object]:
        return {
            "channel_count": self.channel_count,
            "bandwidth": self.bandwidth,
            "fixed_setup_seconds": self.fixed_setup_seconds,
            "next_sequence": self._next_sequence,
            "queue": [asdict(task) for task in self.queue],
            "channels": {str(channel): asdict(task) for channel, task in self.channels.items()},
        }

    @staticmethod
    def _task(value: Mapping[str, object]) -> FetchTask:
        waiting = [WaitingRequest(int(item["request_id"]), float(item["arrival_time"])) for item in value.get("waiting_requests", [])]  # type: ignore[index,union-attr]
        return FetchTask(
            path_index=int(value["path_index"]),
            source=str(value["source"]),
            submit_time=float(value["submit_time"]),
            start_sequence=int(value["start_sequence"]),
            total_size_bytes=int(value["total_size_bytes"]),
            service_start_time=None if value.get("service_start_time") is None else float(value["service_start_time"]),
            ready_time=None if value.get("ready_time") is None else float(value["ready_time"]),
            channel_id=None if value.get("channel_id") is None else int(value["channel_id"]),
            waiting_requests=waiting,
            demand_merged_into_prefetch=bool(value.get("demand_merged_into_prefetch", False)),
        )

    @classmethod
    def restore(cls, value: Mapping[str, object]) -> "FifoChannelScheduler":
        scheduler = cls(int(value["channel_count"]), float(value["bandwidth"]), float(value["fixed_setup_seconds"]))
        scheduler._next_sequence = int(value["next_sequence"])
        scheduler.queue = deque(cls._task(item) for item in value.get("queue", []))  # type: ignore[arg-type]
        raw_channels = value.get("channels", {})
        if not isinstance(raw_channels, Mapping):
            raise DataIntegrityError("调度器快照channels格式错误")
        scheduler.channels = {int(channel): cls._task(item) for channel, item in raw_channels.items()}  # type: ignore[arg-type]
        for task in list(scheduler.queue) + list(scheduler.channels.values()):
            if task.path_index in scheduler.in_flight:
                raise DataIntegrityError("调度器快照存在重复在途对象")
            scheduler.in_flight[task.path_index] = task
        return scheduler

