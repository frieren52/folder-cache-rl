from __future__ import annotations

import heapq
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


class RollingAccessState:
    """Incremental state for recent objects, one-hour popularity and 24-hour activity."""

    def __init__(self, hot_window_seconds: int = 3600, active_window_seconds: int = 86400) -> None:
        self.hot_window_seconds = int(hot_window_seconds)
        self.active_window_seconds = int(active_window_seconds)
        self.last_access: dict[int, int] = {}
        self.hot_counts: Counter[int] = Counter()
        self._hot_last: dict[int, int] = {}
        self._recent_heap: list[tuple[int, int]] = []
        self._hot_heap: list[tuple[int, int, int]] = []
        self._expiry_heap: list[tuple[int, int]] = []
        self._hour_groups: deque[tuple[int, tuple[tuple[int, int], ...]]] = deque()
        self.dirty: set[int] = set()

    def restore(
        self,
        cutoff_time: int,
        last_access: Mapping[int, int],
        hot_groups: Iterable[tuple[int, Mapping[int, int]]],
    ) -> None:
        """Restore a compact warm state without retaining one heap item per historical event."""
        self.last_access = {int(key): int(value) for key, value in last_access.items() if int(value) > cutoff_time - self.active_window_seconds}
        self.hot_counts.clear()
        self._hot_last.clear()
        self._recent_heap.clear()
        self._hot_heap.clear()
        self._expiry_heap.clear()
        self._hour_groups.clear()
        for path_index, event_time in self.last_access.items():
            heapq.heappush(self._recent_heap, (-event_time, path_index))
            heapq.heappush(self._expiry_heap, (event_time + self.active_window_seconds, path_index))
        lower = cutoff_time - self.hot_window_seconds
        for event_time, path_counts in hot_groups:
            if int(event_time) < lower:
                continue
            compact: list[tuple[int, int]] = []
            for raw_path, raw_count in path_counts.items():
                path_index = int(raw_path)
                count = int(raw_count)
                if count <= 0:
                    continue
                compact.append((path_index, count))
                self.hot_counts[path_index] += count
                self._hot_last[path_index] = max(int(event_time), self._hot_last.get(path_index, -1))
            if compact:
                self._hour_groups.append((int(event_time), tuple(compact)))
        for path_index, count in self.hot_counts.items():
            heapq.heappush(self._hot_heap, (-count, -self._hot_last[path_index], path_index))
        self.dirty.clear()

    def ingest_group(self, event_time: int, path_counts: Mapping[int, int]) -> None:
        compact: list[tuple[int, int]] = []
        for raw_path, raw_count in path_counts.items():
            path_index = int(raw_path)
            count = int(raw_count)
            if count <= 0:
                continue
            compact.append((path_index, count))
            self.last_access[path_index] = int(event_time)
            self.hot_counts[path_index] += count
            self._hot_last[path_index] = int(event_time)
            self.dirty.add(path_index)
            heapq.heappush(self._recent_heap, (-int(event_time), path_index))
            heapq.heappush(self._expiry_heap, (int(event_time) + self.active_window_seconds, path_index))
            heapq.heappush(self._hot_heap, (-self.hot_counts[path_index], -int(event_time), path_index))
        if compact:
            self._hour_groups.append((int(event_time), tuple(compact)))

    def advance(self, cutoff_time: int) -> list[int]:
        hot_lower = int(cutoff_time) - self.hot_window_seconds
        while self._hour_groups and self._hour_groups[0][0] < hot_lower:
            event_time, values = self._hour_groups.popleft()
            for path_index, count in values:
                self.hot_counts[path_index] -= count
                if self.hot_counts[path_index] <= 0:
                    self.hot_counts.pop(path_index, None)
                    self._hot_last.pop(path_index, None)
                else:
                    heapq.heappush(
                        self._hot_heap,
                        (-self.hot_counts[path_index], -self._hot_last[path_index], path_index),
                    )
        removed: list[int] = []
        while self._expiry_heap and self._expiry_heap[0][0] <= cutoff_time:
            expiry, path_index = heapq.heappop(self._expiry_heap)
            if self.last_access.get(path_index, -1) + self.active_window_seconds != expiry:
                continue
            if self.last_access[path_index] <= cutoff_time - self.active_window_seconds:
                del self.last_access[path_index]
                removed.append(path_index)
        if len(self._recent_heap) > max(1024, 4 * len(self.last_access)):
            self._recent_heap = [(-event_time, path_index) for path_index, event_time in self.last_access.items()]
            heapq.heapify(self._recent_heap)
        if len(self._expiry_heap) > max(1024, 4 * len(self.last_access)):
            self._expiry_heap = [(event_time + self.active_window_seconds, path_index) for path_index, event_time in self.last_access.items()]
            heapq.heapify(self._expiry_heap)
        if len(self._hot_heap) > max(1024, 4 * len(self.hot_counts)):
            self._hot_heap = [(-count, -self._hot_last[path_index], path_index) for path_index, count in self.hot_counts.items()]
            heapq.heapify(self._hot_heap)
        return removed

    def _recent(self, count: int) -> list[int]:
        selected: list[tuple[int, int]] = []
        result: list[int] = []
        while self._recent_heap and len(result) < count:
            item = heapq.heappop(self._recent_heap)
            event_time = -item[0]
            path_index = item[1]
            if self.last_access.get(path_index) != event_time:
                continue
            selected.append(item)
            result.append(path_index)
        for item in selected:
            heapq.heappush(self._recent_heap, item)
        return result

    def _hot(self, excluded: set[int], count: int) -> list[int]:
        selected: list[tuple[int, int, int]] = []
        result: list[int] = []
        while self._hot_heap and len(result) < count:
            item = heapq.heappop(self._hot_heap)
            negative_count, negative_last, path_index = item
            current = self.hot_counts.get(path_index)
            if current is None or -negative_count != current or -negative_last != self._hot_last.get(path_index):
                continue
            selected.append(item)
            if path_index not in excluded:
                result.append(path_index)
        for item in selected:
            heapq.heappush(self._hot_heap, item)
        return result

    def select_context(self, max_objects: int = 256, recent_objects: int = 128) -> dict[str, np.ndarray]:
        recent = self._recent(recent_objects)
        hot = self._hot(set(recent), max_objects - len(recent))
        ids = recent + hot
        recent_set = set(recent)
        hot_set = set(self._hot(set(), max_objects))
        padded = ids + [-1] * (max_objects - len(ids))
        return {
            "path_indices": np.asarray(padded, dtype=np.int64),
            "valid_mask": np.asarray([value >= 0 for value in padded], dtype=np.bool_),
            "recent_mask": np.asarray([value in recent_set for value in padded], dtype=np.bool_),
            "hot_mask": np.asarray([value in hot_set for value in padded], dtype=np.bool_),
        }

    def recent_ids(self, count: int = 256) -> list[int]:
        return self._recent(int(count))

    def hot_ids(self, count: int = 256) -> list[int]:
        return self._hot(set(), int(count))

    def consume_dirty(self) -> set[int]:
        value = set(self.dirty)
        self.dirty.clear()
        return value


@dataclass
class HistoryIndexEntry:
    vector: np.ndarray
    last_event_time: int
    vector_as_of_time: int
    next_refresh_time: int


class HistoryVectorIndex:
    def __init__(
        self,
        recent_window_seconds: int = 3600,
        active_window_seconds: int = 86400,
        recent_refresh_seconds: int = 300,
        long_refresh_seconds: int = 3600,
    ) -> None:
        self.recent_window_seconds = int(recent_window_seconds)
        self.active_window_seconds = int(active_window_seconds)
        self.recent_refresh_seconds = int(recent_refresh_seconds)
        self.long_refresh_seconds = int(long_refresh_seconds)
        self.entries: dict[int, HistoryIndexEntry] = {}
        self.revision = 0
        self._due_heap: list[tuple[int, int]] = []

    def _next_refresh(self, cutoff_time: int, last_event_time: int) -> int:
        crossing = last_event_time + self.recent_window_seconds
        if cutoff_time < crossing:
            return min(cutoff_time + self.recent_refresh_seconds, crossing)
        return cutoff_time + self.long_refresh_seconds

    def select_changes(
        self,
        cutoff_time: int,
        last_access: Mapping[int, int],
        dirty_ids: Iterable[int],
        context_ids: Iterable[int],
    ) -> tuple[list[int], list[int]]:
        expired = sorted(
            path_index
            for path_index, entry in self.entries.items()
            if last_access.get(path_index, entry.last_event_time) <= cutoff_time - self.active_window_seconds
        )
        refresh = {int(value) for value in dirty_ids}
        refresh.update(int(value) for value in context_ids if int(value) >= 0)
        while self._due_heap and self._due_heap[0][0] <= cutoff_time:
            due, path_index = heapq.heappop(self._due_heap)
            entry = self.entries.get(path_index)
            if entry is not None and entry.next_refresh_time == due:
                refresh.add(path_index)
        if len(self._due_heap) > max(1024, 4 * len(self.entries)):
            self._due_heap = [(entry.next_refresh_time, path_index) for path_index, entry in self.entries.items()]
            heapq.heapify(self._due_heap)
        refresh.difference_update(expired)
        refresh.intersection_update(last_access)
        return sorted(refresh), expired

    def publish(
        self,
        cutoff_time: int,
        path_indices: Sequence[int],
        vectors: np.ndarray,
        last_access: Mapping[int, int],
        removed_ids: Sequence[int] = (),
    ) -> int:
        ids = np.asarray(path_indices, dtype=np.int64)
        values = np.asarray(vectors, dtype=np.float32)
        if values.shape != (len(ids), 128):
            raise ValueError("历史刷新向量必须为[N,128]")
        changed = False
        for path_index in removed_ids:
            changed = self.entries.pop(int(path_index), None) is not None or changed
        for path_index, vector in zip(ids, values):
            key = int(path_index)
            event_time = int(last_access[key])
            next_refresh = self._next_refresh(int(cutoff_time), event_time)
            self.entries[key] = HistoryIndexEntry(
                vector=np.asarray(vector, dtype=np.float32),
                last_event_time=event_time,
                vector_as_of_time=int(cutoff_time),
                next_refresh_time=next_refresh,
            )
            heapq.heappush(self._due_heap, (next_refresh, key))
            changed = True
        if changed:
            self.revision += 1
        return self.revision

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(sorted(self.entries), dtype=np.int64)
        vectors = np.stack([self.entries[int(value)].vector for value in ids]) if len(ids) else np.empty((0, 128), dtype=np.float32)
        return ids, vectors

    def vectors_for(self, path_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        vectors = np.zeros((len(path_indices), 128), dtype=np.float32)
        valid = np.zeros(len(path_indices), dtype=np.bool_)
        for position, path_index in enumerate(path_indices):
            entry = self.entries.get(int(path_index))
            if entry is not None:
                vectors[position] = entry.vector
                valid[position] = True
        return vectors, valid


class NegativeTierState:
    """Maintain the 5m/1h/24h/cold negative pools without rescanning the catalog."""

    def __init__(self, all_path_indices: Iterable[int]) -> None:
        self.last_access: dict[int, int] = {}
        self.tiers = [set(), set(), set(), set(int(value) for value in all_path_indices)]
        self._transitions: list[tuple[int, int, int]] = []

    def restore(self, last_access: Mapping[int, int], cutoff_time: int) -> None:
        all_ids = set().union(*self.tiers)
        self.tiers = [set(), set(), set(), set(all_ids)]
        self.last_access.clear()
        self._transitions.clear()
        boundaries = (0, 300, 3600, 86400)
        for raw_path, raw_time in last_access.items():
            path_index = int(raw_path)
            event_time = int(raw_time)
            age = int(cutoff_time) - event_time
            if age >= 86400:
                continue
            tier = 0 if age < 300 else (1 if age < 3600 else 2)
            self.tiers[3].discard(path_index)
            self.tiers[tier].add(path_index)
            self.last_access[path_index] = event_time
            for target in range(tier + 1, 4):
                heapq.heappush(self._transitions, (event_time + boundaries[target], path_index, target))

    def ingest(self, event_time: int, path_indices: Iterable[int]) -> None:
        for raw_value in path_indices:
            path_index = int(raw_value)
            for tier in self.tiers:
                tier.discard(path_index)
            self.tiers[0].add(path_index)
            self.last_access[path_index] = int(event_time)
            heapq.heappush(self._transitions, (int(event_time) + 300, path_index, 1))
            heapq.heappush(self._transitions, (int(event_time) + 3600, path_index, 2))
            heapq.heappush(self._transitions, (int(event_time) + 86400, path_index, 3))

    def advance(self, cutoff_time: int) -> None:
        boundaries = (0, 300, 3600, 86400)
        while self._transitions and self._transitions[0][0] <= cutoff_time:
            due, path_index, target = heapq.heappop(self._transitions)
            if self.last_access.get(path_index, -1) + boundaries[target] != due:
                continue
            for tier in self.tiers:
                tier.discard(path_index)
            self.tiers[target].add(path_index)
            if target == 3:
                self.last_access.pop(path_index, None)
        if len(self._transitions) > max(1024, 6 * len(self.last_access)):
            boundaries = (0, 300, 3600, 86400)
            rebuilt: list[tuple[int, int, int]] = []
            for path_index, event_time in self.last_access.items():
                age = int(cutoff_time) - event_time
                tier = 0 if age < 300 else (1 if age < 3600 else 2)
                for target in range(tier + 1, 4):
                    rebuilt.append((event_time + boundaries[target], path_index, target))
            self._transitions = rebuilt
            heapq.heapify(self._transitions)

    def pools(self) -> tuple[set[int], set[int], set[int], set[int]]:
        return tuple(self.tiers)  # type: ignore[return-value]
