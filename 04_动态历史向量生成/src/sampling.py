from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


@dataclass(slots=True)
class _Node:
    key: tuple[int, ...]
    entity_id: int
    priority: tuple[int, int]
    left: _Node | None = None
    right: _Node | None = None
    size: int = 1


def _size(node: _Node | None) -> int:
    return 0 if node is None else node.size


def _refresh(node: _Node) -> _Node:
    node.size = 1 + _size(node.left) + _size(node.right)
    return node


def _split(root: _Node | None, key: tuple[int, ...]) -> tuple[_Node | None, _Node | None]:
    if root is None:
        return None, None
    if root.key < key:
        left_of_right, right = _split(root.right, key)
        root.right = left_of_right
        return _refresh(root), right
    left, right_of_left = _split(root.left, key)
    root.left = right_of_left
    return left, _refresh(root)


def _merge(left: _Node | None, right: _Node | None) -> _Node | None:
    if left is None:
        return right
    if right is None:
        return left
    if left.priority < right.priority:
        left.right = _merge(left.right, right)
        return _refresh(left)
    right.left = _merge(left, right.left)
    return _refresh(right)


def _insert(root: _Node | None, node: _Node) -> _Node:
    if root is None:
        return node
    if node.key == root.key:
        raise KeyError(f"重复秩索引键：{node.key}")
    if node.priority < root.priority:
        node.left, node.right = _split(root, node.key)
        return _refresh(node)
    if node.key < root.key:
        root.left = _insert(root.left, node)
    else:
        root.right = _insert(root.right, node)
    return _refresh(root)


def _remove(root: _Node | None, key: tuple[int, ...]) -> tuple[_Node | None, int]:
    if root is None:
        raise KeyError(f"秩索引中不存在键：{key}")
    if key == root.key:
        return _merge(root.left, root.right), root.entity_id
    if key < root.key:
        root.left, entity_id = _remove(root.left, key)
    else:
        root.right, entity_id = _remove(root.right, key)
    return _refresh(root), entity_id


class OrderStatisticIndex:
    """Deterministic ordered set with O(log N) expected rank selection."""

    def __init__(self) -> None:
        self._root: _Node | None = None

    def __len__(self) -> int:
        return _size(self._root)

    def add(self, key: tuple[int, ...], entity_id: int) -> None:
        node = _Node(
            key=key,
            entity_id=entity_id,
            priority=(_splitmix64(entity_id + 2026), entity_id),
        )
        self._root = _insert(self._root, node)

    def remove(self, key: tuple[int, ...]) -> int:
        self._root, entity_id = _remove(self._root, key)
        return entity_id

    def kth(self, rank: int) -> tuple[tuple[int, ...], int]:
        if rank < 0 or rank >= len(self):
            raise IndexError(f"秩越界：rank={rank}, size={len(self)}")
        node = self._root
        while node is not None:
            left_size = _size(node.left)
            if rank < left_size:
                node = node.left
            elif rank == left_size:
                return node.key, node.entity_id
            else:
                rank -= left_size + 1
                node = node.right
        raise AssertionError("不可达的秩索引状态")

    def entities(self) -> list[int]:
        return [self.kth(rank)[1] for rank in range(len(self))]


HISTORY_TIER = {
    "no_history": 0,
    "single_history": 1,
    "low_history": 2,
    "medium_history": 3,
    "high_history": 4,
}


class RollingHistoryPools:
    def __init__(self, path_indices: Iterable[int]) -> None:
        values = np.asarray(list(path_indices), dtype=np.int64)
        self.path_indices = values
        self.event_counts = np.zeros(values.shape[0], dtype=np.int64)
        self.distinct_seconds = np.zeros(values.shape[0], dtype=np.int32)
        self.no_history = OrderStatisticIndex()
        self.single_history = OrderStatisticIndex()
        self.multiple_history = OrderStatisticIndex()
        for position, path_index in enumerate(values.tolist()):
            self.no_history.add((int(path_index),), position)

    def _remove_position(self, position: int) -> None:
        path_index = int(self.path_indices[position])
        distinct = int(self.distinct_seconds[position])
        if distinct == 0:
            self.no_history.remove((path_index,))
        elif distinct == 1:
            self.single_history.remove((path_index,))
        else:
            self.multiple_history.remove((int(self.event_counts[position]), path_index))

    def _add_position(self, position: int) -> None:
        path_index = int(self.path_indices[position])
        count = int(self.event_counts[position])
        distinct = int(self.distinct_seconds[position])
        if count < 0 or distinct < 0 or distinct > count:
            raise ValueError(
                f"滚动历史状态非法：path_index={path_index}, count={count}, distinct={distinct}"
            )
        if distinct == 0:
            if count != 0:
                raise ValueError(f"无历史目录事件数不为 0：path_index={path_index}")
            self.no_history.add((path_index,), position)
        elif distinct == 1:
            self.single_history.add((path_index,), position)
        else:
            self.multiple_history.add((count, path_index), position)

    def apply_deltas(self, deltas: dict[int, tuple[int, int]]) -> None:
        for position in sorted(deltas):
            event_delta, distinct_delta = deltas[position]
            if event_delta == 0 and distinct_delta == 0:
                continue
            self._remove_position(position)
            self.event_counts[position] += int(event_delta)
            self.distinct_seconds[position] += int(distinct_delta)
            self._add_position(position)

    @staticmethod
    def _sample_range(
        index: OrderStatisticIndex,
        start: int,
        end: int,
        target: int,
        generator: np.random.Generator,
    ) -> list[int]:
        available = max(end - start, 0)
        take = min(target, available)
        if take == 0:
            return []
        if take == available:
            ranks = np.arange(start, end, dtype=np.int64)
        else:
            ranks = generator.choice(available, size=take, replace=False).astype(np.int64) + start
        return [index.kth(int(rank))[1] for rank in ranks.tolist()]

    def sample(
        self,
        snapshot_unix_seconds: int,
        sampling_config: dict[str, Any],
    ) -> tuple[list[tuple[int, int]], dict[str, int]]:
        generator = np.random.Generator(
            np.random.PCG64(int(sampling_config["seed"]) + int(snapshot_unix_seconds))
        )
        multiple_size = len(self.multiple_history)
        cut50 = int(np.floor(float(sampling_config["history_low_quantile"]) * multiple_size))
        cut90 = int(np.floor(float(sampling_config["history_high_quantile"]) * multiple_size))
        ranges = {
            "high_history": (
                self.multiple_history,
                cut90,
                multiple_size,
                int(sampling_config["high_history_per_snapshot"]),
            ),
            "medium_history": (
                self.multiple_history,
                cut50,
                cut90,
                int(sampling_config["medium_history_per_snapshot"]),
            ),
            "low_history": (
                self.multiple_history,
                0,
                cut50,
                int(sampling_config["low_history_per_snapshot"]),
            ),
            "single_history": (
                self.single_history,
                0,
                len(self.single_history),
                int(sampling_config["single_history_per_snapshot"]),
            ),
            "no_history": (
                self.no_history,
                0,
                len(self.no_history),
                int(sampling_config["no_history_per_snapshot"]),
            ),
        }
        selected: list[tuple[int, int]] = []
        pool_sizes: dict[str, int] = {}
        for name in (
            "high_history",
            "medium_history",
            "low_history",
            "single_history",
            "no_history",
        ):
            index, start, end, target = ranges[name]
            pool_sizes[name] = max(end - start, 0)
            positions = self._sample_range(index, start, end, target, generator)
            selected.extend(
                (int(self.path_indices[position]), HISTORY_TIER[name]) for position in positions
            )
        selected.sort(key=lambda item: item[0])
        return selected, pool_sizes

    def assert_consistent(self) -> None:
        if len(self.no_history) + len(self.single_history) + len(self.multiple_history) != len(
            self.path_indices
        ):
            raise AssertionError("滚动历史池没有覆盖全部目录")
