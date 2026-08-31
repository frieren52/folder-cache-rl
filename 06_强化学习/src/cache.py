from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .errors import DataIntegrityError


@dataclass(frozen=True)
class EvictedObject:
    path_index: int
    size_bytes: int


@dataclass(frozen=True)
class CacheInsertResult:
    inserted: bool
    evicted: tuple[EvictedObject, ...]


class ByteLRUCache:
    """按字节容量计费、左端为LRU、右端为MRU。"""

    def __init__(self, capacity_bytes: int) -> None:
        if int(capacity_bytes) <= 0:
            raise ValueError("缓存容量必须大于0")
        self.capacity_bytes = int(capacity_bytes)
        self._items: OrderedDict[int, int] = OrderedDict()
        self.occupied_bytes = 0

    def __contains__(self, path_index: int) -> bool:
        return int(path_index) in self._items

    def __len__(self) -> int:
        return len(self._items)

    def size_of(self, path_index: int) -> int | None:
        return self._items.get(int(path_index))

    def touch(self, path_index: int) -> bool:
        key = int(path_index)
        if key not in self._items:
            return False
        self._items.move_to_end(key, last=True)
        return True

    def insert(self, path_index: int, size_bytes: int) -> CacheInsertResult:
        key = int(path_index)
        size = int(size_bytes)
        if size <= 0:
            raise DataIntegrityError(f"对象大小必须为正数：path_index={key}, size={size}")
        if size > self.capacity_bytes:
            return CacheInsertResult(False, ())
        old_size = self._items.pop(key, None)
        if old_size is not None:
            self.occupied_bytes -= old_size
        evicted: list[EvictedObject] = []
        while self._items and self.occupied_bytes + size > self.capacity_bytes:
            evicted_id, evicted_size = self._items.popitem(last=False)
            self.occupied_bytes -= evicted_size
            evicted.append(EvictedObject(evicted_id, evicted_size))
        if self.occupied_bytes + size > self.capacity_bytes:
            raise DataIntegrityError("LRU容量账本不一致")
        self._items[key] = size
        self.occupied_bytes += size
        return CacheInsertResult(True, tuple(evicted))

    def remove(self, path_index: int) -> EvictedObject | None:
        key = int(path_index)
        size = self._items.pop(key, None)
        if size is None:
            return None
        self.occupied_bytes -= size
        return EvictedObject(key, size)

    def lru_items(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._items.items())

    def projected_evictions(self, size_bytes: int) -> tuple[EvictedObject, ...]:
        size = int(size_bytes)
        if size <= 0 or size > self.capacity_bytes:
            return ()
        remaining = max(0, self.occupied_bytes + size - self.capacity_bytes)
        result: list[EvictedObject] = []
        for path_index, object_size in self._items.items():
            if remaining <= 0:
                break
            result.append(EvictedObject(path_index, object_size))
            remaining -= object_size
        return tuple(result)

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "occupied_bytes": self.occupied_bytes,
            "lru_items": [[path_index, size] for path_index, size in self._items.items()],
        }

    @classmethod
    def restore(cls, value: Mapping[str, object]) -> "ByteLRUCache":
        cache = cls(int(value["capacity_bytes"]))
        raw_items = value.get("lru_items")
        if not isinstance(raw_items, Iterable):
            raise DataIntegrityError("缓存快照缺少lru_items")
        for raw in raw_items:
            path_index, size = raw  # type: ignore[misc]
            key = int(path_index)
            object_size = int(size)
            cache._items[key] = object_size
            cache.occupied_bytes += object_size
        if cache.occupied_bytes != int(value["occupied_bytes"]) or cache.occupied_bytes > cache.capacity_bytes:
            raise DataIntegrityError("缓存快照容量账本错误")
        return cache

