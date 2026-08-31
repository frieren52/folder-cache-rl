from __future__ import annotations

import csv
import json
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from .environment import AccessRequest
from .errors import DataIntegrityError
from .utils import SHANGHAI, sha256_file


def load_object_catalog(path: Path) -> tuple[dict[int, int], dict[str, object]]:
    if not path.is_file():
        raise DataIntegrityError(f"对象主表不存在：{path}")
    sizes: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path_index", "total_size_bytes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise DataIntegrityError(f"对象主表缺少字段：{sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                path_index = int(row["path_index"])
                size = int(row["total_size_bytes"])
            except (TypeError, ValueError) as exc:
                raise DataIntegrityError(f"对象主表字段非法：{path}:{line_number}") from exc
            if path_index in sizes or size <= 0:
                raise DataIntegrityError(f"对象主表重复ID或非正大小：{path}:{line_number}")
            sizes[path_index] = size
    if not sizes:
        raise DataIntegrityError("对象主表为空")
    return sizes, {"path": path.as_posix(), "sha256": sha256_file(path), "object_count": len(sizes)}


def _parse_time(date_value: str, time_value: str, path: Path, line_number: int) -> int:
    try:
        value = datetime.strptime(f"{date_value} {time_value}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=SHANGHAI)
    except ValueError as exc:
        raise DataIntegrityError(f"访问时间非法：{path}:{line_number}") from exc
    return int(value.timestamp())


def iter_access_requests(access_dir: Path, object_sizes: dict[int, int]) -> Iterator[AccessRequest]:
    files = sorted(access_dir.glob("access_*.txt"), key=lambda item: item.name)
    if not files:
        raise DataIntegrityError(f"没有访问日志：{access_dir}/access_*.txt")
    request_id = 0
    previous_time: int | None = None
    same_second_sequence = 0
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                parts = line.split()
                if len(parts) != 3:
                    raise DataIntegrityError(f"访问日志必须为 path_index 日期 时间：{path}:{line_number}")
                try:
                    path_index = int(parts[0])
                except ValueError as exc:
                    raise DataIntegrityError(f"访问日志path_index非法：{path}:{line_number}") from exc
                size = object_sizes.get(path_index)
                if size is None:
                    raise DataIntegrityError(f"访问日志对象不在主表：{path}:{line_number}: {path_index}")
                timestamp = _parse_time(parts[1], parts[2], path, line_number)
                if previous_time is not None and timestamp < previous_time:
                    raise DataIntegrityError(f"访问日志时间倒退：{path}:{line_number}")
                same_second_sequence = same_second_sequence + 1 if timestamp == previous_time else 0
                yield AccessRequest(request_id, path_index, float(timestamp), size, same_second_sequence)
                request_id += 1
                previous_time = timestamp


@dataclass(slots=True)
class CompactAccessTrace(Sequence[AccessRequest]):
    """以紧凑数组保存大规模日志，按需构造单条AccessRequest。"""

    path_indices: np.ndarray
    size_by_position: np.ndarray
    position_by_path: dict[int, int]
    event_times: array
    event_positions: array
    events_by_position: list[array]
    is_time_ordered: bool = True

    def __len__(self) -> int:
        return len(self.event_times)

    def __getitem__(self, index: int | slice) -> AccessRequest | list[AccessRequest]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]  # type: ignore[misc]
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        catalog_position = int(self.event_positions[position])
        return AccessRequest(
            position,
            int(self.path_indices[catalog_position]),
            float(self.event_times[position]),
            int(self.size_by_position[catalog_position]),
            position,
        )

    def access_times_for(self, path_index: int) -> np.ndarray:
        position = self.position_by_path.get(int(path_index))
        if position is None:
            return np.empty(0, dtype=np.int64)
        return np.frombuffer(self.events_by_position[position], dtype=np.int64)

    def cursor_at(self, timestamp: float) -> int:
        times = np.frombuffer(self.event_times, dtype=np.int64)
        return int(np.searchsorted(times, float(timestamp), side="left"))

    def summarize(self, start_time: int, end_time: int) -> tuple[int, int]:
        times = np.frombuffer(self.event_times, dtype=np.int64)
        left = int(np.searchsorted(times, int(start_time), side="left"))
        right = int(np.searchsorted(times, int(end_time), side="left"))
        positions = np.frombuffer(self.event_positions, dtype=np.uint32, count=max(0, right - left), offset=left * 4)
        total_bytes = int(self.size_by_position[positions].sum(dtype=np.int64)) if len(positions) else 0
        return right - left, total_bytes


def load_access_requests(access_dir: Path, object_sizes: dict[int, int]) -> CompactAccessTrace:
    files = sorted(access_dir.glob("access_*.txt"), key=lambda item: item.name)
    if not files:
        raise DataIntegrityError(f"没有访问日志：{access_dir}/access_*.txt")
    path_indices = np.asarray(list(object_sizes), dtype=np.int64)
    size_by_position = np.asarray([object_sizes[int(value)] for value in path_indices], dtype=np.int64)
    position_by_path = {int(value): position for position, value in enumerate(path_indices)}
    event_times = array("q")
    event_positions = array("I")
    events_by_position = [array("q") for _ in range(len(path_indices))]
    previous_time: int | None = None
    last_timestamp_text: str | None = None
    last_timestamp_value: int | None = None
    started = time.monotonic()
    total_rows = 0
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            rows = 0
            for line_number, line in enumerate(stream, start=1):
                parts = line.split()
                if len(parts) != 3:
                    raise DataIntegrityError(f"访问日志必须为 path_index 日期 时间：{path}:{line_number}")
                try:
                    path_index = int(parts[0])
                except ValueError as exc:
                    raise DataIntegrityError(f"访问日志path_index非法：{path}:{line_number}") from exc
                catalog_position = position_by_path.get(path_index)
                if catalog_position is None:
                    raise DataIntegrityError(f"访问日志对象不在主表：{path}:{line_number}: {path_index}")
                timestamp_text = f"{parts[1]} {parts[2]}"
                if timestamp_text == last_timestamp_text:
                    assert last_timestamp_value is not None
                    timestamp = last_timestamp_value
                else:
                    timestamp = _parse_time(parts[1], parts[2], path, line_number)
                    last_timestamp_text = timestamp_text
                    last_timestamp_value = timestamp
                if previous_time is not None and timestamp < previous_time:
                    raise DataIntegrityError(f"访问日志时间倒退：{path}:{line_number}")
                event_times.append(timestamp)
                event_positions.append(catalog_position)
                events_by_position[catalog_position].append(timestamp)
                previous_time = timestamp
                rows += 1
                total_rows += 1
                if total_rows % 1_000_000 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "access_load_progress",
                                "rows": total_rows,
                                "file": path.name,
                                "elapsed_seconds": time.monotonic() - started,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            if rows == 0:
                raise DataIntegrityError(f"访问日志为空：{path}")
    if total_rows >= 1_000_000:
        print(
            json.dumps(
                {
                    "event": "access_load_complete",
                    "rows": total_rows,
                    "files": len(files),
                    "elapsed_seconds": time.monotonic() - started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return CompactAccessTrace(
        path_indices,
        size_by_position,
        position_by_path,
        event_times,
        event_positions,
        events_by_position,
    )
