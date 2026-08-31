from __future__ import annotations

import csv
import hashlib
import math
from array import array
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .config import parse_shanghai_time
from .errors import DataIntegrityError
from .sampling import RollingHistoryPools
from .utils import read_json, sha256_file, verify_manifest_files


SHANGHAI = ZoneInfo("Asia/Shanghai")
SCALE_NAMES = ("second", "short", "medium", "long")


@dataclass(slots=True)
class CatalogData:
    path: Path
    path_indices: np.ndarray
    path_to_position: dict[int, int]
    sha256: str


@dataclass(slots=True)
class PreparedEvents:
    catalog: CatalogData
    access_files: list[dict[str, Any]]
    events_by_position: list[array]
    group_times: array
    group_offsets: array
    group_positions: array
    group_counts: array
    total_events: int
    log_start: int
    log_end_exclusive: int


@dataclass(frozen=True, slots=True)
class SnapshotSplit:
    sample_start: int
    latest_snapshot: int
    split_time: int
    stride_seconds: int
    horizon_seconds: int
    candidate_count: int
    train_count: int
    validation_count: int
    purged_count: int

    def iter_candidates(self) -> Iterator[int]:
        value = self.sample_start
        while value <= self.latest_snapshot:
            yield value
            value += self.stride_seconds

    def partition(self, snapshot: int) -> str | None:
        if snapshot + self.horizon_seconds <= self.split_time:
            return "train"
        if snapshot >= self.split_time:
            return "validation"
        return None


def load_catalog(path: Path) -> CatalogData:
    if not path.is_file():
        raise DataIntegrityError(f"目录主表不存在：{path}")
    path_indices: list[int] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or "path_index" not in reader.fieldnames:
                raise DataIntegrityError(f"目录主表缺少 path_index 列：{path}")
            for line_number, row in enumerate(reader, start=2):
                raw_value = row.get("path_index")
                if raw_value is None or raw_value.strip() == "":
                    raise DataIntegrityError(f"目录主表 path_index 为空：{path}:{line_number}")
                try:
                    path_index = int(raw_value)
                except ValueError as exc:
                    raise DataIntegrityError(
                        f"目录主表 path_index 非法：{path}:{line_number}: {raw_value!r}"
                    ) from exc
                if not -(2**63) <= path_index < 2**63:
                    raise DataIntegrityError(f"目录主表 path_index 超出 int64：{path}:{line_number}")
                path_indices.append(path_index)
    except OSError as exc:
        raise DataIntegrityError(f"无法读取目录主表 {path}: {exc}") from exc
    if not path_indices:
        raise DataIntegrityError(f"目录主表为空：{path}")
    if len(set(path_indices)) != len(path_indices):
        raise DataIntegrityError(f"目录主表 path_index 必须唯一：{path}")
    values = np.asarray(path_indices, dtype=np.int64)
    return CatalogData(
        path=path,
        path_indices=values,
        path_to_position={int(value): index for index, value in enumerate(path_indices)},
        sha256=sha256_file(path),
    )


def _parse_timestamp(value: str, path: Path, line_number: int) -> int:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=SHANGHAI)
    except ValueError as exc:
        raise DataIntegrityError(f"非法访问时间：{path}:{line_number}: {value!r}") from exc
    return int(parsed.timestamp())


def load_access_events(catalog: CatalogData, access_dir: Path) -> PreparedEvents:
    files = sorted(access_dir.glob("access_*.txt"), key=lambda item: item.name)
    if not files:
        raise DataIntegrityError(f"没有找到访问日志：{access_dir}/access_*.txt")
    events_by_position = [array("q") for _ in range(len(catalog.path_indices))]
    group_times = array("q")
    group_offsets = array("Q", [0])
    group_positions = array("I")
    group_counts = array("I")
    summaries: list[dict[str, Any]] = []
    total_events = 0
    previous_global_time: int | None = None
    current_group_time: int | None = None
    current_group_counts: dict[int, int] = {}

    def flush_group() -> None:
        nonlocal current_group_time, current_group_counts
        if current_group_time is None:
            return
        group_times.append(current_group_time)
        for position in sorted(current_group_counts):
            count = current_group_counts[position]
            if count > 0xFFFFFFFF:
                raise DataIntegrityError(
                    f"单目录同秒事件数超出 uint32：position={position}, count={count}"
                )
            group_positions.append(position)
            group_counts.append(count)
        group_offsets.append(len(group_positions))
        current_group_time = None
        current_group_counts = {}

    last_timestamp_text: str | None = None
    last_timestamp_value: int | None = None
    for path in files:
        digest = hashlib.sha256()
        rows = 0
        first_time: int | None = None
        last_time: int | None = None
        try:
            with path.open("rb") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    digest.update(raw_line)
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeDecodeError as exc:
                        raise DataIntegrityError(f"日志不是 UTF-8：{path}:{line_number}") from exc
                    parts = line.split()
                    if len(parts) != 3:
                        raise DataIntegrityError(
                            f"日志行必须为 path_index 日期 时间：{path}:{line_number}: {line!r}"
                        )
                    try:
                        path_index = int(parts[0])
                    except ValueError as exc:
                        raise DataIntegrityError(
                            f"非法 path_index：{path}:{line_number}: {parts[0]!r}"
                        ) from exc
                    position = catalog.path_to_position.get(path_index)
                    if position is None:
                        raise DataIntegrityError(
                            f"未知 path_index：{path}:{line_number}: {path_index}"
                        )
                    timestamp_text = f"{parts[1]} {parts[2]}"
                    if timestamp_text == last_timestamp_text:
                        assert last_timestamp_value is not None
                        timestamp = last_timestamp_value
                    else:
                        timestamp = _parse_timestamp(timestamp_text, path, line_number)
                        last_timestamp_text = timestamp_text
                        last_timestamp_value = timestamp
                    if previous_global_time is not None and timestamp < previous_global_time:
                        raise DataIntegrityError(
                            f"访问事件乱序：{path}:{line_number}: {timestamp_text} 早于前一事件"
                        )
                    previous_global_time = timestamp
                    if current_group_time is None:
                        current_group_time = timestamp
                    elif timestamp != current_group_time:
                        flush_group()
                        current_group_time = timestamp
                    current_group_counts[position] = current_group_counts.get(position, 0) + 1
                    events_by_position[position].append(timestamp)
                    rows += 1
                    total_events += 1
                    first_time = timestamp if first_time is None else first_time
                    last_time = timestamp
        except OSError as exc:
            raise DataIntegrityError(f"无法读取访问日志 {path}: {exc}") from exc
        if rows == 0 or first_time is None or last_time is None:
            raise DataIntegrityError(f"访问日志为空：{path}")
        summaries.append(
            {
                "path": path.as_posix(),
                "rows": rows,
                "first_time": datetime.fromtimestamp(first_time, SHANGHAI).isoformat(),
                "last_time": datetime.fromtimestamp(last_time, SHANGHAI).isoformat(),
                "sha256": digest.hexdigest(),
            }
        )
    flush_group()
    if total_events == 0 or not group_times:
        raise DataIntegrityError("解析后的访问事件为空")
    if total_events != sum(item["rows"] for item in summaries):
        raise DataIntegrityError("输入文件总行数与解析事件数不一致")
    return PreparedEvents(
        catalog=catalog,
        access_files=summaries,
        events_by_position=events_by_position,
        group_times=group_times,
        group_offsets=group_offsets,
        group_positions=group_positions,
        group_counts=group_counts,
        total_events=total_events,
        log_start=int(group_times[0]),
        log_end_exclusive=int(group_times[-1]) + 1,
    )


def determine_snapshot_split(config: Mapping[str, Any], events: PreparedEvents) -> SnapshotSplit:
    sample_start = int(parse_shanghai_time(config["history"]["sample_start"], "history.sample_start").timestamp())
    stride = int(config["history"]["snapshot_stride_seconds"])
    horizon = int(config["target"]["horizon_seconds"])
    max_window = max(int(scale["window_seconds"]) for scale in config["history"]["scales"])
    if sample_start - max_window < events.log_start:
        raise DataIntegrityError(
            "sample_start 之前没有完整最长历史窗口："
            f"需要起点 <= {datetime.fromtimestamp(sample_start - max_window, SHANGHAI).isoformat()}"
        )
    latest_allowed = events.log_end_exclusive - horizon
    if latest_allowed < sample_start:
        raise DataIntegrityError("日志末尾不足以生成任何完整标签快照")
    latest_snapshot = sample_start + ((latest_allowed - sample_start) // stride) * stride
    candidate_count = (latest_snapshot - sample_start) // stride + 1
    target_ratio = float(config["split"]["train_ratio"])
    best: tuple[float, int, int, int] | None = None
    split_time = sample_start
    while split_time <= latest_snapshot:
        last_train = split_time - horizon
        train_count = 0 if last_train < sample_start else (last_train - sample_start) // stride + 1
        validation_count = (latest_snapshot - split_time) // stride + 1
        if train_count > 0 and validation_count > 0:
            ratio = train_count / (train_count + validation_count)
            candidate = (abs(ratio - target_ratio), split_time, train_count, validation_count)
            if best is None or candidate < best:
                best = candidate
        split_time += stride
    if best is None:
        raise DataIntegrityError("无法找到同时包含训练和验证快照的切分点")
    _, selected_split, train_count, validation_count = best
    purged_count = candidate_count - train_count - validation_count
    return SnapshotSplit(
        sample_start=sample_start,
        latest_snapshot=latest_snapshot,
        split_time=selected_split,
        stride_seconds=stride,
        horizon_seconds=horizon,
        candidate_count=candidate_count,
        train_count=train_count,
        validation_count=validation_count,
        purged_count=purged_count,
    )


class TimelineCursor:
    def __init__(self, events: PreparedEvents, window_seconds: int) -> None:
        self.events = events
        self.window_seconds = int(window_seconds)
        self.add_group = 0
        self.expire_group = 0

    def _accumulate_group(
        self, group_index: int, sign: int, deltas: dict[int, list[int]]
    ) -> None:
        start = int(self.events.group_offsets[group_index])
        end = int(self.events.group_offsets[group_index + 1])
        for item_index in range(start, end):
            position = int(self.events.group_positions[item_index])
            count = int(self.events.group_counts[item_index])
            value = deltas.setdefault(position, [0, 0])
            value[0] += sign * count
            value[1] += sign

    def advance(self, snapshot: int, pools: RollingHistoryPools) -> None:
        deltas: dict[int, list[int]] = {}
        while (
            self.add_group < len(self.events.group_times)
            and int(self.events.group_times[self.add_group]) < snapshot
        ):
            self._accumulate_group(self.add_group, 1, deltas)
            self.add_group += 1
        lower_bound = snapshot - self.window_seconds
        while (
            self.expire_group < len(self.events.group_times)
            and int(self.events.group_times[self.expire_group]) < lower_bound
        ):
            self._accumulate_group(self.expire_group, -1, deltas)
            self.expire_group += 1
        pools.apply_deltas(
            {position: (value[0], value[1]) for position, value in deltas.items()}
        )


class WelfordState:
    def __init__(self, width: int = 2) -> None:
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values: Sequence[float]) -> None:
        vector = np.asarray(values, dtype=np.float64)
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (vector - self.mean)

    def finalize(self, feature_names: Sequence[str]) -> dict[str, Any]:
        if self.count <= 0:
            raise DataIntegrityError("训练集没有可用于标准化的样本")
        std = np.sqrt(self.m2 / self.count)
        used_std = np.where(std < 1e-6, 1.0, std)
        return {
            "schema_version": "dynamic-history-feature-stats/v1",
            "sample_count": self.count,
            "continuous_features": list(feature_names),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "used_std": used_std.tolist(),
            "missing_flags": ["no_history", "no_valid_interval"],
        }


def _scale_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["name"]): item for item in config["history"]["scales"]}


def build_history_inputs(
    access_times: np.ndarray,
    snapshot: int,
    config: Mapping[str, Any],
    feature_stats: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    if access_times.ndim != 1:
        raise ValueError("access_times 必须是一维数组")
    end = int(np.searchsorted(access_times, snapshot, side="left"))
    max_window = max(int(item["window_seconds"]) for item in config["history"]["scales"])
    history_start = int(np.searchsorted(access_times, snapshot - max_window, side="left"))
    history = access_times[history_start:end]
    interval_clip = int(config["history"]["interval_clip_seconds"])
    if history.size == 0:
        recency = interval_clip
        recency_was_clipped = False
        no_history = 1.0
        interval = interval_clip
        interval_was_clipped = False
        no_valid_interval = 1.0
    else:
        recency_raw = snapshot - int(history[-1])
        recency = min(max(recency_raw, 0), interval_clip)
        recency_was_clipped = recency_raw > interval_clip
        no_history = 0.0
        last_second = int(history[-1])
        previous_index = int(np.searchsorted(history, last_second, side="left")) - 1
        if previous_index < 0:
            interval = interval_clip
            interval_was_clipped = False
            no_valid_interval = 1.0
        else:
            interval_raw = last_second - int(history[previous_index])
            interval = min(max(interval_raw, 0), interval_clip)
            interval_was_clipped = interval_raw > interval_clip
            no_valid_interval = 0.0
    raw_continuous = np.asarray([np.log1p(recency), np.log1p(interval)], dtype=np.float64)
    continuous = raw_continuous.copy()
    if feature_stats is not None:
        mean = np.asarray(feature_stats["mean"], dtype=np.float64)
        used_std = np.asarray(feature_stats["used_std"], dtype=np.float64)
        continuous = (continuous - mean) / used_std
    state = np.asarray(
        [continuous[0], continuous[1], no_history, no_valid_interval], dtype=np.float32
    )
    scale_inputs: dict[str, np.ndarray] = {}
    for name, scale in _scale_map(config).items():
        window = int(scale["window_seconds"])
        bucket = int(scale["bucket_seconds"])
        bucket_count = int(scale["bucket_count"])
        left = int(np.searchsorted(access_times, snapshot - window, side="left"))
        values = access_times[left:end]
        if values.size:
            indices = ((values - (snapshot - window)) // bucket).astype(np.int64)
            counts = np.bincount(indices, minlength=bucket_count)[:bucket_count]
        else:
            counts = np.zeros(bucket_count, dtype=np.int64)
        scale_inputs[f"{name}_counts"] = np.log1p(counts).astype(np.float32)
    auxiliary = {
        "raw_continuous": raw_continuous,
        "recency_clipped": recency_was_clipped,
        "interval_clipped": interval_was_clipped,
    }
    return scale_inputs, state, auxiliary


class FeatureBuilder:
    def __init__(self, config: Mapping[str, Any], events: PreparedEvents) -> None:
        self.config = config
        self.events = events
        self.boundaries = np.asarray(config["target"]["time_boundaries_seconds"], dtype=np.int64)
        self.horizon = int(config["target"]["horizon_seconds"])

    def build(
        self,
        path_index: int,
        history_tier: int,
        snapshot: int,
        feature_stats: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        position = self.events.catalog.path_to_position[path_index]
        times = np.frombuffer(self.events.events_by_position[position], dtype=np.int64)
        inputs, state, auxiliary = build_history_inputs(
            times, snapshot, self.config, feature_stats=feature_stats
        )
        label_start = int(np.searchsorted(times, snapshot, side="left"))
        label_end = int(np.searchsorted(times, snapshot + self.horizon, side="left"))
        y_count = label_end - label_start
        if y_count > np.iinfo(np.int32).max:
            raise DataIntegrityError(
                f"y_count 超出 int32：path_index={path_index}, snapshot={snapshot}, count={y_count}"
            )
        if y_count:
            delta = int(times[label_start]) - snapshot
            y_time = int(np.searchsorted(self.boundaries, delta, side="right") - 1)
            y_access = 1
        else:
            y_time = -1
            y_access = 0
        record: dict[str, Any] = {
            "path_index": int(path_index),
            "snapshot_time": int(snapshot) * 1000,
            "history_tier": int(history_tier),
            **inputs,
            "history_state": state,
            "y_access": y_access,
            "y_time": y_time,
            "y_count": int(y_count),
        }
        return record, auxiliary


def parquet_schema(config: Mapping[str, Any]) -> pa.Schema:
    scales = _scale_map(config)
    fields = [
        pa.field("path_index", pa.int64(), nullable=False),
        pa.field("snapshot_time", pa.timestamp("ms", tz="Asia/Shanghai"), nullable=False),
        pa.field("history_tier", pa.int8(), nullable=False),
    ]
    for name in SCALE_NAMES:
        fields.append(
            pa.field(
                f"{name}_counts",
                pa.list_(pa.float32(), int(scales[name]["bucket_count"])),
                nullable=False,
            )
        )
    fields.extend(
        (
            pa.field("history_state", pa.list_(pa.float32(), 4), nullable=False),
            pa.field("y_access", pa.int8(), nullable=False),
            pa.field("y_time", pa.int8(), nullable=False),
            pa.field("y_count", pa.int32(), nullable=False),
        )
    )
    return pa.schema(fields)


class ParquetShardWriter:
    def __init__(self, root: Path, schema: pa.Schema, storage: Mapping[str, Any]) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.schema = schema
        self.max_shard_rows = int(storage["max_rows_per_shard"])
        self.max_group_rows = int(storage["max_rows_per_row_group"])
        self.compression = str(storage["compression"])
        self.compression_level = int(storage["compression_level"])
        self.parquet_version = str(storage["parquet_version"])
        self.buffer: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.shard_index = 0
        self.shard_rows = 0
        self.shard_row_groups = 0
        self.total_rows = 0
        self.shards: list[dict[str, Any]] = []

    def _open_writer(self) -> None:
        path = self.root / f"part-{self.shard_index:05d}.parquet"
        self.writer = pq.ParquetWriter(
            path,
            self.schema,
            version=self.parquet_version,
            compression=self.compression,
            compression_level=self.compression_level,
            use_dictionary=False,
            use_deprecated_int96_timestamps=False,
        )
        self.shard_rows = 0
        self.shard_row_groups = 0

    def _close_writer(self) -> None:
        if self.writer is None:
            return
        path = self.root / f"part-{self.shard_index:05d}.parquet"
        self.writer.close()
        self.shards.append(
            {
                "path": path,
                "rows": self.shard_rows,
                "row_groups": self.shard_row_groups,
            }
        )
        self.writer = None
        self.shard_index += 1

    def write_record(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.max_group_rows:
            self._flush_buffer(self.max_group_rows)

    def _flush_buffer(self, count: int | None = None) -> None:
        if not self.buffer:
            return
        take = len(self.buffer) if count is None else min(count, len(self.buffer))
        records = self.buffer[:take]
        del self.buffer[:take]
        table = pa.Table.from_pylist(records, schema=self.schema)
        self.write_table(table)

    def write_table(self, table: pa.Table) -> None:
        offset = 0
        while offset < table.num_rows:
            if self.writer is None:
                self._open_writer()
            remaining = self.max_shard_rows - self.shard_rows
            take = min(remaining, table.num_rows - offset, self.max_group_rows)
            piece = table.slice(offset, take)
            assert self.writer is not None
            self.writer.write_table(piece, row_group_size=take)
            self.shard_rows += take
            self.shard_row_groups += 1
            self.total_rows += take
            offset += take
            if self.shard_rows == self.max_shard_rows:
                self._close_writer()

    def close(self) -> list[dict[str, Any]]:
        self._flush_buffer()
        self._close_writer()
        return self.shards


def fixed_list_to_numpy(table: pa.Table, name: str, width: int) -> np.ndarray:
    column = table.column(name).combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(table.num_rows, width)


def replace_history_state(table: pa.Table, feature_stats: Mapping[str, Any]) -> pa.Table:
    state = fixed_list_to_numpy(table, "history_state", 4).copy()
    mean = np.asarray(feature_stats["mean"], dtype=np.float32)
    used_std = np.asarray(feature_stats["used_std"], dtype=np.float32)
    state[:, :2] = (state[:, :2] - mean) / used_std
    flat = pa.array(state.reshape(-1), type=pa.float32())
    replacement = pa.FixedSizeListArray.from_arrays(flat, 4)
    index = table.schema.get_field_index("history_state")
    return table.set_column(index, table.schema.field(index), replacement)


class PreparedParquetDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        split_dir: Path,
        config: Mapping[str, Any],
        sample_count: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.split_dir = split_dir
        self.config = config
        self.sample_count = int(sample_count)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size: int | None = None
        self.shards = sorted(split_dir.glob("part-*.parquet"), key=lambda item: item.name)
        if not self.shards and self.sample_count:
            raise DataIntegrityError(f"样本分片缺失：{split_dir}")
        self.units: list[tuple[int, int]] = []
        self.unit_rows: dict[tuple[int, int], int] = {}
        counted_rows = 0
        for shard_index, path in enumerate(self.shards):
            parquet = pq.ParquetFile(path)
            counted_rows += parquet.metadata.num_rows
            for group in range(parquet.num_row_groups):
                unit = (shard_index, group)
                self.units.append(unit)
                self.unit_rows[unit] = parquet.metadata.row_group(group).num_rows
        if counted_rows != self.sample_count:
            raise DataIntegrityError(
                f"样本数与元数据不一致：{split_dir}，期望 {self.sample_count}，实际 {counted_rows}"
            )

    def __len__(self) -> int:
        if self.batch_size is None:
            return self.sample_count
        return self.batch_count()

    def configure_batching(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.batch_size = int(batch_size)

    def batch_count(self) -> int:
        if self.batch_size is None:
            raise RuntimeError("读取数据前必须先调用 configure_batching")
        return sum(math.ceil(rows / self.batch_size) for rows in self.unit_rows.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_info(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.batch_size is None:
            raise RuntimeError("PreparedParquetDataset 必须通过 make_loader 配置批量大小")
        units = list(self.units)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(units)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        rank, world_size = self._rank_info()
        global_worker = rank * workers + worker_id
        global_workers = world_size * workers
        units = units[global_worker::global_workers]
        scale_widths = {
            name: int(_scale_map(self.config)[name]["bucket_count"]) for name in SCALE_NAMES
        }
        parquet_files: dict[int, pq.ParquetFile] = {}
        for shard_index, row_group in units:
            if shard_index not in parquet_files:
                parquet_files[shard_index] = pq.ParquetFile(self.shards[shard_index])
            parquet = parquet_files[shard_index]
            table = parquet.read_row_group(row_group)
            rows = table.num_rows
            order = np.arange(rows)
            if self.shuffle:
                unit_seed = self.seed + self.epoch * 1_000_003 + shard_index * 10_007 + row_group
                np.random.default_rng(unit_seed).shuffle(order)
            arrays = {
                name: fixed_list_to_numpy(table, f"{name}_counts", width)
                for name, width in scale_widths.items()
            }
            state = fixed_list_to_numpy(table, "history_state", 4)
            path_indices = table.column("path_index").combine_chunks().to_numpy()
            y_access = table.column("y_access").combine_chunks().to_numpy()
            y_time = table.column("y_time").combine_chunks().to_numpy()
            y_count = table.column("y_count").combine_chunks().to_numpy()
            for start in range(0, rows, self.batch_size):
                batch_order = order[start : start + self.batch_size]

                def batch_tensor(values: np.ndarray, dtype: np.dtype) -> torch.Tensor:
                    # 一次只为整批创建连续内存，避免逐样本复制和数百万次 Tensor 分配。
                    return torch.from_numpy(
                        np.array(values[batch_order], dtype=dtype, copy=True, order="C")
                    )

                yield {
                    "path_index": batch_tensor(path_indices, np.dtype(np.int64)),
                    "second_counts": batch_tensor(
                        arrays["second"], np.dtype(np.float32)
                    ).unsqueeze(-1),
                    "short_counts": batch_tensor(
                        arrays["short"], np.dtype(np.float32)
                    ).unsqueeze(-1),
                    "medium_counts": batch_tensor(
                        arrays["medium"], np.dtype(np.float32)
                    ).unsqueeze(-1),
                    "long_counts": batch_tensor(
                        arrays["long"], np.dtype(np.float32)
                    ).unsqueeze(-1),
                    "history_state": batch_tensor(state, np.dtype(np.float32)),
                    "y_access": batch_tensor(y_access, np.dtype(np.int64)),
                    "y_time": batch_tensor(y_time, np.dtype(np.int64)),
                    "y_count": batch_tensor(y_count, np.dtype(np.int64)),
                }


def load_and_verify_dataset(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(dataset_dir / "manifest.json")
    if manifest.get("schema_version") != "dynamic-history-dataset-manifest/v1":
        raise DataIntegrityError(f"不支持的数据集 manifest：{dataset_dir / 'manifest.json'}")
    verify_manifest_files(dataset_dir, manifest.get("files", {}))
    metadata = read_json(dataset_dir / "data_meta.json")
    return metadata, manifest
