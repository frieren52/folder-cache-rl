from __future__ import annotations

import heapq
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import config_sha256, resolve_path
from .errors import DataIntegrityError, OutputExistsError
from .sampling import PairSample, PositiveSample, sample_pairs, sample_positives
from .state import HistoryIndexEntry, HistoryVectorIndex, NegativeTierState, RollingAccessState
from .upstream import HistoryEncoderAdapter, load_access_dataset, release_summary, validate_history_release
from .utils import file_descriptions, parse_time, sha256_file, validate_identifier, write_json
from .vector_store import StaticVectorStore


def split_for_cutoff(cutoff: int, horizon: int, test_start: int, test_end: int) -> str:
    """Classify a cutoff without ever inventing a validation split."""
    if cutoff + horizon <= test_start:
        return "train"
    if cutoff < test_start:
        return "transition"
    if cutoff + horizon <= test_end:
        return "test"
    return "tail"


def _group_values(events: Any, catalog: Any, group_index: int) -> dict[int, int]:
    start = int(events.group_offsets[group_index])
    end = int(events.group_offsets[group_index + 1])
    return {
        int(catalog.path_indices[int(events.group_positions[position])]): int(events.group_counts[position])
        for position in range(start, end)
    }


def _access_times(events: Any, catalog: Any, path_index: int) -> np.ndarray:
    position = catalog.path_to_position[int(path_index)]
    return np.frombuffer(events.events_by_position[position], dtype=np.int64)


class FutureWindow:
    def __init__(self, events: Any, catalog: Any, horizon: int, cutoff: int) -> None:
        self.events = events
        self.catalog = catalog
        self.horizon = int(horizon)
        self.times = np.asarray(events.group_times, dtype=np.int64)
        self.start = int(np.searchsorted(self.times, cutoff, side="left"))
        self.end = self.start
        self.counts: Counter[int] = Counter()
        self.advance(cutoff)

    def _apply(self, group_index: int, sign: int) -> None:
        for path_index, count in _group_values(self.events, self.catalog, group_index).items():
            self.counts[path_index] += sign * count
            if self.counts[path_index] <= 0:
                self.counts.pop(path_index, None)

    def advance(self, cutoff: int) -> None:
        while self.start < len(self.times) and int(self.times[self.start]) < cutoff:
            if self.start < self.end:
                self._apply(self.start, -1)
            self.start += 1
        if self.end < self.start:
            self.end = self.start
        upper = int(cutoff) + self.horizon
        while self.end < len(self.times) and int(self.times[self.end]) < upper:
            self._apply(self.end, 1)
            self.end += 1

    def labels(self, cutoff: int) -> tuple[dict[int, int], dict[int, int]]:
        first_delta: dict[int, int] = {}
        counts = dict(self.counts)
        for path_index in counts:
            values = _access_times(self.events, self.catalog, path_index)
            position = int(np.searchsorted(values, cutoff, side="left"))
            if position >= len(values) or int(values[position]) >= cutoff + self.horizon:
                raise DataIntegrityError(f"未来窗口计数与对象时间线不一致：path_index={path_index}")
            first_delta[path_index] = int(values[position]) - int(cutoff)
        return first_delta, counts


def _sample_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("cutoff_time_seconds", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("history_revision", pa.int64(), nullable=False),
            pa.field("context_folder_ids", pa.list_(pa.int64(), 256), nullable=False),
            pa.field("context_valid_mask", pa.list_(pa.bool_(), 256), nullable=False),
            pa.field("context_recent_mask", pa.list_(pa.bool_(), 256), nullable=False),
            pa.field("context_hot_mask", pa.list_(pa.bool_(), 256), nullable=False),
            pa.field("positive_folder_ids_all", pa.list_(pa.int64()), nullable=False),
            pa.field("pair_positive_ids", pa.list_(pa.int64()), nullable=False),
            pa.field("pair_negative_ids", pa.list_(pa.int64()), nullable=False),
            pa.field("pair_positive_layers", pa.list_(pa.int8()), nullable=False),
            pa.field("pair_weights", pa.list_(pa.float32()), nullable=False),
            pa.field("pair_negative_sources", pa.list_(pa.int8()), nullable=False),
            pa.field("negative_pool_sizes", pa.list_(pa.int32(), 4), nullable=False),
            pa.field("updated_ids", pa.list_(pa.int64()), nullable=False),
            pa.field("removed_ids", pa.list_(pa.int64()), nullable=False),
        ]
    )


def _snapshot_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("persistent_folder_ids", pa.list_(pa.int64()), nullable=False),
            pa.field("persistent_history_vectors", pa.binary(), nullable=False),
            pa.field("persistent_history_valid", pa.list_(pa.bool_()), nullable=False),
            pa.field("current_folder_ids", pa.list_(pa.int64()), nullable=False),
            pa.field("current_history_vectors", pa.binary(), nullable=False),
            pa.field("current_history_valid", pa.list_(pa.bool_()), nullable=False),
            pa.field("candidate_as_of_time", pa.int64(), nullable=False),
        ]
    )


def _restore_index(vector_store_dir: Path, config: Mapping[str, Any]) -> HistoryVectorIndex:
    data = np.load(vector_store_dir / "initial_history_index.npz", allow_pickle=False)
    index = HistoryVectorIndex(
        recent_window_seconds=int(config["history_index"]["recent_window_seconds"]),
        active_window_seconds=int(config["history_index"]["active_window_seconds"]),
        recent_refresh_seconds=int(config["history_index"]["recent_refresh_seconds"]),
        long_refresh_seconds=int(config["history_index"]["long_refresh_seconds"]),
    )
    ids = data["path_indices"].astype(np.int64)
    vectors = data["history_vectors"].astype(np.float32)
    last = data["last_event_times"].astype(np.int64)
    as_of = data["vector_as_of_times"].astype(np.int64)
    due = data["next_refresh_times"].astype(np.int64)
    for path_index, vector, last_time, vector_time, next_time in zip(ids, vectors, last, as_of, due):
        key = int(path_index)
        index.entries[key] = HistoryIndexEntry(vector, int(last_time), int(vector_time), int(next_time))
        heapq.heappush(index._due_heap, (int(next_time), key))
    index.revision = int(data["history_revision"][0])
    return index


def _empty_positive() -> PositiveSample:
    return PositiveSample(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=np.float32),
    )


def _empty_pairs(pool_sizes: list[int]) -> PairSample:
    return PairSample(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int8),
        np.asarray(pool_sizes, dtype=np.int32),
    )


def build_actor_samples(
    config: Mapping[str, Any],
    module_root: Path,
    data_version: str,
    device: str = "auto",
) -> Path:
    data_version = validate_identifier(data_version, "data_version")
    paths = config["paths"]
    version_dir = resolve_path(module_root, paths["dataset_root"]) / data_version
    if version_dir.exists():
        raise OutputExistsError(f"数据版本已存在，拒绝覆盖：{version_dir}")
    staging = version_dir.with_name(f".{version_dir.name}.build-{os.getpid()}")
    vector_store_dir = resolve_path(module_root, paths["vector_store_dir"])
    static_store = StaticVectorStore.load(vector_store_dir)
    repository_root = module_root.parent
    history_source = repository_root / "04_动态历史向量生成" / "src"
    validate_history_release(resolve_path(module_root, paths["history_release"]))
    staging.mkdir(parents=True)
    catalog, events = load_access_dataset(
        history_source,
        resolve_path(module_root, paths["path_catalog"]),
        resolve_path(module_root, paths["access_dir"]),
    )
    history_encoder = HistoryEncoderAdapter(
        history_source,
        resolve_path(module_root, paths["history_release"]),
        device=device,
    )
    train_start = parse_time(str(config["time"]["train_start"]), "time.train_start")
    test_start = parse_time(str(config["time"]["test_start"]), "time.test_start")
    test_end = parse_time(str(config["time"]["test_end"]), "time.test_end")
    stride = int(config["time"]["decision_interval_seconds"])
    horizon = int(config["time"]["target_horizon_seconds"])
    group_times = np.asarray(events.group_times, dtype=np.int64)
    access_state = RollingAccessState(
        int(config["context"]["hot_window_seconds"]),
        int(config["history_index"]["active_window_seconds"]),
    )
    negative_tiers = NegativeTierState(catalog.path_indices)
    group_cursor = 0
    warm_last_access: dict[int, int] = {}
    warm_hot_groups: list[tuple[int, dict[int, int]]] = []
    while group_cursor < len(group_times) and int(group_times[group_cursor]) < train_start:
        values = _group_values(events, catalog, group_cursor)
        event_time = int(group_times[group_cursor])
        for path_index in values:
            warm_last_access[path_index] = event_time
        if event_time >= train_start - int(config["context"]["hot_window_seconds"]):
            warm_hot_groups.append((event_time, values))
        group_cursor += 1
    access_state.restore(train_start, warm_last_access, warm_hot_groups)
    negative_tiers.restore(warm_last_access, train_start)
    future = FutureWindow(events, catalog, horizon, train_start)
    index = _restore_index(vector_store_dir, config)
    sample_path = staging / "actor_samples.parquet"
    snapshot_path = staging / "actor_history_snapshots.parquet"
    sample_writer = pq.ParquetWriter(sample_path, _sample_schema(), compression=str(config["storage"]["parquet_compression"]), version="2.6")
    snapshot_writer = pq.ParquetWriter(snapshot_path, _snapshot_schema(), compression=str(config["storage"]["parquet_compression"]), version="2.6")
    sample_buffer: list[dict[str, Any]] = []
    snapshot_buffer: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    empty_targets = 0
    total_pairs = 0

    def flush() -> None:
        if not sample_buffer:
            return
        sample_writer.write_table(pa.Table.from_pylist(sample_buffer, schema=_sample_schema()), row_group_size=len(sample_buffer))
        snapshot_writer.write_table(pa.Table.from_pylist(snapshot_buffer, schema=_snapshot_schema()), row_group_size=len(snapshot_buffer))
        sample_buffer.clear()
        snapshot_buffer.clear()

    try:
        for cutoff in range(train_start, test_end, stride):
            while group_cursor < len(group_times) and int(group_times[group_cursor]) < cutoff:
                values = _group_values(events, catalog, group_cursor)
                access_state.ingest_group(int(group_times[group_cursor]), values)
                negative_tiers.ingest(int(group_times[group_cursor]), values)
                group_cursor += 1
            removed_from_state = access_state.advance(cutoff)
            negative_tiers.advance(cutoff)
            future.advance(cutoff)
            context = access_state.select_context(
                int(config["context"]["max_objects"]),
                int(config["context"]["recent_objects"]),
            )
            split = split_for_cutoff(cutoff, horizon, test_start, test_end)
            if split in {"train", "test"}:
                first_delta, future_counts = future.labels(cutoff)
                positives = sample_positives(
                    first_delta,
                    future_counts,
                    int(config["sampling"]["seed"]) + cutoff,
                    int(config["sampling"]["max_positive_objects"]),
                    int(config["sampling"]["target_positive_per_layer"]),
                    int(config["sampling"]["deterministic_hot_per_layer"]),
                )
                pairs = sample_pairs(
                    positives,
                    negative_tiers.pools(),
                    config["sampling"]["negative_ratios"],
                    int(config["sampling"]["negatives_per_positive"]),
                    int(config["sampling"]["seed"]) + cutoff + 1_000_000,
                ) if len(positives.sampled_ids) else _empty_pairs([len(value) for value in negative_tiers.pools()])
            else:
                positives = _empty_positive()
                pairs = _empty_pairs([len(value) for value in negative_tiers.pools()])
            dirty = access_state.consume_dirty()
            refresh_ids, expired_ids = index.select_changes(
                cutoff,
                access_state.last_access,
                dirty,
                context["path_indices"][context["valid_mask"]],
            )
            expired_ids = sorted(set(expired_ids) | set(removed_from_state))
            current_ids = sorted(
                (set(pairs.positive_ids.tolist()) | set(pairs.negative_ids.tolist()))
                & set(access_state.last_access)
            )
            encode_ids = sorted(set(refresh_ids) | set(current_ids))
            encoded_map: dict[int, np.ndarray] = {}
            if encode_ids:
                encoded = history_encoder.encode_as_of(
                    cutoff,
                    encode_ids,
                    lambda path_index: _access_times(events, catalog, path_index),
                    batch_size=int(config["history_index"]["encode_batch_size"]),
                )
                encoded_map = {int(path_index): vector for path_index, vector in zip(encoded["path_indices"], encoded["vectors"])}
            refresh_vectors = np.stack([encoded_map[value] for value in refresh_ids]) if refresh_ids else np.empty((0, 128), dtype=np.float32)
            revision = index.publish(cutoff, refresh_ids, refresh_vectors, access_state.last_access, expired_ids)
            referenced_ids = sorted(
                set(int(value) for value in context["path_indices"] if int(value) >= 0)
                | set(pairs.positive_ids.tolist())
                | set(pairs.negative_ids.tolist())
                | set(refresh_ids)
            )
            persistent_vectors, persistent_valid = index.vectors_for(referenced_ids)
            all_current_ids = sorted(set(pairs.positive_ids.tolist()) | set(pairs.negative_ids.tolist()))
            current_vectors = np.zeros((len(all_current_ids), 128), dtype=np.float32)
            current_valid = np.zeros(len(all_current_ids), dtype=np.bool_)
            for position, path_index in enumerate(all_current_ids):
                vector = encoded_map.get(int(path_index))
                if vector is not None:
                    current_vectors[position] = vector
                    current_valid[position] = True
            sample_id = f"actor-{cutoff}"
            sample_buffer.append(
                {
                    "sample_id": sample_id,
                    "cutoff_time_seconds": cutoff,
                    "split": split,
                    "history_revision": revision,
                    "context_folder_ids": context["path_indices"].tolist(),
                    "context_valid_mask": context["valid_mask"].tolist(),
                    "context_recent_mask": context["recent_mask"].tolist(),
                    "context_hot_mask": context["hot_mask"].tolist(),
                    "positive_folder_ids_all": positives.all_ids.tolist(),
                    "pair_positive_ids": pairs.positive_ids.tolist(),
                    "pair_negative_ids": pairs.negative_ids.tolist(),
                    "pair_positive_layers": pairs.positive_layers.tolist(),
                    "pair_weights": pairs.weights.tolist(),
                    "pair_negative_sources": pairs.negative_sources.tolist(),
                    "negative_pool_sizes": pairs.pool_sizes.tolist(),
                    "updated_ids": refresh_ids,
                    "removed_ids": expired_ids,
                }
            )
            snapshot_buffer.append(
                {
                    "sample_id": sample_id,
                    "persistent_folder_ids": referenced_ids,
                    "persistent_history_vectors": persistent_vectors.astype(np.float16).tobytes(order="C"),
                    "persistent_history_valid": persistent_valid.tolist(),
                    "current_folder_ids": all_current_ids,
                    "current_history_vectors": current_vectors.astype(np.float16).tobytes(order="C"),
                    "current_history_valid": current_valid.tolist(),
                    "candidate_as_of_time": cutoff,
                }
            )
            split_counts[split] += 1
            empty_targets += int(split in {"train", "test"} and len(positives.all_ids) == 0)
            total_pairs += len(pairs.positive_ids)
            if len(sample_buffer) >= int(config["storage"]["rows_per_file_batch"]):
                flush()
            if split_counts.total() % 600 == 0:
                print(json.dumps({"stage": "prepare_data", "cutoff": cutoff, "rows": split_counts.total(), "pairs": total_pairs, "history_revision": revision}, ensure_ascii=False), flush=True)
        flush()
    finally:
        sample_writer.close()
        snapshot_writer.close()
    file_records = file_descriptions([sample_path, snapshot_path, vector_store_dir / "initial_history_index.npz"])
    file_records["actor_samples.parquet"]["path"] = "actor_samples.parquet"
    file_records["actor_history_snapshots.parquet"]["path"] = "actor_history_snapshots.parquet"
    file_records["initial_history_index.npz"]["path"] = (vector_store_dir / "initial_history_index.npz").as_posix()
    manifest = {
        "schema_version": "actor-supervision/v3",
        "data_version": data_version,
        "config_sha256": config_sha256(config),
        "decision_interval_seconds": stride,
        "target_horizon_seconds": horizon,
        "timezone": str(config["time"]["timezone"]),
        "split_rows": dict(split_counts),
        "empty_target_rows": empty_targets,
        "pair_count": total_pairs,
        "vector_store_manifest_sha256": sha256_file(vector_store_dir / "vector_store_manifest.json"),
        "history_release": release_summary(resolve_path(module_root, paths["history_release"])),
        "files": file_records,
    }
    write_json(staging / "actor_samples_manifest.json", manifest)
    staging.replace(version_dir)
    fixed_manifest = module_root / "data" / "actor_samples_manifest.json"
    # 只原子更新轻量指针；既有不可变数据版本始终保留。
    write_json(fixed_manifest, {"data_version": data_version, "manifest": (version_dir / "actor_samples_manifest.json").as_posix(), "sha256": sha256_file(version_dir / "actor_samples_manifest.json")})
    print(json.dumps({"data_version_dir": version_dir.as_posix(), "rows": split_counts.total(), "pairs": total_pairs}, ensure_ascii=False), flush=True)
    return version_dir
