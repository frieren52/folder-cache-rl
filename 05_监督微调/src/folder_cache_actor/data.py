from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .errors import DataIntegrityError
from .utils import SHANGHAI
from .vector_store import StaticVectorStore


def time_features(cutoff_time_seconds: int) -> np.ndarray:
    local = datetime.fromtimestamp(int(cutoff_time_seconds), SHANGHAI)
    day_seconds = local.hour * 3600 + local.minute * 60 + local.second
    angle = 2.0 * math.pi * day_seconds / 86400.0
    return np.asarray([math.sin(angle), math.cos(angle)], dtype=np.float32)


def build_context_features(
    static_store: StaticVectorStore,
    context_ids: Sequence[int],
    valid_mask: Sequence[bool],
    recent_mask: Sequence[bool],
    hot_mask: Sequence[bool],
    persistent_vectors: Mapping[int, np.ndarray],
) -> np.ndarray:
    count = len(context_ids)
    result = np.zeros((count, 259), dtype=np.float32)
    for position, raw_id in enumerate(context_ids):
        if not bool(valid_mask[position]):
            continue
        path_index = int(raw_id)
        result[position, :128] = static_store.get([path_index])[0]
        history = persistent_vectors.get(path_index)
        if history is not None:
            result[position, 128:256] = history
            result[position, 256] = 1.0
        result[position, 257] = float(bool(recent_mask[position]))
        result[position, 258] = float(bool(hot_mask[position]))
    return result


def _vector_mapping(ids: Sequence[int], flat_values: Sequence[float], valid: Sequence[bool]) -> dict[int, np.ndarray]:
    if isinstance(flat_values, (bytes, bytearray, memoryview)):
        values = np.frombuffer(flat_values, dtype=np.float16).astype(np.float32)
    else:
        values = np.asarray(flat_values, dtype=np.float16).astype(np.float32)
    if values.size != len(ids) * 128:
        raise DataIntegrityError("历史快照向量长度与对象数不一致")
    matrix = values.reshape(len(ids), 128)
    return {int(path_index): matrix[pos] for pos, path_index in enumerate(ids) if bool(valid[pos])}


class ActorParquetDataset(IterableDataset[dict[str, Any]]):
    """Stream aligned sample/snapshot Parquet row groups with bounded shuffling."""

    def __init__(
        self,
        samples_path: Path,
        snapshots_path: Path,
        split: str,
        shuffle_buffer_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.samples_path = samples_path
        self.snapshots_path = snapshots_path
        self.split = split
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.epoch = 0
        samples = pq.ParquetFile(samples_path)
        snapshots = pq.ParquetFile(snapshots_path)
        if samples.num_row_groups != snapshots.num_row_groups or samples.metadata.num_rows != snapshots.metadata.num_rows:
            raise DataIntegrityError("样本表与历史快照表行数/row group不一致")
        self.num_row_groups = samples.num_row_groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rows(self) -> Iterator[dict[str, Any]]:
        samples = pq.ParquetFile(self.samples_path)
        snapshots = pq.ParquetFile(self.snapshots_path)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        for group_index in range(worker_id, self.num_row_groups, worker_count):
            sample_rows = samples.read_row_group(group_index).to_pylist()
            snapshot_rows = snapshots.read_row_group(group_index).to_pylist()
            if len(sample_rows) != len(snapshot_rows):
                raise DataIntegrityError(f"row group {group_index}未对齐")
            for sample, snapshot in zip(sample_rows, snapshot_rows):
                if sample["sample_id"] != snapshot["sample_id"]:
                    raise DataIntegrityError("样本与历史快照sample_id未对齐")
                if sample["split"] == self.split and len(sample["pair_positive_ids"]) > 0:
                    yield {**sample, **snapshot}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rows = self._rows()
        if self.split != "train" or self.shuffle_buffer_size <= 1:
            yield from rows
            return
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.Generator(np.random.PCG64(self.seed + self.epoch * 1009 + worker_id))
        buffer: list[dict[str, Any]] = []
        for row in rows:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(row)
                continue
            position = int(rng.integers(0, len(buffer)))
            yield buffer[position]
            buffer[position] = row
        rng.shuffle(buffer)
        yield from buffer


def make_collate(static_store: StaticVectorStore):
    def collate(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        contexts: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        times: list[np.ndarray] = []
        pair_sample: list[int] = []
        pair_positive_ids: list[int] = []
        pair_positive_layers: list[int] = []
        pair_weights: list[float] = []
        positive_static: list[np.ndarray] = []
        negative_static: list[np.ndarray] = []
        positive_persistent: list[np.ndarray] = []
        negative_persistent: list[np.ndarray] = []
        persistent_valid: list[bool] = []
        positive_current: list[np.ndarray] = []
        negative_current: list[np.ndarray] = []
        current_valid: list[bool] = []
        for sample_position, row in enumerate(rows):
            persistent = _vector_mapping(row["persistent_folder_ids"], row["persistent_history_vectors"], row["persistent_history_valid"])
            current = _vector_mapping(row["current_folder_ids"], row["current_history_vectors"], row["current_history_valid"])
            contexts.append(
                build_context_features(
                    static_store,
                    row["context_folder_ids"],
                    row["context_valid_mask"],
                    row["context_recent_mask"],
                    row["context_hot_mask"],
                    persistent,
                )
            )
            valid_masks.append(np.asarray(row["context_valid_mask"], dtype=np.bool_))
            times.append(time_features(int(row["cutoff_time_seconds"])))
            for positive_id, negative_id, layer, weight in zip(row["pair_positive_ids"], row["pair_negative_ids"], row["pair_positive_layers"], row["pair_weights"]):
                positive_id = int(positive_id)
                negative_id = int(negative_id)
                pair_sample.append(sample_position)
                pair_positive_ids.append(positive_id)
                pair_positive_layers.append(int(layer))
                pair_weights.append(float(weight))
                positive_static.append(static_store.get([positive_id])[0])
                negative_static.append(static_store.get([negative_id])[0])
                positive_persistent.append(persistent.get(positive_id, np.zeros(128, dtype=np.float32)))
                negative_persistent.append(persistent.get(negative_id, np.zeros(128, dtype=np.float32)))
                persistent_valid.append(positive_id in persistent and negative_id in persistent)
                positive_current.append(current.get(positive_id, np.zeros(128, dtype=np.float32)))
                negative_current.append(current.get(negative_id, np.zeros(128, dtype=np.float32)))
                current_valid.append(positive_id in current and negative_id in current)
        def tensor(values: Any, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(np.asarray(values), dtype=dtype)
        return {
            "context_features": tensor(contexts, torch.float32),
            "context_valid_mask": tensor(valid_masks, torch.bool),
            "time_features": tensor(times, torch.float32),
            "pair_sample_index": tensor(pair_sample, torch.long),
            "pair_positive_ids": tensor(pair_positive_ids, torch.long),
            "pair_positive_layers": tensor(pair_positive_layers, torch.long),
            "pair_weights": tensor(pair_weights, torch.float32),
            "positive_static": tensor(positive_static, torch.float32),
            "negative_static": tensor(negative_static, torch.float32),
            "positive_history_persistent": tensor(positive_persistent, torch.float32),
            "negative_history_persistent": tensor(negative_persistent, torch.float32),
            "persistent_valid": tensor(persistent_valid, torch.bool),
            "positive_history_current": tensor(positive_current, torch.float32),
            "negative_history_current": tensor(negative_current, torch.float32),
            "current_valid": tensor(current_valid, torch.bool),
        }
    return collate
