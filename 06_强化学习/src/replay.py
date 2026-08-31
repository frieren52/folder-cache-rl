from __future__ import annotations

import hashlib
import io
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import DataIntegrityError, OutputExistsError
from .features import CANDIDATE_DIM, RESOURCE_DIM, FeatureNormalizer, NormalizationStats
from .utils import read_json, sha256_file, write_json


REPLAY_SCHEMA_VERSION = "folder-rl-replay/v1"


@dataclass(frozen=True)
class StateArray:
    object_features: np.ndarray
    object_valid_mask: np.ndarray
    time_features: np.ndarray
    resource_features: np.ndarray
    candidate_path_indices: np.ndarray
    candidate_features: np.ndarray
    candidate_valid_mask: np.ndarray
    candidate_action_mask: np.ndarray
    selected_candidate_mask: np.ndarray

    def validate(self) -> None:
        candidate_count = len(self.candidate_path_indices)
        if self.object_features.shape != (256, 259):
            raise DataIntegrityError("Replay object_features必须为[256,259]")
        if self.object_valid_mask.shape != (256,) or self.object_valid_mask.dtype != np.bool_:
            raise DataIntegrityError("Replay object_valid_mask必须为bool[256]")
        if self.time_features.shape != (2,) or self.resource_features.shape != (RESOURCE_DIM,):
            raise DataIntegrityError("Replay时间或资源特征形状错误")
        if self.candidate_features.shape != (candidate_count, CANDIDATE_DIM):
            raise DataIntegrityError("Replay候选特征形状错误")
        for value in (self.candidate_valid_mask, self.candidate_action_mask, self.selected_candidate_mask):
            if value.shape != (candidate_count,) or value.dtype != np.bool_:
                raise DataIntegrityError("Replay候选Mask形状或类型错误")
        for value in (self.object_features, self.time_features, self.resource_features, self.candidate_features):
            if not np.all(np.isfinite(value)):
                raise DataIntegrityError("Replay状态包含非有限值")


@dataclass(frozen=True)
class MacroStepRecord:
    run_id: str
    macro_step_id: int
    cutoff_time: int
    split: str
    policy_name: str
    states: tuple[StateArray, ...]
    action_positions: np.ndarray
    reward: float
    delta_count: int
    delta_bytes: int
    terminal: bool
    actor_sha256: str
    critic_sha256: str | None

    def validate(self) -> None:
        if self.split not in {"train_fit", "train_calibration", "test"}:
            raise DataIntegrityError(f"Replay split非法：{self.split}")
        if not self.states or self.action_positions.shape != (len(self.states),):
            raise DataIntegrityError("Replay微步状态与动作数量不一致")
        if int(self.action_positions[-1]) != -1:
            raise DataIntegrityError("Replay宏步最后一个动作必须为STOP(-1)")
        for state, action in zip(self.states, self.action_positions):
            state.validate()
            if int(action) >= 0:
                if int(action) >= len(state.candidate_path_indices) or not state.candidate_action_mask[int(action)]:
                    raise DataIntegrityError("Replay包含不可执行候选动作")
        if not np.isfinite(self.reward):
            raise DataIntegrityError("Replay奖励不是有限值")


@dataclass(frozen=True)
class Transition:
    state: StateArray
    action_position: int
    reward: float
    discount: float
    next_state: StateArray
    terminal: bool
    macro_step_id: int
    sample_weight: float = 1.0
    return_value: float | None = None


def _stack_states(states: Sequence[StateArray]) -> dict[str, np.ndarray]:
    candidate_count = len(states[0].candidate_path_indices)
    if any(len(state.candidate_path_indices) != candidate_count for state in states):
        raise DataIntegrityError("同一宏步的自然候选数必须保持不变")
    return {
        "object_features": np.asarray(states[0].object_features, dtype=np.float16),
        "object_valid_mask": np.asarray(states[0].object_valid_mask, dtype=np.bool_),
        "time_features": np.asarray(states[0].time_features, dtype=np.float32),
        "candidate_path_indices": np.asarray(states[0].candidate_path_indices, dtype=np.int64),
        "resource_features": np.stack([state.resource_features for state in states]).astype(np.float32),
        "candidate_features": np.stack([state.candidate_features for state in states]).astype(np.float16),
        "candidate_valid_mask": np.stack([state.candidate_valid_mask for state in states]).astype(np.bool_),
        "candidate_action_mask": np.stack([state.candidate_action_mask for state in states]).astype(np.bool_),
        "selected_candidate_mask": np.stack([state.selected_candidate_mask for state in states]).astype(np.bool_),
    }


def serialize_record(record: MacroStepRecord) -> bytes:
    record.validate()
    arrays = _stack_states(record.states)
    arrays["action_positions"] = np.asarray(record.action_positions, dtype=np.int16)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def deserialize_record(metadata: Mapping[str, object], payload: bytes) -> MacroStepRecord:
    with np.load(io.BytesIO(payload), allow_pickle=False) as values:
        actions = values["action_positions"].astype(np.int64)
        states = tuple(
            StateArray(
                values["object_features"].astype(np.float32),
                values["object_valid_mask"].astype(np.bool_),
                values["time_features"].astype(np.float32),
                values["resource_features"][index].astype(np.float32),
                values["candidate_path_indices"].astype(np.int64),
                values["candidate_features"][index].astype(np.float32),
                values["candidate_valid_mask"][index].astype(np.bool_),
                values["candidate_action_mask"][index].astype(np.bool_),
                values["selected_candidate_mask"][index].astype(np.bool_),
            )
            for index in range(len(actions))
        )
    record = MacroStepRecord(
        run_id=str(metadata["run_id"]),
        macro_step_id=int(metadata["macro_step_id"]),
        cutoff_time=int(metadata["cutoff_time"]),
        split=str(metadata["split"]),
        policy_name=str(metadata["policy_name"]),
        states=states,
        action_positions=actions,
        reward=float(metadata["reward"]),
        delta_count=int(metadata["delta_count"]),
        delta_bytes=int(metadata["delta_bytes"]),
        terminal=bool(metadata["terminal"]),
        actor_sha256=str(metadata["actor_sha256"]),
        critic_sha256=None if metadata.get("critic_sha256") in {None, ""} else str(metadata["critic_sha256"]),
    )
    record.validate()
    return record


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("macro_step_id", pa.int64(), nullable=False),
            pa.field("cutoff_time", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("policy_name", pa.string(), nullable=False),
            pa.field("reward", pa.float64(), nullable=False),
            pa.field("delta_count", pa.int64(), nullable=False),
            pa.field("delta_bytes", pa.int64(), nullable=False),
            pa.field("terminal", pa.bool_(), nullable=False),
            pa.field("actor_sha256", pa.string(), nullable=False),
            pa.field("critic_sha256", pa.string(), nullable=True),
            pa.field("payload_sha256", pa.string(), nullable=False),
            pa.field("payload", pa.binary(), nullable=False),
        ]
    )


class ReplayWriter:
    def __init__(
        self,
        root: Path,
        run_id: str,
        config_sha256: str,
        rows_per_shard: int = 256,
        compression: str = "zstd",
    ) -> None:
        if root.exists():
            raise OutputExistsError(f"Replay版本已存在，拒绝覆盖：{root}")
        self.root = root
        self.staging = root.with_name(f".{root.name}.build-{os.getpid()}")
        self.staging.mkdir(parents=True)
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.rows_per_shard = int(rows_per_shard)
        self.compression = compression
        self.buffer: list[dict[str, object]] = []
        self.shards: list[dict[str, object]] = []
        self.record_count = 0

    def append(self, record: MacroStepRecord) -> None:
        if record.run_id != self.run_id:
            raise DataIntegrityError("ReplayWriter收到其他run_id")
        payload = serialize_record(record)
        self.buffer.append(
            {
                "run_id": record.run_id,
                "macro_step_id": record.macro_step_id,
                "cutoff_time": record.cutoff_time,
                "split": record.split,
                "policy_name": record.policy_name,
                "reward": record.reward,
                "delta_count": record.delta_count,
                "delta_bytes": record.delta_bytes,
                "terminal": record.terminal,
                "actor_sha256": record.actor_sha256,
                "critic_sha256": record.critic_sha256,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload": payload,
            }
        )
        self.record_count += 1
        if len(self.buffer) >= self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        name = f"macro_steps-{len(self.shards):05d}.parquet"
        path = self.staging / name
        table = pa.Table.from_pylist(self.buffer, schema=_schema())
        pq.write_table(table, path, compression=self.compression, row_group_size=min(16, len(self.buffer)), version="2.6")
        self.shards.append({"path": name, "rows": len(self.buffer), "sha256": sha256_file(path)})
        self.buffer.clear()

    def close(self, extra_manifest: Mapping[str, object] | None = None) -> Path:
        self._flush()
        manifest = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "record_count": self.record_count,
            "shards": self.shards,
            **({} if extra_manifest is None else dict(extra_manifest)),
        }
        write_json(self.staging / "replay_manifest.json", manifest)
        self.staging.replace(self.root)
        return self.root


@dataclass(frozen=True)
class _Entry:
    shard: int
    row_group: int
    row_in_group: int
    run_id: str
    macro_step_id: int
    split: str
    reward: float
    terminal: bool


class ReplayReader:
    def __init__(self, root: Path, cache_groups: int = 8) -> None:
        self.root = root
        self.manifest = read_json(root / "replay_manifest.json")
        if self.manifest.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise DataIntegrityError("不支持的Replay版本")
        self.candidate_normalizer = None
        self.resource_normalizer = None
        if "candidate_normalization" in self.manifest:
            candidate_stats = NormalizationStats.from_dict(self.manifest["candidate_normalization"])
            self.candidate_normalizer = FeatureNormalizer(
                candidate_stats,
                candidate_stats.dimension,
                candidate_stats.log1p_indices,
            )
        if "resource_normalization" in self.manifest:
            resource_stats = NormalizationStats.from_dict(self.manifest["resource_normalization"])
            self.resource_normalizer = FeatureNormalizer(
                resource_stats,
                resource_stats.dimension,
                resource_stats.log1p_indices,
            )
        self.shard_paths: list[Path] = []
        self.entries: list[_Entry] = []
        for shard_index, item in enumerate(self.manifest["shards"]):
            path = root / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise DataIntegrityError(f"Replay shard缺失或摘要错误：{path}")
            self.shard_paths.append(path)
            parquet = pq.ParquetFile(path)
            for group_index in range(parquet.num_row_groups):
                metadata = parquet.read_row_group(
                    group_index,
                    columns=["run_id", "macro_step_id", "split", "reward", "terminal"],
                ).to_pylist()
                for row_index, row in enumerate(metadata):
                    self.entries.append(
                        _Entry(
                            shard_index,
                            group_index,
                            row_index,
                            str(row["run_id"]),
                            int(row["macro_step_id"]),
                            str(row["split"]),
                            float(row["reward"]),
                            bool(row["terminal"]),
                        )
                    )
        if len(self.entries) != int(self.manifest["record_count"]):
            raise DataIntegrityError("Replay manifest记录数与shard不一致")
        self.cache_groups = int(cache_groups)
        self._cache: OrderedDict[tuple[int, int], list[dict[str, object]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def _rows(self, shard: int, group: int) -> list[dict[str, object]]:
        key = (shard, group)
        rows = self._cache.get(key)
        if rows is not None:
            self._cache.move_to_end(key)
            return rows
        rows = pq.ParquetFile(self.shard_paths[shard]).read_row_group(group).to_pylist()
        self._cache[key] = rows
        while len(self._cache) > self.cache_groups:
            self._cache.popitem(last=False)
        return rows

    def record_at(self, index: int) -> MacroStepRecord:
        entry = self.entries[int(index)]
        row = dict(self._rows(entry.shard, entry.row_group)[entry.row_in_group])
        payload = bytes(row.pop("payload"))
        expected = str(row.pop("payload_sha256"))
        if hashlib.sha256(payload).hexdigest() != expected:
            raise DataIntegrityError(f"Replay payload摘要错误：index={index}")
        record = deserialize_record(row, payload)
        if self.candidate_normalizer is None and self.resource_normalizer is None:
            return record
        states = tuple(
            StateArray(
                state.object_features,
                state.object_valid_mask,
                state.time_features,
                state.resource_features if self.resource_normalizer is None else self.resource_normalizer.transform(state.resource_features),
                state.candidate_path_indices,
                state.candidate_features if self.candidate_normalizer is None else self.candidate_normalizer.transform(state.candidate_features),
                state.candidate_valid_mask,
                state.candidate_action_mask,
                state.selected_candidate_mask,
            )
            for state in record.states
        )
        return MacroStepRecord(
            record.run_id,
            record.macro_step_id,
            record.cutoff_time,
            record.split,
            record.policy_name,
            states,
            record.action_positions,
            record.reward,
            record.delta_count,
            record.delta_bytes,
            record.terminal,
            record.actor_sha256,
            record.critic_sha256,
        )

    def iter_records(self) -> Iterator[MacroStepRecord]:
        for index in range(len(self)):
            yield self.record_at(index)


class ReplaySampler:
    def __init__(
        self,
        reader: ReplayReader,
        split: str,
        seed: int,
        gamma_stop: float,
        n_step: int = 6,
        uniform_probability: float = 0.5,
    ) -> None:
        self.reader = reader
        self.rng = np.random.Generator(np.random.PCG64(int(seed)))
        self.gamma_stop = float(gamma_stop)
        self.n_step = int(n_step)
        self.uniform_probability = float(uniform_probability)
        self.indices = np.asarray([index for index, item in enumerate(reader.entries) if item.split == split], dtype=np.int64)
        self.nonzero = np.asarray([index for index in self.indices if reader.entries[int(index)].reward != 0], dtype=np.int64)
        if not len(self.indices):
            raise DataIntegrityError(f"Replay没有split={split}记录")

    def _sample_index(self) -> tuple[int, float]:
        use_uniform = not len(self.nonzero) or float(self.rng.random()) < self.uniform_probability
        pool = self.indices if use_uniform else self.nonzero
        index = int(pool[int(self.rng.integers(0, len(pool)))])
        if not len(self.nonzero):
            probability = 1.0 / len(self.indices)
        else:
            probability = self.uniform_probability / len(self.indices)
            if index in self.nonzero:
                probability += (1.0 - self.uniform_probability) / len(self.nonzero)
        return index, 1.0 / probability

    def _stop_transition(self, index: int, record: MacroStepRecord, weight: float) -> Transition:
        total = 0.0
        discount = 1.0
        terminal = False
        next_state = record.states[-1]
        steps = 0
        for offset in range(self.n_step):
            target_index = index + offset
            if target_index >= len(self.reader):
                terminal = True
                break
            future = self.reader.record_at(target_index)
            if future.run_id != record.run_id or future.split != record.split:
                terminal = True
                break
            total += discount * future.reward
            steps += 1
            if future.terminal:
                terminal = True
                break
            discount *= self.gamma_stop
        bootstrap_index = index + steps
        if not terminal and bootstrap_index < len(self.reader):
            bootstrap = self.reader.record_at(bootstrap_index)
            if bootstrap.run_id == record.run_id and bootstrap.split == record.split:
                next_state = bootstrap.states[0]
            else:
                terminal = True
        else:
            terminal = True
        return Transition(
            record.states[-1],
            -1,
            total,
            0.0 if terminal else self.gamma_stop**steps,
            next_state,
            terminal,
            record.macro_step_id,
            weight,
        )

    def sample_macro_batch(self, batch_size: int) -> list[Transition]:
        selected: list[tuple[int, MacroStepRecord, float]] = []
        for _ in range(int(batch_size)):
            index, raw_weight = self._sample_index()
            selected.append((index, self.reader.record_at(index), raw_weight))
        mean_weight = sum(item[2] for item in selected) / len(selected)
        transitions: list[Transition] = []
        for index, record, raw_weight in selected:
            weight = raw_weight / mean_weight
            candidate_steps = [position for position, action in enumerate(record.action_positions) if int(action) >= 0]
            transition_weight = weight / (2.0 if candidate_steps else 1.0)
            if candidate_steps:
                step = int(candidate_steps[int(self.rng.integers(0, len(candidate_steps)))])
                transitions.append(
                    Transition(
                        record.states[step],
                        int(record.action_positions[step]),
                        0.0,
                        1.0,
                        record.states[step + 1],
                        False,
                        record.macro_step_id,
                        transition_weight,
                    )
                )
            transitions.append(self._stop_transition(index, record, transition_weight))
        return transitions


def collate_transitions(transitions: Sequence[Transition]) -> dict[str, np.ndarray]:
    if not transitions:
        raise ValueError("transition batch不能为空")
    max_candidates = max(
        max(len(item.state.candidate_path_indices), len(item.next_state.candidate_path_indices))
        for item in transitions
    )
    batch_size = len(transitions)

    def allocate_state() -> dict[str, np.ndarray]:
        return {
            "object_features": np.zeros((batch_size, 256, 259), dtype=np.float32),
            "object_valid_mask": np.zeros((batch_size, 256), dtype=np.bool_),
            "time_features": np.zeros((batch_size, 2), dtype=np.float32),
            "resource_features": np.zeros((batch_size, RESOURCE_DIM), dtype=np.float32),
            "candidate_features": np.zeros((batch_size, max_candidates, CANDIDATE_DIM), dtype=np.float32),
            "candidate_valid_mask": np.zeros((batch_size, max_candidates), dtype=np.bool_),
            "candidate_action_mask": np.zeros((batch_size, max_candidates), dtype=np.bool_),
            "selected_candidate_mask": np.zeros((batch_size, max_candidates), dtype=np.bool_),
        }

    current = allocate_state()
    following = allocate_state()

    def fill(target: dict[str, np.ndarray], row: int, state: StateArray) -> None:
        count = len(state.candidate_path_indices)
        target["object_features"][row] = state.object_features
        target["object_valid_mask"][row] = state.object_valid_mask
        target["time_features"][row] = state.time_features
        target["resource_features"][row] = state.resource_features
        target["candidate_features"][row, :count] = state.candidate_features
        target["candidate_valid_mask"][row, :count] = state.candidate_valid_mask
        target["candidate_action_mask"][row, :count] = state.candidate_action_mask
        target["selected_candidate_mask"][row, :count] = state.selected_candidate_mask

    for row, item in enumerate(transitions):
        fill(current, row, item.state)
        fill(following, row, item.next_state)
    return {
        **{f"state_{name}": value for name, value in current.items()},
        **{f"next_{name}": value for name, value in following.items()},
        "action_position": np.asarray([item.action_position for item in transitions], dtype=np.int64),
        "reward": np.asarray([item.reward for item in transitions], dtype=np.float32),
        "discount": np.asarray([item.discount for item in transitions], dtype=np.float32),
        "terminal": np.asarray([item.terminal for item in transitions], dtype=np.bool_),
        "sample_weight": np.asarray([item.sample_weight for item in transitions], dtype=np.float32),
        "return_value": np.asarray([np.nan if item.return_value is None else item.return_value for item in transitions], dtype=np.float32),
    }
