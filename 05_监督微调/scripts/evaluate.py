from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from folder_cache_actor.config import config_sha256, load_config, resolve_path  # noqa: E402
from folder_cache_actor.data import ActorParquetDataset, build_context_features, make_collate, time_features  # noqa: E402
from folder_cache_actor.errors import ActorError, ArtifactCompatibilityError, OutputExistsError  # noqa: E402
from folder_cache_actor.inference import SupervisedActor  # noqa: E402
from folder_cache_actor.reporting import plot_evaluation_metrics  # noqa: E402
from folder_cache_actor.retrieval import ExactDualRetriever, stable_top_k  # noqa: E402
from folder_cache_actor.state import RollingAccessState  # noqa: E402
from folder_cache_actor.training import run_epoch  # noqa: E402
from folder_cache_actor.upstream import HistoryEncoderAdapter, load_access_dataset, validate_history_release  # noqa: E402
from folder_cache_actor.utils import read_json, sha256_file, validate_identifier, write_json  # noqa: E402
from folder_cache_actor.vector_store import StaticVectorStore  # noqa: E402


HORIZONS = (5, 10, 30, 300, 3600)
CANDIDATE_TYPES = (
    "static_top256",
    "history_top256",
    "retrieval_fusion_top256",
    "current_fusion_top256",
    "recent_top256",
    "hot_top256",
    "union_pool",
)


@dataclass
class MetricSums:
    object_hit: int = 0
    object_total: int = 0
    request_hit: int = 0
    request_total: int = 0
    byte_hit: int = 0
    byte_total: int = 0
    object_macro: float = 0.0
    request_macro: float = 0.0
    byte_macro: float = 0.0
    macro_count: int = 0

    def add(self, candidates: set[int], counts: dict[int, int], sizes: dict[int, int]) -> None:
        if not counts:
            return
        object_hit = sum(path_index in candidates for path_index in counts)
        request_total = sum(counts.values())
        request_hit = sum(count for path_index, count in counts.items() if path_index in candidates)
        byte_total = sum(count * sizes[path_index] for path_index, count in counts.items())
        byte_hit = sum(count * sizes[path_index] for path_index, count in counts.items() if path_index in candidates)
        self.object_hit += object_hit
        self.object_total += len(counts)
        self.request_hit += request_hit
        self.request_total += request_total
        self.byte_hit += byte_hit
        self.byte_total += byte_total
        self.object_macro += object_hit / len(counts)
        self.request_macro += request_hit / request_total
        self.byte_macro += byte_hit / byte_total if byte_total else 0.0
        self.macro_count += 1

    def result(self) -> dict[str, float]:
        return {
            "object_recall_micro": self.object_hit / max(1, self.object_total),
            "object_recall_macro": self.object_macro / max(1, self.macro_count),
            "request_coverage_micro": self.request_hit / max(1, self.request_total),
            "request_coverage_macro": self.request_macro / max(1, self.macro_count),
            "byte_coverage_micro": self.byte_hit / max(1, self.byte_total),
            "byte_coverage_macro": self.byte_macro / max(1, self.macro_count),
        }


def _access_times(events: Any, catalog: Any, path_index: int) -> np.ndarray:
    return np.frombuffer(events.events_by_position[catalog.path_to_position[int(path_index)]], dtype=np.int64)


def _group_values(events: Any, catalog: Any, group_index: int) -> dict[int, int]:
    start = int(events.group_offsets[group_index])
    end = int(events.group_offsets[group_index + 1])
    return {
        int(catalog.path_indices[int(events.group_positions[position])]): int(events.group_counts[position])
        for position in range(start, end)
    }


def _history_map(row: dict[str, Any]) -> dict[int, np.ndarray]:
    ids = row["persistent_folder_ids"]
    matrix = np.frombuffer(row["persistent_history_vectors"], dtype=np.float16).astype(np.float32).reshape(len(ids), 128)
    return {int(path_index): matrix[pos] for pos, path_index in enumerate(ids) if row["persistent_history_valid"][pos]}


def _candidate_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("cutoff_time_seconds", pa.int64(), nullable=False),
            pa.field("path_index", pa.int64(), nullable=False),
            pa.field("from_static", pa.bool_(), nullable=False),
            pa.field("from_history", pa.bool_(), nullable=False),
            pa.field("static_score", pa.float32(), nullable=False),
            pa.field("retrieval_history_score", pa.float32(), nullable=False),
            pa.field("retrieval_fusion_score", pa.float32(), nullable=False),
            pa.field("current_history_score", pa.float32(), nullable=False),
            pa.field("current_fusion_score", pa.float32(), nullable=False),
            pa.field("retrieval_fusion_rank", pa.int32(), nullable=False),
            pa.field("current_fusion_rank", pa.int32(), nullable=False),
            pa.field("history_revision", pa.int64(), nullable=False),
            pa.field("candidate_as_of_time", pa.int64(), nullable=False),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练结束后顺序重放测试集并评价05损失和自然候选召回")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    data_version = validate_identifier(args.data_version, "data_version")
    report_id = validate_identifier(args.report_id, "report_id")
    data_dir = resolve_path(MODULE_ROOT, config["paths"]["dataset_root"]) / data_version
    manifest = read_json(data_dir / "actor_samples_manifest.json")
    checkpoint = args.checkpoint.resolve()
    actor = SupervisedActor.load(checkpoint, device=args.device)
    if int(actor.metadata.get("epoch", -1)) != int(config["training"]["max_epochs"]):
        raise ArtifactCompatibilityError("测试只能使用固定完成第20轮的正式Actor检查点")
    if actor.metadata.get("data_manifest_sha256") != sha256_file(data_dir / "actor_samples_manifest.json"):
        raise ArtifactCompatibilityError("Actor检查点与监督数据版本不一致")
    report_dir = resolve_path(MODULE_ROOT, config["paths"]["outputs_root"]) / "reports" / report_id
    if report_dir.exists():
        raise OutputExistsError(f"report-id已存在，拒绝覆盖：{report_dir}")
    validate_history_release(resolve_path(MODULE_ROOT, config["paths"]["history_release"]))
    report_dir.mkdir(parents=True)
    vector_dir = resolve_path(MODULE_ROOT, config["paths"]["vector_store_dir"])
    static_store = StaticVectorStore.load(vector_dir)
    test_data = ActorParquetDataset(
        data_dir / "actor_samples.parquet",
        data_dir / "actor_history_snapshots.parquet",
        "test",
        1,
        int(config["training"]["seed"]),
    )
    test_batch_size = int(config["training"]["batch_size"])
    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        num_workers=int(config["training"]["dataloader_workers"]),
        collate_fn=make_collate(static_store),
        pin_memory=actor.device.type == "cuda",
    )
    test_batches = int(np.ceil(int(manifest["split_rows"]["test"]) / test_batch_size))
    test_ranking_metrics = run_epoch(
        actor.model,
        test_loader,
        actor.device,
        0,
        None,
        float(config["training"]["gradient_clip_norm"]),
        test_batches,
        int(config["training"]["progress_interval_batches"]),
        int(config["training"]["progress_interval_seconds"]),
        None,
    )
    retriever = ExactDualRetriever(
        static_store.path_indices,
        static_store.vectors,
        int(config["retrieval"]["static_top_k"]),
        int(config["retrieval"]["history_top_k"]),
        int(config["retrieval"]["fusion_top_k"]),
    )
    initial = np.load(vector_dir / "initial_history_index.npz", allow_pickle=False)
    history_index = {
        int(path_index): vector.astype(np.float32)
        for path_index, vector in zip(initial["path_indices"], initial["history_vectors"])
    }
    history_source = MODULE_ROOT.parent / "04_动态历史向量生成" / "src"
    catalog, events = load_access_dataset(
        history_source,
        resolve_path(MODULE_ROOT, config["paths"]["path_catalog"]),
        resolve_path(MODULE_ROOT, config["paths"]["access_dir"]),
    )
    history_encoder = HistoryEncoderAdapter(
        history_source,
        resolve_path(MODULE_ROOT, config["paths"]["history_release"]),
        device=args.device,
    )
    size_frame = pd.read_csv(resolve_path(MODULE_ROOT, config["paths"]["path_catalog"]), usecols=["path_index", "total_size_bytes"])
    sizes = {int(row.path_index): int(row.total_size_bytes) for row in size_frame.itertuples(index=False)}
    metrics = {candidate: {horizon: MetricSums() for horizon in HORIZONS} for candidate in CANDIDATE_TYPES}
    samples = pq.ParquetFile(data_dir / "actor_samples.parquet")
    snapshots = pq.ParquetFile(data_dir / "actor_history_snapshots.parquet")
    access_state = RollingAccessState()
    group_times = np.asarray(events.group_times, dtype=np.int64)
    group_cursor = 0
    first_cutoff = int(initial["snapshot_time"][0])
    warm_last_access: dict[int, int] = {}
    warm_hot_groups: list[tuple[int, dict[int, int]]] = []
    while group_cursor < len(group_times) and int(group_times[group_cursor]) < first_cutoff:
        event_time = int(group_times[group_cursor])
        values = _group_values(events, catalog, group_cursor)
        for path_index in values:
            warm_last_access[path_index] = event_time
        if event_time >= first_cutoff - 3600:
            warm_hot_groups.append((event_time, values))
        group_cursor += 1
    access_state.restore(first_cutoff, warm_last_access, warm_hot_groups)
    candidate_path = report_dir / "natural_candidates_test.parquet"
    candidate_writer = pq.ParquetWriter(candidate_path, _candidate_schema(), compression="zstd", version="2.6")
    candidate_buffer: list[dict[str, Any]] = []
    test_rows = 0
    empty_targets = 0
    started = time.monotonic()
    try:
        for group_index in range(samples.num_row_groups):
            sample_rows = samples.read_row_group(group_index).to_pylist()
            snapshot_rows = snapshots.read_row_group(group_index).to_pylist()
            for row, snapshot in zip(sample_rows, snapshot_rows):
                cutoff = int(row["cutoff_time_seconds"])
                while group_cursor < len(group_times) and int(group_times[group_cursor]) < cutoff:
                    access_state.ingest_group(int(group_times[group_cursor]), _group_values(events, catalog, group_cursor))
                    group_cursor += 1
                access_state.advance(cutoff)
                persistent = _history_map(snapshot)
                for path_index in row["removed_ids"]:
                    history_index.pop(int(path_index), None)
                for path_index in row["updated_ids"]:
                    vector = persistent.get(int(path_index))
                    if vector is None:
                        raise ArtifactCompatibilityError(f"updated_id缺少持久向量：{path_index}")
                    history_index[int(path_index)] = vector
                if row["split"] != "test":
                    continue
                context = build_context_features(
                    static_store,
                    row["context_folder_ids"],
                    row["context_valid_mask"],
                    row["context_recent_mask"],
                    row["context_hot_mask"],
                    persistent,
                )
                output = actor.predict(context[None], np.asarray(row["context_valid_mask"], dtype=np.bool_)[None], time_features(cutoff)[None])
                history_ids = np.asarray(sorted(history_index), dtype=np.int64)
                history_vectors = np.stack([history_index[int(value)] for value in history_ids]) if len(history_ids) else np.empty((0, 128), dtype=np.float32)
                result = retriever.retrieve(output["static_query"][0], output["history_query"][0], output["fusion_weights"][0], history_ids, history_vectors)
                union = result.union_ids
                active_union: list[int] = []
                for path_index in union:
                    values = _access_times(events, catalog, int(path_index))
                    end = int(np.searchsorted(values, cutoff, side="left"))
                    if end and int(values[end - 1]) >= cutoff - 86400:
                        active_union.append(int(path_index))
                current_map: dict[int, np.ndarray] = {}
                if active_union:
                    encoded = history_encoder.encode_as_of(cutoff, active_union, lambda path_index: _access_times(events, catalog, path_index), int(config["history_index"]["encode_batch_size"]))
                    current_map = {int(path_index): vector for path_index, vector in zip(encoded["path_indices"], encoded["vectors"])}
                current_history_scores = np.asarray([
                    float(output["history_query"][0] @ current_map[int(path_index)]) if int(path_index) in current_map else 0.0
                    for path_index in union
                ], dtype=np.float32)
                weights = output["fusion_weights"][0]
                persistent_fusion_all = weights[0] * result.static_scores + weights[1] * result.history_scores
                current_fusion_all = weights[0] * result.static_scores + weights[1] * current_history_scores
                current_ids, _ = stable_top_k(union, current_fusion_all, int(config["retrieval"]["fusion_top_k"]))
                candidates = {
                    "static_top256": set(result.static_ids.tolist()),
                    "history_top256": set(result.history_ids.tolist()),
                    "retrieval_fusion_top256": set(result.fusion_ids.tolist()),
                    "current_fusion_top256": set(current_ids.tolist()),
                    "recent_top256": set(access_state.recent_ids(256)),
                    "hot_top256": set(access_state.hot_ids(256)),
                    "union_pool": set(union.tolist()),
                }
                targets: dict[int, dict[int, int]] = {}
                all_positive = [int(value) for value in row["positive_folder_ids_all"]]
                for horizon in HORIZONS:
                    counts: dict[int, int] = {}
                    for path_index in all_positive:
                        values = _access_times(events, catalog, path_index)
                        start = int(np.searchsorted(values, cutoff, side="left"))
                        end = int(np.searchsorted(values, cutoff + horizon, side="left"))
                        if end > start:
                            counts[path_index] = end - start
                    targets[horizon] = counts
                    for candidate_type, selected in candidates.items():
                        metrics[candidate_type][horizon].add(selected, counts, sizes)
                empty_targets += int(not targets[3600])
                persistent_rank = {int(value): rank + 1 for rank, value in enumerate(result.fusion_ids)}
                current_rank = {int(value): rank + 1 for rank, value in enumerate(current_ids)}
                static_set = set(result.static_ids.tolist())
                history_set = set(result.history_ids.tolist())
                for position, path_index in enumerate(union):
                    candidate_buffer.append(
                        {
                            "cutoff_time_seconds": cutoff,
                            "path_index": int(path_index),
                            "from_static": int(path_index) in static_set,
                            "from_history": int(path_index) in history_set,
                            "static_score": float(result.static_scores[position]),
                            "retrieval_history_score": float(result.history_scores[position]),
                            "retrieval_fusion_score": float(persistent_fusion_all[position]),
                            "current_history_score": float(current_history_scores[position]),
                            "current_fusion_score": float(current_fusion_all[position]),
                            "retrieval_fusion_rank": persistent_rank.get(int(path_index), -1),
                            "current_fusion_rank": current_rank.get(int(path_index), -1),
                            "history_revision": int(row["history_revision"]),
                            "candidate_as_of_time": cutoff,
                        }
                    )
                if len(candidate_buffer) >= 20_000:
                    candidate_writer.write_table(pa.Table.from_pylist(candidate_buffer, schema=_candidate_schema()))
                    candidate_buffer.clear()
                test_rows += 1
                if test_rows % 60 == 0:
                    print(json.dumps({"stage": "evaluate_test", "test_rows": test_rows, "cutoff": cutoff, "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False), flush=True)
        if candidate_buffer:
            candidate_writer.write_table(pa.Table.from_pylist(candidate_buffer, schema=_candidate_schema()))
    finally:
        candidate_writer.close()
    report = {
        "schema_version": "folder-cache-test-evaluation/v1",
        "report_id": report_id,
        "data_version": data_version,
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": config_sha256(config),
        "test_rows": test_rows,
        "empty_target_rows": empty_targets,
        "test_ranking_metrics": test_ranking_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": {
            candidate: {str(horizon): metrics[candidate][horizon].result() for horizon in HORIZONS}
            for candidate in CANDIDATE_TYPES
        },
    }
    report_path = report_dir / "natural_recall_test.json"
    write_json(report_path, report)
    plot_evaluation_metrics(report, report_dir / "natural_recall_metrics.png")
    print(json.dumps({"report_dir": report_dir.as_posix(), "test_rows": test_rows}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except ActorError as exc:
        print(json.dumps({"status": "failed", "stage": "evaluate", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"status": "failed", "stage": "evaluate", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
