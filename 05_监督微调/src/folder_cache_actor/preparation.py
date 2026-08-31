from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import config_sha256, resolve_path
from .errors import DataIntegrityError, OutputExistsError
from .upstream import (
    HistoryEncoderAdapter,
    StaticEncoderAdapter,
    load_access_dataset,
    release_summary,
    validate_history_release,
)
from .utils import file_descriptions, parse_time, sha256_file, write_json
from .vector_store import save_static_vectors


def _access_times_for(events: Any, catalog: Any, path_index: int) -> np.ndarray:
    position = catalog.path_to_position[int(path_index)]
    return np.frombuffer(events.events_by_position[position], dtype=np.int64)


def build_vector_store(config: Mapping[str, Any], module_root: Path, device: str = "auto") -> Path:
    paths = config["paths"]
    output_dir = resolve_path(module_root, paths["vector_store_dir"])
    final_names = (
        "static_path_vectors.npy",
        "static_vector_index.parquet",
        "initial_history_index.npz",
        "vector_store_manifest.json",
    )
    existing = [output_dir / name for name in final_names if (output_dir / name).exists()]
    if existing:
        raise OutputExistsError(f"向量库已存在，拒绝覆盖；请改用新的vector_store_dir：{existing[0]}")
    staging = output_dir / f".build-{os.getpid()}"
    repository_root = module_root.parent
    static_source = repository_root / "03_静态语义向量生成" / "src"
    history_source = repository_root / "04_动态历史向量生成" / "src"
    catalog_path = resolve_path(module_root, paths["path_catalog"])
    records_path = resolve_path(module_root, paths["embedding_records"])
    access_dir = resolve_path(module_root, paths["access_dir"])
    static_release = resolve_path(module_root, paths["static_release"])
    static_base_model = resolve_path(module_root, paths["static_base_model"])
    history_release = resolve_path(module_root, paths["history_release"])
    validate_history_release(history_release)
    staging.mkdir(parents=True, exist_ok=False)
    records = pd.read_csv(records_path)
    required = {"path_index", "embedding_text", "source_text", "instance_context", "date_ordinal", "has_observe_date"}
    missing = sorted(required - set(records.columns))
    if missing:
        raise DataIntegrityError(f"embedding_records.csv缺少字段：{missing}")
    records["path_index"] = pd.to_numeric(records["path_index"], errors="raise").astype("int64")
    if records["path_index"].duplicated().any():
        raise DataIntegrityError("embedding_records.csv的path_index必须唯一")
    catalog, events = load_access_dataset(history_source, catalog_path, access_dir)
    catalog_ids = np.asarray(catalog.path_indices, dtype=np.int64)
    if set(records["path_index"].tolist()) != set(catalog_ids.tolist()):
        raise DataIntegrityError("embedding_records与path_catalog的path_index覆盖不一致")
    records = records.set_index("path_index").loc[catalog_ids].reset_index()
    static_encoder = StaticEncoderAdapter(
        static_source,
        static_release,
        static_base_model,
        device=device,
    )
    static_batches: list[np.ndarray] = []
    batch_size = 512
    for start in range(0, len(records), batch_size):
        batch = records.iloc[start : start + batch_size].where(pd.notna(records.iloc[start : start + batch_size]), None)
        result = static_encoder.encode(batch.to_dict(orient="records"))
        expected = records.iloc[start : start + batch_size]["path_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(result["path_indices"], expected):
            raise DataIntegrityError("03编码输出顺序与输入path_index不一致")
        static_batches.append(result["vectors"])
        print(json.dumps({"stage": "static_encode", "completed": min(start + batch_size, len(records)), "total": len(records)}, ensure_ascii=False), flush=True)
    static_vectors = np.concatenate(static_batches, axis=0)
    static_vectors_path = staging / "static_path_vectors.npy"
    save_static_vectors(static_vectors_path, static_vectors)
    static_index_path = staging / "static_vector_index.parquet"
    pd.DataFrame({"path_index": catalog_ids, "row_position": np.arange(len(catalog_ids), dtype=np.int64)}).to_parquet(static_index_path, index=False)

    history_encoder = HistoryEncoderAdapter(history_source, history_release, device=device)
    snapshot_time = parse_time(str(config["time"]["train_start"]), "time.train_start")
    active_ids: list[int] = []
    last_event_times: list[int] = []
    for path_index in catalog_ids:
        values = _access_times_for(events, catalog, int(path_index))
        end = int(np.searchsorted(values, snapshot_time, side="left"))
        if end and int(values[end - 1]) >= snapshot_time - 86400:
            active_ids.append(int(path_index))
            last_event_times.append(int(values[end - 1]))
    history_result = history_encoder.encode_as_of(
        snapshot_time,
        active_ids,
        lambda path_index: _access_times_for(events, catalog, path_index),
        batch_size=int(config["history_index"]["encode_batch_size"]),
    )
    last_values = np.asarray(last_event_times, dtype=np.int64)
    recent = snapshot_time - last_values < int(config["history_index"]["recent_window_seconds"])
    slots = np.where(
        recent,
        np.asarray(active_ids, dtype=np.int64) % int(config["history_index"]["initial_recent_slots"]),
        np.asarray(active_ids, dtype=np.int64) % int(config["history_index"]["initial_long_slots"]),
    )
    next_refresh = snapshot_time + 10 * (slots + 1)
    next_refresh = np.where(
        recent,
        np.minimum(next_refresh, last_values + int(config["history_index"]["recent_window_seconds"])),
        next_refresh,
    )
    initial_path = staging / "initial_history_index.npz"
    np.savez_compressed(
        initial_path,
        path_indices=np.asarray(active_ids, dtype=np.int64),
        history_vectors=history_result["vectors"].astype(np.float16),
        last_event_times=last_values,
        vector_as_of_times=np.full(len(active_ids), snapshot_time, dtype=np.int64),
        next_refresh_times=next_refresh.astype(np.int64),
        history_revision=np.asarray([1], dtype=np.int64),
        snapshot_time=np.asarray([snapshot_time], dtype=np.int64),
    )
    manifest_path = staging / "vector_store_manifest.json"
    files = file_descriptions([static_vectors_path, static_index_path, initial_path])
    for name, description in files.items():
        description["path"] = name
    manifest = {
        "schema_version": "folder-cache-vector-store/v1",
        "config_sha256": config_sha256(config),
        "snapshot_time": snapshot_time,
        "warmup_start": parse_time(str(config["time"]["warmup_start"]), "time.warmup_start"),
        "vector_dim": 128,
        "catalog_objects": len(catalog_ids),
        "initial_history_objects": len(active_ids),
        "history_revision": 1,
        "inputs": {
            "path_catalog": {"path": catalog_path.as_posix(), "sha256": sha256_file(catalog_path)},
            "embedding_records": {"path": records_path.as_posix(), "sha256": sha256_file(records_path)},
            "static_release": release_summary(static_release),
            "history_release": release_summary(history_release),
        },
        "history_output_contract": history_encoder.contract,
        "files": files,
    }
    write_json(manifest_path, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in final_names:
        (staging / name).replace(output_dir / name)
    staging.rmdir()
    print(json.dumps({"vector_store_dir": output_dir.as_posix(), "catalog_objects": len(catalog_ids), "history_objects": len(active_ids)}, ensure_ascii=False), flush=True)
    return output_dir
