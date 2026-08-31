"""读取 02 产物，生成固定切分、日期特征、标签表和模型输入张量。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


SUPPORTED_QUALITY_TIERS = {"supported_clean", "supported_with_unresolved"}
SEMANTIC_FIELDS = {
    "product": "PRODUCT",
    "instrument": "INSTRUMENT",
    "level": "LEVEL",
}
SOURCE_FIELDS = {
    "archive_root": "ARCHIVE_ROOT",
    "subsystem": "SUBSYSTEM",
}
_SEQUENCE_FIELD = re.compile(r"\[([A-Z_]+)=([^\]]+)\]")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是映射：{config_path}")
    return config


def resolve_path(module_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else (module_root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(semantic_group_id: str) -> str:
    """实现方案中冻结的 SHA-256 语义组切分。"""
    digest = hashlib.sha256(semantic_group_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 10
    return "train" if bucket <= 7 else "validation"


def parse_sequence(value: Any) -> dict[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    return {key: text.strip() for key, text in _SEQUENCE_FIELD.findall(str(value))}


def build_instance_text(source_text: Any, instance_context: Any) -> str:
    parts = [str(value).strip().strip("；") for value in (source_text, instance_context) if pd.notna(value) and str(value).strip()]
    return "；".join(parts)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少字段：{', '.join(missing)}")


def _label_map(values: Iterable[Any]) -> dict[str, int]:
    labels = sorted({str(value) for value in values if pd.notna(value) and str(value)})
    return {label: index for index, label in enumerate(labels)}


def prepare_record_files(config: Mapping[str, Any], module_root: Path) -> tuple[Path, dict[str, Any]]:
    """把 02 的 CSV 转成训练直接读取的两个 Parquet 文件。"""
    paths = config["paths"]
    records_path = resolve_path(module_root, paths["embedding_records"])
    catalog_path = resolve_path(module_root, paths["semantic_catalog"])
    data_dir = resolve_path(module_root, paths["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(records_path, keep_default_na=True)
    catalog = pd.read_csv(catalog_path, keep_default_na=True)
    _require_columns(
        records,
        {
            "path_index",
            "path",
            "semantic_group_id",
            "embedding_text",
            "semantic_sequence",
            "source_context",
            "source_text",
            "instance_context",
            "quality_tier",
            "date_ordinal",
            "has_observe_date",
        },
        records_path,
    )
    _require_columns(catalog, {"semantic_group_id", "embedding_text"}, catalog_path)

    records["path_index"] = pd.to_numeric(records["path_index"], errors="raise").astype("int64")
    if records["path_index"].duplicated().any():
        raise ValueError("embedding_records.csv 的 path_index 必须唯一")
    records["semantic_group_id"] = records["semantic_group_id"].astype(str)
    catalog_ids = set(catalog["semantic_group_id"].astype(str))
    missing_groups = sorted(set(records["semantic_group_id"]) - catalog_ids)
    if missing_groups:
        raise ValueError(f"semantic_catalog.csv 缺少 {len(missing_groups)} 个语义组")

    # 两层文本严格复用 02 已产出的字段，不在 03 重新解释目录层级。
    records["semantic_text"] = records["embedding_text"].fillna("").astype(str)
    records["instance_text"] = [
        build_instance_text(source, context)
        for source, context in zip(records["source_text"], records["instance_context"])
    ]
    records["raw_path"] = records["path"].fillna("").astype(str)
    records["split"] = records["semantic_group_id"].map(stable_split)
    records["is_semantic_supervised"] = records["quality_tier"].isin(SUPPORTED_QUALITY_TIERS)

    semantic_parsed = records["semantic_sequence"].map(parse_sequence)
    source_parsed = records["source_context"].map(parse_sequence)
    for output_name, source_name in SEMANTIC_FIELDS.items():
        records[output_name] = semantic_parsed.map(lambda item, key=source_name: item.get(key, ""))
    for output_name, source_name in SOURCE_FIELDS.items():
        records[output_name] = source_parsed.map(lambda item, key=source_name: item.get(key, ""))

    records["has_observe_date"] = pd.to_numeric(records["has_observe_date"], errors="coerce").fillna(0).astype("int8")
    records["date_ordinal"] = pd.to_numeric(records["date_ordinal"], errors="coerce")
    # 日期统计量只从训练集计算，验证和推理固定复用该均值、标准差。
    valid_train_dates = records.loc[
        (records["split"] == "train") & (records["has_observe_date"] == 1), "date_ordinal"
    ].dropna()
    if valid_train_dates.empty:
        raise ValueError("训练集没有可用于日期归一化的有效日期")
    date_mean = float(valid_train_dates.mean())
    date_std = float(valid_train_dates.std(ddof=0))
    if not math.isfinite(date_std) or date_std <= 0:
        date_std = 1.0
    records["date_value"] = 0.0
    date_mask = (records["has_observe_date"] == 1) & records["date_ordinal"].notna()
    records.loc[date_mask, "date_value"] = (records.loc[date_mask, "date_ordinal"] - date_mean) / date_std

    output_columns = [
        "path_index",
        "semantic_group_id",
        "semantic_text",
        "instance_text",
        "raw_path",
        "quality_tier",
        "is_semantic_supervised",
        "product",
        "instrument",
        "level",
        "archive_root",
        "subsystem",
        "date_ordinal",
        "date_value",
        "has_observe_date",
    ]
    train_records = records.loc[records["split"] == "train", output_columns].sort_values("path_index")
    validation_records = records.loc[records["split"] == "validation", output_columns].sort_values("path_index")
    train_path = data_dir / "train_records.parquet"
    validation_path = data_dir / "validation_records.parquet"
    train_records.to_parquet(train_path, index=False)
    validation_records.to_parquet(validation_path, index=False)

    semantic_train = train_records.loc[train_records["is_semantic_supervised"]]
    label_maps = {
        "product": _label_map(semantic_train["product"]),
        "instrument": _label_map(semantic_train["instrument"]),
        "level": _label_map(semantic_train["level"]),
        "archive_root": _label_map(train_records["archive_root"]),
        "subsystem": _label_map(train_records["subsystem"]),
    }
    metadata: dict[str, Any] = {
        "schema_version": "static-semantic-prepared/v1",
        "input_files": {
            "embedding_records": {"path": str(records_path), "sha256": sha256_file(records_path)},
            "semantic_catalog": {"path": str(catalog_path), "sha256": sha256_file(catalog_path)},
        },
        "split": {
            "algorithm": "sha256-first-8-bytes-big-endian-mod-10",
            "train_buckets": list(range(8)),
            "validation_buckets": [8, 9],
            "train_records": int(len(train_records)),
            "validation_records": int(len(validation_records)),
        },
        "date_normalization": {"mean": date_mean, "std": date_std, "ddof": 0},
        "label_maps": label_maps,
        "quality_tiers": {
            "semantic_supervised": sorted(SUPPORTED_QUALITY_TIERS),
            "instance_only": ["fallback_raw"],
        },
        "bge": {
            "model_name": config["bge"]["model_name"],
            "configured_revision": config["bge"]["revision"],
            "precision": config["bge"]["precision"],
            "resolved_commit": None,
            "tokenizer": None,
            "cache_file": None,
        },
    }
    metadata_path = data_dir / "data_meta.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return data_dir, metadata


def build_bge_cache(
    data_dir: Path,
    encoder: Any,
    cache_dtype: str = "float16",
) -> dict[str, Any]:
    """分别缓存语义文本、实例文本和训练用原始路径的冻结 BGE 表示。"""
    train = pd.read_parquet(data_dir / "train_records.parquet")
    validation = pd.read_parquet(data_dir / "validation_records.parquet")
    records = pd.concat([train, validation], ignore_index=True).sort_values("path_index")
    group_records = records.drop_duplicates("semantic_group_id").sort_values("semantic_group_id")

    print(f"编码语义文本：{len(group_records)} 条", flush=True)
    semantic_vectors = encoder.encode(group_records["semantic_text"].tolist())
    print(f"编码实例文本：{len(records)} 条", flush=True)
    instance_vectors = encoder.encode(records["instance_text"].tolist())
    print(f"编码原始路径：{len(records)} 条", flush=True)
    raw_path_vectors = encoder.encode(records["raw_path"].tolist())
    dtype = np.float16 if cache_dtype == "float16" else np.float32
    cache_path = data_dir / "bge_cache.npz"
    np.savez_compressed(
        cache_path,
        semantic_group_ids=np.asarray(group_records["semantic_group_id"].astype(str).tolist(), dtype=np.str_),
        semantic_vectors=semantic_vectors.astype(dtype),
        path_indices=records["path_index"].to_numpy(dtype=np.int64),
        instance_vectors=instance_vectors.astype(dtype),
        raw_path_vectors=raw_path_vectors.astype(dtype),
    )

    metadata_path = data_dir / "data_meta.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["bge"].update(encoder.metadata())
    metadata["bge"]["cache_file"] = {"path": str(cache_path), "sha256": sha256_file(cache_path)}
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


class PreparedData:
    """把 Parquet 与 BGE 缓存组合成训练脚本需要的批量张量。"""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.train = pd.read_parquet(data_dir / "train_records.parquet")
        self.validation = pd.read_parquet(data_dir / "validation_records.parquet")
        with (data_dir / "data_meta.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        cache_path = data_dir / "bge_cache.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"缺少 {cache_path}，请先运行 prepare_data.py 生成 BGE 缓存")
        with np.load(cache_path, allow_pickle=False) as cache:
            self.semantic_group_ids = cache["semantic_group_ids"].astype(str)
            self.semantic_vectors = cache["semantic_vectors"].astype(np.float32)
            self.path_indices = cache["path_indices"].astype(np.int64)
            self.instance_vectors = cache["instance_vectors"].astype(np.float32)
            self.raw_path_vectors = cache["raw_path_vectors"].astype(np.float32)
        self.group_vector_rows = {value: index for index, value in enumerate(self.semantic_group_ids)}
        self.path_vector_rows = {int(value): index for index, value in enumerate(self.path_indices)}
        all_records = pd.concat([self.train, self.validation], ignore_index=True)
        self.records_by_path = all_records.set_index("path_index", drop=False)
        self.groups = (
            all_records.loc[all_records["is_semantic_supervised"]]
            .drop_duplicates("semantic_group_id")
            .set_index("semantic_group_id", drop=False)
        )
        self.label_maps: dict[str, dict[str, int]] = self.metadata["label_maps"]

    def split_records(self, split: str) -> pd.DataFrame:
        if split == "train":
            return self.train
        if split == "validation":
            return self.validation
        raise ValueError(f"未知数据切分：{split}")

    def _semantic_vectors(self, group_ids: Sequence[str]) -> np.ndarray:
        return np.stack([self.semantic_vectors[self.group_vector_rows[value]] for value in group_ids])

    def _path_vectors(self, path_ids: Sequence[int], kind: str) -> np.ndarray:
        source = self.instance_vectors if kind == "instance" else self.raw_path_vectors
        return np.stack([source[self.path_vector_rows[int(value)]] for value in path_ids])

    def _date_features(self, path_ids: Sequence[int]) -> np.ndarray:
        rows = self.records_by_path.loc[list(path_ids)]
        return rows[["date_value", "has_observe_date"]].to_numpy(dtype=np.float32)

    def _group_labels(self, group_ids: Sequence[str], field: str) -> np.ndarray:
        mapping = self.label_maps[field]
        return np.asarray([mapping.get(str(self.groups.loc[value, field]), -1) for value in group_ids], dtype=np.int64)

    def _path_labels(self, path_ids: Sequence[int], field: str) -> np.ndarray:
        mapping = self.label_maps[field]
        return np.asarray([mapping.get(str(self.records_by_path.loc[value, field]), -1) for value in path_ids], dtype=np.int64)

    def semantic_batch(self, triplets: Sequence[tuple[str, str, str]]) -> dict[str, torch.Tensor]:
        groups = np.asarray(triplets, dtype=str)
        flat = groups.reshape(-1).tolist()
        return {
            "teacher": torch.from_numpy(self._semantic_vectors(flat)).reshape(len(groups), 3, -1),
            "product": torch.from_numpy(self._group_labels(flat, "product")).reshape(len(groups), 3),
            "instrument": torch.from_numpy(self._group_labels(flat, "instrument")).reshape(len(groups), 3),
            "level": torch.from_numpy(self._group_labels(flat, "level")).reshape(len(groups), 3),
        }

    def instance_batch(self, triplets: Sequence[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
        paths = np.asarray(triplets, dtype=np.int64)
        anchor = paths[:, 0].tolist()
        positive = paths[:, 1].tolist()
        negative = paths[:, 2].tolist()
        all_paths = paths.reshape(-1).tolist()
        return {
            "anchor_text": torch.from_numpy(self._path_vectors(anchor, "instance")),
            "positive_path": torch.from_numpy(self._path_vectors(positive, "raw")),
            "negative_path": torch.from_numpy(self._path_vectors(negative, "raw")),
            "date": torch.from_numpy(self._date_features(all_paths)).reshape(len(paths), 3, 2),
            "archive_root": torch.from_numpy(self._path_labels(anchor, "archive_root")),
            "subsystem": torch.from_numpy(self._path_labels(anchor, "subsystem")),
        }

    def final_batch(self, triplets: Sequence[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
        paths = np.asarray(triplets, dtype=np.int64)
        flat_paths = paths.reshape(-1).tolist()
        group_ids = [str(self.records_by_path.loc[value, "semantic_group_id"]) for value in flat_paths]
        return {
            "semantic": torch.from_numpy(self._semantic_vectors(group_ids)).reshape(len(paths), 3, -1),
            "instance": torch.from_numpy(self._path_vectors(flat_paths, "instance")).reshape(len(paths), 3, -1),
            "date": torch.from_numpy(self._date_features(flat_paths)).reshape(len(paths), 3, 2),
        }


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}
