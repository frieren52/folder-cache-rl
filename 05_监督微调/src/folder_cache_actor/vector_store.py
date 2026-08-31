from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import ArtifactCompatibilityError, DataIntegrityError
from .retrieval import normalize_rows
from .utils import read_json, sha256_file


@dataclass(frozen=True)
class StaticVectorStore:
    path_indices: np.ndarray
    vectors: np.ndarray
    positions: dict[int, int]
    manifest: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "StaticVectorStore":
        root = Path(root)
        manifest = read_json(root / "vector_store_manifest.json")
        if manifest.get("schema_version") != "folder-cache-vector-store/v1":
            raise ArtifactCompatibilityError("不支持的向量库manifest")
        for name in ("static_path_vectors.npy", "static_vector_index.parquet", "initial_history_index.npz"):
            expected = manifest.get("files", {}).get(name, {}).get("sha256")
            path = root / name
            if not path.is_file() or not expected or sha256_file(path) != expected:
                raise DataIntegrityError(f"向量库文件缺失或摘要不一致：{path}")
        vectors = np.load(root / "static_path_vectors.npy", mmap_mode="r")
        index = pd.read_parquet(root / "static_vector_index.parquet")
        required = {"path_index", "row_position"}
        if not required.issubset(index.columns):
            raise DataIntegrityError("static_vector_index.parquet字段不完整")
        index = index.sort_values("row_position")
        ids = index["path_index"].to_numpy(dtype=np.int64)
        if vectors.shape != (len(ids), 128) or index["row_position"].tolist() != list(range(len(ids))):
            raise DataIntegrityError("静态向量矩阵与索引不一致")
        if len(set(ids.tolist())) != len(ids):
            raise DataIntegrityError("静态向量索引path_index重复")
        norms = np.linalg.norm(np.asarray(vectors, dtype=np.float32), axis=1)
        if not np.all(np.isfinite(norms)) or not np.allclose(norms, 1.0, atol=1e-4):
            raise DataIntegrityError("静态向量不是有限的L2归一化向量")
        return cls(ids, vectors, {int(value): pos for pos, value in enumerate(ids)}, manifest)

    def get(self, path_indices: np.ndarray | list[int]) -> np.ndarray:
        positions: list[int] = []
        for value in path_indices:
            position = self.positions.get(int(value))
            if position is None:
                raise DataIntegrityError(f"静态向量库缺少path_index={value}")
            positions.append(position)
        return np.asarray(self.vectors[positions], dtype=np.float32)


def save_static_vectors(path: Path, vectors: np.ndarray) -> None:
    values = normalize_rows(np.asarray(vectors, dtype=np.float32))
    if not np.all(np.isfinite(values)):
        raise DataIntegrityError("静态向量包含非有限值")
    np.save(path, values, allow_pickle=False)

