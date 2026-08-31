from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import ArtifactCompatibilityError, DataIntegrityError
from .utils import read_json, sha256_file


def load_source_package(alias: str, source_dir: Path) -> ModuleType:
    """Load an upstream ``src`` package under a stable, collision-free alias."""
    source_dir = source_dir.resolve()
    init_path = source_dir / "__init__.py"
    if not init_path.is_file():
        raise ArtifactCompatibilityError(f"上游Python包缺少__init__.py：{source_dir}")
    existing = sys.modules.get(alias)
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_path != init_path:
            raise ArtifactCompatibilityError(f"命名空间{alias}已绑定其他目录：{existing_path}")
        return existing
    spec = importlib.util.spec_from_file_location(
        alias,
        init_path,
        submodule_search_locations=[str(source_dir)],
    )
    if spec is None or spec.loader is None:
        raise ArtifactCompatibilityError(f"无法创建上游包加载器：{source_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for name in [key for key in sys.modules if key == alias or key.startswith(f"{alias}.")]:
            del sys.modules[name]
        raise
    return module


def _load_submodule(alias: str, source_dir: Path, name: str) -> ModuleType:
    load_source_package(alias, source_dir)
    return __import__(f"{alias}.{name}", fromlist=[name])


class StaticEncoderAdapter:
    def __init__(
        self,
        source_dir: Path,
        release_dir: Path,
        base_model_path: Path,
        device: str = "auto",
    ) -> None:
        manifest = release_summary(release_dir)
        if manifest.get("schema_version") != "static-semantic-encoder-manifest/v1":
            raise ArtifactCompatibilityError("不支持的03静态模型manifest")
        for name in ("model_config.json", "static_encoder.safetensors"):
            expected = manifest.get("files", {}).get(name, {}).get("sha256")
            path = release_dir / name
            if not path.is_file() or not expected or sha256_file(path) != expected:
                raise ArtifactCompatibilityError(f"03模型文件缺失或摘要不一致：{path}")
        inference = _load_submodule("_folder_cache_static_upstream", source_dir, "inference")
        resolved_device = None if device == "auto" else device
        self.encoder = inference.StaticSemanticEncoder.load(
            release_dir,
            device=resolved_device,
            base_model_path=base_model_path,
        )
        self.release_dir = release_dir.resolve()

    def encode(self, records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        result = self.encoder.encode(records)
        vectors = np.asarray(result["vectors"], dtype=np.float32)
        if vectors.shape != (len(records), 128):
            raise ArtifactCompatibilityError(f"03输出形状错误：{vectors.shape}")
        return {"path_indices": np.asarray(result["path_indices"], dtype=np.int64), "vectors": vectors}


class HistoryEncoderAdapter:
    def __init__(self, source_dir: Path, release_dir: Path, device: str = "auto") -> None:
        inference = _load_submodule("_folder_cache_history_upstream", source_dir, "inference")
        self.data_module = _load_submodule("_folder_cache_history_upstream", source_dir, "data")
        self.encoder = inference.DynamicHistoryEncoder.load(release_dir, device=device)
        self.release_dir = release_dir.resolve()
        self.contract = self._validate_contract()

    def _validate_contract(self) -> dict[str, Any]:
        config = self.encoder.config
        boundaries = [int(value) for value in config["target"]["time_boundaries_seconds"]]
        horizon = int(config["target"]["horizon_seconds"])
        if int(config["model"]["vector_dim"]) != 128:
            raise ArtifactCompatibilityError("04历史向量维度必须为128")
        if horizon != 3600 or boundaries != [0, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]:
            raise ArtifactCompatibilityError(
                "04正式release必须使用1小时10档合同；请使用最新04配置重新训练并执行evaluate发布release"
            )
        return {
            "prediction_horizon_seconds": horizon,
            "time_bin_edges_seconds": boundaries,
            "probability_dimension": len(boundaries),
            "no_access_class_index": len(boundaries) - 1,
            "expected_count_horizon_seconds": horizon,
        }

    def encode_as_of(
        self,
        snapshot_time: int,
        path_indices: Sequence[int],
        access_times_for: Callable[[int], Sequence[int] | np.ndarray],
        batch_size: int = 512,
    ) -> dict[str, np.ndarray]:
        ids = np.asarray([int(value) for value in path_indices], dtype=np.int64)
        if not len(ids):
            probability_dim = int(self.contract["probability_dimension"])
            return {
                "path_indices": ids,
                "vectors": np.empty((0, 128), dtype=np.float32),
                "next_access_probs": np.empty((0, probability_dim), dtype=np.float32),
                "expected_access_counts": np.empty(0, dtype=np.float32),
            }
        features: dict[str, list[np.ndarray]] = {
            "second_counts": [],
            "short_counts": [],
            "medium_counts": [],
            "long_counts": [],
            "history_state": [],
        }
        for path_index in ids:
            values = np.asarray(access_times_for(int(path_index)), dtype=np.int64)
            if values.ndim != 1 or (len(values) > 1 and np.any(values[1:] < values[:-1])):
                raise DataIntegrityError(f"path_index={path_index}的访问时间必须是一维非递减数组")
            inputs, state, _ = self.data_module.build_history_inputs(
                values,
                int(snapshot_time),
                self.encoder.config,
                feature_stats=self.encoder.feature_stats,
            )
            for name in ("second", "short", "medium", "long"):
                features[f"{name}_counts"].append(inputs[f"{name}_counts"])
            features["history_state"].append(state)
        stacked = {name: np.stack(values) for name, values in features.items()}
        outputs: dict[str, list[np.ndarray]] = {"vectors": [], "next_access_probs": [], "expected_access_counts": []}
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            predicted = self.encoder.predict_tensors(
                stacked["second_counts"][start:end, :, None],
                stacked["short_counts"][start:end, :, None],
                stacked["medium_counts"][start:end, :, None],
                stacked["long_counts"][start:end, :, None],
                stacked["history_state"][start:end],
            )
            for name in outputs:
                outputs[name].append(predicted[name])
        result = {name: np.concatenate(values, axis=0) for name, values in outputs.items()}
        vectors = np.asarray(result["vectors"], dtype=np.float32)
        if ids.shape != (len(path_indices),) or vectors.shape != (len(path_indices), 128):
            raise ArtifactCompatibilityError(f"04输出形状错误：ids={ids.shape}, vectors={vectors.shape}")
        return {
            "path_indices": ids,
            "vectors": vectors,
            "next_access_probs": np.asarray(result["next_access_probs"], dtype=np.float32),
            "expected_access_counts": np.asarray(result["expected_access_counts"], dtype=np.float32),
        }


def release_summary(release_dir: Path) -> dict[str, Any]:
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DataIntegrityError(f"正式release缺少manifest.json：{release_dir}")
    return read_json(manifest_path)


def validate_history_release(release_dir: Path) -> dict[str, Any]:
    manifest = release_summary(release_dir)
    if manifest.get("schema_version") != "dynamic-history-release-manifest/v1":
        raise ArtifactCompatibilityError("不支持的04历史模型manifest")
    for name in ("model_config.json", "feature_stats.json", "dynamic_encoder.safetensors"):
        expected = manifest.get("files", {}).get(name)
        path = release_dir / name
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise ArtifactCompatibilityError(f"04模型文件缺失或摘要不一致：{path}")
    model_config = read_json(release_dir / "model_config.json")
    contract = model_config.get("output_contract", {})
    expected_boundaries = [0, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
    if (
        int(contract.get("vector_dim", -1)) != 128
        or int(contract.get("horizon_seconds", -1)) != 3600
        or contract.get("time_boundaries_seconds") != expected_boundaries
        or int(contract.get("next_access_probability_dim", -1)) != 10
        or int(contract.get("no_access_class_index", -1)) != 9
    ):
        raise ArtifactCompatibilityError("04正式release不是1小时10档、128维输出合同")
    return manifest


def load_access_dataset(source_dir: Path, catalog_path: Path, access_dir: Path) -> tuple[Any, Any]:
    data = _load_submodule("_folder_cache_history_upstream", source_dir, "data")
    catalog = data.load_catalog(catalog_path)
    events = data.load_access_events(catalog, access_dir)
    return catalog, events
