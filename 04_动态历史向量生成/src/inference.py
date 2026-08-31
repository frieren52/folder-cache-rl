from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
from safetensors.torch import load_file

from .config import validate_config
from .data import SCALE_NAMES, build_history_inputs
from .errors import ArtifactCompatibilityError
from .model import DynamicHistoryModel
from .utils import read_json, resolve_device, verify_manifest_files


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _parse_aware_second(value: Any, field_name: str) -> int:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} 不是合法 ISO 时间：{value!r}") from exc
    else:
        raise TypeError(f"{field_name} 必须是带时区 datetime 或 ISO 字符串")
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} 必须包含时区")
    if parsed.microsecond != 0:
        raise ValueError(f"{field_name} 必须对齐到整秒")
    return int(parsed.astimezone(SHANGHAI).timestamp())


class DynamicHistoryEncoder:
    def __init__(
        self,
        model: DynamicHistoryModel,
        config: Mapping[str, Any],
        feature_stats: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.config = config
        self.feature_stats = feature_stats
        self.device = device

    @classmethod
    def load(cls, release_dir: str | Path, device: str = "auto") -> "DynamicHistoryEncoder":
        root = Path(release_dir)
        manifest = read_json(root / "manifest.json")
        if manifest.get("schema_version") != "dynamic-history-release-manifest/v1":
            raise ArtifactCompatibilityError(f"不支持的模型 manifest：{root / 'manifest.json'}")
        try:
            verify_manifest_files(root, manifest.get("files", {}))
        except Exception as exc:
            raise ArtifactCompatibilityError(f"模型发布文件校验失败：{exc}") from exc
        model_config = read_json(root / "model_config.json")
        if model_config.get("schema_version") != "dynamic-history-encoder/v1":
            raise ArtifactCompatibilityError("不支持的 model_config schema")
        config = model_config["resolved_config"]
        validate_config(config)
        boundaries = [int(value) for value in config["target"]["time_boundaries_seconds"]]
        expected_contract = {
            "vector_dim": int(config["model"]["vector_dim"]),
            "next_access_probability_dim": len(boundaries),
            "time_boundaries_seconds": boundaries,
            "no_access_class_index": len(boundaries) - 1,
            "horizon_seconds": int(config["target"]["horizon_seconds"]),
        }
        if model_config.get("output_contract") != expected_contract:
            raise ArtifactCompatibilityError("model_config 输出接口与模型配置不一致")
        feature_stats = read_json(root / "feature_stats.json")
        if feature_stats.get("schema_version") != "dynamic-history-feature-stats/v1":
            raise ArtifactCompatibilityError("不支持的 feature_stats schema")
        resolved_device = resolve_device(device)
        model = DynamicHistoryModel(config).to(resolved_device)
        state = load_file(str(root / "dynamic_encoder.safetensors"), device=str(resolved_device))
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ArtifactCompatibilityError(f"模型参数形状不兼容：{exc}") from exc
        return cls(model, config, feature_stats, resolved_device)

    @torch.no_grad()
    def predict_tensors(
        self,
        second_counts: np.ndarray,
        short_counts: np.ndarray,
        medium_counts: np.ndarray,
        long_counts: np.ndarray,
        history_state: np.ndarray,
    ) -> dict[str, np.ndarray]:
        arrays = (second_counts, short_counts, medium_counts, long_counts, history_state)
        tensors = [torch.as_tensor(value, dtype=torch.float32, device=self.device) for value in arrays]
        outputs = self.model(*tensors)
        result = {
            "vectors": outputs["vectors"].float().cpu().numpy(),
            "next_access_probs": outputs["next_access_probs"].float().cpu().numpy(),
            "expected_access_counts": outputs["expected_access_counts"].float().cpu().numpy(),
        }
        for name, values in result.items():
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"模型输出包含非有限值：{name}")
        return result

    def encode(
        self,
        history_records: Sequence[Mapping[str, Any]],
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        path_indices: list[int] = []
        features: dict[str, list[np.ndarray]] = {
            "second_counts": [],
            "short_counts": [],
            "medium_counts": [],
            "long_counts": [],
            "history_state": [],
        }
        for record_index, record in enumerate(history_records):
            path_index = record.get("path_index")
            if isinstance(path_index, bool) or not isinstance(path_index, (int, np.integer)):
                raise TypeError(f"history_records[{record_index}].path_index 必须是 int64")
            path_index = int(path_index)
            if not -(2**63) <= path_index < 2**63:
                raise ValueError(f"history_records[{record_index}].path_index 超出 int64")
            snapshot = _parse_aware_second(
                record.get("snapshot_time"), f"history_records[{record_index}].snapshot_time"
            )
            raw_access_times = record.get("access_times")
            if not isinstance(raw_access_times, Sequence) or isinstance(raw_access_times, (str, bytes)):
                raise TypeError(f"history_records[{record_index}].access_times 必须是时间序列")
            parsed_times = np.asarray(
                [
                    _parse_aware_second(
                        value, f"history_records[{record_index}].access_times[{item_index}]"
                    )
                    for item_index, value in enumerate(raw_access_times)
                ],
                dtype=np.int64,
            )
            if parsed_times.size and np.any(parsed_times[1:] < parsed_times[:-1]):
                raise ValueError(f"history_records[{record_index}].access_times 必须非递减")
            if parsed_times.size and int(parsed_times[-1]) >= snapshot:
                raise ValueError(f"history_records[{record_index}] 含等于或晚于快照的事件")
            max_window = max(
                int(item["window_seconds"]) for item in self.config["history"]["scales"]
            )
            parsed_times = parsed_times[parsed_times >= snapshot - max_window]
            scale_inputs, state, _ = build_history_inputs(
                parsed_times,
                snapshot,
                self.config,
                feature_stats=self.feature_stats,
            )
            path_indices.append(path_index)
            for name in SCALE_NAMES:
                features[f"{name}_counts"].append(scale_inputs[f"{name}_counts"])
            features["history_state"].append(state)
        if not path_indices:
            vector_dim = int(self.config["model"]["vector_dim"])
            probability_dim = len(self.config["target"]["time_boundaries_seconds"])
            return {
                "path_indices": np.empty(0, dtype=np.int64),
                "vectors": np.empty((0, vector_dim), dtype=np.float32),
                "next_access_probs": np.empty((0, probability_dim), dtype=np.float32),
                "expected_access_counts": np.empty(0, dtype=np.float32),
            }
        stacked = {name: np.stack(values) for name, values in features.items()}
        outputs = {"vectors": [], "next_access_probs": [], "expected_access_counts": []}
        for start in range(0, len(path_indices), batch_size):
            end = min(start + batch_size, len(path_indices))
            predicted = self.predict_tensors(
                stacked["second_counts"][start:end, :, None],
                stacked["short_counts"][start:end, :, None],
                stacked["medium_counts"][start:end, :, None],
                stacked["long_counts"][start:end, :, None],
                stacked["history_state"][start:end],
            )
            for name in outputs:
                outputs[name].append(predicted[name])
        return {
            "path_indices": np.asarray(path_indices, dtype=np.int64),
            **{name: np.concatenate(values, axis=0) for name, values in outputs.items()},
        }
