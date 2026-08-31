"""模型制品加载、正式推理接口和交付清单生成。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from .data import build_instance_text, sha256_file
from .model import FrozenBGEEncoder, StaticSemanticModel


def load_trained_model(output_dir: Path, device: torch.device) -> tuple[StaticSemanticModel, dict[str, Any]]:
    config_path = output_dir / "model_config.json"
    weights_path = output_dir / "static_encoder.safetensors"
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    label_sizes = {field: len(values) for field, values in model_config["label_maps"].items()}
    model = StaticSemanticModel(model_config["architecture"], label_sizes)
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    model.to(device).eval()
    return model, model_config


def write_manifest(output_dir: Path, model_config: Mapping[str, Any]) -> Path:
    artifact_names = [
        "static_encoder.safetensors",
        "model_config.json",
        "evaluation_report.json",
        "loss_history.jsonl",
        "loss_curves.png",
    ]
    files = {}
    for name in artifact_names:
        path = output_dir / name
        if path.exists():
            files[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "static-semantic-encoder-manifest/v1",
        "model_version": model_config["schema_version"],
        "data": model_config["data"],
        "bge": {
            "model_name": model_config["bge"]["model_name"],
            "revision": model_config["bge"]["revision"],
            "resolved_commit": model_config["bge"]["resolved_commit"],
        },
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


class StaticSemanticEncoder:
    """后续模块唯一需要调用的静态语义编码接口。"""
    def __init__(
        self,
        model: StaticSemanticModel,
        bge: FrozenBGEEncoder,
        model_config: Mapping[str, Any],
        device: torch.device,
    ):
        self.model = model
        self.bge = bge
        self.model_config = dict(model_config)
        self.device = device

    @classmethod
    def load(
        cls,
        output_dir: str | Path,
        device: str | torch.device | None = None,
        base_model_path: str | Path | None = None,
    ) -> "StaticSemanticEncoder":
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        output_path = Path(output_dir).resolve()
        model, model_config = load_trained_model(output_path, resolved_device)
        bge_config = model_config["bge"]
        bge = FrozenBGEEncoder(
            model_name=bge_config["model_name"],
            revision=bge_config["revision"],
            max_length=int(bge_config["max_length"]),
            batch_size=int(bge_config["batch_size"]),
            precision=bge_config["precision"],
            device=resolved_device,
            local_path=base_model_path,
        )
        expected_commit = bge_config["resolved_commit"]
        if expected_commit and bge.resolved_commit != expected_commit:
            raise ValueError(f"推理 BGE commit 不一致：模型={expected_commit}，当前={bge.resolved_commit}")
        return cls(model=model, bge=bge, model_config=model_config, device=resolved_device)

    @torch.inference_mode()
    def encode(self, records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        """按输入顺序返回 path_indices 和 L2 归一化的 128 维向量。"""
        if not records:
            output_dim = int(self.model_config["architecture"]["output_dim"])
            return {
                "path_indices": np.empty((0,), dtype=np.int64),
                "vectors": np.empty((0, output_dim), dtype=np.float32),
            }
        path_indices: list[int] = []
        semantic_texts: list[str] = []
        instance_texts: list[str] = []
        date_features: list[list[float]] = []
        date_mean = float(self.model_config["date_normalization"]["mean"])
        date_std = float(self.model_config["date_normalization"]["std"])
        for position, record in enumerate(records):
            path_indices.append(int(record.get("path_index", position)))
            semantic = record.get("semantic_text", record.get("embedding_text", ""))
            if not str(semantic).strip():
                raise ValueError(f"第 {position} 条记录缺少 semantic_text/embedding_text")
            semantic_texts.append(str(semantic))
            instance = record.get("instance_text")
            if instance is None:
                instance = build_instance_text(record.get("source_text", ""), record.get("instance_context", ""))
            instance_texts.append(str(instance))
            has_date = int(record.get("has_observe_date", 0)) == 1
            ordinal = record.get("date_ordinal")
            if has_date and ordinal is not None and math.isfinite(float(ordinal)):
                date_features.append([(float(ordinal) - date_mean) / date_std, 1.0])
            else:
                date_features.append([0.0, 0.0])

        semantic_bge = torch.from_numpy(self.bge.encode(semantic_texts)).to(self.device)
        instance_bge = torch.from_numpy(self.bge.encode(instance_texts)).to(self.device)
        dates = torch.tensor(date_features, dtype=torch.float32, device=self.device)
        vectors = self.model.encode_final(semantic_bge, instance_bge, dates).float().cpu().numpy()
        if not np.isfinite(vectors).all():
            raise ValueError("模型输出包含 NaN 或无穷值")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("模型输出未完成 L2 归一化")
        return {"path_indices": np.asarray(path_indices, dtype=np.int64), "vectors": vectors.astype(np.float32)}
