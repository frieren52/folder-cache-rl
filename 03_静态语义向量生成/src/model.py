"""静态语义双分支模型，以及只负责文本编码的冻结 BGE-M3。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, eps: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim, eps=eps),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(values), p=2, dim=-1)


class StaticSemanticModel(nn.Module):
    """训练部分：语义投影、日期投影、实例投影、融合层和辅助头。"""
    def __init__(self, architecture: Mapping[str, Any], label_sizes: Mapping[str, int]):
        super().__init__()
        bge_dim = int(architecture["bge_dim"])
        hidden_dim = int(architecture["hidden_dim"])
        semantic_dim = int(architecture["semantic_dim"])
        date_dim = int(architecture["date_dim"])
        instance_dim = int(architecture["instance_dim"])
        output_dim = int(architecture["output_dim"])
        dropout = float(architecture["dropout"])
        eps = float(architecture["layer_norm_eps"])

        if semantic_dim + instance_dim != output_dim:
            raise ValueError("首版融合层要求 semantic_dim + instance_dim == output_dim")
        self.architecture = dict(architecture)
        self.label_sizes = {key: max(1, int(value)) for key, value in label_sizes.items()}
        self.semantic_head = ProjectionHead(bge_dim, hidden_dim, semantic_dim, dropout, eps)
        self.date_projection = nn.Sequential(nn.Linear(2, date_dim, bias=True), nn.GELU())
        self.instance_head = ProjectionHead(bge_dim + date_dim, hidden_dim, instance_dim, dropout, eps)
        self.fusion = nn.Sequential(
            nn.LayerNorm(semantic_dim + instance_dim, eps=eps),
            nn.Linear(semantic_dim + instance_dim, output_dim, bias=True),
        )
        self.attribute_heads = nn.ModuleDict(
            {
                field: nn.Linear(semantic_dim, self.label_sizes[field], bias=True)
                for field in ("product", "instrument", "level")
            }
        )
        self.date_head = nn.Linear(instance_dim, 1, bias=True)
        self.source_heads = nn.ModuleDict(
            {
                field: nn.Linear(instance_dim, self.label_sizes[field], bias=True)
                for field in ("archive_root", "subsystem")
            }
        )

    def encode_semantic(self, bge_vectors: torch.Tensor) -> torch.Tensor:
        return self.semantic_head(bge_vectors)

    def encode_instance(self, bge_vectors: torch.Tensor, date_features: torch.Tensor) -> torch.Tensor:
        date_vectors = self.date_projection(date_features)
        return self.instance_head(torch.cat((bge_vectors, date_vectors), dim=-1))

    def encode_final(
        self,
        semantic_bge_vectors: torch.Tensor,
        instance_bge_vectors: torch.Tensor,
        date_features: torch.Tensor,
    ) -> torch.Tensor:
        # 最终向量由模型学习融合，不使用人工设置的语义/实例固定权重。
        semantic = self.encode_semantic(semantic_bge_vectors)
        instance = self.encode_instance(instance_bge_vectors, date_features)
        return F.normalize(self.fusion(torch.cat((semantic, instance), dim=-1)), p=2, dim=-1)

    def attribute_logits(self, semantic_vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        return {field: head(semantic_vectors) for field, head in self.attribute_heads.items()}

    def source_logits(self, instance_vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        return {field: head(instance_vectors) for field, head in self.source_heads.items()}


class FrozenBGEEncoder:
    """统一数据准备和正式推理使用的 BGE 文本编码入口。"""
    def __init__(
        self,
        model_name: str,
        revision: str,
        max_length: int,
        batch_size: int,
        precision: str,
        device: torch.device,
        local_path: str | Path | None = None,
    ):
        if not revision:
            raise ValueError("BGE revision 不能为空")
        self.model_name = model_name
        self.revision = revision
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.precision = precision
        self.device = device
        source = str(Path(local_path).resolve()) if local_path else model_name
        load_options: dict[str, Any] = {"trust_remote_code": False}
        if local_path:
            load_options["local_files_only"] = True
        else:
            load_options["revision"] = revision
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source, **load_options)
            model_options = dict(load_options)
            if device.type == "cuda" and precision == "float16":
                model_options["torch_dtype"] = torch.float16
            self.model = AutoModel.from_pretrained(source, **model_options)
        except Exception as exc:
            location = f"本地路径 {source}" if local_path else f"{model_name}@{revision}"
            raise RuntimeError(f"无法加载 BGE-M3（{location}），请检查配置中的 local_path 或网络访问") from exc
        self.model.requires_grad_(False)
        self.model.eval().to(device)
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if hidden_size != 1024:
            raise ValueError(f"BGE-M3 hidden_size 应为 1024，实际为 {hidden_size}")
        resolved = getattr(self.model.config, "_commit_hash", None)
        if resolved and len(revision) == 40 and resolved != revision:
            raise ValueError(f"BGE commit 不一致：配置={revision}，实际={resolved}")
        self.resolved_commit = resolved or revision

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        outputs: list[np.ndarray] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        report_every = max(1, (total_batches + 9) // 10)
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = [str(text) for text in texts[start : start + self.batch_size]]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            hidden = self.model(**tokens).last_hidden_state[:, 0, :]
            hidden = F.normalize(hidden.float(), p=2, dim=-1)
            outputs.append(hidden.cpu().numpy())
            if batch_index % report_every == 0 or batch_index == total_batches:
                processed = min(start + self.batch_size, len(texts))
                print(f"BGE 编码：{processed}/{len(texts)}", flush=True)
        return np.concatenate(outputs, axis=0)

    def metadata(self) -> dict[str, Any]:
        tokenizer = {
            "class": type(self.tokenizer).__name__,
            "model_max_length": int(self.tokenizer.model_max_length),
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
            "vocab_size": int(self.tokenizer.vocab_size),
            "special_tokens_map": self.tokenizer.special_tokens_map,
        }
        return {
            "model_name": self.model_name,
            "configured_revision": self.revision,
            "resolved_commit": self.resolved_commit,
            "precision": self.precision if self.device.type == "cuda" else "float32",
            "tokenizer": tokenizer,
        }
