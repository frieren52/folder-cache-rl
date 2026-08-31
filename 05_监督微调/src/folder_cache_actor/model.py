from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActorConfig:
    input_dim: int = 259
    vector_dim: int = 128
    hidden_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ffn_dim: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    pooling_mode: str = "dot_product"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActorConfig":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_normalize(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


class FolderCacheActor(nn.Module):
    """Permutation-invariant set actor used by supervised and RL stages."""

    def __init__(self, config: ActorConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config if isinstance(config, ActorConfig) else ActorConfig.from_mapping(config)
        cfg = self.config
        if cfg.pooling_mode not in {"dot_product", "mha"}:
            raise ValueError(f"不支持的pooling_mode：{cfg.pooling_mode}")
        self.object_projection = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_eps),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers, norm=None)
        self.pool_query = nn.Parameter(torch.empty(cfg.hidden_dim))
        nn.init.normal_(self.pool_query, mean=0.0, std=cfg.hidden_dim**-0.5)
        self.pool_attention = (
            nn.MultiheadAttention(
                cfg.hidden_dim,
                cfg.transformer_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            if cfg.pooling_mode == "mha"
            else None
        )
        self.time_fusion = nn.Linear(cfg.hidden_dim + 2, cfg.hidden_dim)
        self.static_head = nn.Linear(cfg.hidden_dim, cfg.vector_dim)
        self.history_head = nn.Linear(cfg.hidden_dim, cfg.vector_dim)
        self.fusion_head = nn.Linear(cfg.hidden_dim, 2)

    def _pool(self, encoded: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size = encoded.shape[0]
        all_padding = ~valid_mask.any(dim=1)
        if self.pool_attention is None:
            scores = torch.einsum("bsh,h->bs", encoded, self.pool_query) / math.sqrt(self.config.hidden_dim)
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
            scores = torch.where(all_padding[:, None], torch.zeros_like(scores), scores)
            weights = torch.softmax(scores, dim=1).masked_fill(~valid_mask, 0.0)
            pooled = torch.einsum("bs,bsh->bh", weights, encoded)
        else:
            query = self.pool_query.view(1, 1, -1).expand(batch_size, -1, -1)
            pooled, _ = self.pool_attention(
                query,
                encoded,
                encoded,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            pooled = pooled[:, 0]
        return torch.where(all_padding[:, None], torch.zeros_like(pooled), pooled)

    def forward(
        self,
        context_features: torch.Tensor,
        context_valid_mask: torch.Tensor,
        time_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context_features.ndim != 3 or context_features.shape[-1] != self.config.input_dim:
            raise ValueError(f"context_features必须为[B,S,{self.config.input_dim}]")
        if context_valid_mask.shape != context_features.shape[:2] or context_valid_mask.dtype != torch.bool:
            raise ValueError("context_valid_mask形状或类型错误")
        if time_features.shape != (context_features.shape[0], 2):
            raise ValueError("time_features必须为[B,2]")
        all_padding = ~context_valid_mask.any(dim=1)
        safe_mask = context_valid_mask.clone()
        if torch.any(all_padding):
            safe_mask[all_padding, 0] = True
        projected = self.object_projection(context_features)
        encoded = self.encoder(projected, src_key_padding_mask=~safe_mask)
        pooled = self._pool(encoded, safe_mask)
        pooled = torch.where(all_padding[:, None], torch.zeros_like(pooled), pooled)
        state = self.time_fusion(torch.cat((pooled, time_features), dim=-1))
        static_query = safe_normalize(self.static_head(state))
        history_query = safe_normalize(self.history_head(state))
        fusion_weights = torch.softmax(self.fusion_head(state), dim=-1)
        outputs = {
            "static_query": static_query,
            "history_query": history_query,
            "fusion_weights": fusion_weights,
        }
        if not all(torch.isfinite(value).all() for value in outputs.values()):
            raise FloatingPointError("Actor输出包含非有限值")
        return outputs


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

