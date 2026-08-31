from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from folder_cache_actor.model import ActorConfig, FolderCacheActor
from torch import nn


@dataclass(frozen=True)
class CriticConfig:
    actor_state_dim: int = 256
    resource_input_dim: int = 10
    resource_hidden_dim: int = 64
    critic_state_dim: int = 256
    candidate_input_dim: int = 274
    candidate_hidden_dim: int = 256
    candidate_embedding_dim: int = 128
    head_hidden_dim: int = 256
    head_bottleneck_dim: int = 64
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CriticConfig":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriticOutput:
    candidate_q_values: torch.Tensor
    stop_q_value: torch.Tensor


class DemandStateEncoder(nn.Module):
    """复制05 Actor状态编码部分，不使用三个策略输出头。"""

    def __init__(self, actor: FolderCacheActor) -> None:
        super().__init__()
        self.actor_config = actor.config
        self.object_projection = copy.deepcopy(actor.object_projection)
        self.encoder = copy.deepcopy(actor.encoder)
        self.pool_query = nn.Parameter(actor.pool_query.detach().clone())
        self.pool_attention = copy.deepcopy(actor.pool_attention)
        self.time_fusion = copy.deepcopy(actor.time_fusion)

    def _pool(self, encoded: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size = encoded.shape[0]
        all_padding = ~valid_mask.any(dim=1)
        if self.pool_attention is None:
            scores = torch.einsum("bsh,h->bs", encoded, self.pool_query) / self.actor_config.hidden_dim**0.5
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
        object_features: torch.Tensor,
        object_valid_mask: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        if object_features.ndim != 3 or object_features.shape[1:] != (256, self.actor_config.input_dim):
            raise ValueError(f"object_features必须为[B,256,{self.actor_config.input_dim}]")
        if object_valid_mask.shape != object_features.shape[:2] or object_valid_mask.dtype != torch.bool:
            raise ValueError("object_valid_mask形状或类型错误")
        if time_features.shape != (object_features.shape[0], 2):
            raise ValueError("time_features必须为[B,2]")
        all_padding = ~object_valid_mask.any(dim=1)
        safe_mask = object_valid_mask.clone()
        if torch.any(all_padding):
            safe_mask[all_padding, 0] = True
        projected = self.object_projection(object_features)
        encoded = self.encoder(projected, src_key_padding_mask=~safe_mask)
        pooled = self._pool(encoded, safe_mask)
        state = self.time_fusion(torch.cat((pooled, time_features), dim=-1))
        return torch.where(all_padding[:, None], torch.zeros_like(state), state)


class FolderCacheCritic(nn.Module):
    def __init__(self, actor: FolderCacheActor, config: CriticConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config if isinstance(config, CriticConfig) else CriticConfig.from_mapping(config)
        cfg = self.config
        if actor.config.hidden_dim != cfg.actor_state_dim:
            raise ValueError("Actor隐藏维度与Critic状态合同不一致")
        self.demand_encoder = DemandStateEncoder(actor)
        self.resource_encoder = nn.Sequential(
            nn.Linear(cfg.resource_input_dim, cfg.resource_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.resource_hidden_dim, eps=cfg.layer_norm_eps),
            nn.Linear(cfg.resource_hidden_dim, cfg.resource_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.resource_hidden_dim, eps=cfg.layer_norm_eps),
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(cfg.actor_state_dim + cfg.resource_hidden_dim, cfg.critic_state_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.critic_state_dim, eps=cfg.layer_norm_eps),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(cfg.candidate_input_dim, cfg.candidate_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.candidate_hidden_dim, eps=cfg.layer_norm_eps),
            nn.Linear(cfg.candidate_hidden_dim, cfg.candidate_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.candidate_embedding_dim, eps=cfg.layer_norm_eps),
        )
        batch_dim = cfg.candidate_embedding_dim * 2
        self.candidate_head = nn.Sequential(
            nn.Linear(cfg.critic_state_dim + cfg.candidate_embedding_dim + batch_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.head_hidden_dim, eps=cfg.layer_norm_eps),
            nn.Linear(cfg.head_hidden_dim, cfg.head_bottleneck_dim),
            nn.GELU(),
            nn.Linear(cfg.head_bottleneck_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(cfg.critic_state_dim + batch_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.head_hidden_dim, eps=cfg.layer_norm_eps),
            nn.Linear(cfg.head_hidden_dim, cfg.head_bottleneck_dim),
            nn.GELU(),
            nn.Linear(cfg.head_bottleneck_dim, 1),
        )

    @staticmethod
    def _selected_summary(candidate_embeddings: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        selected = selected_mask.unsqueeze(-1)
        counts = selected.sum(dim=1).clamp_min(1)
        mean = (candidate_embeddings * selected).sum(dim=1) / counts
        negative = torch.finfo(candidate_embeddings.dtype).min
        maximum = candidate_embeddings.masked_fill(~selected, negative).max(dim=1).values
        empty = ~selected_mask.any(dim=1)
        mean = torch.where(empty[:, None], torch.zeros_like(mean), mean)
        maximum = torch.where(empty[:, None], torch.zeros_like(maximum), maximum)
        return torch.cat((mean, maximum), dim=-1)

    def forward(
        self,
        object_features: torch.Tensor,
        object_valid_mask: torch.Tensor,
        time_features: torch.Tensor,
        resource_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        selected_candidate_mask: torch.Tensor,
    ) -> CriticOutput:
        batch_size, candidate_count, feature_dim = candidate_features.shape
        if feature_dim != self.config.candidate_input_dim:
            raise ValueError(f"candidate_features末维必须为{self.config.candidate_input_dim}")
        if resource_features.shape != (batch_size, self.config.resource_input_dim):
            raise ValueError(f"resource_features必须为[B,{self.config.resource_input_dim}]")
        for name, value in (("candidate_valid_mask", candidate_valid_mask), ("selected_candidate_mask", selected_candidate_mask)):
            if value.shape != (batch_size, candidate_count) or value.dtype != torch.bool:
                raise ValueError(f"{name}形状或类型错误")
        demand = self.demand_encoder(object_features, object_valid_mask, time_features)
        resource = self.resource_encoder(resource_features)
        state = self.state_fusion(torch.cat((demand, resource), dim=-1))
        candidates = self.candidate_encoder(candidate_features)
        candidates = candidates.masked_fill(~candidate_valid_mask.unsqueeze(-1), 0.0)
        batch_summary = self._selected_summary(candidates, selected_candidate_mask & candidate_valid_mask)
        expanded_state = state[:, None, :].expand(-1, candidate_count, -1)
        expanded_batch = batch_summary[:, None, :].expand(-1, candidate_count, -1)
        candidate_q = self.candidate_head(torch.cat((expanded_state, candidates, expanded_batch), dim=-1)).squeeze(-1)
        stop_q = self.stop_head(torch.cat((state, batch_summary), dim=-1))
        if not torch.isfinite(candidate_q).all() or not torch.isfinite(stop_q).all():
            raise FloatingPointError("Critic输出包含非有限值")
        return CriticOutput(candidate_q, stop_q)


class TwinCritic(nn.Module):
    """两个独立在线Critic及各自Target。"""

    def __init__(
        self,
        actor: FolderCacheActor,
        critic_config: CriticConfig | Mapping[str, Any],
        q1_seed: int = 2026,
        q2_seed: int = 2027,
    ) -> None:
        super().__init__()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(q1_seed))
            self.q1 = FolderCacheCritic(actor, critic_config)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(q2_seed))
            self.q2 = FolderCacheCritic(actor, critic_config)
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()
        for parameter in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            parameter.requires_grad_(False)

    def online(self, *args: torch.Tensor) -> tuple[CriticOutput, CriticOutput]:
        return self.q1(*args), self.q2(*args)

    @torch.no_grad()
    def target(self, *args: torch.Tensor) -> tuple[CriticOutput, CriticOutput]:
        self.q1_target.eval()
        self.q2_target.eval()
        return self.q1_target(*args), self.q2_target(*args)

    @torch.no_grad()
    def soft_update(self, tau: float) -> None:
        value = float(tau)
        if not 0 < value <= 1:
            raise ValueError("tau必须位于(0,1]")
        for online, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for source, destination in zip(online.parameters(), target.parameters()):
                destination.lerp_(source, value)
            for source, destination in zip(online.buffers(), target.buffers()):
                destination.copy_(source)


def min_q(first: CriticOutput, second: CriticOutput) -> CriticOutput:
    return CriticOutput(
        torch.minimum(first.candidate_q_values, second.candidate_q_values),
        torch.minimum(first.stop_q_value, second.stop_q_value),
    )


def make_actor_from_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> FolderCacheActor:
    actor = FolderCacheActor(ActorConfig.from_mapping(checkpoint["model_config"]))
    actor.load_state_dict(checkpoint["model_state"], strict=True)
    return actor

