from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


LOSS_FIELDS = ("total", "static", "history", "fusion")


def _hierarchical_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    sample_index: torch.Tensor,
    positive_ids: torch.Tensor,
    layers: torch.Tensor,
    batch_size: int,
    include_empty_cutoffs: bool = False,
) -> torch.Tensor:
    """negative mean -> weighted positive mean -> layer mean -> cutoff mean."""
    group_rows = torch.stack((sample_index.long(), positive_ids.long(), layers.long()), dim=1)
    unique_rows, inverse = torch.unique(group_rows, dim=0, return_inverse=True)
    group_count = torch.zeros(len(unique_rows), dtype=values.dtype, device=values.device)
    group_sum = torch.zeros_like(group_count)
    group_weight_sum = torch.zeros_like(group_count)
    active = mask.to(values.dtype)
    group_count.scatter_add_(0, inverse, active)
    group_sum.scatter_add_(0, inverse, values * active)
    group_weight_sum.scatter_add_(0, inverse, weights * active)
    positive_valid = group_count > 0
    positive_loss = group_sum / group_count.clamp_min(1.0)
    positive_weight = group_weight_sum / group_count.clamp_min(1.0)
    layer_key = unique_rows[:, 0] * 4 + unique_rows[:, 2]
    layer_numerator = torch.zeros(batch_size * 4, dtype=values.dtype, device=values.device)
    layer_denominator = torch.zeros_like(layer_numerator)
    layer_numerator.scatter_add_(0, layer_key, positive_loss * positive_weight * positive_valid)
    layer_denominator.scatter_add_(0, layer_key, positive_weight * positive_valid)
    layer_loss = layer_numerator / layer_denominator.clamp_min(1.0)
    layer_valid = (layer_denominator > 0).reshape(batch_size, 4)
    layer_loss = layer_loss.reshape(batch_size, 4)
    cutoff_loss = (layer_loss * layer_valid).sum(dim=1) / layer_valid.sum(dim=1).clamp_min(1)
    cutoff_valid = layer_valid.any(dim=1)
    if include_empty_cutoffs:
        return cutoff_loss.mean()
    if not torch.any(cutoff_valid):
        return values.sum() * 0.0
    return cutoff_loss[cutoff_valid].mean()


def compute_pairwise_losses(
    actor_outputs: Mapping[str, torch.Tensor],
    pair_sample_index: torch.Tensor,
    pair_positive_ids: torch.Tensor,
    pair_positive_layers: torch.Tensor,
    positive_static: torch.Tensor,
    negative_static: torch.Tensor,
    positive_history_persistent: torch.Tensor,
    negative_history_persistent: torch.Tensor,
    persistent_valid: torch.Tensor,
    positive_history_current: torch.Tensor,
    negative_history_current: torch.Tensor,
    current_valid: torch.Tensor,
    pair_weights: torch.Tensor,
    pair_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the three equally weighted ranking losses for flattened pairs."""
    pair_sample_index = pair_sample_index.long()
    if pair_valid is None:
        pair_valid = torch.ones_like(pair_weights, dtype=torch.bool)
    static_query = actor_outputs["static_query"][pair_sample_index]
    history_query = actor_outputs["history_query"][pair_sample_index]
    fusion_weights = actor_outputs["fusion_weights"][pair_sample_index]
    static_positive_score = (static_query * positive_static).sum(dim=-1)
    static_negative_score = (static_query * negative_static).sum(dim=-1)
    static_pair = F.softplus(static_negative_score - static_positive_score)
    persistent_positive_score = (history_query * positive_history_persistent).sum(dim=-1)
    persistent_negative_score = (history_query * negative_history_persistent).sum(dim=-1)
    history_pair = F.softplus(persistent_negative_score - persistent_positive_score)
    current_positive_score = (history_query * positive_history_current).sum(dim=-1)
    current_negative_score = (history_query * negative_history_current).sum(dim=-1)
    fusion_positive = fusion_weights[:, 0] * static_positive_score + fusion_weights[:, 1] * current_positive_score
    fusion_negative = fusion_weights[:, 0] * static_negative_score + fusion_weights[:, 1] * current_negative_score
    fusion_pair = F.softplus(fusion_negative - fusion_positive)
    batch_size = int(actor_outputs["static_query"].shape[0])
    reduction = lambda values, mask, include_empty=False: _hierarchical_mean(
        values,
        pair_weights,
        mask,
        pair_sample_index,
        pair_positive_ids,
        pair_positive_layers,
        batch_size,
        include_empty,
    )
    static_loss = reduction(static_pair, pair_valid)
    history_loss = reduction(history_pair, pair_valid & persistent_valid, True)
    # 融合任务覆盖全部合法pair；历史缺失端已由调用方填0，不能因此丢掉静态监督。
    fusion_loss = reduction(fusion_pair, pair_valid)
    total = static_loss + history_loss + fusion_loss
    return {"total": total, "static": static_loss, "history": history_loss, "fusion": fusion_loss}


def ranking_sums(
    actor_outputs: Mapping[str, torch.Tensor],
    pair_sample_index: torch.Tensor,
    positive_static: torch.Tensor,
    negative_static: torch.Tensor,
    positive_history: torch.Tensor,
    negative_history: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, int]:
    query_static = actor_outputs["static_query"][pair_sample_index.long()]
    query_history = actor_outputs["history_query"][pair_sample_index.long()]
    static_ok = ((query_static * positive_static).sum(-1) > (query_static * negative_static).sum(-1))
    history_ok = ((query_history * positive_history).sum(-1) > (query_history * negative_history).sum(-1)) & valid
    return {
        "static_correct": int(static_ok.sum().item()),
        "static_pairs": int(static_ok.numel()),
        "history_correct": int(history_ok.sum().item()),
        "history_pairs": int(valid.sum().item()),
    }
