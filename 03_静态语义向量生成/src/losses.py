"""详细设计第三节中的全部损失函数。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .model import StaticSemanticModel


def _similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dim=-1)


def triplet_ranking(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    return F.relu(margin + _similarity(anchor, negative) - _similarity(anchor, positive)).mean()


def _masked_multilabel_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0.0
    targets = F.one_hot(labels[valid], num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[valid], targets)


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


def collapse_loss(values: torch.Tensor, target_std: float, eps: float) -> torch.Tensor:
    std = torch.sqrt(values.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


def compute_multitask_loss(
    model: StaticSemanticModel,
    semantic_batch: Mapping[str, torch.Tensor],
    instance_batch: Mapping[str, torch.Tensor],
    final_batch: Mapping[str, torch.Tensor],
    loss_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """一次计算语义、实例、最终表示和防坍塌损失。"""
    margin = float(loss_config["margin"])
    batch_size = semantic_batch["teacher"].shape[0]

    # 语义任务：三元组排序 + 冻结 BGE 相似度蒸馏 + 三类业务属性监督。
    semantic_teacher = F.normalize(semantic_batch["teacher"].float(), p=2, dim=-1)
    semantic_vectors = model.encode_semantic(semantic_teacher.reshape(-1, semantic_teacher.shape[-1])).reshape(batch_size, 3, -1)
    sem_anchor, sem_positive, sem_negative = semantic_vectors.unbind(dim=1)
    teacher_anchor, teacher_positive, teacher_negative = semantic_teacher.unbind(dim=1)
    semantic_rank = triplet_ranking(sem_anchor, sem_positive, sem_negative, margin)
    semantic_distill = 0.5 * (
        (_similarity(sem_anchor, sem_positive) - _similarity(teacher_anchor, teacher_positive)).pow(2)
        + (_similarity(sem_anchor, sem_negative) - _similarity(teacher_anchor, teacher_negative)).pow(2)
    ).mean()
    attribute_logits = model.attribute_logits(semantic_vectors.reshape(-1, semantic_vectors.shape[-1]))
    attribute_loss = sum(
        _masked_multilabel_loss(attribute_logits[field], semantic_batch[field].reshape(-1))
        for field in ("product", "instrument", "level")
    )
    semantic_loss = (
        semantic_rank
        + float(loss_config["semantic_distill_weight"]) * semantic_distill
        + float(loss_config["attribute_weight"]) * attribute_loss
    )

    # 实例任务：结构化实例文本与同目录原始路径对齐，并保留日期和来源。
    dates = instance_batch["date"].float()
    instance_anchor = model.encode_instance(instance_batch["anchor_text"].float(), dates[:, 0, :])
    instance_positive = model.encode_instance(instance_batch["positive_path"].float(), dates[:, 1, :])
    instance_negative = model.encode_instance(instance_batch["negative_path"].float(), dates[:, 2, :])
    instance_align = triplet_ranking(instance_anchor, instance_positive, instance_negative, margin)
    valid_date = dates[:, 0, 1] > 0.5
    if valid_date.any():
        date_loss = F.mse_loss(model.date_head(instance_anchor[valid_date]).squeeze(-1), dates[valid_date, 0, 0])
    else:
        date_loss = instance_anchor.sum() * 0.0
    source_logits = model.source_logits(instance_anchor)
    source_loss = 0.5 * sum(
        _masked_cross_entropy(source_logits[field], instance_batch[field])
        for field in ("archive_root", "subsystem")
    )
    instance_loss = (
        instance_align
        + float(loss_config["date_weight"]) * date_loss
        + float(loss_config["source_weight"]) * source_loss
    )

    # 最终表示任务直接约束正式交付的 128 维向量空间。
    final_semantic = final_batch["semantic"].float()
    final_instance = final_batch["instance"].float()
    final_dates = final_batch["date"].float()
    final_vectors = model.encode_final(
        final_semantic.reshape(-1, final_semantic.shape[-1]),
        final_instance.reshape(-1, final_instance.shape[-1]),
        final_dates.reshape(-1, 2),
    ).reshape(batch_size, 3, -1)
    final_anchor, final_positive, final_negative = final_vectors.unbind(dim=1)
    final_loss = triplet_ranking(final_anchor, final_positive, final_negative, margin)

    semantic_positive_similarity = _similarity(sem_anchor, sem_positive)
    semantic_negative_similarity = _similarity(sem_anchor, sem_negative)
    instance_positive_similarity = _similarity(instance_anchor, instance_positive)
    instance_negative_similarity = _similarity(instance_anchor, instance_negative)
    final_positive_similarity = _similarity(final_anchor, final_positive)
    final_negative_similarity = _similarity(final_anchor, final_negative)

    # 防坍塌约束同时覆盖语义、实例和最终表示三个空间。
    collapse = (
        collapse_loss(semantic_vectors.reshape(-1, semantic_vectors.shape[-1]), float(loss_config["collapse_target_std"]), float(loss_config["collapse_eps"]))
        + collapse_loss(
            torch.cat((instance_anchor, instance_positive, instance_negative), dim=0),
            float(loss_config["collapse_target_std"]),
            float(loss_config["collapse_eps"]),
        )
        + collapse_loss(final_vectors.reshape(-1, final_vectors.shape[-1]), float(loss_config["collapse_target_std"]), float(loss_config["collapse_eps"]))
    ) / 3.0
    total = (
        semantic_loss
        + float(loss_config["instance_task_weight"]) * instance_loss
        + float(loss_config["final_task_weight"]) * final_loss
        + float(loss_config["collapse_weight"]) * collapse
    )
    return {
        "total": total,
        "semantic": semantic_loss,
        "semantic_rank": semantic_rank,
        "semantic_distill": semantic_distill,
        "attribute": attribute_loss,
        "instance": instance_loss,
        "instance_align": instance_align,
        "date": date_loss,
        "source": source_loss,
        "final": final_loss,
        "collapse": collapse,
        "semantic_positive_similarity": semantic_positive_similarity.mean(),
        "semantic_negative_similarity": semantic_negative_similarity.mean(),
        "semantic_triplet_accuracy": (semantic_positive_similarity > semantic_negative_similarity).float().mean(),
        "semantic_mean_margin": (semantic_positive_similarity - semantic_negative_similarity).mean(),
        "instance_positive_similarity": instance_positive_similarity.mean(),
        "instance_negative_similarity": instance_negative_similarity.mean(),
        "instance_triplet_accuracy": (instance_positive_similarity > instance_negative_similarity).float().mean(),
        "instance_mean_margin": (instance_positive_similarity - instance_negative_similarity).mean(),
        "final_positive_similarity": final_positive_similarity.mean(),
        "final_negative_similarity": final_negative_similarity.mean(),
        "final_triplet_accuracy": (final_positive_similarity > final_negative_similarity).float().mean(),
        "final_mean_margin": (final_positive_similarity - final_negative_similarity).mean(),
    }
