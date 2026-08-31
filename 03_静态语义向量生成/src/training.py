"""训练和评估共用的单轮执行逻辑。"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .data import PreparedData, move_batch
from .losses import compute_multitask_loss
from .model import StaticSemanticModel
from .sampling import TripletSets, iter_triplet_batches


def run_epoch(
    model: StaticSemanticModel,
    data: PreparedData,
    triplets: TripletSets,
    config: Mapping[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    shuffle_seed: int | None = None,
) -> dict[str, float]:
    """执行一轮三任务训练；optimizer 为空时只评估、不更新参数。"""
    is_training = optimizer is not None
    model.train(is_training)
    metric_sums: dict[str, float] = {}
    sample_count = 0

    batches = iter_triplet_batches(
        triplets,
        batch_size=int(config["train"]["batch_size"]),
        seed=shuffle_seed,
    )
    for semantic_items, instance_items, final_items in batches:
        # 每次更新各取等量的语义、实例和最终表示样本，对应方案中的 1:1:1。
        semantic_batch = move_batch(data.semantic_batch(semantic_items), device)
        instance_batch = move_batch(data.instance_batch(instance_items), device)
        final_batch = move_batch(data.final_batch(final_items), device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            losses = compute_multitask_loss(model, semantic_batch, instance_batch, final_batch, config["loss"])
            if is_training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["gradient_clip_norm"]))
                optimizer.step()

        batch_size = len(semantic_items)
        for name, value in losses.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("当前切分没有可执行的三任务样本")
    return {name: value / sample_count for name, value in metric_sums.items()}
