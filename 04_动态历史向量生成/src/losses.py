from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional as F


def compute_losses(
    outputs: Mapping[str, torch.Tensor],
    y_access: torch.Tensor,
    y_time: torch.Tensor,
    y_count: torch.Tensor,
    loss_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    access_targets = y_access.to(dtype=outputs["access_logits"].dtype)
    count_targets = torch.log1p(y_count.to(dtype=outputs["predicted_log_counts"].dtype))
    access_loss = F.binary_cross_entropy_with_logits(
        outputs["access_logits"], access_targets
    )
    # -1 是“未来窗口无访问”的正式标签。使用 ignore_index 可避免每批次
    # torch.any(...)->CPU 的同步；全为负样本时分母取 1，结果仍严格为 0。
    positive_count = (y_time != -1).sum()
    time_sum = F.cross_entropy(
        outputs["time_logits"],
        y_time.long(),
        ignore_index=-1,
        reduction="sum",
    )
    time_loss = time_sum / positive_count.clamp_min(1)
    count_loss = F.huber_loss(
        outputs["predicted_log_counts"],
        count_targets,
        delta=float(loss_config["huber_delta"]),
    )
    total = (
        float(loss_config["access_weight"]) * access_loss
        + float(loss_config["time_weight"]) * time_loss
        + float(loss_config["count_weight"]) * count_loss
    )
    return {
        "total": total,
        "access": access_loss,
        "time": time_loss,
        "count": count_loss,
    }


def loss_sums(
    outputs: Mapping[str, torch.Tensor],
    y_access: torch.Tensor,
    y_time: torch.Tensor,
    y_count: torch.Tensor,
    loss_config: Mapping[str, Any],
) -> dict[str, float | int]:
    access_targets = y_access.to(dtype=outputs["access_logits"].dtype)
    positive = y_access == 1
    access_sum = F.binary_cross_entropy_with_logits(
        outputs["access_logits"], access_targets, reduction="sum"
    )
    if torch.any(positive):
        time_sum = F.cross_entropy(
            outputs["time_logits"][positive], y_time[positive].long(), reduction="sum"
        )
    else:
        time_sum = outputs["time_logits"].sum() * 0.0
    count_sum = F.huber_loss(
        outputs["predicted_log_counts"],
        torch.log1p(y_count.to(dtype=outputs["predicted_log_counts"].dtype)),
        delta=float(loss_config["huber_delta"]),
        reduction="sum",
    )
    return {
        "access_sum": float(access_sum.detach().cpu()),
        "time_sum": float(time_sum.detach().cpu()),
        "count_sum": float(count_sum.detach().cpu()),
        "samples": int(y_access.numel()),
        "positive_samples": int(positive.sum().item()),
    }


def finalize_loss_sums(
    sums: Mapping[str, float | int], loss_config: Mapping[str, Any]
) -> dict[str, float]:
    samples = int(sums["samples"])
    positives = int(sums["positive_samples"])
    if samples <= 0:
        raise ValueError("损失统计没有样本")
    access = float(sums["access_sum"]) / samples
    time = float(sums["time_sum"]) / positives if positives else 0.0
    count = float(sums["count_sum"]) / samples
    total = (
        float(loss_config["access_weight"]) * access
        + float(loss_config["time_weight"]) * time
        + float(loss_config["count_weight"]) * count
    )
    return {"total": total, "access": access, "time": time, "count": count}
