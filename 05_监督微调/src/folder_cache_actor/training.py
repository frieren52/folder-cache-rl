from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch

from .losses import LOSS_FIELDS, compute_pairwise_losses, ranking_sums


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _empty_sums() -> dict[str, float]:
    return {**{name: 0.0 for name in LOSS_FIELDS}, "loss_weight": 0.0, "pairs": 0.0, "static_correct": 0.0, "static_pairs": 0.0, "history_correct": 0.0, "history_pairs": 0.0, "static_weight_sum": 0.0, "states": 0.0}


def _finalize(sums: Mapping[str, float]) -> dict[str, float]:
    loss_weight = max(1.0, sums["loss_weight"])
    return {
        **{name: sums[name] / loss_weight for name in LOSS_FIELDS},
        "static_accuracy": sums["static_correct"] / max(1.0, sums["static_pairs"]),
        "history_accuracy": sums["history_correct"] / max(1.0, sums["history_pairs"]),
        "history_valid_ratio": sums["history_pairs"] / max(1.0, sums["static_pairs"]),
        "static_weight": sums["static_weight_sum"] / max(1.0, sums["states"]),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    expected_batches: int,
    progress_interval_batches: int,
    progress_interval_seconds: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = _empty_sums()
    interval = _empty_sums()
    started = time.monotonic()
    interval_started = started
    last_report = started
    batches = 0
    for batches, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        pair_count = int(batch["pair_sample_index"].numel())
        if pair_count <= 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(batch["context_features"], batch["context_valid_mask"], batch["time_features"])
            losses = compute_pairwise_losses(
                outputs,
                batch["pair_sample_index"],
                batch["pair_positive_ids"],
                batch["pair_positive_layers"],
                batch["positive_static"],
                batch["negative_static"],
                batch["positive_history_persistent"],
                batch["negative_history_persistent"],
                batch["persistent_valid"],
                batch["positive_history_current"],
                batch["negative_history_current"],
                batch["current_valid"],
                batch["pair_weights"],
            )
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError(f"epoch={epoch}, batch={batches}出现非有限损失")
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm), error_if_nonfinite=True)
                optimizer.step()
        ranking = ranking_sums(
            outputs,
            batch["pair_sample_index"],
            batch["positive_static"],
            batch["negative_static"],
            batch["positive_history_persistent"],
            batch["negative_history_persistent"],
            batch["persistent_valid"],
        )
        state_count = int(outputs["fusion_weights"].shape[0])
        for target in (totals, interval):
            for name in LOSS_FIELDS:
                target[name] += float(losses[name].detach().cpu()) * state_count
            target["loss_weight"] += state_count
            target["pairs"] += pair_count
            for name, value in ranking.items():
                target[name] += value
            target["static_weight_sum"] += float(outputs["fusion_weights"][:, 0].sum().detach().cpu())
            target["states"] += state_count
        now = time.monotonic()
        should_report = training and (
            batches % int(progress_interval_batches) == 0
            or now - last_report >= int(progress_interval_seconds)
        )
        if should_report:
            elapsed = max(now - interval_started, 1e-9)
            record: dict[str, Any] = {
                "event": "train_progress",
                "epoch": epoch,
                "batch": batches,
                "expected_batches": expected_batches,
                "progress": min(1.0, batches / max(1, expected_batches)),
                "loss_interval": {name: interval[name] / max(1.0, interval["loss_weight"]) for name in LOSS_FIELDS},
                "loss_running": {name: totals[name] / max(1.0, totals["loss_weight"]) for name in LOSS_FIELDS},
                "interval_pairs_per_second": interval["pairs"] / elapsed,
                "elapsed_seconds": now - started,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if device.type == "cuda":
                record["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
                record["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if progress_callback is not None:
                progress_callback(record)
            interval = _empty_sums()
            interval_started = now
            last_report = now
    if training and interval["pairs"] > 0:
        now = time.monotonic()
        elapsed = max(now - interval_started, 1e-9)
        record = {
            "event": "train_progress",
            "epoch": epoch,
            "batch": batches,
            "expected_batches": expected_batches,
            "progress": min(1.0, batches / max(1, expected_batches)),
            "loss_interval": {name: interval[name] / max(1.0, interval["loss_weight"]) for name in LOSS_FIELDS},
            "loss_running": {name: totals[name] / max(1.0, totals["loss_weight"]) for name in LOSS_FIELDS},
            "interval_pairs_per_second": interval["pairs"] / elapsed,
            "elapsed_seconds": now - started,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if device.type == "cuda":
            record["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
            record["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if progress_callback is not None:
            progress_callback(record)
    if totals["pairs"] <= 0:
        raise ValueError("当前数据划分没有有效正负样本pair")
    return _finalize(totals)
