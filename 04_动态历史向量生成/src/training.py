from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from contextlib import nullcontext
from itertools import islice
from typing import Any, Callable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import PreparedParquetDataset
from .losses import compute_losses, finalize_loss_sums


def make_loader(
    dataset: PreparedParquetDataset,
    batch_size: int,
    workers: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset.configure_batching(batch_size)
    return DataLoader(
        dataset,
        # Dataset 已直接产出整批 Tensor；关闭 DataLoader 的逐样本自动拼批。
        batch_size=None,
        num_workers=workers,
        # 每轮重建 worker，使 dataset.set_epoch(epoch) 的确定性洗牌种子生效。
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def forward_batch(
    model: torch.nn.Module, batch: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return model(
        batch["second_counts"],
        batch["short_counts"],
        batch["medium_counts"],
        batch["long_counts"],
        batch["history_state"],
    )


def _empty_sums(device: torch.device) -> dict[str, torch.Tensor | int]:
    return {
        "access_sum": torch.zeros((), device=device),
        "time_sum": torch.zeros((), device=device),
        "count_sum": torch.zeros((), device=device),
        "samples": 0,
        "positive_samples": 0,
    }


def _update_sums(
    target: dict[str, torch.Tensor | int],
    losses: Mapping[str, torch.Tensor],
    samples: int,
    positives: int,
) -> None:
    target["access_sum"] += losses["access"].detach() * samples
    target["time_sum"] += losses["time"].detach() * positives
    target["count_sum"] += losses["count"].detach() * samples
    target["samples"] += samples
    target["positive_samples"] += positives


def _finalize_sums(
    sums: Mapping[str, torch.Tensor | int], loss_config: Mapping[str, Any]
) -> dict[str, float]:
    host_sums = {
        "access_sum": float(sums["access_sum"].detach().cpu()),
        "time_sum": float(sums["time_sum"].detach().cpu()),
        "count_sum": float(sums["count_sum"].detach().cpu()),
        "samples": int(sums["samples"]),
        "positive_samples": int(sums["positive_samples"]),
    }
    return finalize_loss_sums(host_sums, loss_config)


def _print_progress(
    stage: str,
    epoch: int,
    batch_index: int,
    total_batches: int,
    sample_count: int,
    sums: Mapping[str, torch.Tensor | int],
    interval_sums: Mapping[str, torch.Tensor | int],
    loss_config: Mapping[str, Any],
    started_at: float,
    interval_started_at: float,
    learning_rate: float | None = None,
    callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    processed = int(sums["samples"])
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    interval_elapsed = max(time.perf_counter() - interval_started_at, 1e-9)
    interval_samples = int(interval_sums["samples"])
    samples_per_second = processed / elapsed
    record: dict[str, Any] = {
        "event": f"{stage}_progress",
        "epoch": epoch,
        "batch": batch_index,
        "batches": total_batches,
        "processed_samples": processed,
        "total_samples": sample_count,
        "progress": processed / sample_count,
        "elapsed_seconds": elapsed,
        "eta_seconds": (sample_count - processed) / max(samples_per_second, 1e-9),
        "samples_per_second": samples_per_second,
        "interval_samples_per_second": interval_samples / interval_elapsed,
        "loss_running": _finalize_sums(sums, loss_config),
        "loss_interval": _finalize_sums(interval_sums, loss_config),
    }
    sum_device = sums["access_sum"].device
    if sum_device.type == "cuda":
        record["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        record["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
    if learning_rate is not None:
        record["learning_rate"] = learning_rate
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if callback is not None:
        callback(record)


def _autocast_context(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def run_training_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Mapping[str, Any],
    device: torch.device,
    precision: str,
    sample_count: int,
    epoch: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, float]:
    model.train()
    training = config["training"]
    accumulation = int(training["gradient_accumulation_steps"])
    progress_interval = int(training["progress_interval_batches"])
    progress_seconds = float(training["progress_interval_seconds"])
    micro_batches = len(loader)
    sums = _empty_sums(device)
    interval_sums = _empty_sums(device)
    optimizer.zero_grad(set_to_none=True)
    started_at = time.perf_counter()
    interval_started_at = started_at
    iterator = iter(loader)
    batch_index = 0
    loss_config = config["loss"]
    while True:
        host_group = list(islice(iterator, accumulation))
        if not host_group:
            break
        group_samples = sum(int(batch["y_access"].numel()) for batch in host_group)
        group_positives = sum(
            int((batch["y_access"] == 1).sum().item()) for batch in host_group
        )
        for host_batch in host_group:
            samples = int(host_batch["y_access"].numel())
            positives = int((host_batch["y_access"] == 1).sum().item())
            batch = move_batch(host_batch, device)
            with _autocast_context(device, precision):
                outputs = forward_batch(model, batch)
                losses = compute_losses(
                    outputs,
                    batch["y_access"],
                    batch["y_time"],
                    batch["y_count"],
                    loss_config,
                )
                weighted_total = (
                    float(loss_config["access_weight"])
                    * losses["access"]
                    * (samples / group_samples)
                    + float(loss_config["count_weight"])
                    * losses["count"]
                    * (samples / group_samples)
                )
                if group_positives:
                    weighted_total = (
                        weighted_total
                        + float(loss_config["time_weight"])
                        * losses["time"]
                        * (positives / group_positives)
                    )
            weighted_total.backward()
            _update_sums(sums, losses, samples, positives)
            _update_sums(interval_sums, losses, samples, positives)
            batch_index += 1
            now = time.perf_counter()
            should_report = (
                batch_index % progress_interval == 0
                or now - interval_started_at >= progress_seconds
                or batch_index == micro_batches
            )
            if should_report:
                _print_progress(
                    "train",
                    epoch,
                    batch_index,
                    micro_batches,
                    sample_count,
                    sums,
                    interval_sums,
                    loss_config,
                    started_at,
                    interval_started_at,
                    float(optimizer.param_groups[0]["lr"]),
                    progress_callback,
                )
                interval_sums = _empty_sums(device)
                interval_started_at = time.perf_counter()
        clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    if int(sums["samples"]) != sample_count:
        raise RuntimeError(
            f"训练轮样本数不一致：期望 {sample_count}，实际 {sums['samples']}"
        )
    return _finalize_sums(sums, loss_config)


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: Mapping[str, Any],
    device: torch.device,
    precision: str,
    sample_count: int,
    epoch: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, float]:
    model.eval()
    sums = _empty_sums(device)
    interval_sums = _empty_sums(device)
    progress_interval = int(config["training"]["progress_interval_batches"])
    progress_seconds = float(config["training"]["progress_interval_seconds"])
    total_batches = len(loader)
    started_at = time.perf_counter()
    interval_started_at = started_at
    for batch_index, host_batch in enumerate(loader, start=1):
        samples = int(host_batch["y_access"].numel())
        positives = int((host_batch["y_access"] == 1).sum().item())
        batch = move_batch(host_batch, device)
        with _autocast_context(device, precision):
            outputs = forward_batch(model, batch)
            losses = compute_losses(
                outputs,
                batch["y_access"],
                batch["y_time"],
                batch["y_count"],
                config["loss"],
            )
        _update_sums(sums, losses, samples, positives)
        _update_sums(interval_sums, losses, samples, positives)
        now = time.perf_counter()
        should_report = (
            batch_index % progress_interval == 0
            or now - interval_started_at >= progress_seconds
            or batch_index == total_batches
        )
        if should_report:
            _print_progress(
                "validation",
                epoch,
                batch_index,
                total_batches,
                sample_count,
                sums,
                interval_sums,
                config["loss"],
                started_at,
                interval_started_at,
                callback=progress_callback,
            )
            interval_sums = _empty_sums(device)
            interval_started_at = time.perf_counter()
    if int(sums["samples"]) != sample_count:
        raise RuntimeError(
            f"验证轮样本数不一致：期望 {sample_count}，实际 {sums['samples']}"
        )
    return _finalize_sums(sums, config["loss"])


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    minimum_learning_rate: float,
    base_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    minimum_ratio = minimum_learning_rate / base_learning_rate

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, minimum_ratio)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
