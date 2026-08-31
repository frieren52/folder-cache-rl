from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .losses import LOSS_FIELDS


def append_jsonl(path: Path, record: Mapping[str, object], reset: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if reset else "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_epoch_log(
    path: Path,
    epoch: int,
    train: Mapping[str, float],
    learning_rate: float,
    reset: bool = False,
) -> None:
    append_jsonl(
        path,
        {
            "schema_version": "folder-cache-actor-loss-history/v1",
            "epoch": int(epoch),
            "learning_rate": float(learning_rate),
            "train": {name: float(train[name]) for name in LOSS_FIELDS},
            "diagnostics": {
                "train_static_accuracy": float(train["static_accuracy"]),
                "train_history_accuracy": float(train["history_accuracy"]),
                "train_static_weight": float(train["static_weight"]),
            },
        },
        reset=reset,
    )


def _atomic_save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    temporary.replace(path)


def plot_epoch_log(log_path: Path, output_path: Path) -> None:
    history = read_jsonl(log_path)
    if not history:
        raise ValueError(f"损失日志为空：{log_path}")
    epochs = [int(record["epoch"]) for record in history]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, name in zip(axes.reshape(-1)[:4], LOSS_FIELDS):
        axis.plot(epochs, [float(record["train"][name]) for record in history], label="train")
        axis.set_title(name)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    accuracy = axes.reshape(-1)[4]
    accuracy.plot(epochs, [float(record["diagnostics"]["train_static_accuracy"]) for record in history], label="static")
    accuracy.plot(epochs, [float(record["diagnostics"]["train_history_accuracy"]) for record in history], label="history")
    accuracy.set_title("train pair accuracy")
    accuracy.set_xlabel("epoch")
    accuracy.grid(alpha=0.3)
    accuracy.legend()
    weights = axes.reshape(-1)[5]
    weights.plot(epochs, [float(record["diagnostics"]["train_static_weight"]) for record in history], label="train static")
    weights.set_title("mean fusion static weight")
    weights.set_xlabel("epoch")
    weights.grid(alpha=0.3)
    weights.legend()
    figure.suptitle("Folder-cache Actor training-only history", fontsize=15)
    _atomic_save(figure, output_path)


def plot_progress_log(log_path: Path, output_path: Path) -> None:
    history = [record for record in read_jsonl(log_path) if record.get("event") == "train_progress"]
    if not history:
        return
    x_values = [float(record["epoch"]) - 1.0 + float(record["progress"]) for record in history]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, name in zip(axes.reshape(-1)[:4], LOSS_FIELDS):
        axis.plot(x_values, [float(record["loss_running"][name]) for record in history], label="running")
        axis.plot(x_values, [float(record["loss_interval"][name]) for record in history], label="interval", alpha=0.6)
        axis.set_title(name)
        axis.set_xlabel("epoch progress")
        axis.grid(alpha=0.3)
        axis.legend()
    throughput = axes.reshape(-1)[4]
    throughput.plot(x_values, [float(record["interval_pairs_per_second"]) for record in history])
    throughput.set_title("pair throughput")
    throughput.set_xlabel("epoch progress")
    throughput.set_ylabel("pairs / second")
    throughput.grid(alpha=0.3)
    memory = axes.reshape(-1)[5]
    memory.plot(x_values, [float(record.get("gpu_memory_allocated_mb", 0.0)) for record in history], label="allocated")
    memory.plot(x_values, [float(record.get("gpu_memory_reserved_mb", 0.0)) for record in history], label="reserved")
    memory.set_title("GPU memory")
    memory.set_xlabel("epoch progress")
    memory.set_ylabel("MiB")
    memory.grid(alpha=0.3)
    memory.legend()
    figure.suptitle("Folder-cache Actor live training metrics", fontsize=15)
    _atomic_save(figure, output_path)


def plot_evaluation_metrics(report: Mapping[str, object], output_path: Path) -> None:
    horizons = ["5", "10", "30", "300", "3600"]
    candidate_types = ["static_top256", "history_top256", "retrieval_fusion_top256", "current_fusion_top256"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, metric in zip(axes, ("object_recall_micro", "request_coverage_micro", "byte_coverage_micro")):
        for candidate in candidate_types:
            values = [float(report["metrics"].get(candidate, {}).get(horizon, {}).get(metric, 0.0)) for horizon in horizons]
            axis.plot(horizons, values, marker="o", label=candidate)
        axis.set_title(metric)
        axis.set_xlabel("horizon seconds")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Test-set natural candidate metrics", fontsize=15)
    _atomic_save(figure, output_path)
