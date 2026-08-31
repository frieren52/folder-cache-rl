"""Loss history persistence and non-interactive chart rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


LOSS_FIELDS = ("total", "access", "time", "count")


def save_loss_log(
    output_path: Path,
    epoch: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    learning_rate: float,
    reset: bool = False,
) -> None:
    record = {
        "schema_version": "dynamic-history-loss-history/v1",
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "train": {name: float(train_metrics[name]) for name in LOSS_FIELDS},
        "validation": {name: float(validation_metrics[name]) for name in LOSS_FIELDS},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w" if reset else "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_loss_log(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def plot_loss_log(log_path: Path, output_path: Path) -> None:
    history = read_loss_log(log_path)
    if not history:
        raise ValueError(f"损失日志为空：{log_path}")
    epochs = [int(record["epoch"]) for record in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, field in zip(axes.reshape(-1), LOSS_FIELDS):
        axis.plot(
            epochs,
            [float(record["train"][field]) for record in history],
            label="train",
        )
        axis.plot(
            epochs,
            [float(record["validation"][field]) for record in history],
            label="validation",
        )
        axis.set_title(field)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Dynamic history encoder loss history", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_progress_log(output_path: Path, record: Mapping[str, object], reset: bool = False) -> None:
    value = {
        "schema_version": "dynamic-history-progress/v1",
        **record,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w" if reset else "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def plot_progress_log(log_path: Path, output_path: Path) -> None:
    history = [
        record
        for record in read_loss_log(log_path)
        if record.get("event") == "train_progress"
    ]
    if not history:
        raise ValueError(f"训练进度日志为空：{log_path}")
    x_values = [
        float(record["epoch"]) - 1.0 + float(record["progress"])
        for record in history
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, field in zip(axes.reshape(-1)[:4], LOSS_FIELDS):
        axis.plot(
            x_values,
            [float(record["loss_interval"][field]) for record in history],
            label="interval",
            alpha=0.65,
        )
        axis.plot(
            x_values,
            [float(record["loss_running"][field]) for record in history],
            label="epoch running",
        )
        axis.set_title(field)
        axis.set_xlabel("epoch progress")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.3)
        axis.legend()
    throughput_axis = axes.reshape(-1)[4]
    throughput_axis.plot(
        x_values,
        [float(record["interval_samples_per_second"]) for record in history],
    )
    throughput_axis.set_title("throughput")
    throughput_axis.set_xlabel("epoch progress")
    throughput_axis.set_ylabel("samples / second")
    throughput_axis.grid(alpha=0.3)
    memory_axis = axes.reshape(-1)[5]
    if any("gpu_memory_allocated_mb" in record for record in history):
        memory_axis.plot(
            x_values,
            [float(record.get("gpu_memory_allocated_mb", 0.0)) for record in history],
            label="allocated",
        )
        memory_axis.plot(
            x_values,
            [float(record.get("gpu_memory_reserved_mb", 0.0)) for record in history],
            label="reserved",
        )
        memory_axis.legend()
    memory_axis.set_title("GPU memory")
    memory_axis.set_xlabel("epoch progress")
    memory_axis.set_ylabel("MiB")
    memory_axis.grid(alpha=0.3)
    figure.suptitle("Dynamic history encoder live training metrics", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    temporary.replace(output_path)
