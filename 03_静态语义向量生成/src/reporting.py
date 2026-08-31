"""训练损失日志与曲线绘制。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


# 这里只记录损失项；相似度和准确率继续由训练控制台与评估报告负责。
LOSS_FIELDS = (
    "total",
    "semantic",
    "semantic_rank",
    "semantic_distill",
    "attribute",
    "instance",
    "instance_align",
    "date",
    "source",
    "final",
    "collapse",
)


def save_loss_log(
    output_path: Path,
    epoch: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    learning_rate: float,
    reset: bool = False,
) -> None:
    """把一轮训练集、验证集的全部损失追加为一行 JSON。"""
    record = {
        "schema_version": "static-semantic-loss-history/v1",
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "train": {name: float(train_metrics[name]) for name in LOSS_FIELDS},
        "validation": {name: float(validation_metrics[name]) for name in LOSS_FIELDS},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if reset else "a"
    with output_path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def plot_loss_log(log_path: Path, output_path: Path) -> None:
    """从损失日志生成逐损失训练/验证曲线。"""
    with log_path.open("r", encoding="utf-8") as handle:
        history = [json.loads(line) for line in handle if line.strip()]
    if not history:
        raise ValueError(f"损失日志为空：{log_path}")

    epochs = [record["epoch"] for record in history]
    columns = 3
    rows = math.ceil(len(LOSS_FIELDS) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.2 * rows), constrained_layout=True)
    flat_axes = axes.reshape(-1)

    for axis, name in zip(flat_axes, LOSS_FIELDS):
        axis.plot(epochs, [record["train"][name] for record in history], label="train")
        axis.plot(epochs, [record["validation"][name] for record in history], label="validation")
        axis.set_title(name)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    for axis in flat_axes[len(LOSS_FIELDS) :]:
        axis.set_visible(False)

    figure.suptitle("Static semantic encoder loss history", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
