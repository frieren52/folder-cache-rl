from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def append_jsonl(path: Path, record: Mapping[str, object], reset: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if reset else "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def plot_critic_history(path: Path, output: Path) -> None:
    rows = read_jsonl(path)
    if not rows:
        return
    updates = [int(row["update"]) for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fields = ("loss", "q_mean", "target_mean", "td_abs_mean", "calibration_mae", "learning_rate")
    for axis, field in zip(axes.reshape(-1), fields):
        axis.plot(updates, [float(row.get(field, float("nan"))) for row in rows])
        axis.set_title(field)
        axis.set_xlabel("update")
        axis.grid(alpha=0.3)
    figure.suptitle("Folder-cache twin Critic training", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    temporary.replace(output)


def plot_actor_history(path: Path, output: Path) -> None:
    rows = read_jsonl(path)
    if not rows:
        return
    updates = [int(row["update"]) for row in rows]
    fields = (
        "total",
        "advantage_total",
        "advantage_static",
        "advantage_history",
        "advantage_fusion",
        "anchor_total",
        "calibration_advantage_loss",
        "learning_rate",
    )
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for axis, field in zip(axes.reshape(-1), fields):
        axis.plot(updates, [float(row.get(field, float("nan"))) for row in rows])
        axis.set_title(field)
        axis.set_xlabel("update")
        axis.grid(alpha=0.3)
    figure.suptitle("Folder-cache Actor RL training", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    temporary.replace(output)
