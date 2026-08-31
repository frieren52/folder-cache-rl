from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from .errors import ConfigError
from .utils import sha256_json


SHANGHAI = ZoneInfo("Asia/Shanghai")


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"无法读取配置 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"配置根节点必须是映射：{path}")
    config = deepcopy(value)
    validate_config(config)
    return config


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"配置缺少 {section}.{key}")
    return mapping[key]


def parse_shanghai_time(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} 不是合法 ISO 时间：{value!r}") from exc
    if parsed.tzinfo is None:
        raise ConfigError(f"{field_name} 必须包含时区：{value!r}")
    return parsed.astimezone(SHANGHAI)


def validate_config(config: dict[str, Any]) -> None:
    for section in (
        "history",
        "model",
        "target",
        "sampling",
        "storage",
        "split",
        "loss",
        "training",
        "evaluation",
    ):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"配置缺少映射段：{section}")

    history = config["history"]
    parse_shanghai_time(_require(history, "sample_start", "history"), "history.sample_start")
    stride = int(_require(history, "snapshot_stride_seconds", "history"))
    if stride <= 0:
        raise ConfigError("history.snapshot_stride_seconds 必须大于 0")
    scales = _require(history, "scales", "history")
    if not isinstance(scales, list) or not scales:
        raise ConfigError("history.scales 必须是非空列表")
    names: set[str] = set()
    for index, scale in enumerate(scales):
        if not isinstance(scale, dict):
            raise ConfigError(f"history.scales[{index}] 必须是映射")
        name = str(_require(scale, "name", f"history.scales[{index}]"))
        window = int(_require(scale, "window_seconds", f"history.scales[{index}]"))
        bucket = int(_require(scale, "bucket_seconds", f"history.scales[{index}]"))
        count = int(_require(scale, "bucket_count", f"history.scales[{index}]"))
        if name in names or window <= 0 or bucket <= 0 or count <= 0:
            raise ConfigError(f"history.scales[{index}] 参数非法或名称重复")
        if bucket * count != window:
            raise ConfigError(f"history.scales[{index}] 的 bucket_count × bucket_seconds 必须等于 window_seconds")
        names.add(name)
    if names != {"second", "short", "medium", "long"}:
        raise ConfigError("history.scales 必须且只能包含 second、short、medium、long")

    boundaries = [int(item) for item in config["target"].get("time_boundaries_seconds", [])]
    horizon = int(_require(config["target"], "horizon_seconds", "target"))
    if len(boundaries) != 10 or boundaries[0] != 0 or boundaries[-1] != horizon:
        raise ConfigError(
            "target.time_boundaries_seconds 必须含 10 个边界，形成 9 个访问档，"
            "并覆盖 0 到 horizon_seconds"
        )
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ConfigError("target.time_boundaries_seconds 必须严格递增")

    sampling = config["sampling"]
    for key in (
        "high_history_per_snapshot",
        "medium_history_per_snapshot",
        "low_history_per_snapshot",
        "single_history_per_snapshot",
        "no_history_per_snapshot",
    ):
        if int(_require(sampling, key, "sampling")) < 0:
            raise ConfigError(f"sampling.{key} 不能为负数")
    low_q = float(_require(sampling, "history_low_quantile", "sampling"))
    high_q = float(_require(sampling, "history_high_quantile", "sampling"))
    if not 0.0 < low_q < high_q < 1.0:
        raise ConfigError("sampling 的历史分位点必须满足 0 < low < high < 1")

    split = config["split"]
    train_ratio = float(_require(split, "train_ratio", "split"))
    validation_ratio = float(_require(split, "validation_ratio", "split"))
    if abs(train_ratio + validation_ratio - 1.0) > 1e-9:
        raise ConfigError("split.train_ratio + split.validation_ratio 必须等于 1")
    if split.get("ratio_basis") != "usable_snapshot_count":
        raise ConfigError("首版 split.ratio_basis 必须为 usable_snapshot_count")

    model = config["model"]
    if int(model.get("state_input_dim", -1)) != 4:
        raise ConfigError("首版 model.state_input_dim 必须为 4")
    if int(model.get("vector_dim", 0)) <= 0:
        raise ConfigError("model.vector_dim 必须大于 0")

    storage = config["storage"]
    shard_rows = int(_require(storage, "max_rows_per_shard", "storage"))
    group_rows = int(_require(storage, "max_rows_per_row_group", "storage"))
    if shard_rows <= 0 or group_rows <= 0 or shard_rows % group_rows != 0:
        raise ConfigError("storage 行数必须为正，且分片行数必须能被 Row Group 行数整除")

    training = config["training"]
    for key in (
        "dataloader_workers",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "progress_interval_batches",
        "progress_interval_seconds",
    ):
        value = int(_require(training, key, "training"))
        invalid = value < 0 if key == "dataloader_workers" else value <= 0
        if invalid:
            raise ConfigError(f"training.{key} 参数非法：{value}")

    medium = next(item for item in scales if item["name"] == "medium")
    baseline_window = int(_require(config["evaluation"], "baseline_window_seconds", "evaluation"))
    medium_window = int(medium["window_seconds"])
    medium_bucket = int(medium["bucket_seconds"])
    if (
        baseline_window <= 0
        or baseline_window > medium_window
        or baseline_window % medium_bucket != 0
    ):
        raise ConfigError(
            "evaluation.baseline_window_seconds 必须不超过 medium 窗口且能被其桶宽整除"
        )


def config_sha256(config: dict[str, Any]) -> str:
    return sha256_json(config)


def data_config_sha256(config: dict[str, Any]) -> str:
    return sha256_json(
        {
            section: config[section]
            for section in ("history", "target", "sampling", "storage", "split")
        }
    )
