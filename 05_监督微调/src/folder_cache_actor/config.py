from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .utils import parse_time, sha256_json


def _require(mapping: Mapping[str, Any], name: str, location: str) -> Any:
    if name not in mapping:
        raise ConfigError(f"{location} 缺少配置项 {name}")
    return mapping[name]


def resolve_path(module_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (module_root / path).resolve()


def validate_config(config: Mapping[str, Any]) -> None:
    for section in ("paths", "time", "context", "history_index", "retrieval", "sampling", "model", "training", "storage"):
        value = _require(config, section, "根配置")
        if not isinstance(value, Mapping):
            raise ConfigError(f"{section} 必须是映射")
    time = config["time"]
    obsolete_time = {"validation_start", "validation_end"} & set(time)
    if obsolete_time:
        raise ConfigError(f"05不设置验证集，请删除旧配置：{sorted(obsolete_time)}")
    if int(time["decision_interval_seconds"]) != 10:
        raise ConfigError("首版 decision_interval_seconds 必须为10")
    if int(time["target_horizon_seconds"]) != 3600:
        raise ConfigError("首版 target_horizon_seconds 必须为3600")
    boundaries = [parse_time(str(time[name]), f"time.{name}") for name in ("warmup_start", "train_start", "test_start", "test_end")]
    if boundaries != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ConfigError("预热、训练和测试时间边界必须严格递增")
    context = config["context"]
    if int(context["max_objects"]) != 256 or int(context["recent_objects"]) != 128:
        raise ConfigError("首版上下文必须为最近128个并补齐至256个")
    model = config["model"]
    if int(model["input_dim"]) != 259 or int(model["vector_dim"]) != 128:
        raise ConfigError("Actor输入维度必须为259，查询维度必须为128")
    if int(model["transformer_layers"]) != 4:
        raise ConfigError("首版Actor固定4层Transformer")
    if str(model["pooling_mode"]) not in {"dot_product", "mha"}:
        raise ConfigError("pooling_mode只支持dot_product或mha")
    ratios = [float(value) for value in config["sampling"]["negative_ratios"]]
    if len(ratios) != 4 or abs(sum(ratios) - 1.0) > 1e-9 or any(value < 0 for value in ratios):
        raise ConfigError("negative_ratios必须包含4个非负且和为1的比例")
    training = config["training"]
    obsolete_training = {"early_stopping", "early_stopping_patience", "patience"} & set(training)
    if obsolete_training:
        raise ConfigError(f"05固定训练20轮且不早停，请删除旧配置：{sorted(obsolete_training)}")
    if str(training["optimizer"]).lower() != "adamw":
        raise ConfigError("首版training.optimizer必须为adamw")
    if bool(training["amp"]):
        raise ConfigError("首版关闭AMP，training.amp必须为false")
    for name in ("batch_size", "max_epochs", "progress_interval_batches", "progress_interval_seconds"):
        if int(training[name]) <= 0:
            raise ConfigError(f"training.{name}必须大于0")
    if int(training["max_epochs"]) != 20:
        raise ConfigError("首版固定训练20轮，training.max_epochs必须为20")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ConfigError(f"配置文件必须是映射：{path}")
    validate_config(value)
    return value


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_json(config)
