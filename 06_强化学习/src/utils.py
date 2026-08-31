from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from zoneinfo import ZoneInfo

from .errors import ConfigError, DataIntegrityError


SHANGHAI = ZoneInfo("Asia/Shanghai")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def parse_time(value: str, location: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{location}不是合法ISO时间：{value!r}") from exc
    if parsed.tzinfo is None:
        raise ConfigError(f"{location}必须包含时区：{value!r}")
    return int(parsed.timestamp())


def validate_identifier(value: str, location: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ConfigError(f"{location}只能包含字母、数字、点、下划线和连字符：{value!r}")
    return value


def resolve_path(module_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (module_root / path).resolve()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("配置要求CUDA，但当前容器没有可用GPU")
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"无法读取JSON：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataIntegrityError(f"JSON根节点必须是对象：{path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def finite_or_raise(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise DataIntegrityError(f"{name}包含非有限值")

