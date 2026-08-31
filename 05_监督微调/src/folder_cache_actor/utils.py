from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import torch

from .errors import ConfigError, DataIntegrityError


SHANGHAI = ZoneInfo("Asia/Shanghai")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_identifier(value: str, field_name: str) -> str:
    if not value or IDENTIFIER_RE.fullmatch(value) is None:
        raise ConfigError(f"{field_name} 只能包含字母、数字、点、下划线和短横线：{value!r}")
    return value


def parse_time(value: str, field_name: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{field_name} 不是合法ISO时间：{value!r}") from exc
    if parsed.tzinfo is None:
        raise ConfigError(f"{field_name} 必须包含时区")
    return int(parsed.astimezone(SHANGHAI).timestamp())


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"无法读取JSON文件 {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def file_descriptions(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(f"非法device：{requested}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("配置要求CUDA，但容器未检测到GPU")
    return device

