from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .errors import ConfigError, DataIntegrityError


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_identifier(value: str, field_name: str) -> str:
    if not value or IDENTIFIER_RE.fullmatch(value) is None:
        raise ConfigError(
            f"{field_name} 只能包含字母、数字、点、下划线和短横线，实际值：{value!r}"
        )
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"无法读取 JSON 文件 {path}: {exc}") from exc


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
        raise ConfigError(f"非法 device 配置：{requested}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("配置要求 CUDA，但当前容器未检测到可用 GPU")
    return device


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested not in {"auto", "fp32", "bf16"}:
        raise ConfigError(f"precision 只支持 auto、fp32、bf16，实际值：{requested}")
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"
    if requested == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ConfigError("当前设备不支持配置要求的 BF16")
    return requested


def verify_manifest_files(root: Path, files: dict[str, str]) -> None:
    for relative_path, expected_sha in files.items():
        path = root / relative_path
        if not path.is_file():
            raise DataIntegrityError(f"manifest 文件缺失：{path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise DataIntegrityError(
                f"文件摘要不匹配：{path}，期望 {expected_sha}，实际 {actual_sha}"
            )


def file_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(paths, key=lambda item: item.as_posix())
    }
