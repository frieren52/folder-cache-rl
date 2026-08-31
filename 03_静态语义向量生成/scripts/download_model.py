"""下载本模块实际使用的 BGE-M3 文件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from src.data import load_config, resolve_path  # noqa: E402


# 当前模型只使用 Hugging Face 的 AutoModel 和 AutoTokenizer，ONNX、图片及
# sentence-transformers 附加文件都不参与编码，因此不下载。
REQUIRED_FILES = [
    "config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载固定版本的 BGE-M3")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    bge = config["bge"]
    model_dir = resolve_path(MODULE_ROOT, bge["local_path"])
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=bge["model_name"],
        revision=bge["revision"],
        local_dir=model_dir,
        endpoint=bge["download_endpoint"],
        allow_patterns=REQUIRED_FILES,
        max_workers=int(bge["download_workers"]),
    )

    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"模型下载不完整：{', '.join(missing)}")
    total_bytes = sum((model_dir / name).stat().st_size for name in REQUIRED_FILES)
    print(
        json.dumps(
            {
                "model": bge["model_name"],
                "revision": bge["revision"],
                "local_path": str(model_dir),
                "size_bytes": total_bytes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
