"""数据准备入口：CSV 切分、特征统计和冻结 BGE 离线编码。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from src.data import build_bge_cache, load_config, prepare_record_files, resolve_path  # noqa: E402
from src.model import FrozenBGEEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备静态语义编码模型的训练数据和离线 BGE 缓存")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--device", default=None, help="默认自动选择 cuda/cpu")
    parser.add_argument("--bge-local-path", type=Path, default=None, help="覆盖配置中的 BGE 本地目录")
    parser.add_argument("--skip-bge-cache", action="store_true", help="只生成切分数据和元数据")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    data_dir, metadata = prepare_record_files(config, MODULE_ROOT)
    if not args.skip_bge_cache:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        configured_local = config["bge"].get("local_path")
        local_path = args.bge_local_path
        if local_path is None and configured_local:
            local_path = resolve_path(MODULE_ROOT, configured_local)
        encoder = FrozenBGEEncoder(
            model_name=config["bge"]["model_name"],
            revision=config["bge"]["revision"],
            max_length=int(config["bge"]["max_length"]),
            batch_size=int(config["bge"]["batch_size"]),
            precision=config["bge"]["precision"],
            device=device,
            local_path=local_path,
        )
        metadata = build_bge_cache(data_dir, encoder, cache_dtype=config["bge"]["cache_dtype"])
    summary = {
        "data_dir": str(data_dir),
        "train_records": metadata["split"]["train_records"],
        "validation_records": metadata["split"]["validation_records"],
        "bge_cache": metadata["bge"]["cache_file"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
