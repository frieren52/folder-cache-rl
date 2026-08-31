"""Validate raw access events and build the fixed train/validation dataset."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.config import load_config  # noqa: E402
from src.errors import DynamicHistoryError  # noqa: E402
from src.preparation import prepare_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造动态历史模型训练与验证数据")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPOSITORY_ROOT / "02_全局数据" / "artifacts" / "vocab" / "path_catalog.csv",
    )
    parser.add_argument(
        "--access-dir",
        type=Path,
        default=REPOSITORY_ROOT / "02_全局数据" / "raw",
    )
    parser.add_argument("--data-root", type=Path, default=MODULE_ROOT / "data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    output = prepare_dataset(
        config,
        args.dataset_id,
        args.catalog.resolve(),
        args.access_dir.resolve(),
        args.data_root.resolve(),
    )
    print(json.dumps({"dataset_dir": output.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except DynamicHistoryError as exc:
        print(
            json.dumps(
                {"status": "failed", "stage": "prepare_data", "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover - last-resort diagnostic
        print(
            json.dumps(
                {"status": "failed", "stage": "prepare_data", "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1) from exc
