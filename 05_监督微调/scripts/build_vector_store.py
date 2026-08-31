from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from folder_cache_actor.config import load_config  # noqa: E402
from folder_cache_actor.errors import ActorError  # noqa: E402
from folder_cache_actor.preparation import build_vector_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="使用03/04正式模型构建05双路向量库")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    build_vector_store(load_config(args.config.resolve()), MODULE_ROOT, args.device)


if __name__ == "__main__":
    try:
        main()
    except ActorError as exc:
        print(json.dumps({"status": "failed", "stage": "build_vector_store", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"status": "failed", "stage": "build_vector_store", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc

