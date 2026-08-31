from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import torch
from folder_cache_actor.config import load_config as load_actor_config
from folder_cache_actor.data import ActorParquetDataset, make_collate
from folder_cache_actor.inference import SupervisedActor
from folder_cache_actor.utils import read_json
from folder_cache_actor.vector_store import StaticVectorStore
from torch.utils.data import DataLoader


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.actor_rl import ActorRLTrainer, generate_advantage_macros, train_actor_loop  # noqa: E402
from src.config import config_sha256, load_config_set  # noqa: E402
from src.critic import CriticConfig, TwinCritic  # noqa: E402
from src.errors import ArtifactCompatibilityError, RLError  # noqa: E402
from src.replay import ReplayReader  # noqa: E402
from src.utils import resolve_device, resolve_path, sha256_file, validate_identifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用Critic优势强化更新05 Actor")
    parser.add_argument("--config-dir", type=Path, default=MODULE_ROOT / "config")
    parser.add_argument("--actor-config", type=Path, default=REPOSITORY_ROOT / "05_监督微调" / "config" / "config.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--actor-data-version", required=True)
    parser.add_argument("--train-replay", type=Path, required=True)
    parser.add_argument("--calibration-replay", type=Path, required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--advantage-margin", type=float, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_set(args.config_dir.resolve())
    actor_config = load_actor_config(args.actor_config.resolve())
    run_id = validate_identifier(args.run_id, "run_id")
    data_version = validate_identifier(args.actor_data_version, "actor_data_version")
    device = resolve_device(args.device or str(config["training"]["runtime"]["device"]))
    actor_path = (
        args.actor_checkpoint
        or resolve_path(MODULE_ROOT, str(config["environment"]["paths"]["actor_checkpoint"]))
    ).resolve()
    supervised = SupervisedActor.load(actor_path, device="cpu")
    actor_sha256 = sha256_file(actor_path)
    critic_checkpoint = torch.load(args.critic_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if critic_checkpoint.get("schema_version") != "folder-rl-critic-checkpoint/v1":
        raise ArtifactCompatibilityError("不支持的Critic检查点")
    if critic_checkpoint.get("actor_checkpoint_sha256") != actor_sha256:
        raise ArtifactCompatibilityError("Critic与待强化Actor不匹配")
    model = TwinCritic(
        supervised.model,
        CriticConfig.from_mapping(config["critic"]["model"]),
        int(config["critic"]["seeds"]["q1"]),
        int(config["critic"]["seeds"]["q2"]),
    ).to(device)
    for name in ("q1", "q2", "q1_target", "q2_target"):
        getattr(model, name).load_state_dict(critic_checkpoint[f"{name}_state"], strict=True)
        getattr(model, name).eval()
    train_reader = ReplayReader(args.train_replay.resolve())
    calibration_reader = ReplayReader(args.calibration_replay.resolve())
    for reader in (train_reader, calibration_reader):
        if reader.manifest.get("actor_sha256") != actor_sha256:
            raise ArtifactCompatibilityError("优势Replay与待强化Actor不匹配")
        for name in ("candidate_normalization", "resource_normalization"):
            if reader.manifest.get(name) != critic_checkpoint.get(name):
                raise ArtifactCompatibilityError(f"优势Replay的{name}与Critic不匹配")
    actor_training = config["training"]["actor"]
    train_advantages = generate_advantage_macros(
        train_reader,
        model,
        device,
        float(args.advantage_margin),
        int(actor_training["max_advantage_pairs_per_macro_step"]),
        int(config["training"]["sampling"]["seed"]),
        "train_fit",
    )
    calibration_advantages = generate_advantage_macros(
        calibration_reader,
        model,
        device,
        float(args.advantage_margin),
        int(actor_training["max_advantage_pairs_per_macro_step"]),
        int(config["training"]["sampling"]["seed"]) + 1,
        "train_calibration",
    )
    actor_root = REPOSITORY_ROOT / "05_监督微调"
    data_dir = resolve_path(actor_root, str(actor_config["paths"]["dataset_root"])) / data_version
    manifest = read_json(data_dir / "actor_samples_manifest.json")
    vector_store = StaticVectorStore.load(resolve_path(actor_root, str(actor_config["paths"]["vector_store_dir"])))
    anchor_data = ActorParquetDataset(
        data_dir / "actor_samples.parquet",
        data_dir / "actor_history_snapshots.parquet",
        "train",
        int(actor_config["sampling"]["shuffle_buffer_size"]),
        int(config["training"]["sampling"]["seed"]),
    )
    anchor_loader = DataLoader(
        anchor_data,
        batch_size=int(actor_training["batch_macro_steps"]),
        num_workers=int(config["training"]["runtime"]["dataloader_workers"]),
        collate_fn=make_collate(vector_store),
        pin_memory=device.type == "cuda",
    )
    trainer = ActorRLTrainer(
        supervised.model,
        device,
        float(actor_training["learning_rate"]),
        float(actor_training["weight_decay"]),
        float(actor_training["gradient_clip_norm"]),
        float(actor_training["lambda_anchor"]),
    )
    output_dir = resolve_path(MODULE_ROOT, str(config["environment"]["paths"]["outputs_root"])) / "runs" / run_id / "actor_rl"
    if output_dir.exists():
        raise ArtifactCompatibilityError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    def checkpoint_builder(update: int, calibration: dict[str, float]) -> dict:
        return {
            "schema_version": "folder-cache-actor-checkpoint/v2",
            "epoch": 0,
            "rl_update": update,
            "rl_calibration": calibration,
            "config": config,
            "config_sha256": config_sha256(config),
            "data_version": data_version,
            "data_manifest_sha256": sha256_file(data_dir / "actor_samples_manifest.json"),
            "source_supervised_actor_sha256": actor_sha256,
            "source_critic_sha256": sha256_file(args.critic_checkpoint.resolve()),
            "train_replay_manifest_sha256": sha256_file(args.train_replay.resolve() / "replay_manifest.json"),
            "calibration_replay_manifest_sha256": sha256_file(args.calibration_replay.resolve() / "replay_manifest.json"),
            "advantage_margin": float(args.advantage_margin),
            "input_contract": supervised.metadata.get("input_contract"),
            "retrieval_contract": supervised.metadata.get("retrieval_contract"),
            "test_data_used": False,
        }

    result = train_actor_loop(
        trainer,
        train_advantages,
        calibration_advantages,
        anchor_loader,
        int(actor_training["batch_macro_steps"]),
        int(actor_training["max_updates"]),
        int(actor_training["calibration_interval_updates"]),
        int(actor_training["early_stopping_patience"]),
        int(config["training"]["sampling"]["seed"]),
        output_dir,
        checkpoint_builder,
        int(config["training"]["runtime"]["progress_interval_updates"]),
        int(config["training"]["runtime"]["progress_interval_seconds"]),
    )
    print(
        json.dumps(
            {
                "event": "actor_rl_train_complete",
                **result,
                "train_advantage_macro_steps": len(train_advantages),
                "calibration_advantage_macro_steps": len(calibration_advantages),
                "anchor_train_rows": int(manifest["split_rows"]["train"]),
                "test_data_used": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except RLError as exc:
        print(json.dumps({"status": "failed", "stage": "train_actor_rl", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"status": "failed", "stage": "train_actor_rl", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
