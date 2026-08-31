from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from folder_cache_actor.inference import SupervisedActor


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from src.config import config_sha256, load_config_set  # noqa: E402
from src.critic import CriticConfig, TwinCritic  # noqa: E402
from src.errors import ArtifactCompatibilityError, RLError  # noqa: E402
from src.replay import ReplayReader, ReplaySampler, Transition  # noqa: E402
from src.training import CriticTrainer, train_critic_loop  # noqa: E402
from src.utils import resolve_device, resolve_path, sha256_file, validate_identifier  # noqa: E402


class MixedSampler:
    def __init__(self, first: ReplaySampler, second: ReplaySampler, seed: int, first_probability: float) -> None:
        self.first = first
        self.second = second
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.first_probability = float(first_probability)
        if not 0 <= self.first_probability <= 1:
            raise ValueError("初始Replay采样概率必须位于[0,1]")

    def sample_macro_batch(self, batch_size: int) -> list[Transition]:
        count = int(batch_size)
        first_count = int(round(count * self.first_probability))
        values = [] if first_count == 0 else self.first.sample_macro_batch(first_count)
        if first_count < count:
            values.extend(self.second.sample_macro_batch(count - first_count))
        self.rng.shuffle(values)
        return values


def _require_matching_replays(readers: list[ReplayReader], actor_sha256: str) -> None:
    reference = readers[0].manifest
    for reader in readers:
        manifest = reader.manifest
        if manifest.get("actor_sha256") != actor_sha256:
            raise ArtifactCompatibilityError("Replay绑定的Actor与当前Actor不一致")
        for name in ("candidate_normalization", "resource_normalization"):
            if name not in manifest or manifest[name] != reference.get(name):
                raise ArtifactCompatibilityError(f"Replay的{name}不一致")


def calibration_transitions(reader: ReplayReader, gamma: float) -> list[Transition]:
    records = [reader.record_at(index) for index, item in enumerate(reader.entries) if item.split == "train_calibration"]
    if not records:
        raise ArtifactCompatibilityError("校准Replay没有train_calibration记录")
    returns = np.zeros(len(records), dtype=np.float64)
    running = 0.0
    for index in range(len(records) - 1, -1, -1):
        if records[index].terminal or (index + 1 < len(records) and records[index + 1].run_id != records[index].run_id):
            running = records[index].reward
        else:
            running = records[index].reward + gamma * running
        returns[index] = running
    result: list[Transition] = []
    for record, value in zip(records, returns):
        for state, action in zip(record.states, record.action_positions):
            result.append(Transition(state, int(action), 0.0, 0.0, state, True, record.macro_step_id, 1.0, float(value)))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练06双Critic")
    parser.add_argument("--config-dir", type=Path, default=MODULE_ROOT / "config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("initial", "final"), required=True)
    parser.add_argument("--initial-replay", type=Path, required=True)
    parser.add_argument("--epsilon-replay", type=Path)
    parser.add_argument("--calibration-replay", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--resume-critic", type=Path)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_set(args.config_dir.resolve())
    run_id = validate_identifier(args.run_id, "run_id")
    env_paths = config["environment"]["paths"]
    actor_path = (args.actor_checkpoint or resolve_path(MODULE_ROOT, str(env_paths["actor_checkpoint"]))).resolve()
    device = resolve_device(args.device or str(config["training"]["runtime"]["device"]))
    actor = SupervisedActor.load(actor_path, device="cpu")
    critic_config = CriticConfig.from_mapping(config["critic"]["model"])
    model = TwinCritic(
        actor.model,
        critic_config,
        int(config["critic"]["seeds"]["q1"]),
        int(config["critic"]["seeds"]["q2"]),
    )
    actor_sha256 = sha256_file(actor_path)
    if args.stage == "final" and args.resume_critic is None:
        raise ValueError("final阶段必须通过--resume-critic从初始Critic继续训练")
    if args.resume_critic is not None:
        checkpoint = torch.load(args.resume_critic.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "folder-rl-critic-checkpoint/v1":
            raise ArtifactCompatibilityError("不支持的续训Critic检查点")
        if checkpoint.get("actor_checkpoint_sha256") != actor_sha256:
            raise ArtifactCompatibilityError("续训Critic绑定的Actor与当前Actor不一致")
        for name in ("q1", "q2", "q1_target", "q2_target"):
            model_part = getattr(model, name)
            model_part.load_state_dict(checkpoint[f"{name}_state"], strict=True)
    target = config["critic"]["target"]
    decision_interval = int(config["environment"]["time"]["decision_interval_seconds"])
    gamma_stop = 2 ** (-decision_interval / float(target["reward_half_life_seconds"]))
    initial_reader = ReplayReader(args.initial_replay.resolve())
    initial_sampler = ReplaySampler(
        initial_reader,
        "train_fit",
        int(config["training"]["sampling"]["seed"]),
        gamma_stop,
        int(target["n_step"]),
        float(config["training"]["sampling"]["uniform_macro_probability"]),
    )
    sampler: object = initial_sampler
    readers = [initial_reader]
    if args.stage == "final":
        if args.epsilon_replay is None:
            raise ValueError("final阶段必须提供--epsilon-replay")
        epsilon_reader = ReplayReader(args.epsilon_replay.resolve())
        readers.append(epsilon_reader)
        epsilon_sampler = ReplaySampler(
            epsilon_reader,
            "train_fit",
            int(config["training"]["sampling"]["seed"]) + 1,
            gamma_stop,
            int(target["n_step"]),
            float(config["training"]["sampling"]["uniform_macro_probability"]),
        )
        initial_probability = float(config["training"]["sampling"]["initial_replay_probability"])
        epsilon_probability = float(config["training"]["sampling"]["epsilon_replay_probability"])
        if not np.isclose(initial_probability + epsilon_probability, 1.0, atol=1e-9):
            raise ValueError("初始与epsilon Replay采样概率之和必须为1")
        sampler = MixedSampler(
            initial_sampler,
            epsilon_sampler,
            int(config["training"]["sampling"]["seed"]) + 2,
            initial_probability,
        )
    calibration_reader = ReplayReader(args.calibration_replay.resolve())
    readers.append(calibration_reader)
    _require_matching_replays(readers, actor_sha256)
    fixed_calibration = calibration_transitions(calibration_reader, gamma_stop)
    training = config["training"]
    trainer = CriticTrainer(
        model,
        device,
        float(training["critic"]["learning_rate"]),
        float(training["critic"]["weight_decay"]),
        float(training["critic"]["huber_delta"]),
        float(training["critic"]["gradient_clip_norm"]),
        float(target["polyak_tau"]),
    )
    output_dir = resolve_path(MODULE_ROOT, str(env_paths["outputs_root"])) / "runs" / run_id / f"critic_{args.stage}"
    if output_dir.exists():
        raise ArtifactCompatibilityError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    def checkpoint_builder(update: int, calibration: dict[str, float]) -> dict:
        return {
            "schema_version": "folder-rl-critic-checkpoint/v1",
            "run_id": run_id,
            "stage": args.stage,
            "update": update,
            "calibration": calibration,
            "config": config,
            "config_sha256": config_sha256(config),
            "critic_config": critic_config.to_dict(),
            "actor_checkpoint_sha256": actor_sha256,
            "candidate_normalization": initial_reader.manifest["candidate_normalization"],
            "resource_normalization": initial_reader.manifest["resource_normalization"],
            "decision_interval_seconds": decision_interval,
            "gamma_stop": gamma_stop,
            "initial_replay_manifest_sha256": sha256_file(args.initial_replay.resolve() / "replay_manifest.json"),
            "epsilon_replay_manifest_sha256": None if args.epsilon_replay is None else sha256_file(args.epsilon_replay.resolve() / "replay_manifest.json"),
            "calibration_replay_manifest_sha256": sha256_file(args.calibration_replay.resolve() / "replay_manifest.json"),
            "test_data_used": False,
        }

    max_updates = int(training["critic"]["initial_max_updates"] if args.stage == "initial" else training["critic"]["final_max_updates"])
    result = train_critic_loop(
        trainer,
        sampler,  # type: ignore[arg-type]
        fixed_calibration,
        max_updates,
        int(training["sampling"]["batch_macro_steps"]),
        int(training["critic"]["calibration_interval_updates"]),
        int(training["critic"]["early_stopping_patience"]),
        output_dir,
        checkpoint_builder,
        int(training["runtime"]["progress_interval_updates"]),
        int(training["runtime"]["progress_interval_seconds"]),
    )
    print(json.dumps({"event": "critic_train_complete", **result, "test_data_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except RLError as exc:
        print(json.dumps({"status": "failed", "stage": "train_critic", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"status": "failed", "stage": "train_critic", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
