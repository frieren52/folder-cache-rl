from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from folder_cache_actor.inference import SupervisedActor
from folder_cache_actor.upstream import HistoryEncoderAdapter, validate_history_release
from folder_cache_actor.vector_store import StaticVectorStore


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.candidate import CandidateEngine  # noqa: E402
from src.config import config_sha256, load_config_set  # noqa: E402
from src.critic import CriticConfig, TwinCritic  # noqa: E402
from src.environment import FolderCacheEnvironment  # noqa: E402
from src.errors import RLError  # noqa: E402
from src.features import FeatureNormalizer, NormalizationStats  # noqa: E402
from src.policy_router import PolicyRouter, epsilon_at  # noqa: E402
from src.replay import ReplayWriter  # noqa: E402
from src.rewards import RewardScales, reward_percentiles  # noqa: E402
from src.runtime import (  # noqa: E402
    RuntimeStateProvider,
    SimulationRunner,
    TorchCriticScorer,
    load_runtime_inputs,
    make_feature_accumulators,
    summarize_requests,
    update_feature_accumulators,
)
from src.utils import parse_time, read_json, resolve_device, resolve_path, sha256_file, validate_identifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建06强化学习Replay")
    parser.add_argument("--config-dir", type=Path, default=MODULE_ROOT / "config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replay-version", required=True)
    parser.add_argument("--split", choices=("train_fit", "train_calibration"), required=True)
    parser.add_argument("--behavior", choices=("initial", "epsilon"), default="initial")
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--critic-checkpoint", type=Path)
    parser.add_argument("--normalization-from", type=Path)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_normalization(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = read_json(path / "replay_manifest.json")
    if "candidate_normalization" not in manifest or "resource_normalization" not in manifest:
        raise ValueError("指定Replay没有归一化统计")
    return manifest["candidate_normalization"], manifest["resource_normalization"]  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    config = load_config_set(args.config_dir.resolve())
    run_id = validate_identifier(args.run_id, "run_id")
    replay_version = validate_identifier(args.replay_version, "replay_version")
    environment_file = config["environment"]
    paths = environment_file["paths"]
    times = environment_file["time"]
    env_config = environment_file["environment"]
    storage = environment_file["storage"]
    train_start = parse_time(str(times["train_start"]), "time.train_start")
    test_start = parse_time(str(times["test_start"]), "time.test_start")
    calibration_start = test_start - (int(times["calibration_hours"]) + int(times["settlement_hours"])) * 3600
    calibration_end = test_start - int(times["settlement_hours"]) * 3600
    if args.split == "train_fit":
        split_start, scoring_end, action_end = train_start, calibration_start, calibration_start
    else:
        split_start, scoring_end, action_end = calibration_start, test_start, calibration_end
        if args.normalization_from is None:
            raise ValueError("train_calibration Replay必须通过--normalization-from复用train_fit统计")
    device = resolve_device(args.device or str(config["training"]["runtime"]["device"]))
    actor_checkpoint = (args.actor_checkpoint or resolve_path(MODULE_ROOT, str(paths["actor_checkpoint"]))).resolve()
    actor = SupervisedActor.load(actor_checkpoint, device=str(device))
    actor_sha = sha256_file(actor_checkpoint)
    vector_store_dir = resolve_path(MODULE_ROOT, str(paths["vector_store_dir"]))
    static_store = StaticVectorStore.load(vector_store_dir)
    history_release = resolve_path(MODULE_ROOT, str(paths["history_release"]))
    validate_history_release(history_release)
    history_encoder = HistoryEncoderAdapter(REPOSITORY_ROOT / "04_动态历史向量生成" / "src", history_release, device=str(device))
    object_sizes, requests = load_runtime_inputs(MODULE_ROOT, environment_file)
    total_catalog_bytes = sum(object_sizes.values())
    cache_capacity = int(total_catalog_bytes * float(env_config["cache_capacity_ratio"]))
    warmup_start = parse_time(str(times["warmup_start"]), "time.warmup_start")
    base_environment = FolderCacheEnvironment(
        requests,
        object_sizes,
        cache_capacity,
        int(env_config["channel_count"]),
        float(env_config["bandwidth_bytes_per_second_per_channel"]),
        float(env_config["fixed_setup_seconds"]),
        warmup_start,
        max(60, 2 * int(times["decision_interval_seconds"])),
    )
    base_environment.advance_to(split_start)
    policy_environment = base_environment.clone()
    baseline_environment = base_environment.clone()
    state_provider = RuntimeStateProvider(
        requests,
        train_start,
        static_store,
        actor,
        history_encoder,
        vector_store_dir / "initial_history_index.npz",
        int(times["decision_interval_seconds"]),
    )
    for cutoff in range(train_start, split_start, int(times["decision_interval_seconds"])):
        state_provider.state_at(cutoff)
    candidate_engine = CandidateEngine(
        static_store,
        object_sizes,
        state_provider.encode_current,
        int(config["policy"]["retrieval"]["static_top_k"]),
        int(config["policy"]["retrieval"]["history_top_k"]),
        int(times["max_completion_horizon_seconds"]),
    )
    router = PolicyRouter(int(env_config["max_prefetch_per_macro_step"]), float(times["decision_latency_seconds"]))
    training_macro_steps = (test_start - train_start) // int(times["decision_interval_seconds"])
    training_request_count, training_request_bytes = summarize_requests(requests, train_start, test_start)
    scales = RewardScales(
        training_request_count / training_macro_steps,
        training_request_bytes / training_macro_steps,
    )
    rng = np.random.Generator(np.random.PCG64(int(config["training"]["sampling"]["seed"])))
    runner = SimulationRunner(
        policy_environment,
        baseline_environment,
        state_provider,
        candidate_engine,
        router,
        scales,
        actor_sha,
        run_id,
        scoring_end,
        action_end,
        rng,
        int(times["decision_interval_seconds"]),
        config["policy"]["initial_exploration"],
        config["policy"]["epsilon_greedy"],
    )
    scorer_factory = None
    critic_sha = None
    candidate_normalization: dict[str, object] | None = None
    resource_normalization: dict[str, object] | None = None
    if args.normalization_from is not None:
        candidate_normalization, resource_normalization = _load_normalization(args.normalization_from.resolve())
    if args.behavior == "epsilon":
        if args.critic_checkpoint is None or candidate_normalization is None or resource_normalization is None:
            raise ValueError("epsilon Replay要求--critic-checkpoint和--normalization-from")
        checkpoint = torch.load(args.critic_checkpoint.resolve(), map_location=device, weights_only=False)
        model = TwinCritic(
            actor.model,
            CriticConfig.from_mapping(config["critic"]["model"]),
            int(config["critic"]["seeds"]["q1"]),
            int(config["critic"]["seeds"]["q2"]),
        ).to(device)
        for name in ("q1", "q2", "q1_target", "q2_target"):
            getattr(model, name).load_state_dict(checkpoint[f"{name}_state"], strict=True)
            getattr(model, name).eval()
        critic_sha = sha256_file(args.critic_checkpoint.resolve())
        candidate_stats = NormalizationStats.from_dict(candidate_normalization)
        resource_stats = NormalizationStats.from_dict(resource_normalization)
        candidate_normalizer = FeatureNormalizer(candidate_stats, 274, candidate_stats.log1p_indices)
        resource_normalizer = FeatureNormalizer(resource_stats, 10, resource_stats.log1p_indices)
        scorer_factory = lambda actor_state: TorchCriticScorer(
            model, actor_state, device, candidate_normalizer, resource_normalizer
        )
    replay_root = resolve_path(MODULE_ROOT, str(paths["replay_root"])) / replay_version
    writer = ReplayWriter(
        replay_root,
        run_id,
        config_sha256(config),
        int(storage["rows_per_shard"]),
        str(storage["parquet_compression"]),
    )
    candidate_accumulator, resource_accumulator = make_feature_accumulators()
    rewards: list[float] = []
    total_steps = (scoring_end - split_start) // int(times["decision_interval_seconds"])
    for macro_step_id, cutoff in enumerate(range(split_start, scoring_end, int(times["decision_interval_seconds"]))):
        epsilon = epsilon_at(
            macro_step_id,
            total_steps,
            float(config["policy"]["epsilon_greedy"]["epsilon_start"]),
            float(config["policy"]["epsilon_greedy"]["epsilon_end"]),
        )
        result = runner.macro_step(
            macro_step_id,
            cutoff,
            args.split,
            args.behavior,
            scorer_factory,
            critic_sha,
            epsilon,
        )
        writer.append(result.record)
        rewards.append(result.record.reward)
        if args.split == "train_fit" and args.behavior == "initial":
            update_feature_accumulators(candidate_accumulator, resource_accumulator, result.record)
        if (macro_step_id + 1) % 600 == 0:
            print(
                json.dumps(
                    {
                        "event": "replay_progress",
                        "split": args.split,
                        "behavior": args.behavior,
                        "macro_steps": macro_step_id + 1,
                        "total_macro_steps": total_steps,
                        "cutoff_time": cutoff,
                        "reward_mean": float(np.mean(rewards)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if candidate_normalization is None or resource_normalization is None:
        candidate_normalization = candidate_accumulator.finalize().to_dict()
        resource_normalization = resource_accumulator.finalize().to_dict()
    writer.close(
        {
            "split": args.split,
            "behavior": args.behavior,
            "actor_sha256": actor_sha,
            "critic_sha256": critic_sha,
            "history_release_sha256": sha256_file(history_release / "manifest.json"),
            "candidate_normalization": candidate_normalization,
            "resource_normalization": resource_normalization,
            "reward_scales": {"count_scale": scales.count_scale, "bytes_scale": scales.bytes_scale},
            "reward_summary": reward_percentiles(rewards),
            "test_data_used": False,
        }
    )
    print(
        json.dumps(
            {
                "event": "replay_complete",
                "replay_dir": replay_root.as_posix(),
                "macro_steps": total_steps,
                "reward_summary": reward_percentiles(rewards),
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
        print(json.dumps({"status": "failed", "stage": "build_replay", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"status": "failed", "stage": "build_replay", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
