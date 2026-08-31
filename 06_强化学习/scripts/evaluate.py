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
from src.evaluation import (  # noqa: E402
    POLICY_COMPLEXITY,
    PolicyMetrics,
    policy_metrics_from_environment,
    protection_failures,
    select_frozen_policy,
)
from src.errors import ArtifactCompatibilityError, RLError  # noqa: E402
from src.features import FeatureNormalizer, NormalizationStats  # noqa: E402
from src.policy_router import PolicyRouter  # noqa: E402
from src.rewards import RewardScales  # noqa: E402
from src.runtime import (  # noqa: E402
    RuntimeStateProvider,
    SimulationRunner,
    TorchCriticScorer,
    load_runtime_inputs,
    summarize_requests,
)
from src.utils import (  # noqa: E402
    parse_time,
    read_json,
    resolve_device,
    resolve_path,
    sha256_file,
    validate_identifier,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行06策略回放，或汇总指标并冻结策略")
    parser.add_argument("--config-dir", type=Path, default=MODULE_ROOT / "config")
    parser.add_argument("--phase", choices=("calibration", "test"), required=True)
    parser.add_argument("--policy", choices=POLICY_COMPLEXITY, help="运行单个策略并生成指标")
    parser.add_argument("--metrics", type=Path, nargs="+", help="汇总已有单策略指标")
    parser.add_argument("--run-id")
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--critic-checkpoint", type=Path)
    parser.add_argument("--channel-count", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--actor-recall-ok", action="store_true")
    parser.add_argument("--frozen-policy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.policy is None) == (args.metrics is None):
        parser.error("必须且只能提供--policy或--metrics")
    return args


def _phase_boundaries(config: dict[str, object], phase: str) -> tuple[int, int, int]:
    times = config["environment"]["time"]  # type: ignore[index]
    test_start = parse_time(str(times["test_start"]), "time.test_start")
    settlement_seconds = int(times["settlement_hours"]) * 3600
    if phase == "calibration":
        start = test_start - (int(times["calibration_hours"]) * 3600 + settlement_seconds)
        return start, test_start - settlement_seconds, test_start
    end = parse_time(str(times["test_end"]), "time.test_end")
    return test_start, end - settlement_seconds, end


def _load_scorer(
    actor: SupervisedActor,
    actor_path: Path,
    checkpoint_path: Path,
    config: dict[str, object],
    device: torch.device,
) -> tuple[TwinCritic, FeatureNormalizer, FeatureNormalizer, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != "folder-rl-critic-checkpoint/v1":
        raise ArtifactCompatibilityError("不支持的Critic检查点")
    if checkpoint.get("actor_checkpoint_sha256") != sha256_file(actor_path):
        raise ArtifactCompatibilityError("Critic与评价Actor不匹配")
    for name in ("candidate_normalization", "resource_normalization"):
        if name not in checkpoint:
            raise ArtifactCompatibilityError(f"Critic检查点缺少{name}")
    critic = TwinCritic(
        actor.model,
        CriticConfig.from_mapping(config["critic"]["model"]),  # type: ignore[index]
        int(config["critic"]["seeds"]["q1"]),  # type: ignore[index]
        int(config["critic"]["seeds"]["q2"]),  # type: ignore[index]
    ).to(device)
    for name in ("q1", "q2", "q1_target", "q2_target"):
        getattr(critic, name).load_state_dict(checkpoint[f"{name}_state"], strict=True)
        getattr(critic, name).eval()
    candidate_stats = NormalizationStats.from_dict(checkpoint["candidate_normalization"])
    resource_stats = NormalizationStats.from_dict(checkpoint["resource_normalization"])
    return (
        critic,
        FeatureNormalizer(candidate_stats, candidate_stats.dimension, candidate_stats.log1p_indices),
        FeatureNormalizer(resource_stats, resource_stats.dimension, resource_stats.log1p_indices),
        sha256_file(checkpoint_path),
    )


def _run_policy(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    if not args.run_id:
        raise ValueError("策略回放必须提供--run-id")
    run_id = validate_identifier(args.run_id, "run_id")
    policy_name = str(args.policy)
    environment_file = config["environment"]
    paths = environment_file["paths"]
    times = environment_file["time"]
    env_config = environment_file["environment"]
    split_start, action_end, scoring_end = _phase_boundaries(config, args.phase)
    interval = int(times["decision_interval_seconds"])
    device = resolve_device(args.device or str(config["training"]["runtime"]["device"]))
    object_sizes, requests = load_runtime_inputs(MODULE_ROOT, environment_file)
    cache_capacity = int(sum(object_sizes.values()) * float(env_config["cache_capacity_ratio"]))
    channel_count = int(args.channel_count or env_config["channel_count"])
    if channel_count <= 0:
        raise ValueError("channel_count必须大于0")
    base = FolderCacheEnvironment(
        requests,
        object_sizes,
        cache_capacity,
        channel_count,
        float(env_config["bandwidth_bytes_per_second_per_channel"]),
        float(env_config["fixed_setup_seconds"]),
        parse_time(str(times["warmup_start"]), "time.warmup_start"),
        max(60, 2 * interval),
    )
    base.advance_to(split_start)
    base.track_wait_samples = True
    base.reset_metrics(split_start)
    if policy_name == "no_prefetch":
        base.advance_to(scoring_end)
        metrics, diagnostics = policy_metrics_from_environment(policy_name, base, split_start, scoring_end)
        return {
            "schema_version": "folder-rl-policy-metrics/v1",
            **metrics.__dict__,
            "diagnostics": diagnostics,
            "phase": args.phase,
            "run_id": run_id,
            "channel_count": channel_count,
            "config_sha256": config_sha256(config),
            "actor_sha256": None,
            "critic_sha256": None,
            "test_data_used": args.phase == "test",
        }

    actor_path = (args.actor_checkpoint or resolve_path(MODULE_ROOT, str(paths["actor_checkpoint"]))).resolve()
    actor = SupervisedActor.load(actor_path, device=str(device))
    actor_sha256 = sha256_file(actor_path)
    vector_store_dir = resolve_path(MODULE_ROOT, str(paths["vector_store_dir"]))
    static_store = StaticVectorStore.load(vector_store_dir)
    history_release = resolve_path(MODULE_ROOT, str(paths["history_release"]))
    validate_history_release(history_release)
    history_encoder = HistoryEncoderAdapter(
        REPOSITORY_ROOT / "04_动态历史向量生成" / "src",
        history_release,
        device=str(device),
    )
    train_start = parse_time(str(times["train_start"]), "time.train_start")
    state_provider = RuntimeStateProvider(
        requests,
        train_start,
        static_store,
        actor,
        history_encoder,
        vector_store_dir / "initial_history_index.npz",
        interval,
    )
    for cutoff in range(train_start, split_start, interval):
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
    scorer_factory = None
    critic_sha256 = None
    if policy_name in {"critic", "actor_critic"}:
        if args.critic_checkpoint is None:
            raise ValueError(f"{policy_name}回放必须提供--critic-checkpoint")
        critic, candidate_normalizer, resource_normalizer, critic_sha256 = _load_scorer(
            actor, actor_path, args.critic_checkpoint.resolve(), config, device
        )
        scorer_factory = lambda actor_state: TorchCriticScorer(
            critic, actor_state, device, candidate_normalizer, resource_normalizer
        )
    test_start = parse_time(str(times["test_start"]), "time.test_start")
    train_steps = max(1, (test_start - train_start) // interval)
    train_request_count, train_request_bytes = summarize_requests(requests, train_start, test_start)
    scales = RewardScales(
        train_request_count / train_steps,
        train_request_bytes / train_steps,
    )
    runner = SimulationRunner(
        base.clone(),
        base.clone(),
        state_provider,
        candidate_engine,
        router,
        scales,
        actor_sha256,
        run_id,
        scoring_end,
        action_end,
        np.random.Generator(np.random.PCG64(int(config["training"]["sampling"]["seed"]))),
        interval,
        config["policy"]["initial_exploration"],
        config["policy"]["epsilon_greedy"],
    )
    total_steps = (scoring_end - split_start) // interval
    for macro_step_id, cutoff in enumerate(range(split_start, scoring_end, interval)):
        runner.macro_step(
            macro_step_id,
            cutoff,
            "train_calibration" if args.phase == "calibration" else "test",
            policy_name,
            scorer_factory,
            critic_sha256,
        )
        if (macro_step_id + 1) % 600 == 0:
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "policy_name": policy_name,
                        "phase": args.phase,
                        "macro_steps": macro_step_id + 1,
                        "total_macro_steps": total_steps,
                        "cutoff_time": cutoff,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    metrics, diagnostics = policy_metrics_from_environment(policy_name, runner.policy, split_start, scoring_end)
    return {
        "schema_version": "folder-rl-policy-metrics/v1",
        **metrics.__dict__,
        "diagnostics": diagnostics,
        "phase": args.phase,
        "run_id": run_id,
        "channel_count": channel_count,
        "config_sha256": config_sha256(config),
        "actor_sha256": actor_sha256,
        "critic_sha256": critic_sha256,
        "history_release_sha256": sha256_file(history_release / "manifest.json"),
        "test_data_used": args.phase == "test",
    }


def _summarize(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    raw_values = [read_json(path.resolve()) for path in args.metrics]
    phases = {str(value.get("phase")) for value in raw_values}
    channels = {int(value.get("channel_count", -1)) for value in raw_values}
    if phases != {args.phase} or len(channels) != 1:
        raise ArtifactCompatibilityError("待汇总指标的phase或channel_count不一致")
    values = {str(raw["policy_name"]): PolicyMetrics.from_mapping(raw) for raw in raw_values}
    protection = config["training"]["protection"]
    if args.phase == "calibration":
        selected, failures = select_frozen_policy(values, protection, args.actor_recall_ok)
        return {
            "schema_version": "folder-rl-policy-selection/v1",
            "phase": "train_calibration",
            "selected_policy": selected,
            "protection_failures": failures,
            "metrics": {name: metric.__dict__ for name, metric in values.items()},
            "channel_count": next(iter(channels)),
            "test_data_used": False,
        }
    if not args.frozen_policy or args.frozen_policy not in values or "no_prefetch" not in values:
        raise ValueError("test汇总必须提供已冻结且存在指标的--frozen-policy，并包含no_prefetch")
    return {
        "schema_version": "folder-rl-test-report/v1",
        "phase": "test",
        "frozen_policy": args.frozen_policy,
        "protection_failures": protection_failures(values[args.frozen_policy], values["no_prefetch"], protection),
        "metrics": {name: metric.__dict__ for name, metric in values.items()},
        "channel_count": next(iter(channels)),
        "selection_changed_from_test": False,
    }


def main() -> None:
    args = parse_args()
    config = load_config_set(args.config_dir.resolve())
    report = _run_policy(args, config) if args.policy is not None else _summarize(args, config)
    write_json(args.output.resolve(), report)
    print(json.dumps({"event": "evaluation_complete", **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except RLError as exc:
        print(json.dumps({"status": "failed", "stage": "evaluate", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"status": "failed", "stage": "evaluate", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
