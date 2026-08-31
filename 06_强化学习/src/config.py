from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .utils import parse_time, sha256_json


CONFIG_FILES = ("environment", "policy", "critic", "training")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location}必须是映射")
    return value


def _positive(value: Any, location: str) -> float:
    number = float(value)
    if number <= 0:
        raise ConfigError(f"{location}必须大于0")
    return number


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigError(f"无法读取配置：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"配置根节点必须是映射：{path}")
    return value


def load_config_set(config_dir: Path) -> dict[str, Any]:
    config = {name: load_yaml(config_dir / f"{name}.yaml") for name in CONFIG_FILES}
    validate_config_set(config)
    return config


def validate_config_set(config: Mapping[str, Any]) -> None:
    missing = set(CONFIG_FILES) - set(config)
    if missing:
        raise ConfigError(f"缺少配置文件：{sorted(missing)}")
    environment_file = _mapping(config["environment"], "environment.yaml")
    for name in ("paths", "time", "environment", "storage"):
        _mapping(environment_file.get(name), f"environment.{name}")
    time = environment_file["time"]
    boundaries = [parse_time(str(time[name]), f"time.{name}") for name in ("warmup_start", "train_start", "test_start", "test_end")]
    if boundaries != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ConfigError("warmup、train、test时间边界必须严格递增")
    if int(time["decision_interval_seconds"]) <= 0:
        raise ConfigError("decision_interval_seconds必须大于0")
    latency = float(time["decision_latency_seconds"])
    if latency < 0 or latency >= float(time["decision_interval_seconds"]):
        raise ConfigError("decision_latency_seconds必须位于[0, decision_interval_seconds)")
    _positive(time["max_completion_horizon_seconds"], "time.max_completion_horizon_seconds")
    env = environment_file["environment"]
    _positive(env["channel_count"], "environment.channel_count")
    _positive(env["bandwidth_bytes_per_second_per_channel"], "environment.bandwidth")
    ratio = float(env["cache_capacity_ratio"])
    if not 0 < ratio <= 1:
        raise ConfigError("cache_capacity_ratio必须位于(0,1]")
    if int(env["max_prefetch_per_macro_step"]) <= 0:
        raise ConfigError("max_prefetch_per_macro_step必须大于0")
    if env["queue_policy"] != "fifo" or env["eviction_policy"] != "lru":
        raise ConfigError("首版固定使用FIFO和LRU")
    policy = _mapping(config["policy"], "policy.yaml")
    retrieval = _mapping(policy.get("retrieval"), "policy.retrieval")
    if int(retrieval["static_top_k"]) != 256 or int(retrieval["history_top_k"]) != 256 or int(retrieval["union_max"]) != 512:
        raise ConfigError("首版候选必须为双路Top-256、并集最多512")
    initial = _mapping(policy.get("initial_exploration"), "policy.initial_exploration")
    initial_route_probabilities = [
        float(initial[name])
        for name in ("simple_greedy_probability", "direct_stop_probability", "random_probability")
    ]
    if any(value < 0 for value in initial_route_probabilities) or abs(sum(initial_route_probabilities) - 1.0) > 1e-9:
        raise ConfigError("初始探索三种行为概率之和必须为1")
    random_probabilities = [float(initial["random_top32_probability"]), float(initial["random_all_probability"])]
    if any(value < 0 for value in random_probabilities) or abs(sum(random_probabilities) - 1.0) > 1e-9:
        raise ConfigError("随机候选两种来源概率之和必须为1")
    epsilon = _mapping(policy.get("epsilon_greedy"), "policy.epsilon_greedy")
    for name in ("epsilon_start", "epsilon_end", "exploration_stop_probability"):
        if not 0 <= float(epsilon[name]) <= 1:
            raise ConfigError(f"policy.epsilon_greedy.{name}必须位于[0,1]")
    if int(epsilon["exploration_top_k"]) <= 0:
        raise ConfigError("policy.epsilon_greedy.exploration_top_k必须大于0")
    critic = _mapping(config["critic"], "critic.yaml")
    model = _mapping(critic.get("model"), "critic.model")
    if int(model["resource_input_dim"]) != 10 or int(model["candidate_input_dim"]) != 274:
        raise ConfigError("Critic资源维度必须为10、候选维度必须为274")
    if int(critic["target"]["n_step"]) <= 0:
        raise ConfigError("critic.target.n_step必须大于0")
    training = _mapping(config["training"], "training.yaml")
    for section in ("runtime", "sampling", "critic", "actor", "protection"):
        _mapping(training.get(section), f"training.{section}")
    replay_probabilities = [
        float(training["sampling"]["initial_replay_probability"]),
        float(training["sampling"]["epsilon_replay_probability"]),
    ]
    if any(value < 0 for value in replay_probabilities) or abs(sum(replay_probabilities) - 1.0) > 1e-9:
        raise ConfigError("初始与epsilon Replay采样概率必须非负且和为1")
    uniform_probability = float(training["sampling"]["uniform_macro_probability"])
    nonzero_probability = float(training["sampling"]["nonzero_reward_probability"])
    if not 0 <= uniform_probability <= 1 or not 0 <= nonzero_probability <= 1:
        raise ConfigError("宏步采样概率必须位于[0,1]")
    if abs(uniform_probability + nonzero_probability - 1.0) > 1e-9:
        raise ConfigError("uniform与nonzero宏步采样概率之和必须为1")
    for location, value in (
        ("critic.batch_macro_steps", training["sampling"]["batch_macro_steps"]),
        ("critic.initial_max_updates", training["critic"]["initial_max_updates"]),
        ("critic.final_max_updates", training["critic"]["final_max_updates"]),
        ("actor.max_updates", training["actor"]["max_updates"]),
    ):
        _positive(value, location)


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_json(config)
