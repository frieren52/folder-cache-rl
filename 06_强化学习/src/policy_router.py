from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .candidate import CandidateBatch, CandidateEngine
from .environment import FolderCacheEnvironment
from .errors import ArtifactCompatibilityError


@dataclass(frozen=True)
class ScoreResult:
    candidate_q1: np.ndarray
    candidate_q2: np.ndarray
    stop_q1: float
    stop_q2: float


class CriticScorer(Protocol):
    def __call__(self, batch: CandidateBatch) -> ScoreResult: ...


@dataclass(frozen=True)
class PolicyDecision:
    cutoff_time: int
    policy_name: str
    selected_path_indices: tuple[int, ...]
    selected_scores: tuple[float, ...]
    stop_reason: str
    decision_latency_seconds: float
    actor_sha256: str
    critic_sha256: str | None


@dataclass(frozen=True)
class RoutedDecision:
    decision: PolicyDecision
    states: tuple[CandidateBatch, ...]
    action_positions: tuple[int, ...]


class PolicyRouter:
    ROUTES = {"no_prefetch", "simple_greedy", "critic", "actor_critic"}

    def __init__(self, max_prefetch_per_macro_step: int, decision_latency_seconds: float) -> None:
        if int(max_prefetch_per_macro_step) <= 0:
            raise ValueError("每宏步预取上限必须大于0")
        self.max_prefetch = int(max_prefetch_per_macro_step)
        self.decision_latency = float(decision_latency_seconds)

    @staticmethod
    def _best_position(values: np.ndarray, legal: np.ndarray, path_indices: np.ndarray) -> int | None:
        positions = np.flatnonzero(legal)
        if not len(positions):
            return None
        order = sorted(positions.tolist(), key=lambda pos: (-float(values[pos]), int(path_indices[pos])))
        return int(order[0])

    def decide(
        self,
        route: str,
        batch: CandidateBatch,
        shadow_environment: FolderCacheEnvironment,
        candidate_engine: CandidateEngine,
        scoring_end_time: int,
        actor_sha256: str,
        critic_sha256: str | None = None,
        scorer: CriticScorer | None = None,
    ) -> PolicyDecision:
        return self.decide_with_trace(
            route,
            batch,
            shadow_environment,
            candidate_engine,
            scoring_end_time,
            actor_sha256,
            critic_sha256,
            scorer,
        ).decision

    def decide_with_trace(
        self,
        route: str,
        batch: CandidateBatch,
        shadow_environment: FolderCacheEnvironment,
        candidate_engine: CandidateEngine,
        scoring_end_time: int,
        actor_sha256: str,
        critic_sha256: str | None = None,
        scorer: CriticScorer | None = None,
    ) -> RoutedDecision:
        if route not in self.ROUTES:
            raise ValueError(f"未知策略路由：{route}")
        if route == "no_prefetch":
            decision = PolicyDecision(batch.cutoff_time, route, (), (), "stop", self.decision_latency, actor_sha256, None)
            return RoutedDecision(decision, (batch,), (-1,))
        if route in {"critic", "actor_critic"} and (scorer is None or not critic_sha256):
            raise ArtifactCompatibilityError(f"{route}缺少匹配Critic")
        selected_ids: list[int] = []
        selected_scores: list[float] = []
        current = batch
        stop_reason = "stop"
        trace_states: list[CandidateBatch] = []
        trace_actions: list[int] = []
        while len(selected_ids) < self.max_prefetch:
            legal = current.valid_mask & current.action_mask & ~current.selected_mask
            if not np.any(legal):
                stop_reason = "no_legal_candidate"
                break
            if route == "simple_greedy":
                values = np.asarray(
                    [record.probability_after_completion * record.total_size_bytes for record in current.records],
                    dtype=np.float64,
                )
                position = self._best_position(values, legal, current.path_indices)
                if position is None or values[position] <= 0:
                    break
                score = float(values[position])
            else:
                assert scorer is not None
                result = scorer(current)
                if result.candidate_q1.shape != (len(current.records),) or result.candidate_q2.shape != (len(current.records),):
                    raise ValueError("Critic scorer候选输出形状错误")
                values = np.minimum(result.candidate_q1, result.candidate_q2)
                position = self._best_position(values, legal, current.path_indices)
                stop_value = min(float(result.stop_q1), float(result.stop_q2))
                if position is None or float(values[position]) <= stop_value:
                    break
                score = float(values[position])
            trace_states.append(current)
            trace_actions.append(position)
            selected_ids.append(int(current.path_indices[position]))
            selected_scores.append(score)
            current = candidate_engine.select_in_shadow(current, shadow_environment, position, scoring_end_time)
        if len(selected_ids) >= self.max_prefetch:
            stop_reason = "max_prefetch_reached"
        trace_states.append(current)
        trace_actions.append(-1)
        decision = PolicyDecision(
            batch.cutoff_time,
            route,
            tuple(selected_ids),
            tuple(selected_scores),
            stop_reason,
            self.decision_latency,
            actor_sha256,
            critic_sha256 if route in {"critic", "actor_critic"} else None,
        )
        return RoutedDecision(decision, tuple(trace_states), tuple(trace_actions))

    def random_with_trace(
        self,
        batch: CandidateBatch,
        shadow_environment: FolderCacheEnvironment,
        candidate_engine: CandidateEngine,
        scoring_end_time: int,
        actor_sha256: str,
        rng: np.random.Generator,
        top_k_probability: float = 0.8,
        top_k: int = 32,
    ) -> RoutedDecision:
        current = batch
        selected_ids: list[int] = []
        selected_scores: list[float] = []
        trace_states: list[CandidateBatch] = []
        trace_actions: list[int] = []
        while len(selected_ids) < self.max_prefetch:
            legal = current.valid_mask & current.action_mask & ~current.selected_mask
            positions = np.flatnonzero(legal)
            if not len(positions):
                break
            if float(rng.random()) < float(top_k_probability):
                positions = np.asarray(
                    sorted(
                        positions.tolist(),
                        key=lambda pos: (-current.records[pos].current_fusion_score, int(current.path_indices[pos])),
                    )[: int(top_k)],
                    dtype=np.int64,
                )
            position = int(positions[int(rng.integers(0, len(positions)))])
            trace_states.append(current)
            trace_actions.append(position)
            selected_ids.append(int(current.path_indices[position]))
            selected_scores.append(float(current.records[position].current_fusion_score))
            current = candidate_engine.select_in_shadow(current, shadow_environment, position, scoring_end_time)
        trace_states.append(current)
        trace_actions.append(-1)
        reason = "max_prefetch_reached" if len(selected_ids) >= self.max_prefetch else "no_legal_candidate"
        decision = PolicyDecision(
            batch.cutoff_time,
            "random",
            tuple(selected_ids),
            tuple(selected_scores),
            reason,
            self.decision_latency,
            actor_sha256,
            None,
        )
        return RoutedDecision(decision, tuple(trace_states), tuple(trace_actions))


def initial_exploration_route(
    rng: np.random.Generator,
    simple_greedy_probability: float = 0.50,
    direct_stop_probability: float = 0.25,
    random_probability: float = 0.25,
) -> str:
    probabilities = np.asarray(
        [simple_greedy_probability, direct_stop_probability, random_probability],
        dtype=np.float64,
    )
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-9):
        raise ValueError("初始探索概率必须非负且和为1")
    value = float(rng.random())
    if value < probabilities[0]:
        return "simple_greedy"
    if value < probabilities[0] + probabilities[1]:
        return "no_prefetch"
    return "random"


def epsilon_at(position: int, count: int, start: float = 0.10, end: float = 0.01) -> float:
    if count <= 1:
        return float(end)
    fraction = min(1.0, max(0.0, int(position) / (int(count) - 1)))
    return float(start) + fraction * (float(end) - float(start))
