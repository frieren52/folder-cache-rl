from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from folder_cache_actor.losses import compute_pairwise_losses
from folder_cache_actor.model import FolderCacheActor

from .critic import TwinCritic, min_q
from .replay import ReplayReader, StateArray
from .reporting import append_jsonl, plot_actor_history


@dataclass(frozen=True)
class AdvantagePair:
    positive_position: int
    negative_position: int
    advantage: float


@dataclass(frozen=True)
class AdvantageMacro:
    macro_step_id: int
    state: StateArray
    pairs: tuple[AdvantagePair, ...]


def _state_tensors(state: StateArray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(state.object_features[None], dtype=torch.float32, device=device),
        torch.as_tensor(state.object_valid_mask[None], dtype=torch.bool, device=device),
        torch.as_tensor(state.time_features[None], dtype=torch.float32, device=device),
        torch.as_tensor(state.resource_features[None], dtype=torch.float32, device=device),
        torch.as_tensor(state.candidate_features[None], dtype=torch.float32, device=device),
        torch.as_tensor(state.candidate_valid_mask[None], dtype=torch.bool, device=device),
        torch.as_tensor(state.selected_candidate_mask[None], dtype=torch.bool, device=device),
    )


@torch.no_grad()
def critic_advantages(model: TwinCritic, state: StateArray, device: torch.device) -> np.ndarray:
    first, second = model.target(*_state_tensors(state, device))
    output = min_q(first, second)
    advantages = output.candidate_q_values[0] - output.stop_q_value[0, 0]
    return advantages.float().cpu().numpy()


def build_advantage_pairs(
    advantages: np.ndarray,
    legal_mask: np.ndarray,
    margin: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[AdvantagePair, ...]:
    values = np.asarray(advantages, dtype=np.float32)
    legal = np.asarray(legal_mask, dtype=np.bool_)
    positives = sorted(np.flatnonzero(legal & (values > float(margin))).tolist(), key=lambda pos: (-float(values[pos]), pos))
    negatives = sorted(np.flatnonzero(legal & (values <= 0.0)).tolist(), key=lambda pos: (float(values[pos]), pos))
    if not positives or not negatives:
        return ()
    combinations = [(positive, negative) for positive in positives for negative in negatives]
    chosen: list[tuple[int, int]] = []
    for positive in positives:
        chosen.append((positive, negatives[0]))
        if len(chosen) >= int(max_pairs):
            break
    remaining = [item for item in combinations if item not in set(chosen)]
    if remaining and len(chosen) < int(max_pairs):
        order = rng.permutation(len(remaining))[: int(max_pairs) - len(chosen)]
        chosen.extend(remaining[int(index)] for index in order)
    return tuple(
        AdvantagePair(positive, negative, float(values[positive] - values[negative]))
        for positive, negative in chosen
    )


def generate_advantage_macros(
    reader: ReplayReader,
    model: TwinCritic,
    device: torch.device,
    margin: float,
    max_pairs: int,
    seed: int,
    split: str = "train_fit",
) -> list[AdvantageMacro]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    result: list[AdvantageMacro] = []
    model.q1_target.eval()
    model.q2_target.eval()
    for index, entry in enumerate(reader.entries):
        if entry.split != split:
            continue
        record = reader.record_at(index)
        state = record.states[0]
        advantages = critic_advantages(model, state, device)
        legal = state.candidate_valid_mask & state.candidate_action_mask
        pairs = build_advantage_pairs(advantages, legal, margin, max_pairs, rng)
        if pairs:
            result.append(AdvantageMacro(record.macro_step_id, state, pairs))
    return result


def collate_advantage_macros(rows: Sequence[AdvantageMacro], device: torch.device) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("优势宏步batch不能为空")
    contexts = np.stack([row.state.object_features for row in rows])
    valid = np.stack([row.state.object_valid_mask for row in rows])
    times = np.stack([row.state.time_features for row in rows])
    sample_indices: list[int] = []
    positive_ids: list[int] = []
    positive_static: list[np.ndarray] = []
    negative_static: list[np.ndarray] = []
    positive_history: list[np.ndarray] = []
    negative_history: list[np.ndarray] = []
    history_valid: list[bool] = []
    weights: list[float] = []
    for sample_index, row in enumerate(rows):
        for pair in row.pairs:
            positive = row.state.candidate_features[pair.positive_position]
            negative = row.state.candidate_features[pair.negative_position]
            sample_indices.append(sample_index)
            positive_ids.append(int(row.state.candidate_path_indices[pair.positive_position]))
            positive_static.append(positive[:128])
            negative_static.append(negative[:128])
            positive_history.append(positive[128:256])
            negative_history.append(negative[128:256])
            history_valid.append(bool(positive[256] > 0.5 and negative[256] > 0.5))
            weights.append(1.0)

    def tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

    return {
        "context_features": tensor(contexts, torch.float32),
        "context_valid_mask": tensor(valid, torch.bool),
        "time_features": tensor(times, torch.float32),
        "pair_sample_index": tensor(sample_indices, torch.long),
        "pair_positive_ids": tensor(positive_ids, torch.long),
        "pair_positive_layers": torch.zeros(len(sample_indices), dtype=torch.long, device=device),
        "positive_static": tensor(positive_static, torch.float32),
        "negative_static": tensor(negative_static, torch.float32),
        "positive_history": tensor(positive_history, torch.float32),
        "negative_history": tensor(negative_history, torch.float32),
        "history_valid": tensor(history_valid, torch.bool),
        "pair_weights": tensor(weights, torch.float32),
    }


def _move_anchor(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@dataclass(frozen=True)
class ActorUpdateMetrics:
    total: float
    advantage_total: float
    advantage_static: float
    advantage_history: float
    advantage_fusion: float
    anchor_total: float
    gradient_norm: float


class ActorRLTrainer:
    def __init__(
        self,
        actor: FolderCacheActor,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        gradient_clip_norm: float,
        lambda_anchor: float,
    ) -> None:
        self.actor = actor.to(device)
        self.device = device
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.lambda_anchor = float(lambda_anchor)
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    @staticmethod
    def _loss(actor: FolderCacheActor, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = actor(batch["context_features"], batch["context_valid_mask"], batch["time_features"])
        return compute_pairwise_losses(
            outputs,
            batch["pair_sample_index"],
            batch["pair_positive_ids"],
            batch["pair_positive_layers"],
            batch["positive_static"],
            batch["negative_static"],
            batch["positive_history"],
            batch["negative_history"],
            batch["history_valid"],
            batch["positive_history"],
            batch["negative_history"],
            batch["history_valid"],
            batch["pair_weights"],
        )

    def update(self, advantage_rows: Sequence[AdvantageMacro], anchor_batch: Mapping[str, torch.Tensor]) -> ActorUpdateMetrics:
        advantage = collate_advantage_macros(advantage_rows, self.device)
        anchor = _move_anchor(anchor_batch, self.device)
        self.actor.train()
        self.optimizer.zero_grad(set_to_none=True)
        advantage_losses = self._loss(self.actor, advantage)
        anchor_outputs = self.actor(anchor["context_features"], anchor["context_valid_mask"], anchor["time_features"])
        anchor_losses = compute_pairwise_losses(
            anchor_outputs,
            anchor["pair_sample_index"],
            anchor["pair_positive_ids"],
            anchor["pair_positive_layers"],
            anchor["positive_static"],
            anchor["negative_static"],
            anchor["positive_history_persistent"],
            anchor["negative_history_persistent"],
            anchor["persistent_valid"],
            anchor["positive_history_current"],
            anchor["negative_history_current"],
            anchor["current_valid"],
            anchor["pair_weights"],
        )
        total = advantage_losses["total"] + self.lambda_anchor * anchor_losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("Actor强化损失包含非有限值")
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm, error_if_nonfinite=True)
        self.optimizer.step()
        return ActorUpdateMetrics(
            float(total.detach().cpu()),
            float(advantage_losses["total"].detach().cpu()),
            float(advantage_losses["static"].detach().cpu()),
            float(advantage_losses["history"].detach().cpu()),
            float(advantage_losses["fusion"].detach().cpu()),
            float(anchor_losses["total"].detach().cpu()),
            float(torch.as_tensor(gradient).detach().cpu()),
        )


def train_actor_loop(
    trainer: ActorRLTrainer,
    train_rows: Sequence[AdvantageMacro],
    calibration_rows: Sequence[AdvantageMacro],
    anchor_iterator: Any,
    batch_macro_steps: int,
    max_updates: int,
    calibration_interval: int,
    patience: int,
    seed: int,
    output_dir: Path,
    checkpoint_builder: Callable[[int, Mapping[str, float]], dict[str, Any]],
    progress_interval_updates: int = 100,
    progress_interval_seconds: int = 60,
) -> dict[str, Any]:
    if not train_rows or not calibration_rows:
        raise ValueError("Actor强化训练和校准优势对均不能为空")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    loss_path = output_dir / "actor_loss_history.jsonl"
    plot_path = output_dir / "actor_loss_curves.png"
    checkpoint_path = output_dir / "actor_rl_best.pt"
    best_loss = float("inf")
    best_update = 0
    stale = 0
    started = time.monotonic()
    last_progress = started
    anchor = iter(anchor_iterator)
    for update in range(1, int(max_updates) + 1):
        positions = rng.integers(0, len(train_rows), size=int(batch_macro_steps))
        try:
            anchor_batch = next(anchor)
        except StopIteration:
            anchor = iter(anchor_iterator)
            anchor_batch = next(anchor)
        metrics = trainer.update([train_rows[int(position)] for position in positions], anchor_batch)
        now = time.monotonic()
        if update == 1 or update % int(progress_interval_updates) == 0 or now - last_progress >= int(progress_interval_seconds):
            record = {
                "event": "actor_rl_train_progress",
                "update": update,
                **metrics.__dict__,
                "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": now - started,
            }
            if trainer.device.type == "cuda":
                record["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(trainer.device) / 1024**2
                record["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(trainer.device) / 1024**2
            print(json.dumps(record, ensure_ascii=False), flush=True)
            last_progress = now
        if update % int(calibration_interval) != 0:
            continue
        trainer.actor.eval()
        with torch.no_grad():
            losses: list[float] = []
            for start in range(0, len(calibration_rows), int(batch_macro_steps)):
                batch = collate_advantage_macros(calibration_rows[start : start + int(batch_macro_steps)], trainer.device)
                losses.append(float(trainer._loss(trainer.actor, batch)["total"].cpu()))
        calibration_loss = float(np.mean(losses))
        row = {
            "schema_version": "folder-rl-actor-loss-history/v1",
            "update": update,
            **metrics.__dict__,
            "calibration_advantage_loss": calibration_loss,
            "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
        }
        append_jsonl(loss_path, row, reset=update == int(calibration_interval))
        plot_actor_history(loss_path, plot_path)
        print(json.dumps({"event": "actor_rl_calibration", **row}, ensure_ascii=False), flush=True)
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_update = update
            stale = 0
            value = checkpoint_builder(update, {"advantage_loss": calibration_loss})
            value.update(
                {
                    "model_state": trainer.actor.state_dict(),
                    "optimizer_state": trainer.optimizer.state_dict(),
                    "model_config": trainer.actor.config.to_dict(),
                }
            )
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
            torch.save(value, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_update == 0:
        raise RuntimeError("Actor强化训练没有产生检查点")
    return {"best_update": best_update, "best_calibration_advantage_loss": best_loss, "checkpoint": checkpoint_path.as_posix()}
