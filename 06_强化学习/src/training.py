from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .critic import CriticOutput, TwinCritic, min_q
from .replay import ReplaySampler, Transition, collate_transitions
from .reporting import append_jsonl, plot_critic_history


STATE_NAMES = (
    "object_features",
    "object_valid_mask",
    "time_features",
    "resource_features",
    "candidate_features",
    "candidate_valid_mask",
    "candidate_action_mask",
    "selected_candidate_mask",
)


def _tensor_batch(values: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if value.dtype == np.bool_:
            dtype = torch.bool
        elif np.issubdtype(value.dtype, np.integer):
            dtype = torch.long
        else:
            dtype = torch.float32
        result[name] = torch.as_tensor(value, dtype=dtype, device=device)
    return result


def _critic_args(batch: Mapping[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, ...]:
    return tuple(batch[f"{prefix}{name}"] for name in STATE_NAMES if name != "candidate_action_mask")


def _executed_q(output: CriticOutput, action_position: torch.Tensor) -> torch.Tensor:
    candidate_action = action_position >= 0
    safe_position = action_position.clamp_min(0)
    candidate = output.candidate_q_values.gather(1, safe_position[:, None]).squeeze(1)
    return torch.where(candidate_action, candidate, output.stop_q_value.squeeze(1))


@torch.no_grad()
def _next_target_value(model: TwinCritic, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    model.q1.eval()
    model.q2.eval()
    online = min_q(*model.online(*_critic_args(batch, "next_")))
    target_first, target_second = model.target(*_critic_args(batch, "next_"))
    target = min_q(target_first, target_second)
    legal = batch["next_candidate_valid_mask"] & batch["next_candidate_action_mask"] & ~batch["next_selected_candidate_mask"]
    masked_online = online.candidate_q_values.masked_fill(~legal, torch.finfo(online.candidate_q_values.dtype).min)
    candidate_value, candidate_position = masked_online.max(dim=1)
    choose_candidate = legal.any(dim=1) & (candidate_value > online.stop_q_value.squeeze(1))
    target_candidate = target.candidate_q_values.gather(1, candidate_position[:, None]).squeeze(1)
    return torch.where(choose_candidate, target_candidate, target.stop_q_value.squeeze(1))


@dataclass(frozen=True)
class UpdateMetrics:
    loss: float
    q_mean: float
    target_mean: float
    td_abs_mean: float
    gradient_norm: float


class CriticTrainer:
    def __init__(
        self,
        model: TwinCritic,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        huber_delta: float,
        gradient_clip_norm: float,
        polyak_tau: float,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.huber_delta = float(huber_delta)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.polyak_tau = float(polyak_tau)
        self.optimizer = torch.optim.AdamW(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    def update(self, transitions: Sequence[Transition]) -> UpdateMetrics:
        batch = _tensor_batch(collate_transitions(transitions), self.device)
        self.model.q1.train()
        self.model.q2.train()
        self.optimizer.zero_grad(set_to_none=True)
        first, second = self.model.online(*_critic_args(batch, "state_"))
        q1 = _executed_q(first, batch["action_position"])
        q2 = _executed_q(second, batch["action_position"])
        with torch.no_grad():
            next_value = _next_target_value(self.model, batch)
            target = batch["reward"] + batch["discount"] * next_value
        first_loss = F.huber_loss(q1, target, reduction="none", delta=self.huber_delta)
        second_loss = F.huber_loss(q2, target, reduction="none", delta=self.huber_delta)
        per_item = 0.5 * (first_loss + second_loss)
        weights = batch["sample_weight"] / batch["sample_weight"].mean().clamp_min(1e-12)
        loss = (per_item * weights).mean()
        if not torch.isfinite(loss) or not torch.isfinite(target).all():
            raise FloatingPointError("Critic loss或TD目标包含非有限值")
        loss.backward()
        parameters = list(self.model.q1.parameters()) + list(self.model.q2.parameters())
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.model.soft_update(self.polyak_tau)
        q_min = torch.minimum(q1, q2)
        return UpdateMetrics(
            float(loss.detach().cpu()),
            float(q_min.mean().detach().cpu()),
            float(target.mean().detach().cpu()),
            float((q_min - target).abs().mean().detach().cpu()),
            float(torch.as_tensor(gradient_norm).detach().cpu()),
        )

    @torch.no_grad()
    def calibration(self, transitions: Sequence[Transition], batch_size: int = 128) -> dict[str, float]:
        self.model.q1.eval()
        self.model.q2.eval()
        collected: list[np.ndarray] = []
        for start in range(0, len(transitions), int(batch_size)):
            values = collate_transitions(transitions[start : start + int(batch_size)])
            batch = _tensor_batch(values, self.device)
            output = min_q(*self.model.online(*_critic_args(batch, "state_")))
            predicted = _executed_q(output, batch["action_position"])
            returns = batch["return_value"]
            valid = torch.isfinite(returns)
            if torch.any(valid):
                collected.append((predicted[valid] - returns[valid]).abs().cpu().numpy())
        if not collected:
            raise ValueError("校准transition缺少固定回报")
        errors = np.concatenate(collected)
        return {"mae": float(np.mean(errors)), "p90": float(np.percentile(errors, 90))}


def train_critic_loop(
    trainer: CriticTrainer,
    sampler: ReplaySampler,
    calibration_transitions: Sequence[Transition],
    max_updates: int,
    batch_macro_steps: int,
    calibration_interval: int,
    patience: int,
    output_dir: Path,
    checkpoint_builder: Callable[[int, Mapping[str, float]], dict[str, Any]],
    progress_interval_updates: int = 100,
    progress_interval_seconds: int = 60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.jsonl"
    plot_path = output_dir / "loss_curves.png"
    checkpoint_path = output_dir / "critic_best.pt"
    best_mae = float("inf")
    best_update = 0
    stale = 0
    started = time.monotonic()
    last_progress = started
    latest: UpdateMetrics | None = None
    for update in range(1, int(max_updates) + 1):
        latest = trainer.update(sampler.sample_macro_batch(batch_macro_steps))
        now = time.monotonic()
        if update == 1 or update % int(progress_interval_updates) == 0 or now - last_progress >= int(progress_interval_seconds):
            record = {
                "event": "critic_train_progress",
                "update": update,
                **latest.__dict__,
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
        calibration = trainer.calibration(calibration_transitions)
        row = {
            "schema_version": "folder-rl-critic-loss-history/v1",
            "update": update,
            **latest.__dict__,
            "calibration_mae": calibration["mae"],
            "calibration_p90": calibration["p90"],
            "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
        }
        append_jsonl(history_path, row, reset=update == int(calibration_interval))
        plot_critic_history(history_path, plot_path)
        print(json.dumps({"event": "critic_calibration", **row}, ensure_ascii=False), flush=True)
        if calibration["mae"] < best_mae:
            best_mae = calibration["mae"]
            best_update = update
            stale = 0
            value = checkpoint_builder(update, calibration)
            value.update(
                {
                    "q1_state": trainer.model.q1.state_dict(),
                    "q2_state": trainer.model.q2.state_dict(),
                    "q1_target_state": trainer.model.q1_target.state_dict(),
                    "q2_target_state": trainer.model.q2_target.state_dict(),
                    "optimizer_state": trainer.optimizer.state_dict(),
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
        raise RuntimeError("Critic训练没有产生校准检查点")
    return {"best_update": best_update, "best_calibration_mae": best_mae, "checkpoint": checkpoint_path.as_posix()}
