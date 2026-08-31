from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from folder_cache_actor.model import ActorConfig, FolderCacheActor

from common import MODULE_ROOT, state  # noqa: F401
from src.critic import CriticConfig, TwinCritic
from src.replay import MacroStepRecord, ReplayReader, ReplaySampler, ReplayWriter, Transition, collate_transitions
from src.training import CriticTrainer


class CriticTests(unittest.TestCase):
    def _model(self) -> TwinCritic:
        actor = FolderCacheActor(
            ActorConfig(
                input_dim=259,
                vector_dim=128,
                hidden_dim=256,
                transformer_layers=1,
                transformer_heads=4,
                transformer_ffn_dim=128,
                dropout=0.0,
            )
        )
        return TwinCritic(actor, CriticConfig())

    def test_twin_critic_shapes_and_independent_heads(self) -> None:
        model = self._model().eval()
        args = (
            torch.zeros(1, 256, 259),
            torch.ones(1, 256, dtype=torch.bool),
            torch.zeros(1, 2),
            torch.zeros(1, 10),
            torch.zeros(1, 3, 274),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, dtype=torch.bool),
        )
        first, second = model.online(*args)
        self.assertEqual(first.candidate_q_values.shape, (1, 3))
        self.assertEqual(first.stop_q_value.shape, (1, 1))
        self.assertFalse(torch.equal(model.q1.candidate_head[0].weight, model.q2.candidate_head[0].weight))
        self.assertEqual(second.candidate_q_values.shape, (1, 3))

    def test_single_update_is_finite_and_updates_target(self) -> None:
        actor = FolderCacheActor(
            ActorConfig(
                input_dim=259,
                vector_dim=128,
                hidden_dim=16,
                transformer_layers=1,
                transformer_heads=4,
                transformer_ffn_dim=32,
                dropout=0.0,
            )
        )
        config = CriticConfig(
            actor_state_dim=16,
            resource_hidden_dim=8,
            critic_state_dim=16,
            candidate_hidden_dim=16,
            candidate_embedding_dim=8,
            head_hidden_dim=16,
            head_bottleneck_dim=8,
        )
        model = TwinCritic(actor, config)
        before = model.q1_target.stop_head[-1].weight.detach().clone()
        trainer = CriticTrainer(model, torch.device("cpu"), 1e-3, 0.0, 1.0, 1.0, 0.5)
        current = state(2)
        following = state(2, selected=(0,))
        transitions = [
            Transition(current, 0, 0.0, 1.0, following, False, 0),
            Transition(following, -1, 1.0, 0.0, following, True, 0),
        ]
        metrics = trainer.update(transitions)
        self.assertTrue(np.isfinite(metrics.loss))
        self.assertFalse(torch.equal(before, model.q1_target.stop_head[-1].weight))


class ReplayTests(unittest.TestCase):
    def _record(self, step: int, reward: float, terminal: bool = False) -> MacroStepRecord:
        return MacroStepRecord(
            "run",
            step,
            step * 10,
            "train_fit",
            "simple_greedy",
            (state(2), state(2, selected=(0,))),
            np.asarray([0, -1], dtype=np.int64),
            reward,
            0,
            0,
            terminal,
            "actor",
            None,
        )

    def test_roundtrip_sampler_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "replay"
            writer = ReplayWriter(root, "run", "config", rows_per_shard=2)
            writer.append(self._record(0, 1.0))
            writer.append(self._record(1, 0.0, terminal=True))
            writer.close()
            reader = ReplayReader(root)
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader.record_at(0).reward, 1.0)
            self.assertEqual(reader.record_at(0).reward, 1.0)
            sampler = ReplaySampler(reader, "train_fit", 2026, 0.9, n_step=2)
            transitions = sampler.sample_macro_batch(2)
            self.assertEqual(len(transitions), 4)
            self.assertAlmostEqual(transitions[0].sample_weight, transitions[1].sample_weight)
            batch = collate_transitions(transitions)
            self.assertEqual(batch["action_position"].ndim, 1)
            self.assertEqual(batch["state_candidate_features"].shape[-1], 274)


if __name__ == "__main__":
    unittest.main()
