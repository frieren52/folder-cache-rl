from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from folder_cache_actor.errors import ArtifactCompatibilityError
from folder_cache_actor.inference import SupervisedActor
from folder_cache_actor.model import ActorConfig, FolderCacheActor, trainable_parameter_count


class ActorModelTests(unittest.TestCase):
    def test_output_contract_and_all_padding(self) -> None:
        model = FolderCacheActor(ActorConfig(dropout=0.0)).eval()
        features = torch.randn(2, 256, 259)
        valid = torch.ones(2, 256, dtype=torch.bool)
        valid[1] = False
        output = model(features, valid, torch.zeros(2, 2))
        self.assertEqual(output["static_query"].shape, (2, 128))
        self.assertTrue(torch.allclose(output["static_query"].norm(dim=1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(output["history_query"].norm(dim=1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(output["fusion_weights"].sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_permutation_invariance(self) -> None:
        model = FolderCacheActor(ActorConfig(dropout=0.0)).eval()
        features = torch.randn(1, 256, 259)
        valid = torch.zeros(1, 256, dtype=torch.bool)
        valid[:, :20] = True
        permutation = torch.randperm(256)
        first = model(features, valid, torch.randn(1, 2))
        second = model(features[:, permutation], valid[:, permutation], torch.randn(1, 2))
        # Use the same time feature for the actual comparison.
        times = torch.tensor([[0.25, -0.75]])
        first = model(features, valid, times)
        second = model(features[:, permutation], valid[:, permutation], times)
        for name in first:
            self.assertTrue(torch.allclose(first[name], second[name], atol=1e-5), name)

    def test_parameter_counts(self) -> None:
        self.assertEqual(trainable_parameter_count(FolderCacheActor(ActorConfig(pooling_mode="dot_product"))), 2_308_354)
        self.assertEqual(trainable_parameter_count(FolderCacheActor(ActorConfig(pooling_mode="mha"))), 2_571_522)

    def test_inference_accepts_only_v2_checkpoint(self) -> None:
        config = ActorConfig(dropout=0.0)
        model = FolderCacheActor(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor.pt"
            torch.save({"schema_version": "folder-cache-actor-checkpoint/v2", "model_config": config.to_dict(), "model_state": model.state_dict()}, checkpoint)
            loaded = SupervisedActor.load(checkpoint, device="cpu")
            self.assertEqual(loaded.metadata["schema_version"], "folder-cache-actor-checkpoint/v2")
            torch.save({"schema_version": "folder-cache-actor-checkpoint/v1", "model_config": config.to_dict(), "model_state": model.state_dict()}, checkpoint)
            with self.assertRaises(ArtifactCompatibilityError):
                SupervisedActor.load(checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
