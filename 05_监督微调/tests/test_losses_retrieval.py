from __future__ import annotations

import unittest

import numpy as np
import torch

from folder_cache_actor.losses import compute_pairwise_losses
from folder_cache_actor.retrieval import ExactDualRetriever, stable_top_k


class LossTests(unittest.TestCase):
    def test_three_losses_are_finite_and_backward(self) -> None:
        query = torch.randn(2, 128, requires_grad=True)
        query = query / query.norm(dim=1, keepdim=True)
        outputs = {
            "static_query": query,
            "history_query": query,
            "fusion_weights": torch.softmax(torch.randn(2, 2, requires_grad=True), dim=1),
        }
        pair_sample = torch.tensor([0, 0, 1, 1])
        positive_ids = torch.tensor([10, 10, 20, 20])
        layers = torch.tensor([0, 0, 3, 3])
        positive = torch.randn(4, 128)
        negative = torch.randn(4, 128)
        losses = compute_pairwise_losses(
            outputs, pair_sample, positive_ids, layers,
            positive, negative, positive, negative, torch.tensor([True, True, False, False]),
            positive, negative, torch.ones(4, dtype=torch.bool), torch.ones(4),
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()


class RetrievalTests(unittest.TestCase):
    def test_stable_tie_break_and_union(self) -> None:
        ids, _ = stable_top_k(np.asarray([3, 1, 2]), np.ones(3), 2)
        self.assertEqual(ids.tolist(), [1, 2])
        static_ids = np.arange(300, dtype=np.int64)
        static = np.eye(300, 128, dtype=np.float32)
        retriever = ExactDualRetriever(static_ids, static, 2, 2, 2)
        result = retriever.retrieve(
            np.ones(128, dtype=np.float32),
            np.ones(128, dtype=np.float32),
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([298, 299], dtype=np.int64),
            np.ones((2, 128), dtype=np.float32),
        )
        self.assertLessEqual(len(result.union_ids), 4)


if __name__ == "__main__":
    unittest.main()

