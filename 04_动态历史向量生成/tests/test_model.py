from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import small_config  # noqa: E402
from src.config import validate_config  # noqa: E402
from src.errors import ConfigError  # noqa: E402
from src.losses import compute_losses  # noqa: E402
from src.model import DynamicHistoryModel, ScaleTCN  # noqa: E402
from src.reporting import (  # noqa: E402
    plot_loss_log,
    plot_progress_log,
    save_loss_log,
    save_progress_log,
)


class ModelTests(unittest.TestCase):
    def test_eight_class_contract_is_rejected(self) -> None:
        config = deepcopy(small_config())
        config["target"]["time_boundaries_seconds"] = [0, 1, 2, 3, 4, 5, 7, 10]
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_shapes_probabilities_and_losses(self) -> None:
        config = small_config()
        model = DynamicHistoryModel(config).eval()
        counts = [torch.zeros(2, 4, 1) for _ in range(4)]
        outputs = model(*counts, torch.zeros(2, 4))
        self.assertEqual(outputs["vectors"].shape, (2, 16))
        self.assertEqual(outputs["time_logits"].shape, (2, 9))
        self.assertEqual(outputs["next_access_probs"].shape, (2, 10))
        torch.testing.assert_close(outputs["next_access_probs"].sum(dim=1), torch.ones(2))
        torch.testing.assert_close(torch.linalg.vector_norm(outputs["vectors"], dim=1), torch.ones(2))
        losses = compute_losses(
            outputs,
            torch.zeros(2, dtype=torch.int64),
            torch.full((2,), -1, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            config["loss"],
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["time"]), 0.0)

    def test_tcn_does_not_leak_right_context(self) -> None:
        torch.manual_seed(7)
        model = ScaleTCN(8, 3, [1, 2], 0.0).eval()
        first = torch.zeros(1, 12, 1)
        second = first.clone()
        second[:, -1] = 10.0
        with torch.no_grad():
            first_output = model(first)
            second_output = model(second)
        torch.testing.assert_close(first_output[:, :-1], second_output[:, :-1])

    def test_loss_log_and_plot(self) -> None:
        metrics = {"total": 1.0, "access": 0.4, "time": 0.3, "count": 0.3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "loss_history.jsonl"
            image = root / "loss_curves.png"
            save_loss_log(log, 1, metrics, metrics, 1e-4, reset=True)
            plot_loss_log(log, image)
            self.assertGreater(image.stat().st_size, 0)

            progress_log = root / "training_progress.jsonl"
            progress_image = root / "live_training_metrics.png"
            save_progress_log(
                progress_log,
                {
                    "event": "train_progress",
                    "epoch": 1,
                    "progress": 0.5,
                    "interval_samples_per_second": 100.0,
                    "loss_running": metrics,
                    "loss_interval": metrics,
                },
                reset=True,
            )
            plot_progress_log(progress_log, progress_image)
            self.assertGreater(progress_image.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
