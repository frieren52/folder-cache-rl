from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from folder_cache_actor.config import load_config, validate_config
from folder_cache_actor.errors import ConfigError
from folder_cache_actor.reporting import append_jsonl, plot_progress_log, save_epoch_log, plot_epoch_log


MODULE_ROOT = Path(__file__).resolve().parents[1]


class ConfigReportingTests(unittest.TestCase):
    def test_default_config_and_plots(self) -> None:
        config = load_config(MODULE_ROOT / "config" / "config.yaml")
        self.assertEqual(config["training"]["progress_interval_seconds"], 60)
        self.assertEqual(config["training"]["max_epochs"], 20)
        self.assertNotIn("validation_start", config["time"])
        self.assertNotIn("validation_end", config["time"])
        self.assertFalse({"early_stopping", "early_stopping_patience", "patience"} & set(config["training"]))
        metrics = {"total": 3.0, "static": 1.0, "history": 1.0, "fusion": 1.0, "static_accuracy": 0.6, "history_accuracy": 0.5, "static_weight": 0.4}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_epoch_log(root / "loss.jsonl", 1, metrics, 1e-4, reset=True)
            plot_epoch_log(root / "loss.jsonl", root / "loss.png")
            append_jsonl(root / "progress.jsonl", {"event": "train_progress", "epoch": 1, "progress": 0.5, "loss_running": {"total": 3.0, "static": 1.0, "history": 1.0, "fusion": 1.0}, "loss_interval": {"total": 3.0, "static": 1.0, "history": 1.0, "fusion": 1.0}, "interval_pairs_per_second": 10.0}, reset=True)
            plot_progress_log(root / "progress.jsonl", root / "progress.png")
            self.assertTrue((root / "loss.png").is_file())
            self.assertTrue((root / "progress.png").is_file())

    def test_rejects_validation_and_early_stopping_configuration(self) -> None:
        config = load_config(MODULE_ROOT / "config" / "config.yaml")
        validation_config = copy.deepcopy(config)
        validation_config["time"]["validation_start"] = validation_config["time"]["test_start"]
        with self.assertRaises(ConfigError):
            validate_config(validation_config)
        early_stopping_config = copy.deepcopy(config)
        early_stopping_config["training"]["early_stopping_patience"] = 3
        with self.assertRaises(ConfigError):
            validate_config(early_stopping_config)


if __name__ == "__main__":
    unittest.main()
