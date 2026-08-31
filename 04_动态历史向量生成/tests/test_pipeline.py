from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import small_config, write_small_raw  # noqa: E402
from src.data import PreparedParquetDataset  # noqa: E402
from src.inference import DynamicHistoryEncoder  # noqa: E402
from src.preparation import prepare_dataset  # noqa: E402
from src.training import make_loader  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_small_data_train_evaluate_and_inference(self) -> None:
        with tempfile.TemporaryDirectory(dir=MODULE_ROOT / "data") as directory:
            root = Path(directory)
            catalog, access_dir = write_small_raw(root)
            config = small_config()
            config_path = root / "config.yaml"
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
            data_root = root / "prepared"
            dataset_dir = prepare_dataset(
                config, "tiny-dataset", catalog, access_dir, data_root
            )
            metadata = json.loads((dataset_dir / "data_meta.json").read_text(encoding="utf-8"))
            report = json.loads((dataset_dir / "data_report.json").read_text(encoding="utf-8"))
            self.assertGreater(metadata["samples"]["train"], 0)
            self.assertGreater(metadata["samples"]["validation"], 0)
            self.assertEqual(
                len(
                    report["splits"]["train"]["overall"]["labels"][
                        "time_bucket_counts"
                    ]
                ),
                9,
            )
            prepared_train = PreparedParquetDataset(
                dataset_dir / "train",
                config,
                metadata["samples"]["train"],
                shuffle=False,
                seed=7,
            )
            prepared_loader = make_loader(prepared_train, batch_size=4, workers=0)
            prepared_batches = list(prepared_loader)
            self.assertEqual(len(prepared_batches), len(prepared_loader))
            self.assertEqual(
                sum(int(batch["y_access"].numel()) for batch in prepared_batches),
                metadata["samples"]["train"],
            )
            self.assertEqual(prepared_batches[0]["second_counts"].shape[1:], (4, 1))
            outputs_root = root / "outputs"
            train_result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_ROOT / "scripts" / "train.py"),
                    "--config",
                    str(config_path),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--run-id",
                    "tiny-run",
                    "--outputs-root",
                    str(outputs_root),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"event": "train_progress"', train_result.stdout)
            self.assertIn('"event": "validation_progress"', train_result.stdout)
            run_dir = outputs_root / "runs" / "tiny-run"
            self.assertTrue((run_dir / "loss_history.jsonl").is_file())
            self.assertGreater((run_dir / "loss_curves.png").stat().st_size, 0)
            self.assertTrue((run_dir / "training_progress.jsonl").is_file())
            self.assertGreater((run_dir / "live_training_metrics.png").stat().st_size, 0)
            subprocess.run(
                [
                    sys.executable,
                    str(MODULE_ROOT / "scripts" / "evaluate.py"),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--run-dir",
                    str(run_dir),
                    "--model-id",
                    "tiny-model",
                    "--outputs-root",
                    str(outputs_root),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
            )
            release = outputs_root / "releases" / "tiny-model"
            self.assertTrue((release / "loss_history.jsonl").is_file())
            self.assertTrue((release / "loss_curves.png").is_file())
            encoder = DynamicHistoryEncoder.load(release, device="cpu")
            shanghai = ZoneInfo("Asia/Shanghai")
            snapshot = datetime(2026, 1, 1, 0, 1, tzinfo=shanghai)
            result = encoder.encode(
                [
                    {
                        "path_index": 7,
                        "snapshot_time": snapshot,
                        "access_times": [snapshot - timedelta(seconds=4)],
                    },
                    {
                        "path_index": 8,
                        "snapshot_time": snapshot,
                        "access_times": [],
                    },
                ]
            )
            self.assertEqual(result["vectors"].shape, (2, 16))
            self.assertEqual(result["next_access_probs"].shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
