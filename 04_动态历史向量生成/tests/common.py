from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


SHANGHAI = ZoneInfo("Asia/Shanghai")
MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent


def small_config(sample_start: str = "2026-01-01T00:00:20+08:00") -> dict:
    with (MODULE_ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as stream:
        config = deepcopy(yaml.safe_load(stream))
    config["history"].update(
        {
            "sample_start": sample_start,
            "snapshot_stride_seconds": 2,
            "interval_clip_seconds": 20,
            "scales": [
                {"name": "second", "window_seconds": 4, "bucket_seconds": 1, "bucket_count": 4},
                {"name": "short", "window_seconds": 8, "bucket_seconds": 2, "bucket_count": 4},
                {"name": "medium", "window_seconds": 12, "bucket_seconds": 3, "bucket_count": 4},
                {"name": "long", "window_seconds": 20, "bucket_seconds": 5, "bucket_count": 4},
            ],
        }
    )
    config["model"].update(
        {
            "vector_dim": 16,
            "state_input_dim": 4,
            "tcn_channels": 16,
            "tcn_dilations": [1, 2],
            "transformer_layers": 1,
            "transformer_heads": 4,
            "transformer_ffn_dim": 32,
            "state_hidden_dim": 8,
            "fusion_hidden_dim": 24,
            "count_hidden_dim": 8,
            "dropout": 0.0,
        }
    )
    config["target"] = {
        "horizon_seconds": 10,
        "time_boundaries_seconds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    }
    config["sampling"].update(
        {
            "high_history_per_snapshot": 1,
            "medium_history_per_snapshot": 1,
            "low_history_per_snapshot": 1,
            "single_history_per_snapshot": 1,
            "no_history_per_snapshot": 1,
        }
    )
    config["storage"].update(
        {"max_rows_per_shard": 10, "max_rows_per_row_group": 5}
    )
    config["training"].update(
        {
            "device": "cpu",
            "precision": "fp32",
            "distributed": False,
            "dataloader_workers": 0,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "progress_interval_batches": 1,
            "progress_interval_seconds": 60,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "warmup_ratio": 0.1,
        }
    )
    config["evaluation"].update(
        {
            "baseline_window_seconds": 6,
            "min_brier_relative_improvement": -100.0,
            "min_count_mae_relative_improvement": -100.0,
            "max_nll_relative_increase": 100.0,
        }
    )
    return config


def write_small_raw(root: Path) -> tuple[Path, Path]:
    catalog_path = root / "path_catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path_index", "path", "access_count", "total_size_bytes"))
        for path_index in range(10):
            writer.writerow((path_index, f"/path/{path_index}", 0, 1))
    access_dir = root / "raw"
    access_dir.mkdir()
    start = datetime(2026, 1, 1, tzinfo=SHANGHAI)
    events: list[tuple[datetime, int]] = []
    for second in range(81):
        moment = start + timedelta(seconds=second)
        if second % 2 == 0:
            events.extend(((moment, 0), (moment, 0)))
        if second % 5 == 0:
            events.append((moment, 1))
        if second == 10:
            events.append((moment, 2))
        if second % 7 == 0:
            events.append((moment, 3))
        if second in (12, 13, 44):
            events.append((moment, 4))
    events.sort(key=lambda item: (item[0], item[1]))
    access_path = access_dir / "access_20260101.txt"
    with access_path.open("w", encoding="utf-8", newline="\n") as stream:
        for moment, path_index in events:
            stream.write(f"{path_index} {moment:%Y-%m-%d %H:%M:%S}\n")
    return catalog_path, access_dir
