from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from folder_cache_actor.data import ActorParquetDataset, make_collate
from folder_cache_actor.model import ActorConfig, FolderCacheActor
from folder_cache_actor.sample_builder import _sample_schema, _snapshot_schema
from folder_cache_actor.training import run_epoch
from folder_cache_actor.vector_store import StaticVectorStore


class DataTrainingTests(unittest.TestCase):
    def test_parquet_to_training_step_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {
                "sample_id": "sample-1",
                "cutoff_time_seconds": 1_782_316_800,
                "split": "train",
                "history_revision": 1,
                "context_folder_ids": [1, 2] + [-1] * 254,
                "context_valid_mask": [True, True] + [False] * 254,
                "context_recent_mask": [True, False] + [False] * 254,
                "context_hot_mask": [True, True] + [False] * 254,
                "positive_folder_ids_all": [1],
                "pair_positive_ids": [1],
                "pair_negative_ids": [2],
                "pair_positive_layers": [0],
                "pair_weights": [1.0],
                "pair_negative_sources": [0],
                "negative_pool_sizes": [1, 0, 0, 0],
                "updated_ids": [1, 2],
                "removed_ids": [],
            }
            vector = np.zeros(128, dtype=np.float16)
            vector[0] = 1.0
            snapshot = {
                "sample_id": "sample-1",
                "persistent_folder_ids": [1, 2],
                "persistent_history_vectors": np.concatenate((vector, vector)).tobytes(),
                "persistent_history_valid": [True, True],
                "current_folder_ids": [1, 2],
                "current_history_vectors": np.concatenate((vector, vector)).tobytes(),
                "current_history_valid": [True, True],
                "candidate_as_of_time": 1_782_316_800,
            }
            pq.write_table(pa.Table.from_pylist([sample], schema=_sample_schema()), root / "samples.parquet")
            pq.write_table(pa.Table.from_pylist([snapshot], schema=_snapshot_schema()), root / "snapshots.parquet")
            static = np.zeros((2, 128), dtype=np.float32)
            static[0, 0] = 1.0
            static[1, 1] = 1.0
            store = StaticVectorStore(np.asarray([1, 2]), static, {1: 0, 2: 1}, {})
            dataset = ActorParquetDataset(root / "samples.parquet", root / "snapshots.parquet", "train", 1, 7)
            loader = DataLoader(dataset, batch_size=1, collate_fn=make_collate(store))
            model = FolderCacheActor(ActorConfig(transformer_layers=1, dropout=0.0))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            progress: list[dict] = []
            metrics = run_epoch(model, loader, torch.device("cpu"), 1, optimizer, 1.0, 1, 1, 60, progress.append)
            self.assertTrue(np.isfinite(metrics["total"]))
            self.assertEqual(len(progress), 1)


if __name__ == "__main__":
    unittest.main()
