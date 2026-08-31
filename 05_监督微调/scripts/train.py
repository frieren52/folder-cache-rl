from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from folder_cache_actor import __version__  # noqa: E402
from folder_cache_actor.config import config_sha256, load_config, resolve_path  # noqa: E402
from folder_cache_actor.data import ActorParquetDataset, make_collate  # noqa: E402
from folder_cache_actor.errors import ActorError, ArtifactCompatibilityError, OutputExistsError  # noqa: E402
from folder_cache_actor.model import ActorConfig, FolderCacheActor, trainable_parameter_count  # noqa: E402
from folder_cache_actor.reporting import append_jsonl, plot_epoch_log, plot_progress_log, read_jsonl, save_epoch_log  # noqa: E402
from folder_cache_actor.training import run_epoch  # noqa: E402
from folder_cache_actor.utils import read_json, resolve_device, set_global_seed, sha256_file, validate_identifier, write_json  # noqa: E402
from folder_cache_actor.vector_store import StaticVectorStore  # noqa: E402


def _save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _checkpoint(
    model: FolderCacheActor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    data_version: str,
    data_manifest_sha: str,
    vector_manifest_sha: str,
) -> dict:
    return {
        "schema_version": "folder-cache-actor-checkpoint/v2",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "config": config,
        "config_sha256": config_sha256(config),
        "folder_cache_actor_version": __version__,
        "data_version": data_version,
        "data_manifest_sha256": data_manifest_sha,
        "vector_store_manifest_sha256": vector_manifest_sha,
        "input_contract": {"context_shape": [256, 259], "time_features": ["time_sin", "time_cos"]},
        "retrieval_contract": {"static_top_k": 256, "history_top_k": 256, "union_max": 512, "fusion_top_k": 256},
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练05监督Actor")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    data_version = validate_identifier(args.data_version, "data_version")
    run_id = validate_identifier(args.run_id, "run_id")
    data_dir = resolve_path(MODULE_ROOT, config["paths"]["dataset_root"]) / data_version
    data_manifest_path = data_dir / "actor_samples_manifest.json"
    data_manifest = read_json(data_manifest_path)
    if data_manifest.get("schema_version") != "actor-supervision/v3" or data_manifest.get("config_sha256") != config_sha256(config):
        raise ArtifactCompatibilityError("监督数据版本与当前配置不一致")
    data_manifest_sha = sha256_file(data_manifest_path)
    vector_store_dir = resolve_path(MODULE_ROOT, config["paths"]["vector_store_dir"])
    vector_manifest_sha = sha256_file(vector_store_dir / "vector_store_manifest.json")
    if data_manifest.get("vector_store_manifest_sha256") != vector_manifest_sha:
        raise ArtifactCompatibilityError("监督数据绑定的向量库与当前向量库不一致")
    static_store = StaticVectorStore.load(vector_store_dir)
    outputs_root = resolve_path(MODULE_ROOT, config["paths"]["outputs_root"])
    run_dir = outputs_root / "runs" / run_id
    checkpoints = run_dir / "checkpoints"
    resume = args.resume.resolve() if args.resume else None
    if resume is None:
        if run_dir.exists():
            raise OutputExistsError(f"run-id已存在，拒绝覆盖：{run_dir}")
        checkpoints.mkdir(parents=True)
    elif not resume.is_file() or not run_dir.is_dir():
        raise ArtifactCompatibilityError("恢复训练要求既有run目录及检查点")
    training = config["training"]
    seed = int(training["seed"])
    set_global_seed(seed)
    device = resolve_device(str(training["device"]))
    model = FolderCacheActor(ActorConfig.from_mapping(config["model"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    start_epoch = 1
    last_checkpoint: dict | None = None
    if resume is not None:
        value = torch.load(resume, map_location=device, weights_only=False)
        if value.get("schema_version") != "folder-cache-actor-checkpoint/v2":
            raise ArtifactCompatibilityError("不支持的恢复检查点")
        if value.get("config_sha256") != config_sha256(config) or value.get("data_manifest_sha256") != data_manifest_sha:
            raise ArtifactCompatibilityError("恢复检查点与配置或数据版本不一致")
        model.load_state_dict(value["model_state"], strict=True)
        optimizer.load_state_dict(value["optimizer_state"])
        random.setstate(value["python_random_state"])
        np.random.set_state(value["numpy_random_state"])
        torch.set_rng_state(value["torch_random_state"])
        if torch.cuda.is_available() and value.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(value["cuda_random_state"])
        start_epoch = int(value["epoch"]) + 1
        last_checkpoint = value
    train_data = ActorParquetDataset(data_dir / "actor_samples.parquet", data_dir / "actor_history_snapshots.parquet", "train", int(config["sampling"]["shuffle_buffer_size"]), seed)
    collate = make_collate(static_store)
    batch_size = int(training["batch_size"])
    workers = int(training["dataloader_workers"])
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers, collate_fn=collate, pin_memory=device.type == "cuda")
    loss_log = run_dir / "loss_history.jsonl"
    loss_plot = run_dir / "loss_curves.png"
    progress_log = run_dir / "training_progress.jsonl"
    progress_plot = run_dir / "live_training_metrics.png"
    progress_initialized = resume is not None and progress_log.exists()

    def record_progress(record: dict) -> None:
        nonlocal progress_initialized
        append_jsonl(progress_log, {"schema_version": "folder-cache-actor-progress/v1", **record}, reset=not progress_initialized)
        progress_initialized = True
        plot_progress_log(progress_log, progress_plot)

    train_batches = math.ceil(int(data_manifest["split_rows"]["train"]) / batch_size)
    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        train_data.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, epoch, optimizer, float(training["gradient_clip_norm"]), train_batches,
            int(training["progress_interval_batches"]), int(training["progress_interval_seconds"]), record_progress,
        )
        save_epoch_log(loss_log, epoch, train_metrics, float(optimizer.param_groups[0]["lr"]), reset=resume is None and epoch == 1)
        plot_epoch_log(loss_log, loss_plot)
        last_checkpoint = _checkpoint(model, optimizer, epoch, config, data_version, data_manifest_sha, vector_manifest_sha)
        _save_checkpoint(checkpoints / "actor_supervised_last.pt", last_checkpoint)
        print(json.dumps({"event": "epoch_complete", "epoch": epoch, "train": train_metrics}, ensure_ascii=False), flush=True)
    if last_checkpoint is None:
        raise ArtifactCompatibilityError("没有可发布的末轮检查点")
    final_epoch = int(last_checkpoint["epoch"])
    if final_epoch != int(training["max_epochs"]):
        raise ArtifactCompatibilityError(f"固定训练尚未完成：当前epoch={final_epoch}，要求={training['max_epochs']}")
    _save_checkpoint(checkpoints / "actor_supervised_final.pt", last_checkpoint)
    final_train = read_jsonl(loss_log)[-1]["train"]
    write_json(run_dir / "run_meta.json", {"schema_version": "folder-cache-actor-run/v2", "run_id": run_id, "data_version": data_version, "trainable_parameters": trainable_parameter_count(model), "device": str(device), "final_epoch": final_epoch, "final_train_total": float(final_train["total"]), "test_data_used_during_training": False, "loss_history_sha256": sha256_file(loss_log), "loss_curves_sha256": sha256_file(loss_plot), "progress_log_sha256": sha256_file(progress_log), "progress_plot_sha256": sha256_file(progress_plot), "final_checkpoint_sha256": sha256_file(checkpoints / "actor_supervised_final.pt")})
    print(json.dumps({"run_dir": run_dir.as_posix(), "final_epoch": final_epoch, "final_train_total": float(final_train["total"]), "test_data_used_during_training": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except ActorError as exc:
        print(json.dumps({"status": "failed", "stage": "train", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"status": "failed", "stage": "train", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
