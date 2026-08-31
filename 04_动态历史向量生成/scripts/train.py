"""Train the dynamic history encoder and keep resumable run artifacts."""

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
import yaml


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from src.config import config_sha256, data_config_sha256, load_config  # noqa: E402
from src.data import PreparedParquetDataset, load_and_verify_dataset  # noqa: E402
from src.errors import (  # noqa: E402
    ArtifactCompatibilityError,
    DynamicHistoryError,
    OutputExistsError,
)
from src.model import DynamicHistoryModel, trainable_parameter_count  # noqa: E402
from src.reporting import (  # noqa: E402
    plot_loss_log,
    plot_progress_log,
    save_loss_log,
    save_progress_log,
)
from src.training import (  # noqa: E402
    make_loader,
    make_scheduler,
    run_training_epoch,
    run_validation_epoch,
)
from src.utils import (  # noqa: E402
    resolve_device,
    resolve_precision,
    set_global_seed,
    sha256_file,
    validate_identifier,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练动态历史编码模型")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--outputs-root", type=Path, default=MODULE_ROOT / "outputs")
    return parser.parse_args()


def _checkpoint_value(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_epoch: int,
    best_loss: float,
    stale_epochs: int,
    config: dict,
    dataset_id: str,
    dataset_manifest_sha: str,
) -> dict:
    return {
        "schema_version": "dynamic-history-checkpoint/v1",
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_total": best_loss,
        "stale_epochs": stale_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config,
        "config_sha256": config_sha256(config),
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": dataset_manifest_sha,
    }


def _save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ArtifactCompatibilityError("--max-epochs 必须大于0")
        config["training"]["max_epochs"] = args.max_epochs
    if bool(config["training"]["distributed"]):
        raise ArtifactCompatibilityError("首版训练入口尚未开放 distributed=true")
    run_id = validate_identifier(args.run_id, "run_id")
    dataset_dir = args.dataset_dir.resolve()
    metadata, dataset_manifest = load_and_verify_dataset(dataset_dir)
    expected_data_config_sha = data_config_sha256(config)
    if metadata.get("data_config_sha256") != expected_data_config_sha:
        raise ArtifactCompatibilityError("数据集元信息与当前数据构造配置不一致，请重新构造数据集")
    if dataset_manifest.get("data_config_sha256") != expected_data_config_sha:
        raise ArtifactCompatibilityError("数据集 manifest 与当前数据构造配置不一致，请重新构造数据集")
    dataset_id = str(metadata["dataset_id"])
    dataset_manifest_sha = sha256_file(dataset_dir / "manifest.json")
    run_dir = args.outputs_root.resolve() / "runs" / run_id
    checkpoints = run_dir / "checkpoints"
    resume_path = args.resume.resolve() if args.resume is not None else None
    if resume_path is None:
        if run_dir.exists():
            raise OutputExistsError(f"run_id 已存在，拒绝覆盖：{run_dir}")
        checkpoints.mkdir(parents=True)
        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    else:
        if not run_dir.is_dir() or not resume_path.is_file():
            raise ArtifactCompatibilityError(
                f"恢复训练要求既有 run 目录和检查点：run={run_dir}, checkpoint={resume_path}"
            )

    seed = int(config["training"]["seed"])
    set_global_seed(seed)
    device = resolve_device(str(config["training"]["device"]))
    precision = resolve_precision(str(config["training"]["precision"]), device)
    model = DynamicHistoryModel(config).to(device)
    parameter_count = trainable_parameter_count(model)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["adam_betas"]),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_samples = int(metadata["samples"]["train"])
    validation_samples = int(metadata["samples"]["validation"])
    batch_size = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    train_data = PreparedParquetDataset(
        dataset_dir / "train", config, train_samples, shuffle=True, seed=seed
    )
    validation_data = PreparedParquetDataset(
        dataset_dir / "validation", config, validation_samples, shuffle=False, seed=seed
    )
    workers = int(training["dataloader_workers"])
    train_loader = make_loader(train_data, batch_size, workers)
    validation_loader = make_loader(validation_data, batch_size, workers)
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * int(training["max_epochs"])
    scheduler = make_scheduler(
        optimizer,
        total_updates,
        float(training["warmup_ratio"]),
        float(training["min_learning_rate"]),
        float(training["learning_rate"]),
    )
    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != "dynamic-history-checkpoint/v1":
            raise ArtifactCompatibilityError(f"不支持的检查点：{resume_path}")
        if checkpoint.get("config_sha256") != config_sha256(config):
            raise ArtifactCompatibilityError("恢复检查点与当前配置不一致")
        if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha:
            raise ArtifactCompatibilityError("恢复检查点与当前数据集摘要不一致")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_total"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])

    loss_log = run_dir / "loss_history.jsonl"
    loss_plot = run_dir / "loss_curves.png"
    progress_log = run_dir / "training_progress.jsonl"
    progress_plot = run_dir / "live_training_metrics.png"
    progress_initialized = resume_path is not None

    def record_progress(record: dict) -> None:
        nonlocal progress_initialized
        save_progress_log(
            progress_log,
            record,
            reset=not progress_initialized,
        )
        progress_initialized = True
        if record.get("event") == "train_progress":
            plot_progress_log(progress_log, progress_plot)

    min_delta = float(training["early_stopping_min_delta"])
    patience = int(training["early_stopping_patience"])
    max_epochs = int(training["max_epochs"])
    for epoch in range(start_epoch, max_epochs + 1):
        train_data.set_epoch(epoch)
        train_metrics = run_training_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            device,
            precision,
            train_samples,
            epoch,
            record_progress,
        )
        validation_metrics = run_validation_epoch(
            model,
            validation_loader,
            config,
            device,
            precision,
            validation_samples,
            epoch,
            record_progress,
        )
        save_loss_log(
            loss_log,
            epoch,
            train_metrics,
            validation_metrics,
            optimizer.param_groups[0]["lr"],
            reset=resume_path is None and epoch == 1,
        )
        plot_loss_log(loss_log, loss_plot)
        improved = validation_metrics["total"] < best_loss - min_delta
        if improved:
            best_loss = validation_metrics["total"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint_value = _checkpoint_value(
            model,
            optimizer,
            scheduler,
            epoch,
            best_epoch,
            best_loss,
            stale_epochs,
            config,
            dataset_id,
            dataset_manifest_sha,
        )
        _save_checkpoint(checkpoints / "last.pt", checkpoint_value)
        if improved:
            _save_checkpoint(checkpoints / "best.pt", checkpoint_value)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stale_epochs >= patience:
            break
    write_json(
        run_dir / "run_meta.json",
        {
            "schema_version": "dynamic-history-run/v1",
            "run_id": run_id,
            "dataset_id": dataset_id,
            "dataset_manifest_sha256": dataset_manifest_sha,
            "config_sha256": config_sha256(config),
            "trainable_parameters": parameter_count,
            "device": str(device),
            "precision": precision,
            "best_epoch": best_epoch,
            "best_validation_total": best_loss,
            "loss_history_sha256": sha256_file(loss_log),
            "loss_curves_sha256": sha256_file(loss_plot),
            "training_progress_sha256": sha256_file(progress_log),
            "live_training_metrics_sha256": sha256_file(progress_plot),
        },
    )
    print(
        json.dumps(
            {"run_dir": run_dir.as_posix(), "best_epoch": best_epoch, "best_loss": best_loss},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except DynamicHistoryError as exc:
        print(
            json.dumps({"status": "failed", "stage": "train", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(
            json.dumps({"status": "failed", "stage": "train", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1) from exc
