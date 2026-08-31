"""训练入口：按验证集总损失选择并保存最佳模型。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import save_file


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from src.data import (  # noqa: E402
    PreparedData,
    load_config,
    resolve_path,
)
from src.inference import write_manifest  # noqa: E402
from src.model import StaticSemanticModel  # noqa: E402
from src.reporting import plot_loss_log, save_loss_log  # noqa: E402
from src.sampling import build_triplet_sets  # noqa: E402
from src.training import run_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练静态语义编码模型")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--device", default=None, help="默认自动选择 cuda/cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """创建按验证损失自动降低学习率的调度器。"""
    scheduler_config = train_config["scheduler"]
    if scheduler_config["name"] != "ReduceLROnPlateau":
        raise ValueError(f"不支持的学习率调度器：{scheduler_config['name']}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        threshold=float(scheduler_config["threshold"]),
        threshold_mode="rel",
        min_lr=float(scheduler_config["min_learning_rate"]),
    )


def make_model_config(config: Mapping[str, Any], data: PreparedData) -> dict[str, Any]:
    bge_meta = data.metadata["bge"]
    if not bge_meta.get("resolved_commit") or not bge_meta.get("cache_file"):
        raise ValueError("BGE 缓存未绑定 resolved_commit，请重新运行 prepare_data.py")
    expected_revision = str(config["bge"]["revision"])
    if bge_meta["resolved_commit"] != expected_revision:
        raise ValueError(
            f"BGE 缓存版本与配置不一致：缓存={bge_meta['resolved_commit']}，配置={expected_revision}"
        )
    return {
        "schema_version": "static-semantic-encoder/v1",
        "architecture": dict(config["model"]),
        "preprocessing": {
            "semantic_text": "embedding_text",
            "instance_text": ["source_text", "instance_context"],
            "raw_path_training_only": True,
        },
        "date_normalization": data.metadata["date_normalization"],
        "label_maps": data.metadata["label_maps"],
        "bge": {
            "model_name": config["bge"]["model_name"],
            "revision": expected_revision,
            "resolved_commit": bge_meta["resolved_commit"],
            "max_length": int(config["bge"]["max_length"]),
            "batch_size": int(config["bge"]["batch_size"]),
            "precision": config["bge"]["precision"],
            "tokenizer": bge_meta["tokenizer"],
        },
        "sampling": dict(config["sampling"]),
        "loss": dict(config["loss"]),
        "training": dict(config["train"]),
        "data": {
            "schema_version": data.metadata["schema_version"],
            "input_files": data.metadata["input_files"],
            "split": data.metadata["split"],
            "bge_cache_sha256": bge_meta["cache_file"]["sha256"],
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    seed = int(config["sampling"]["seed"])
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = resolve_path(MODULE_ROOT, config["paths"]["data_dir"])
    output_dir = resolve_path(MODULE_ROOT, config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = output_dir / "loss_history.jsonl"
    loss_plot_path = output_dir / "loss_curves.png"
    data = PreparedData(data_dir)
    model_config = make_model_config(config, data)
    label_sizes = {field: len(values) for field, values in data.label_maps.items()}
    model = StaticSemanticModel(config["model"], label_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, config["train"])
    validation_triplets = build_triplet_sets(data.validation, config, seed=seed + 100_000)
    best_loss = float("inf")
    stale_epochs = 0
    max_epochs = int(config["train"]["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        training_triplets = build_triplet_sets(data.train, config, seed=seed + epoch)
        train_metrics = run_epoch(
            model, data, training_triplets, config, device, optimizer, shuffle_seed=seed + epoch
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, data, validation_triplets, config, device, optimizer=None, shuffle_seed=None
            )
        # 每轮立即保存全部损失，即使后续训练中断也能保留已完成的历史。
        save_loss_log(
            loss_log_path,
            epoch,
            train_metrics,
            validation_metrics,
            learning_rate,
            reset=epoch == 1,
        )
        scheduler.step(validation_metrics["total"])
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "train_total": train_metrics["total"],
                    "validation_total": validation_metrics["total"],
                    "validation_semantic_accuracy": validation_metrics["semantic_triplet_accuracy"],
                    "validation_instance_accuracy": validation_metrics["instance_triplet_accuracy"],
                    "validation_final_accuracy": validation_metrics["final_triplet_accuracy"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            stale_epochs = 0
            state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
            save_file(state, str(output_dir / "static_encoder.safetensors"))
            model_config["best_epoch"] = epoch
            model_config["best_validation_total"] = best_loss
            with (output_dir / "model_config.json").open("w", encoding="utf-8") as handle:
                json.dump(model_config, handle, ensure_ascii=False, indent=2)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["train"]["early_stopping_patience"]):
                break
    plot_loss_log(loss_log_path, loss_plot_path)
    write_manifest(output_dir, model_config)
    print(json.dumps({"best_epoch": model_config["best_epoch"], "best_validation_total": best_loss}, ensure_ascii=False))


if __name__ == "__main__":
    main()
