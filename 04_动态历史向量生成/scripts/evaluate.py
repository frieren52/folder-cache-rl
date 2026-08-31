"""Evaluate the best checkpoint and publish a self-contained model release."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow
import torch
from safetensors.torch import load_file, save_file


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.config import config_sha256, validate_config  # noqa: E402
from src.data import PreparedParquetDataset, load_and_verify_dataset  # noqa: E402
from src.errors import (  # noqa: E402
    ArtifactCompatibilityError,
    DynamicHistoryError,
    OutputExistsError,
)
from src.inference import DynamicHistoryEncoder  # noqa: E402
from src.losses import finalize_loss_sums, loss_sums  # noqa: E402
from src.model import DynamicHistoryModel, trainable_parameter_count  # noqa: E402
from src.training import forward_batch, make_loader, move_batch  # noqa: E402
from src.utils import (  # noqa: E402
    file_hashes,
    read_json,
    resolve_device,
    resolve_precision,
    sha256_file,
    validate_identifier,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证并发布动态历史编码模型")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--outputs-root", type=Path, default=MODULE_ROOT / "outputs")
    return parser.parse_args()


def _relative_change(baseline: float, model: float, lower_is_better: bool = True) -> float:
    if baseline == 0.0:
        return 0.0 if model == 0.0 else (-math.inf if lower_is_better else math.inf)
    return (baseline - model) / baseline if lower_is_better else (model - baseline) / baseline


def _source_state() -> dict:
    git_root = REPOSITORY_ROOT / ".git"
    source_paths = [
        *sorted((MODULE_ROOT / "src").glob("*.py")),
        *sorted((MODULE_ROOT / "scripts").glob("*.py")),
        MODULE_ROOT / "config" / "config.yaml",
    ]
    return {
        "git_available": git_root.exists(),
        "files": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        },
    }


@torch.no_grad()
def evaluate_model(
    model: DynamicHistoryModel,
    loader,
    config: dict,
    device: torch.device,
    precision: str,
    sample_count: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    loss_totals: dict[str, float | int] = {
        "access_sum": 0.0,
        "time_sum": 0.0,
        "count_sum": 0.0,
        "samples": 0,
        "positive_samples": 0,
    }
    brier = nll = count_mae = 0.0
    baseline_brier = baseline_nll = baseline_count_mae = 0.0
    epsilon = float(config["evaluation"]["probability_clip_epsilon"])
    boundaries = np.asarray(config["target"]["time_boundaries_seconds"], dtype=np.float64)
    horizon = int(config["target"]["horizon_seconds"])
    time_bin_count = len(boundaries) - 1
    output_class_count = time_bin_count + 1
    baseline_window = int(config["evaluation"]["baseline_window_seconds"])
    captured: dict[str, np.ndarray] | None = None
    for host_batch in loader:
        batch = move_batch(host_batch, device)
        context = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if precision == "bf16"
            else nullcontext()
        )
        with context:
            outputs = forward_batch(model, batch)
        values = loss_sums(
            outputs,
            batch["y_access"],
            batch["y_time"],
            batch["y_count"],
            config["loss"],
        )
        for key in loss_totals:
            loss_totals[key] += values[key]
        probabilities = outputs["next_access_probs"].float().cpu().numpy().astype(np.float64)
        counts = outputs["expected_access_counts"].float().cpu().numpy().astype(np.float64)
        y_access = batch["y_access"].cpu().numpy()
        y_time = batch["y_time"].cpu().numpy()
        y_count = batch["y_count"].cpu().numpy().astype(np.float64)
        classes = np.where(y_access == 1, y_time, time_bin_count).astype(np.int64)
        one_hot = np.eye(output_class_count, dtype=np.float64)[classes]
        brier += float(np.square(probabilities - one_hot).sum())
        nll += float(-np.log(np.clip(probabilities[np.arange(len(classes)), classes], epsilon, 1.0)).sum())
        count_mae += float(np.abs(counts - y_count).sum())

        medium = batch["medium_counts"].float().cpu().numpy()[:, :, 0]
        bucket_seconds = next(
            int(item["bucket_seconds"])
            for item in config["history"]["scales"]
            if item["name"] == "medium"
        )
        baseline_buckets = baseline_window // bucket_seconds
        recent_count = np.expm1(medium[:, -baseline_buckets:]).sum(axis=1).astype(np.float64)
        rate = recent_count / baseline_window
        baseline_probabilities = np.empty(
            (len(rate), output_class_count), dtype=np.float64
        )
        for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            baseline_probabilities[:, index] = np.exp(-rate * left) - np.exp(-rate * right)
        baseline_probabilities[:, time_bin_count] = np.exp(-rate * boundaries[-1])
        baseline_brier += float(np.square(baseline_probabilities - one_hot).sum())
        baseline_nll += float(
            -np.log(
                np.clip(
                    baseline_probabilities[np.arange(len(classes)), classes], epsilon, 1.0
                )
            ).sum()
        )
        baseline_expected_count = rate * horizon
        baseline_count_mae += float(np.abs(baseline_expected_count - y_count).sum())
        if captured is None:
            captured = {
                "second_counts": batch["second_counts"].float().cpu().numpy(),
                "short_counts": batch["short_counts"].float().cpu().numpy(),
                "medium_counts": batch["medium_counts"].float().cpu().numpy(),
                "long_counts": batch["long_counts"].float().cpu().numpy(),
                "history_state": batch["history_state"].float().cpu().numpy(),
                "vectors": outputs["vectors"].float().cpu().numpy(),
                "next_access_probs": outputs["next_access_probs"].float().cpu().numpy(),
                "expected_access_counts": outputs["expected_access_counts"].float().cpu().numpy(),
            }
    if int(loss_totals["samples"]) != sample_count or captured is None:
        raise RuntimeError(
            f"验证样本数不一致：期望 {sample_count}，实际 {loss_totals['samples']}"
        )
    metrics = {
        "losses": finalize_loss_sums(loss_totals, config["loss"]),
        "model": {
            "brier": brier / sample_count,
            "nll": nll / sample_count,
            "count_mae": count_mae / sample_count,
        },
        "baseline": {
            "brier": baseline_brier / sample_count,
            "nll": baseline_nll / sample_count,
            "count_mae": baseline_count_mae / sample_count,
        },
    }
    metrics["relative"] = {
        "brier_improvement": _relative_change(
            metrics["baseline"]["brier"], metrics["model"]["brier"]
        ),
        "nll_increase": _relative_change(
            metrics["baseline"]["nll"], metrics["model"]["nll"], lower_is_better=False
        ),
        "count_mae_improvement": _relative_change(
            metrics["baseline"]["count_mae"], metrics["model"]["count_mae"]
        ),
    }
    return metrics, captured


def main() -> None:
    args = parse_args()
    model_id = validate_identifier(args.model_id, "model_id")
    dataset_dir = args.dataset_dir.resolve()
    run_dir = args.run_dir.resolve()
    metadata, _ = load_and_verify_dataset(dataset_dir)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise ArtifactCompatibilityError(f"最佳检查点不存在：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "dynamic-history-checkpoint/v1":
        raise ArtifactCompatibilityError(f"不支持的检查点：{checkpoint_path}")
    config = checkpoint["config"]
    validate_config(config)
    dataset_manifest_sha = sha256_file(dataset_dir / "manifest.json")
    if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha:
        raise ArtifactCompatibilityError("检查点与验证数据集摘要不一致")
    device = resolve_device(str(config["training"]["device"]))
    precision = resolve_precision(str(config["training"]["precision"]), device)
    model = DynamicHistoryModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    validation_samples = int(metadata["samples"]["validation"])
    validation_data = PreparedParquetDataset(
        dataset_dir / "validation",
        config,
        validation_samples,
        shuffle=False,
        seed=int(config["training"]["seed"]),
    )
    validation_loader = make_loader(
        validation_data,
        int(config["training"]["micro_batch_size"]),
        int(config["training"]["dataloader_workers"]),
    )
    metrics, captured = evaluate_model(
        model, validation_loader, config, device, precision, validation_samples
    )
    thresholds = config["evaluation"]
    provisional_pass = (
        metrics["relative"]["brier_improvement"]
        >= float(thresholds["min_brier_relative_improvement"])
        and metrics["relative"]["count_mae_improvement"]
        >= float(thresholds["min_count_mae_relative_improvement"])
        and metrics["relative"]["nll_increase"]
        <= float(thresholds["max_nll_relative_increase"])
    )

    releases_root = args.outputs_root.resolve() / "releases"
    final_root = releases_root / model_id
    staging_root = releases_root / f".staging-{model_id}"
    if final_root.exists() or staging_root.exists():
        raise OutputExistsError(
            f"模型发布目录已存在，拒绝覆盖：staging={staging_root}, release={final_root}"
        )
    staging_root.mkdir(parents=True)
    state = {
        key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
    }
    save_file(state, str(staging_root / "dynamic_encoder.safetensors"))
    shutil.copy2(dataset_dir / "feature_stats.json", staging_root / "feature_stats.json")
    shutil.copy2(run_dir / "loss_history.jsonl", staging_root / "loss_history.jsonl")
    shutil.copy2(run_dir / "loss_curves.png", staging_root / "loss_curves.png")
    model_config = {
        "schema_version": "dynamic-history-encoder/v1",
        "model_id": model_id,
        "run_id": run_dir.name,
        "dataset_id": metadata["dataset_id"],
        "dataset_manifest_sha256": dataset_manifest_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trainable_parameters": trainable_parameter_count(model),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_total": float(checkpoint["best_validation_total"]),
        "output_contract": {
            "vector_dim": int(config["model"]["vector_dim"]),
            "next_access_probability_dim": len(
                config["target"]["time_boundaries_seconds"]
            ),
            "time_boundaries_seconds": config["target"]["time_boundaries_seconds"],
            "no_access_class_index": len(
                config["target"]["time_boundaries_seconds"]
            ) - 1,
            "horizon_seconds": int(config["target"]["horizon_seconds"]),
        },
        "resolved_config": config,
    }
    write_json(staging_root / "model_config.json", model_config)

    exported = DynamicHistoryModel(config).to(device)
    exported.load_state_dict(
        load_file(str(staging_root / "dynamic_encoder.safetensors"), device=str(device)), strict=True
    )
    exported.eval()
    comparison_inputs = [
        torch.as_tensor(captured[name], device=device)
        for name in (
            "second_counts",
            "short_counts",
            "medium_counts",
            "long_counts",
            "history_state",
        )
    ]
    with torch.no_grad():
        reference_outputs = model(*comparison_inputs)
        exported_outputs = exported(*comparison_inputs)
    export_differences = {
        "vectors": float(
            np.max(
                np.abs(
                    exported_outputs["vectors"].float().cpu().numpy()
                    - reference_outputs["vectors"].float().cpu().numpy()
                )
            )
        ),
        "next_access_probs": float(
            np.max(
                np.abs(
                    exported_outputs["next_access_probs"].float().cpu().numpy()
                    - reference_outputs["next_access_probs"].float().cpu().numpy()
                )
            )
        ),
        "expected_access_counts": float(
            np.max(
                np.abs(
                    exported_outputs["expected_access_counts"].float().cpu().numpy()
                    - reference_outputs["expected_access_counts"].float().cpu().numpy()
                )
            )
        ),
    }
    if max(export_differences.values()) > 1e-5:
        raise ArtifactCompatibilityError(f"导出模型输出不一致：{export_differences}")
    report = {
        "schema_version": "dynamic-history-evaluation/v1",
        "model_id": model_id,
        "dataset_id": metadata["dataset_id"],
        "validation_samples": validation_samples,
        **metrics,
        "thresholds_provisional": bool(thresholds["thresholds_provisional"]),
        "provisional_pass": provisional_pass,
        "export_consistency_max_abs_difference": export_differences,
    }
    write_json(staging_root / "evaluation_report.json", report)
    artifact_paths = [path for path in staging_root.iterdir() if path.is_file()]
    manifest = {
        "schema_version": "dynamic-history-release-manifest/v1",
        "model_id": model_id,
        "run_id": run_dir.name,
        "dataset_id": metadata["dataset_id"],
        "dataset_manifest_sha256": dataset_manifest_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": config_sha256(config),
        "source": _source_state(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "precision": precision,
        },
        "trainable_parameters": trainable_parameter_count(model),
        "files": file_hashes(staging_root, artifact_paths),
    }
    write_json(staging_root / "manifest.json", manifest)
    DynamicHistoryEncoder.load(staging_root, device=str(device))
    staging_root.replace(final_root)
    print(
        json.dumps(
            {"release_dir": final_root.as_posix(), "provisional_pass": provisional_pass},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except DynamicHistoryError as exc:
        print(
            json.dumps({"status": "failed", "stage": "evaluate", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:  # pragma: no cover
        print(
            json.dumps({"status": "failed", "stage": "evaluate", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1) from exc
