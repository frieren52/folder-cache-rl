"""在固定验证集上评估最佳模型并生成正式报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from src.data import PreparedData, load_config, resolve_path  # noqa: E402
from src.inference import load_trained_model, write_manifest  # noqa: E402
from src.sampling import build_triplet_sets  # noqa: E402
from src.training import run_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估静态语义编码模型")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config" / "config.yaml")
    parser.add_argument("--device", default=None, help="默认自动选择 cuda/cpu")
    return parser.parse_args()


@torch.inference_mode()
def semantic_recall_at_k(model: torch.nn.Module, data: PreparedData, device: torch.device, k: int) -> dict[str, Any]:
    """按产品、仪器、等级完全一致的强正样本计算语义 Recall@K。"""
    groups = (
        data.validation.loc[data.validation["is_semantic_supervised"]]
        .drop_duplicates("semantic_group_id")
        .sort_values("semantic_group_id")
        .reset_index(drop=True)
    )
    group_ids = groups["semantic_group_id"].astype(str).tolist()
    teacher_vectors = torch.from_numpy(data._semantic_vectors(group_ids)).to(device)
    semantic_vectors = model.encode_semantic(teacher_vectors).cpu().numpy()
    similarities = semantic_vectors @ semantic_vectors.T

    # 先按强正样本键分组，避免把训练时的弱正样本混入正式召回指标。
    relevance_groups: dict[tuple[str, str, str], list[int]] = {}
    for index, row in groups.iterrows():
        key = (str(row["product"]), str(row["instrument"]), str(row["level"]))
        if all(key):
            relevance_groups.setdefault(key, []).append(index)

    recalls: list[float] = []
    for indices in relevance_groups.values():
        if len(indices) < 2:
            continue
        relevant_set = set(indices)
        for query_index in indices:
            relevant = relevant_set - {query_index}
            scores = similarities[query_index].copy()
            scores[query_index] = -np.inf
            top_k = np.argsort(-scores)[: min(k, len(scores) - 1)]
            recalls.append(len(relevant.intersection(top_k.tolist())) / len(relevant))

    total_queries = len(groups)
    valid_queries = len(recalls)
    return {
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        "valid_queries": valid_queries,
        "total_reliable_groups": total_queries,
        "query_coverage": valid_queries / total_queries if total_queries else 0.0,
    }


def evaluate_thresholds(report: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """只检查配置中明确给值的门槛；空值表示暂不冻结该门槛。"""
    actual = {
        "semantic_recall_at_10": report["semantic_recall"]["recall_at_10"],
        "semantic_triplet_accuracy": report["triplet_metrics"]["semantic"]["accuracy"],
        "instance_triplet_accuracy": report["triplet_metrics"]["instance"]["accuracy"],
        "final_triplet_accuracy": report["triplet_metrics"]["final"]["accuracy"],
    }
    checks = {}
    for name, minimum in thresholds.items():
        checks[name] = {
            "minimum": minimum,
            "actual": actual[name],
            "passed": True if minimum is None else actual[name] >= float(minimum),
        }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = resolve_path(MODULE_ROOT, config["paths"]["data_dir"])
    output_dir = resolve_path(MODULE_ROOT, config["paths"]["output_dir"])
    data = PreparedData(data_dir)
    model, model_config = load_trained_model(output_dir, device)

    # 验证样本使用固定种子，保证重复评估得到同一组正负关系。
    seed = int(config["sampling"]["seed"])
    triplets = build_triplet_sets(data.validation, config, seed=seed + 100_000)
    losses = run_epoch(model, data, triplets, config, device)
    recall_k = int(config["evaluation"]["semantic_recall_k"])
    recall = semantic_recall_at_k(model, data, device, recall_k)
    report: dict[str, Any] = {
        "schema_version": "static-semantic-evaluation/v1",
        "sample_counts": {
            "semantic": len(triplets.semantic),
            "instance": len(triplets.instance),
            "final": len(triplets.final),
        },
        "losses": {name: losses[name] for name in ("total", "semantic", "instance", "final", "collapse")},
        "triplet_metrics": {
            task: {
                "accuracy": losses[f"{task}_triplet_accuracy"],
                "mean_margin": losses[f"{task}_mean_margin"],
            }
            for task in ("semantic", "instance", "final")
        },
        "semantic_recall": recall,
    }
    report["thresholds"] = evaluate_thresholds(report, config["evaluation"]["thresholds"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    write_manifest(output_dir, model_config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["thresholds"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
