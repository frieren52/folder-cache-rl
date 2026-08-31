from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

from .config import config_sha256, data_config_sha256
from .data import (
    FeatureBuilder,
    ParquetShardWriter,
    TimelineCursor,
    WelfordState,
    determine_snapshot_split,
    load_access_events,
    load_catalog,
    parquet_schema,
    replace_history_state,
)
from .errors import DataIntegrityError, OutputExistsError
from .sampling import HISTORY_TIER, RollingHistoryPools
from .utils import file_hashes, sha256_file, validate_identifier, write_json


SHANGHAI = ZoneInfo("Asia/Shanghai")
TIER_NAMES = {value: key for key, value in HISTORY_TIER.items()}
TARGET_BY_TIER = {
    0: "no_history_per_snapshot",
    1: "single_history_per_snapshot",
    2: "low_history_per_snapshot",
    3: "medium_history_per_snapshot",
    4: "high_history_per_snapshot",
}


def _nearest_rank(counter: Counter[int], quantile: float) -> int | None:
    total = sum(counter.values())
    if total == 0:
        return None
    target = max(1, math.ceil(quantile * total))
    cumulative = 0
    for value, count in sorted(counter.items()):
        cumulative += count
        if cumulative >= target:
            return int(value)
    raise AssertionError("不可达的分位数状态")


class GroupStatistics:
    def __init__(self, scale_widths: Mapping[str, int], time_bin_count: int) -> None:
        self.count = 0
        self.scale_widths = dict(scale_widths)
        self.time_bin_count = time_bin_count
        self.all_zero = Counter({name: 0 for name in scale_widths})
        self.nonzero_hist = {name: Counter() for name in scale_widths}
        self.trailing_zero_hist = {name: Counter() for name in scale_widths}
        self.y_access = 0
        self.y_time = Counter()
        self.y_count = Counter()
        self.y_count_sum = 0
        self.flags = np.zeros(2, dtype=np.int64)
        self.continuous_sum = np.zeros(2, dtype=np.float64)
        self.continuous_sumsq = np.zeros(2, dtype=np.float64)
        self.clipped = np.zeros(2, dtype=np.int64)

    def update(self, record: Mapping[str, Any], auxiliary: Mapping[str, Any]) -> None:
        self.count += 1
        for name, width in self.scale_widths.items():
            values = np.asarray(record[f"{name}_counts"], dtype=np.float32)
            nonzero_positions = np.flatnonzero(values)
            nonzero = int(nonzero_positions.size)
            trailing = width if nonzero == 0 else width - 1 - int(nonzero_positions[-1])
            self.all_zero[name] += int(nonzero == 0)
            self.nonzero_hist[name][nonzero] += 1
            self.trailing_zero_hist[name][trailing] += 1
        y_access = int(record["y_access"])
        y_count = int(record["y_count"])
        self.y_access += y_access
        if y_access:
            self.y_time[int(record["y_time"])] += 1
        self.y_count[y_count] += 1
        self.y_count_sum += y_count
        state = np.asarray(record["history_state"], dtype=np.float64)
        self.flags += state[2:].astype(np.int64)
        continuous = np.asarray(auxiliary["raw_continuous"], dtype=np.float64)
        self.continuous_sum += continuous
        self.continuous_sumsq += continuous * continuous
        self.clipped[0] += int(auxiliary["recency_clipped"])
        self.clipped[1] += int(auxiliary["interval_clipped"])

    def to_dict(self) -> dict[str, Any]:
        continuous_mean = self.continuous_sum / max(self.count, 1)
        variance = self.continuous_sumsq / max(self.count, 1) - continuous_mean**2
        continuous_std = np.sqrt(np.maximum(variance, 0.0))
        scale_report: dict[str, Any] = {}
        for name in self.scale_widths:
            nonzero = self.nonzero_hist[name]
            trailing = self.trailing_zero_hist[name]
            scale_report[name] = {
                "all_zero_count": int(self.all_zero[name]),
                "all_zero_rate": self.all_zero[name] / max(self.count, 1),
                "nonzero_bucket_mean": (
                    sum(value * count for value, count in nonzero.items()) / max(self.count, 1)
                ),
                "nonzero_bucket_quantiles": {
                    key: _nearest_rank(nonzero, value)
                    for key, value in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
                },
                "trailing_zero_quantiles": {
                    key: _nearest_rank(trailing, value)
                    for key, value in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
                },
            }
        return {
            "samples": self.count,
            "scales": scale_report,
            "labels": {
                "access_count": self.y_access,
                "access_rate": self.y_access / max(self.count, 1),
                "time_bucket_counts": [
                    int(self.y_time[index]) for index in range(self.time_bin_count)
                ],
                "time_bucket_rates_given_access": [
                    self.y_time[index] / max(self.y_access, 1)
                    for index in range(self.time_bin_count)
                ],
                "count_mean": self.y_count_sum / max(self.count, 1),
                "count_quantiles": {
                    key: _nearest_rank(self.y_count, value)
                    for key, value in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
                },
                "count_max": max(self.y_count, default=0),
            },
            "state": {
                "continuous_mean": continuous_mean.tolist(),
                "continuous_std": continuous_std.tolist(),
                "clipped_count": self.clipped.tolist(),
                "clipped_rate": (self.clipped / max(self.count, 1)).tolist(),
                "missing_flag_count": self.flags.tolist(),
                "missing_flag_rate": (self.flags / max(self.count, 1)).tolist(),
            },
        }


class DataReportAccumulator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.scale_widths = {
            str(item["name"]): int(item["bucket_count"])
            for item in config["history"]["scales"]
        }
        self.time_bin_count = len(config["target"]["time_boundaries_seconds"]) - 1
        self.groups = {
            split: {
                "all": GroupStatistics(self.scale_widths, self.time_bin_count),
                **{
                    TIER_NAMES[tier]: GroupStatistics(
                        self.scale_widths, self.time_bin_count
                    )
                    for tier in sorted(TIER_NAMES)
                },
            }
            for split in ("train", "validation")
        }
        self.snapshot_count = Counter()
        self.samples_per_snapshot = {
            "train": Counter(),
            "validation": Counter(),
        }
        self.actual_tiers = {
            "train": Counter(),
            "validation": Counter(),
        }
        self.target_tiers = {
            "train": Counter(),
            "validation": Counter(),
        }
        self.shortfall_snapshots = {
            "train": Counter(),
            "validation": Counter(),
        }
        self.directories = {"train": set(), "validation": set()}
        self.sampling = config["sampling"]
        self.integrity = {
            "non_finite_values": 0,
            "fixed_length_errors": 0,
            "row_count_mismatches": 0,
        }

    def observe_snapshot(
        self,
        split: str,
        selected: list[tuple[int, int]],
        pool_sizes: Mapping[str, int],
    ) -> None:
        self.snapshot_count[split] += 1
        self.samples_per_snapshot[split][len(selected)] += 1
        actual = Counter(tier for _, tier in selected)
        for tier, config_key in TARGET_BY_TIER.items():
            target = int(self.sampling[config_key])
            name = TIER_NAMES[tier]
            self.actual_tiers[split][name] += actual[tier]
            self.target_tiers[split][name] += target
            if int(pool_sizes[name]) < target:
                self.shortfall_snapshots[split][name] += 1

    def observe_record(
        self,
        split: str,
        record: Mapping[str, Any],
        auxiliary: Mapping[str, Any],
    ) -> None:
        tier_name = TIER_NAMES[int(record["history_tier"])]
        self.groups[split]["all"].update(record, auxiliary)
        self.groups[split][tier_name].update(record, auxiliary)
        self.directories[split].add(int(record["path_index"]))
        for name, width in self.scale_widths.items():
            values = np.asarray(record[f"{name}_counts"])
            if values.shape != (width,):
                self.integrity["fixed_length_errors"] += 1
            self.integrity["non_finite_values"] += int((~np.isfinite(values)).sum())
        state = np.asarray(record["history_state"])
        if state.shape != (4,):
            self.integrity["fixed_length_errors"] += 1
        self.integrity["non_finite_values"] += int((~np.isfinite(state)).sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "dynamic-history-data-report/v1",
            "quantile_method": "nearest_rank",
            "splits": {
                split: {
                    "snapshots": int(self.snapshot_count[split]),
                    "samples": self.groups[split]["all"].count,
                    "covered_directories": len(self.directories[split]),
                    "samples_per_snapshot_histogram": {
                        str(key): value
                        for key, value in sorted(self.samples_per_snapshot[split].items())
                    },
                    "sampling": {
                        "target_counts": dict(self.target_tiers[split]),
                        "actual_counts": dict(self.actual_tiers[split]),
                        "shortfall_snapshots": dict(self.shortfall_snapshots[split]),
                    },
                    "overall": self.groups[split]["all"].to_dict(),
                    "by_history_tier": {
                        name: self.groups[split][name].to_dict()
                        for name in TIER_NAMES.values()
                    },
                }
                for split in ("train", "validation")
            },
            "integrity": dict(self.integrity),
        }


def _render_report(report: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    lines = [
        "# 动态历史样本构造报告",
        "",
        f"- 数据集：`{metadata['dataset_id']}`",
        f"- 事件数：{metadata['input']['total_events']}",
        f"- 目录数：{metadata['input']['catalog_rows']}",
        f"- 采样起点：`{metadata['split']['sample_start']}`",
        f"- 切分点：`{metadata['split']['split_time']}`",
        "",
        "## 切分与采样",
        "",
        "| 数据集 | 快照 | 样本 | 覆盖目录 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, label in (("train", "训练"), ("validation", "验证")):
        item = report["splits"][split]
        lines.append(
            f"| {label} | {item['snapshots']} | {item['samples']} | {item['covered_directories']} |"
        )
    lines.extend(("", "## 输入稀疏性", ""))
    for split, label in (("train", "训练"), ("validation", "验证")):
        lines.extend((f"### {label}集", "", "| 尺度 | 全零率 | 非零桶P50 | P90 | P95 | P99 |", "| --- | ---: | ---: | ---: | ---: | ---: |"))
        scales = report["splits"][split]["overall"]["scales"]
        for name in ("second", "short", "medium", "long"):
            values = scales[name]
            quantiles = values["nonzero_bucket_quantiles"]
            lines.append(
                f"| {name} | {values['all_zero_rate']:.4f} | {quantiles['p50']} | {quantiles['p90']} | {quantiles['p95']} | {quantiles['p99']} |"
            )
        lines.append("")
    integrity = report["integrity"]
    lines.extend(
        (
            "## 完整性",
            "",
            f"- 非有限值：{integrity['non_finite_values']}",
            f"- 定长字段错误：{integrity['fixed_length_errors']}",
            f"- 行数汇总错误：{integrity['row_count_mismatches']}",
            "",
        )
    )
    return "\n".join(lines)


def _shard_summaries(dataset_root: Path, shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in shards:
        path = Path(item["path"])
        parquet = pq.ParquetFile(path)
        first = parquet.read_row_group(0, columns=["snapshot_time"])["snapshot_time"][0].as_py()
        last_group = parquet.num_row_groups - 1
        last_column = parquet.read_row_group(last_group, columns=["snapshot_time"])["snapshot_time"]
        last = last_column[len(last_column) - 1].as_py()
        result.append(
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "rows": int(item["rows"]),
                "row_groups": int(item["row_groups"]),
                "first_snapshot": first.isoformat(),
                "last_snapshot": last.isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return result


def _transform_training(
    raw_root: Path,
    final_root: Path,
    schema: Any,
    storage: Mapping[str, Any],
    feature_stats: Mapping[str, Any],
) -> tuple[ParquetShardWriter, list[dict[str, Any]]]:
    writer = ParquetShardWriter(final_root, schema, storage)
    for path in sorted(raw_root.glob("part-*.parquet"), key=lambda item: item.name):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            writer.write_table(replace_history_state(parquet.read_row_group(row_group), feature_stats))
    shards = writer.close()
    return writer, shards


def prepare_dataset(
    config: Mapping[str, Any],
    dataset_id: str,
    catalog_path: Path,
    access_dir: Path,
    data_root: Path,
) -> Path:
    dataset_id = validate_identifier(dataset_id, "dataset_id")
    staging_root = data_root / "staging" / dataset_id
    final_root = data_root / "datasets" / dataset_id
    if staging_root.exists() or final_root.exists():
        raise OutputExistsError(
            f"数据运行目录已存在，拒绝覆盖：staging={staging_root}, dataset={final_root}"
        )
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()

    catalog = load_catalog(catalog_path)
    events = load_access_events(catalog, access_dir)
    split = determine_snapshot_split(config, events)
    schema = parquet_schema(config)
    raw_train_root = staging_root / "raw_train"
    raw_writer = ParquetShardWriter(raw_train_root, schema, config["storage"])
    report = DataReportAccumulator(config)
    state_stats = WelfordState(2)
    pools = RollingHistoryPools(catalog.path_indices)
    max_window = max(int(item["window_seconds"]) for item in config["history"]["scales"])
    timeline = TimelineCursor(events, max_window)
    builder = FeatureBuilder(config, events)
    feature_stats: dict[str, Any] | None = None
    validation_writer: ParquetShardWriter | None = None
    training_shards: list[dict[str, Any]] | None = None

    for snapshot in split.iter_candidates():
        timeline.advance(snapshot, pools)
        partition = split.partition(snapshot)
        if partition is None:
            continue
        if partition == "validation" and feature_stats is None:
            raw_writer.close()
            feature_stats = state_stats.finalize(
                ("log_recency_seconds", "log_last_distinct_interval_seconds")
            )
            _, training_shards = _transform_training(
                raw_train_root,
                staging_root / "train",
                schema,
                config["storage"],
                feature_stats,
            )
            if raw_train_root.parent != staging_root:
                raise AssertionError("拒绝删除不在当前 staging 下的临时训练目录")
            shutil.rmtree(raw_train_root)
            validation_writer = ParquetShardWriter(
                staging_root / "validation", schema, config["storage"]
            )
        selected, pool_sizes = pools.sample(snapshot, dict(config["sampling"]))
        report.observe_snapshot(partition, selected, pool_sizes)
        for path_index, history_tier in selected:
            record, auxiliary = builder.build(
                path_index,
                history_tier,
                snapshot,
                feature_stats=feature_stats if partition == "validation" else None,
            )
            report.observe_record(partition, record, auxiliary)
            if partition == "train":
                state_stats.update(auxiliary["raw_continuous"])
                raw_writer.write_record(record)
            else:
                assert validation_writer is not None
                validation_writer.write_record(record)

    if feature_stats is None or training_shards is None or validation_writer is None:
        raise DataIntegrityError("数据构造未进入验证区间，无法发布完整数据集")
    validation_shards = validation_writer.close()
    pools.assert_consistent()

    report_value = report.to_dict()
    train_rows = sum(int(item["rows"]) for item in training_shards)
    validation_rows = sum(int(item["rows"]) for item in validation_shards)
    if train_rows != report_value["splits"]["train"]["samples"]:
        report_value["integrity"]["row_count_mismatches"] += 1
    if validation_rows != report_value["splits"]["validation"]["samples"]:
        report_value["integrity"]["row_count_mismatches"] += 1
    if any(int(value) != 0 for value in report_value["integrity"].values()):
        raise DataIntegrityError(
            f"数据完整性检查失败：{json.dumps(report_value['integrity'], ensure_ascii=False)}"
        )

    input_value = {
        "catalog": {
            "path": catalog_path.as_posix(),
            "rows": len(catalog.path_indices),
            "sha256": catalog.sha256,
        },
        "access_files": events.access_files,
        "total_events": events.total_events,
        "log_start": datetime.fromtimestamp(events.log_start, SHANGHAI).isoformat(),
        "log_end_exclusive": datetime.fromtimestamp(
            events.log_end_exclusive, SHANGHAI
        ).isoformat(),
    }
    metadata: dict[str, Any] = {
        "schema_version": "dynamic-history-data-meta/v1",
        "dataset_id": dataset_id,
        "input": {
            **input_value,
            "catalog_rows": len(catalog.path_indices),
        },
        "split": {
            "sample_start": datetime.fromtimestamp(split.sample_start, SHANGHAI).isoformat(),
            "latest_snapshot": datetime.fromtimestamp(split.latest_snapshot, SHANGHAI).isoformat(),
            "split_time": datetime.fromtimestamp(split.split_time, SHANGHAI).isoformat(),
            "candidate_snapshots": split.candidate_count,
            "train_snapshots": split.train_count,
            "validation_snapshots": split.validation_count,
            "purged_snapshots": split.purged_count,
        },
        "samples": {"train": train_rows, "validation": validation_rows},
        "config_sha256": config_sha256(dict(config)),
        "data_config_sha256": data_config_sha256(dict(config)),
    }
    write_json(staging_root / "feature_stats.json", feature_stats)
    write_json(staging_root / "data_report.json", report_value)
    (staging_root / "data_report.md").write_text(
        _render_report(report_value, metadata), encoding="utf-8", newline="\n"
    )
    metadata["shards"] = {
        "train": _shard_summaries(staging_root, training_shards),
        "validation": _shard_summaries(staging_root, validation_shards),
    }
    write_json(staging_root / "data_meta.json", metadata)
    manifest_paths = [path for path in staging_root.rglob("*") if path.is_file()]
    manifest = {
        "schema_version": "dynamic-history-dataset-manifest/v1",
        "dataset_id": dataset_id,
        "config_sha256": config_sha256(dict(config)),
        "data_config_sha256": data_config_sha256(dict(config)),
        "inputs": input_value,
        "files": file_hashes(staging_root, manifest_paths),
    }
    write_json(staging_root / "manifest.json", manifest)
    staging_root.replace(final_root)
    return final_root
