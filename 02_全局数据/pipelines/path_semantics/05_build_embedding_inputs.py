from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELD_SPECS = [
    ("data_domain", "数据域", "DOMAIN"),
    ("data_family", "数据族", "FAMILY"),
    ("platform", "平台或模型", "PLATFORM"),
    ("satellite", "卫星或航天平台", "SATELLITE"),
    ("satellite_qualifiers", "平台限定", "SATELLITE_QUALIFIER"),
    ("instrument", "载荷或仪器", "INSTRUMENT"),
    ("data_level", "数据等级", "LEVEL"),
    ("data_sublevel", "子等级", "SUBLEVEL"),
    ("product", "产品", "PRODUCT"),
    ("product_variants", "产品变体", "PRODUCT_VARIANT"),
    ("region_types", "区域或轨道类型", "REGION_TYPE"),
    ("spatial_subregions", "空间子区域", "SUBREGION"),
    ("aggregation_period", "产品聚合周期", "AGGREGATION"),
    ("time_slot", "产品时次", "TIME_SLOT"),
    ("temporal_phase", "昼夜阶段", "TEMPORAL_PHASE"),
    ("projection", "投影", "PROJECTION"),
    ("resolution", "空间分辨率", "RESOLUTION"),
    ("orbit_direction", "轨道方向", "ORBIT_DIRECTION"),
    ("processing_stages", "处理阶段", "PROCESSING_STAGE"),
    ("format_hint", "格式", "FORMAT"),
]

SCALAR_NAMES = {
    "data_domain",
    "data_family",
    "platform",
    "satellite",
    "instrument",
    "data_level",
    "data_sublevel",
    "product",
    "aggregation_period",
    "time_slot",
    "temporal_phase",
    "projection",
    "resolution",
    "orbit_direction",
    "format_hint",
}

LIST_NAMES = {
    "satellite_qualifiers",
    "region_types",
    "spatial_subregions",
    "processing_stages",
    "product_variants",
}

STRICT_CONFIDENCE = 0.70


def split_pipe(value: str) -> list[str]:
    return [item for item in value.split("|") if item]


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def value_confidences(row: dict[str, str], field_name: str) -> list[tuple[str, float]]:
    field_meta = json.loads(row["field_metadata_json"] or "{}")
    list_meta = json.loads(row["list_metadata_json"] or "{}")
    if field_name in SCALAR_NAMES:
        item = field_meta.get(field_name)
        return [(item["value"], float(item["confidence"]))] if item else []
    return [
        (item["value"], float(item["confidence"]))
        for item in list_meta.get(field_name, [])
    ]


def build_semantic_views(row: dict[str, str]) -> dict[str, Any]:
    strict_phrases: list[str] = []
    enriched_phrases: list[str] = []
    strict_sequence: list[str] = []
    enriched_sequence: list[str] = []
    all_confidences: list[float] = []

    for field_name, chinese_label, sequence_label in FIELD_SPECS:
        pairs = value_confidences(row, field_name)
        if not pairs:
            continue
        values = stable_unique(value for value, _ in pairs)
        confidences = [confidence for _, confidence in pairs]
        all_confidences.extend(confidences)
        enriched_phrases.append(f"{chinese_label} " + "、".join(values))
        enriched_sequence.extend(f"[{sequence_label}={value}]" for value in values)

        strict_values = stable_unique(value for value, confidence in pairs if confidence >= STRICT_CONFIDENCE)
        if strict_values:
            strict_phrases.append(f"{chinese_label} " + "、".join(strict_values))
            strict_sequence.extend(f"[{sequence_label}={value}]" for value in strict_values)

    extras = split_pipe(row.get("extra_tokens", ""))
    if extras:
        enriched_phrases.append("原始业务标记 " + "、".join(stable_unique(extras)))
        enriched_sequence.extend(f"[EXTRA={value}]" for value in stable_unique(extras))

    strict_text = "；".join(strict_phrases)
    enriched_text = "；".join(enriched_phrases)
    if not enriched_text:
        # 极少量日志/备份路径没有产品字段，仍给出可区分的保守文本，不伪装成卫星语义。
        source_values = stable_unique([row.get("archive_root", ""), row.get("subsystem", "")])
        enriched_text = "目录来源 " + "、".join(source_values)
        enriched_sequence = [f"[SOURCE={value}]" for value in source_values]
    if not strict_text:
        # 严格版本不回填未解释Token；没有高置信语义时，只保留来源类别。
        source_values = stable_unique([row.get("archive_root", ""), row.get("subsystem", "")])
        strict_text = "目录来源 " + "、".join(source_values)
        strict_sequence = [f"[SOURCE={value}]" for value in source_values]

    quality_tier = "supported_clean"
    if row["route_supported"] != "1":
        quality_tier = "fallback_raw"
    elif extras:
        quality_tier = "supported_with_unresolved"
    elif row.get("conflicts_json", "") not in {"", "[]"}:
        quality_tier = "supported_with_conflict"

    return {
        "semantic_text_strict": strict_text,
        "embedding_text": enriched_text,
        "semantic_sequence_strict": "".join(strict_sequence),
        "semantic_sequence": "".join(enriched_sequence),
        "quality_tier": quality_tier,
        "mean_semantic_confidence": round(sum(all_confidences) / max(1, len(all_confidences)), 4),
        "min_semantic_confidence": round(min(all_confidences), 4) if all_confidences else 0.0,
        "has_unresolved_tokens": int(bool(extras)),
    }


def semantic_group_id(sequence: str) -> str:
    return "sem_" + hashlib.blake2b(sequence.encode("utf-8"), digest_size=10).hexdigest()


def make_source_context(row: dict[str, str]) -> str:
    values = [
        f"[ARCHIVE_ROOT={row['archive_root']}]" if row.get("archive_root") else "",
        f"[SUBSYSTEM={row['subsystem']}]" if row.get("subsystem") else "",
        f"[ROUTE={row['parser_route']}]" if row.get("parser_route") else "",
    ]
    return "".join(value for value in values if value)


def make_instance_context(row: dict[str, str]) -> str:
    values = [
        f"[OBSERVE_YEAR={row['observe_year']}]" if row.get("observe_year") else "",
        f"[OBSERVE_DATE={row['observe_date']}]" if row.get("observe_date") else "",
    ]
    values.extend(f"[INSTANCE_ID={value}]" for value in split_pipe(row.get("instance_ids", "")))
    return "".join(value for value in values if value)


def assert_no_instance_time_leak(record: dict[str, Any], row: dict[str, str]) -> None:
    for field_name in ("observe_date", "observe_year"):
        value = row.get(field_name, "")
        if value and value in record["embedding_text"]:
            raise AssertionError(
                f"实例时间泄漏到静态Embedding文本: path_index={row['path_index']} field={field_name} value={value}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤5：生成静态语义Embedding输入")
    parser.add_argument("--parsed-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with gzip.open(args.parsed_input.resolve(), "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            views = build_semantic_views(row)
            group_id = semantic_group_id(views["semantic_sequence"])
            record = {
                "path_index": int(row["path_index"]),
                "path": row["path"],
                "semantic_group_id": group_id,
                "embedding_text": views["embedding_text"],
                "semantic_text_strict": views["semantic_text_strict"],
                "semantic_sequence": views["semantic_sequence"],
                "semantic_sequence_strict": views["semantic_sequence_strict"],
                "source_context": make_source_context(row),
                "instance_context": make_instance_context(row),
                "quality_tier": views["quality_tier"],
                "parser_route": row["parser_route"],
                "parse_status": row["parse_status"],
                "route_supported": int(row["route_supported"]),
                "has_unresolved_tokens": views["has_unresolved_tokens"],
                "mean_semantic_confidence": views["mean_semantic_confidence"],
                "min_semantic_confidence": views["min_semantic_confidence"],
                # 以下是检索/预取系统的旁路特征，不进入静态语义Embedding。
                "access_count": int(row["access_count"]),
                "total_size_bytes": int(row["total_size_bytes"]),
                "observe_year": row["observe_year"],
                "observe_date": row["observe_date"],
                "instance_ids": row["instance_ids"],
            }
            assert_no_instance_time_leak(record, row)
            records.append(record)

    record_path = output_dir / "embedding_records.csv.gz"
    with gzip.open(record_path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    catalog: dict[str, dict[str, Any]] = {}
    for record in records:
        item = catalog.setdefault(
            record["semantic_group_id"],
            {
                "semantic_group_id": record["semantic_group_id"],
                "embedding_text": record["embedding_text"],
                "semantic_text_strict": record["semantic_text_strict"],
                "semantic_sequence": record["semantic_sequence"],
                "path_count": 0,
                "total_access_count": 0,
                "total_size_bytes": 0,
                "quality_tiers": Counter(),
                "parser_routes": Counter(),
                "example_paths": [],
            },
        )
        item["path_count"] += 1
        item["total_access_count"] += record["access_count"]
        item["total_size_bytes"] += record["total_size_bytes"]
        item["quality_tiers"][record["quality_tier"]] += 1
        item["parser_routes"][record["parser_route"]] += 1
        if len(item["example_paths"]) < 3:
            item["example_paths"].append(record["path"])

    catalog_rows: list[dict[str, Any]] = []
    for item in sorted(catalog.values(), key=lambda value: (-value["path_count"], value["semantic_group_id"])):
        catalog_rows.append(
            {
                "semantic_group_id": item["semantic_group_id"],
                "embedding_text": item["embedding_text"],
                "semantic_text_strict": item["semantic_text_strict"],
                "semantic_sequence": item["semantic_sequence"],
                "path_count": item["path_count"],
                "total_access_count": item["total_access_count"],
                "total_size_bytes": item["total_size_bytes"],
                "quality_tiers_json": json.dumps(item["quality_tiers"], ensure_ascii=False, separators=(",", ":")),
                "parser_routes_json": json.dumps(item["parser_routes"], ensure_ascii=False, separators=(",", ":")),
                "example_paths": " || ".join(item["example_paths"]),
            }
        )

    with gzip.open(output_dir / "semantic_catalog.csv.gz", "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog_rows[0]))
        writer.writeheader()
        writer.writerows(catalog_rows)

    quality_counts = Counter(record["quality_tier"] for record in records)
    route_counts = Counter(record["parser_route"] for record in records)
    sample_indices: set[int] = set()
    randomizer = random.Random(20260820)
    by_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_quality[record["quality_tier"]].append(record)
    per_tier = max(1, args.sample_size // max(1, len(by_quality)))
    for group in by_quality.values():
        take = min(per_tier, len(group))
        sample_indices.update(item["path_index"] for item in randomizer.sample(group, take))
    samples = [record for record in records if record["path_index"] in sample_indices][: args.sample_size]
    with (output_dir / "embedding_record_samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in samples:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [len(record["embedding_text"]) for record in records]
    summary = {
        "input_parsed_paths": len(records),
        "output_embedding_records": len(records),
        "unique_semantic_groups": len(catalog_rows),
        "deduplication_ratio": round(len(catalog_rows) / max(1, len(records)), 4),
        "quality_tier_counts": dict(quality_counts),
        "supported_record_percentage": round(
            100.0 * sum(record["route_supported"] for record in records) / max(1, len(records)), 4
        ),
        "records_with_unresolved_tokens": sum(record["has_unresolved_tokens"] for record in records),
        "records_with_observe_date_separated": sum(bool(record["observe_date"]) for record in records),
        "records_with_observe_year_separated": sum(bool(record["observe_year"]) for record in records),
        "embedding_text_length_min": min(lengths),
        "embedding_text_length_mean": round(sum(lengths) / len(lengths), 2),
        "embedding_text_length_max": max(lengths),
        "instance_time_leak_checks": "passed",
        "recommended_embedding_column": "embedding_text",
        "recommended_strict_ablation_column": "semantic_text_strict",
    }
    (output_dir / "step5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    quality_table = "\n".join(f"| {name} | {count} |" for name, count in quality_counts.most_common())
    top_catalog_table = "\n".join(
        f"| {row['path_count']} | {row['embedding_text'][:120]} | {row['example_paths'].split(' || ')[0][:100]} |"
        for row in catalog_rows[:15]
    )
    route_table = "\n".join(f"| {name} | {count} |" for name, count in route_counts.most_common(15))

    report = f"""# 步骤5：Embedding输入生成与全流程总结

## 1. 最终结果

已将步骤4的{len(records):,}条结构化路径全部转换为可直接送入文本Embedding模型的输入，并同时生成逐路径记录和去重语义目录。

- 逐路径Embedding记录：{len(records):,}条。
- 去重后的语义组：{len(catalog_rows):,}组。
- 专用路由记录占比：{summary['supported_record_percentage']}%。
- 观测日期已旁路分离：{summary['records_with_observe_date_separated']:,}条。
- 观测年份已旁路分离：{summary['records_with_observe_year_separated']:,}条。
- 静态文本时间泄漏检查：通过。
- 推荐送入BGE的列：`embedding_text`。

## 2. 本步骤输入了什么

输入是步骤4产生的`parsed_paths.csv.gz`，每条记录包含：

1. 原始目录路径、访问次数、聚合数据大小。
2. 解析路由与解析状态。
3. 卫星/平台、载荷、数据等级、产品、区域、周期、投影、分辨率、方向、处理阶段、格式等规范字段。
4. 每个字段的原始值、规范值、来源深度、置信度和证据。
5. 未确认业务Token、字段冲突和告警。

这里输入的仍然是目录级聚合数据，不是假设出来的具体文件名。

## 3. 是怎样处理的

### 3.1 构造静态语义文本

按固定顺序把规范字段写成带标签的短文本，例如：

```text
数据域 FY3；卫星或航天平台 FY3H；载荷或仪器 MERSI；数据等级 L2L3；产品 OCA；区域或轨道类型 ORBIT；空间分辨率 10KM
```

这种写法让BGE既看到代码，也看到代码在路径中的角色，比直接Embedding整条原始路径更稳定。

### 3.2 保留两种文本

- `embedding_text`：推荐主输入。包含全部已解释字段；不确定缩写以“原始业务标记”保留，但不虚构释义。
- `semantic_text_strict`：只保留置信度不低于{STRICT_CONFIDENCE}的解释字段，不加入未解释Token，适合做消融对比。

同时输出`semantic_sequence`，形式为：

```text
[DOMAIN=FY3][SATELLITE=FY3H][INSTRUMENT=MERSI][LEVEL=L2L3][PRODUCT=OCA]
```

它适合后续训练结构化Embedding模型或作为可审计输入。

### 3.3 将不应进入静态语义向量的信息旁路分离

以下字段不拼入`embedding_text`：

- `observe_year`、`observe_date`：它们表示数据实例时间，不是长期稳定含义。
- `access_count`：属于动态访问行为特征。
- `total_size_bytes`：属于资源和预取成本特征。
- `instance_ids`：单独放入实例上下文，避免编号主导语义距离。
- `archive_root`、`subsystem`、解析路由：放入`source_context`，可作为类别特征使用，但不让常见目录名前缀淹没产品语义。

产品自身的`DAILY`、`00HOUR`、`DAY/NIGHT`仍保留，因为它们定义产品类型或时次，不是某一天的实例日期。

### 3.4 质量分层

| 质量层 | 路径数 |
| --- | ---: |
{quality_table}

- `supported_clean`：专用路由且没有未解释Token，可直接用于第一版Embedding。
- `supported_with_unresolved`：主结构可信，但仍有业务缩写；建议保留并在后续代码表确认。
- `fallback_raw`：非卫星目录或低频结构；保守生成文本，不把它伪装为已确认卫星产品。

### 3.5 语义去重

对排除观测日期后的`semantic_sequence`做稳定哈希，生成`semantic_group_id`。同一产品不同日期通常共享一个语义组，因此只需对`semantic_catalog.csv.gz`中的每个语义组计算一次Embedding，再把向量映射回路径，可减少重复推理和存储。

## 4. 输出了什么

### `embedding_records.csv.gz`

每条原始路径一行，关键列为：

- `embedding_text`：推荐的BGE输入。
- `semantic_text_strict`：严格置信版本。
- `semantic_sequence`：结构化语义序列。
- `semantic_group_id`：去重语义ID。
- `source_context`：存储域、子系统和解析路由。
- `instance_context`：观测年份、日期和实例编号。
- `access_count`、`total_size_bytes`：预取模型旁路特征。
- `quality_tier`和置信度：训练过滤与抽样依据。

### `semantic_catalog.csv.gz`

每个唯一静态语义一行，同时聚合路径数、访问次数、数据大小、路由分布和示例路径。实际批量调用BGE时，优先对这个文件的`embedding_text`编码。

### 其他文件

- `embedding_record_samples.jsonl`：可读样例。
- `step5_summary.json`：机器可读统计。
- 本报告：输入、输出、处理过程和使用建议。

## 5. 高频去重语义示例

| 对应路径数 | Embedding文本 | 示例路径 |
| ---: | --- | --- |
{top_catalog_table}

## 6. 主要路由规模

| 路由 | 记录数 |
| --- | ---: |
{route_table}

## 7. 完整流程总结

```text
49,922条目录聚合路径
  → 审计真实骨架，否定固定第N层规则
  → Token候选识别，保留未知与歧义
  → 按目录分支建立状态机解析器
  → 扩展高价值分支，专用覆盖达到99.5673%
  → 规范字段与原始值并存，记录置信度和证据
  → 分离实例时间、访问热度、大小和存储位置
  → 生成带字段标签的静态语义文本
  → 按静态语义去重
  → 得到可供BGE编码的语义目录和逐路径映射
```

## 8. 使用建议

第一版建议直接使用BGE对`semantic_catalog.csv.gz`的`embedding_text`编码，不必立即微调。将得到的静态语义向量通过`semantic_group_id`映射回每条路径；访问次数、长期访问序列、文件大小和实例时间进入历史行为模型或最终打分层，不要拼进BGE文本。

评估时至少比较三组：原始路径文本、`semantic_text_strict`、推荐的`embedding_text`。最终指标应使用预取任务的Recall@K、字节命中率和无效预取率，而不能只看文本相似度。
"""
    (output_dir / "步骤5_Embedding输入生成与全流程总结.md").write_text(report, encoding="utf-8")
    print("STEP5_EMBEDDING_INPUTS_COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
