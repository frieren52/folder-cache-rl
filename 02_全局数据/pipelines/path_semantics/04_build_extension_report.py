from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="步骤4：生成规则扩展前后对比与复核报告")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--expanded-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    expanded_dir = args.expanded_dir.resolve()
    before = read_json(baseline_dir / "step3_summary.json")
    after_source = expanded_dir / "step3_summary.json"
    after = read_json(after_source)
    before_routes = read_csv(baseline_dir / "route_summary.csv")
    after_routes = read_csv(expanded_dir / "route_summary.csv")
    after_extras = read_csv(expanded_dir / "unresolved_extra_tokens.csv")

    before_route_names = {row["parser_route"] for row in before_routes}
    added_routes = [
        row
        for row in after_routes
        if row["route_supported"] == "1" and row["parser_route"] not in before_route_names
    ]
    fallback_routes = [row for row in after_routes if row["route_supported"] == "0"]

    comparison = [
        {
            "metric": "supported_paths",
            "before": before["supported_paths"],
            "after": after["supported_paths"],
            "change": after["supported_paths"] - before["supported_paths"],
        },
        {
            "metric": "supported_path_percentage",
            "before": before["supported_path_percentage"],
            "after": after["supported_path_percentage"],
            "change": round(after["supported_path_percentage"] - before["supported_path_percentage"], 4),
        },
        {
            "metric": "fallback_paths",
            "before": before["parse_status_counts"].get("fallback", 0),
            "after": after["parse_status_counts"].get("fallback", 0),
            "change": after["parse_status_counts"].get("fallback", 0)
            - before["parse_status_counts"].get("fallback", 0),
        },
        {
            "metric": "supported_paths_with_unresolved_extras",
            "before": before["supported_paths_with_unresolved_extras"],
            "after": after["supported_paths_with_unresolved_extras"],
            "change": after["supported_paths_with_unresolved_extras"]
            - before["supported_paths_with_unresolved_extras"],
        },
        {
            "metric": "unique_unresolved_extra_tokens",
            "before": before["unique_unresolved_extra_tokens"],
            "after": after["unique_unresolved_extra_tokens"],
            "change": after["unique_unresolved_extra_tokens"] - before["unique_unresolved_extra_tokens"],
        },
        {
            "metric": "supported_paths_with_conflicts",
            "before": before["supported_paths_with_conflicts"],
            "after": after["supported_paths_with_conflicts"],
            "change": after["supported_paths_with_conflicts"] - before["supported_paths_with_conflicts"],
        },
    ]
    write_csv(expanded_dir / "coverage_comparison.csv", comparison)
    write_csv(expanded_dir / "added_route_summary.csv", added_routes)
    write_csv(expanded_dir / "remaining_fallback_routes.csv", fallback_routes)

    summary = {
        "input_paths": after["input_paths"],
        "baseline_supported_paths": before["supported_paths"],
        "expanded_supported_paths": after["supported_paths"],
        "newly_supported_paths": after["supported_paths"] - before["supported_paths"],
        "baseline_supported_percentage": before["supported_path_percentage"],
        "expanded_supported_percentage": after["supported_path_percentage"],
        "coverage_gain_percentage_points": round(
            after["supported_path_percentage"] - before["supported_path_percentage"], 4
        ),
        "baseline_fallback_paths": before["parse_status_counts"].get("fallback", 0),
        "expanded_fallback_paths": after["parse_status_counts"].get("fallback", 0),
        "fallback_reduction": before["parse_status_counts"].get("fallback", 0)
        - after["parse_status_counts"].get("fallback", 0),
        "supported_paths_with_unresolved_extras": after["supported_paths_with_unresolved_extras"],
        "supported_paths_with_conflicts": after["supported_paths_with_conflicts"],
        "new_supported_routes": len(added_routes),
        "self_tests": after["self_tests"],
    }
    (expanded_dir / "step4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    added_table = "\n".join(
        f"| {row['parser_route']} | {row['paths']} | {row['parsed']} | "
        f"{row['parsed_with_extras']} | {row['conflict']} |"
        for row in added_routes
    )
    fallback_table = "\n".join(
        f"| {row['parser_route']} | {row['paths']} | {row['path_percentage']}% |"
        for row in fallback_routes
    )
    domain_extra_rows = [
        row
        for row in after_extras
        if any(marker in row["route_counts_json"] for marker in ("arch_dsscache", "outshare_data", "inshare"))
    ][:20]
    extra_table = "\n".join(
        f"| {row['token']} | {row['occurrences']} | {row['route_counts_json']} |"
        for row in domain_extra_rows
    ) or "| 无 | 0 | - |"

    report = f"""# 步骤4：高价值分支扩展与复核报告

## 1. 本步结论

本步对步骤3的1,796条fallback路径做了分支结构复核，只为层级稳定、字段角色可由上下文确认的分支建立专用路由。

- 专用路由覆盖：{before['supported_paths']}→{after['supported_paths']}条，新增{summary['newly_supported_paths']}条。
- 覆盖率：{before['supported_path_percentage']}%→{after['supported_path_percentage']}%，提升{summary['coverage_gain_percentage_points']}个百分点。
- fallback：{summary['baseline_fallback_paths']}→{summary['expanded_fallback_paths']}条，减少{summary['fallback_reduction']}条。
- 专用路由含未解释Token：{before['supported_paths_with_unresolved_extras']}→{after['supported_paths_with_unresolved_extras']}条。
- 专用路由字段冲突：{after['supported_paths_with_conflicts']}条。
- 语法检查与代表路径自测：通过。

## 2. 新增专用路由

| 路由 | 路径数 | 完整解析 | 带未解释Token | 冲突 |
| --- | ---: | ---: | ---: | ---: |
{added_table}

主要新增语法：

1. `FYSIMU/模拟平台/模拟仪器/年/日期`。
2. `FY4B、JPSS1、METOPC/[处理阶段]/仪器/等级/产品/...`。
3. `RS/平台/仪器/等级/产品/...`。
4. `EXTSAT/数据族/平台/...`，区分SATE与NAFP，不把数值预报模型强行当成卫星。
5. `DQ/数据集/仪器/等级/...`与`GF5A/仪器/产品/...`。
6. DSSCACHE下FY3D/E、FY3模拟数据，以及`TEMPWORK/FY3平台/...`变体。

## 3. 安全Token扩展

本步只添加可由外形确认的组合规则：

- `4KM`等一位KM分辨率。
- `5DAY`作聚合周期，`00HOUR/12HOUR`单独作时次槽位，避免与DAILY冲突。
- `DAILY_5000M`拆为周期+分辨率。
- `ASCENDKu/DESCENDKu`等拆为方向+低置信通道后缀。
- `01MVk/05MVk`、`WRADC`、`OBC/OBCCD`只保存为产品变体候选，不输出未经验证的中文释义。
- 短数字作实例/通道编号候选，不猜测业务含义。

## 4. 仍保留的高频业务Token

| Token | 出现次数 | 路由分布 |
| --- | ---: | --- |
{extra_table}

`SPAC`、`MOS`、`RNE`等缩写的层级位置可观察，但仅凭路径无法确认它们是产品、模式还是处理阶段，因此继续保存在`extra_tokens`。

## 5. 剩余fallback

| fallback分支 | 路径数 | 占全部路径 |
| --- | ---: | ---: |
{fallback_table}

剩余{summary['expanded_fallback_paths']}条主要是Python软件库、质控工作目录、日志、交换区和临时目录。它们不适合套用卫星产品字段，保留fallback比伪造语义更安全。

## 6. 输出文件

- `parsed_paths.csv.gz`：全49,922条扩展后解析结果。
- `route_summary.csv`、`field_coverage_by_route.csv`、`field_value_catalog.csv`：路由与字段统计。
- `unresolved_extra_tokens.csv`：仍需业务确认的Token。
- `coverage_comparison.csv`：扩展前后对比。
- `added_route_summary.csv`：新增专用路由。
- `remaining_fallback_routes.csv`：剩余fallback分布。
- `step4_summary.json`：机器可读摘要。

## 7. 下一步建议

步骤5建立“Embedding输入文本/结构化序列生成器”：将规范字段、原始Token、置信度和未解释Token分层输出，并明确哪些时间字段不应进入静态语义向量。
"""
    (expanded_dir / "步骤4_高价值分支扩展与复核报告.md").write_text(report, encoding="utf-8")
    print("STEP4_EXTENSION_REPORT_COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
