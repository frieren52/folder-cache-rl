from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PIPELINE_ROOT = Path(__file__).resolve().parent
ROOT = PIPELINE_ROOT.parent.parent / "analysis" / "lru_miss_attribution_analysis"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORT_PATH = ROOT / "report.md"
MANIFEST_PATH = ROOT / "manifest.json"
FLOW_PATH = FIGURES / "miss_attribution_flow.png"
COMPOSITION_PATH = FIGURES / "miss_cause_composition.png"

CAUSES = (
    "cold_start",
    "capacity_eviction",
    "oversize",
    "bandwidth_queue",
    "bandwidth_transfer",
    "bandwidth_state_divergence",
)
CAUSE_LABELS = {
    "cold_start": "冷启动",
    "capacity_eviction": "容量淘汰后再访问",
    "oversize": "对象超过缓存容量",
    "bandwidth_queue": "带宽：仍在排队",
    "bandwidth_transfer": "带宽：仍在传输",
    "bandwidth_state_divergence": "带宽：状态分叉后重新取回",
}
OP_LABELS = {
    "trigger_new_transfer": "新触发物理取回",
    "queued_merge": "合并到排队任务",
    "inflight_merge": "合并到在途任务",
}
WAIT_LABELS = {
    "lt_1s": "<1 秒",
    "1_5s": "1–5 秒",
    "5_30s": "5–30 秒",
    "30_60s": "30–60 秒",
    "1_5m": "1–5 分钟",
    "5_30m": "5–30 分钟",
    "30_60m": "30–60 分钟",
    "ge_1h": "≥1 小时",
}
WAIT_ORDER = tuple(WAIT_LABELS)
SCENARIOS = (
    "瞬时取回",
    "10通道 × 200 MiB/s",
    "30通道 × 200 MiB/s",
    "60通道 × 200 MiB/s",
)
SHORT_SCENARIOS = {
    "瞬时取回": "瞬时",
    "10通道 × 200 MiB/s": "10 通道",
    "30通道 × 200 MiB/s": "30 通道",
    "60通道 × 200 MiB/s": "60 通道",
}

BACKGROUND = (250, 251, 253)
FOREGROUND = (28, 37, 50)
MUTED = (87, 99, 116)
GRID = (218, 224, 232)
FRAME = (151, 164, 183)
NODE = (238, 243, 249)
DECISION = (230, 238, 248)
CAUSE_COLORS = {
    "cold_start": (55, 111, 176),
    "capacity_eviction": (213, 94, 0),
    "oversize": (180, 55, 87),
    "bandwidth_queue": (126, 87, 194),
    "bandwidth_transfer": (24, 145, 90),
    "bandwidth_state_divergence": (117, 127, 145),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_pairs = (
        (
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/msyhbd.ttc"),
        ),
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    )
    for regular_path, bold_path in font_pairs:
        candidate = bold_path if bold else regular_path
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise RuntimeError("未找到可用字体；Windows 需要微软雅黑，Linux 需要 Noto Sans CJK 或 DejaVu Sans")


TITLE_FONT = font(38, True)
SUBTITLE_FONT = font(26, True)
BODY_FONT = font(20)
BODY_BOLD_FONT = font(20, True)
SMALL_FONT = font(17)
TINY_FONT = font(15)


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_data() -> dict[str, list[dict[str, str]] | dict[str, str]]:
    with (TABLES / "dataset_summary.csv").open("r", encoding="utf-8-sig", newline="") as file:
        summary = next(csv.DictReader(file))
    data: dict[str, list[dict[str, str]] | dict[str, str]] = {
        "summary": summary,
        "results": read_csv("lru_results.csv"),
        "causes": read_csv("miss_cause_summary.csv"),
        "operations": read_csv("miss_operational_state.csv"),
        "waits": read_csv("miss_wait_buckets.csv"),
        "diagnostics": read_csv("miss_diagnostics.csv"),
        "top": read_csv("miss_top_objects.csv"),
    }
    validate(data)
    return data


def validate(data: dict[str, list[dict[str, str]] | dict[str, str]]) -> None:
    results = data["results"]
    causes = data["causes"]
    operations = data["operations"]
    waits = data["waits"]
    diagnostics = data["diagnostics"]
    assert isinstance(results, list) and len(results) == 12
    assert isinstance(causes, list) and len(causes) == 12 * len(CAUSES)
    assert isinstance(operations, list) and len(operations) == 9 * 3
    assert isinstance(waits, list) and len(waits) == 9 * len(WAIT_ORDER)
    assert isinstance(diagnostics, list) and len(diagnostics) == 12
    assert all(row["validation_status"] == "passed" for row in results)
    assert all(row["validation_status"] == "passed" for row in diagnostics)

    grouped_causes: defaultdict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in causes:
        grouped_causes[(row["scenario"], int(row["capacity_percent"]))].append(row)
    for key, rows in grouped_causes.items():
        miss_events = int(rows[0]["miss_events"])
        miss_bytes = int(rows[0]["miss_bytes"])
        if sum(int(row["cause_events"]) for row in rows) != miss_events:
            raise RuntimeError(f"原因次数未闭合：{key}")
        if sum(int(row["cause_bytes"]) for row in rows) != miss_bytes:
            raise RuntimeError(f"原因字节未闭合：{key}")


def center_text(draw: ImageDraw.ImageDraw, center: tuple[float, float], text: str, text_font, fill=FOREGROUND) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=5, align="center")
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(
        (center[0] - width / 2, center[1] - height / 2),
        text,
        font=text_font,
        fill=fill,
        spacing=5,
        align="center",
    )


def node(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill=NODE, outline=FRAME) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)
    center_text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, BODY_FONT)


def decision(draw: ImageDraw.ImageDraw, center: tuple[int, int], width: int, height: int, text: str) -> None:
    x, y = center
    points = ((x, y - height // 2), (x + width // 2, y), (x, y + height // 2), (x - width // 2, y))
    draw.polygon(points, fill=DECISION)
    draw.line((*points, points[0]), fill=FRAME, width=2, joint="curve")
    center_text(draw, center, text, BODY_FONT)


def arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    label: str | None = None,
    label_at: tuple[int, int] | None = None,
) -> None:
    draw.line(points, fill=MUTED, width=3, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 or 1
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    px, py = -uy, ux
    tip = (x2, y2)
    left = (int(x2 - ux * 15 + px * 7), int(y2 - uy * 15 + py * 7))
    right = (int(x2 - ux * 15 - px * 7), int(y2 - uy * 15 - py * 7))
    draw.polygon((tip, left, right), fill=MUTED)
    if label and label_at:
        draw.text(label_at, label, font=SMALL_FONT, fill=MUTED)


def render_flow() -> None:
    image = Image.new("RGB", (2000, 1120), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), "LRU MISS 逐事件反事实归因流程", font=TITLE_FONT, fill=FOREGROUND)
    draw.text((60, 88), "有限带宽模型与同容量瞬时 LRU 在同一访问事件上对照", font=BODY_FONT, fill=MUTED)

    node(draw, (55, 430, 245, 530), "请求到达")
    decision(draw, (425, 480), 280, 170, "有限带宽模型\n当前命中？")
    node(draw, (330, 170, 520, 260), "计为 HIT", fill=(226, 244, 236), outline=(83, 151, 116))
    node(draw, (615, 430, 850, 530), "计为有限模型 MISS")
    decision(draw, (1050, 480), 310, 180, "同容量瞬时 LRU\n此事件命中？")

    node(draw, (910, 170, 1190, 270), "检查有限模型对象状态")
    node(draw, (1320, 95, 1700, 180), "bandwidth_queue\n对象仍在中央 FIFO")
    node(draw, (1320, 220, 1700, 305), "bandwidth_transfer\n对象已占通道传输")
    node(draw, (1320, 345, 1700, 430), "bandwidth_state_divergence\n对象已完全缺失，重新取回")

    node(draw, (900, 710, 1200, 810), "沿用瞬时 LRU 的 MISS 原因")
    node(draw, (1320, 625, 1700, 710), "cold_start\n轨迹内首次观测")
    node(draw, (1320, 750, 1700, 835), "capacity_eviction\n曾访问但已被 LRU 淘汰")
    node(draw, (1320, 875, 1700, 960), "oversize\n对象本身超过该容量")

    arrow(draw, [(245, 480), (285, 480)], None)
    arrow(draw, [(425, 395), (425, 260)], "是", (445, 310))
    arrow(draw, [(565, 480), (615, 480)], "否", (575, 445))
    arrow(draw, [(850, 480), (895, 480)])
    arrow(draw, [(1050, 390), (1050, 270)], "是", (1070, 320))
    arrow(draw, [(1050, 570), (1050, 710)], "否", (1070, 625))

    for y1, y2 in ((205, 137), (220, 262), (235, 387)):
        arrow(draw, [(1190, y1), (1245, y1), (1245, y2), (1320, y2)])
    for y1, y2 in ((740, 667), (760, 792), (780, 917)):
        arrow(draw, [(1200, y1), (1245, y1), (1245, y2), (1320, y2)])

    draw.rounded_rectangle((55, 900, 1160, 1050), radius=18, fill=(244, 246, 249), outline=GRID, width=2)
    draw.text((82, 925), "有限带宽 MISS 同时记录一条操作状态轴", font=BODY_BOLD_FONT, fill=FOREGROUND)
    draw.text(
        (82, 970),
        "完全缺失 → 新触发物理取回　｜　排队中 → 合并到排队任务　｜　传输中 → 合并到在途任务",
        font=BODY_FONT,
        fill=MUTED,
    )
    draw.text((82, 1010), "操作状态用于工程诊断，不与上方六类主原因重复计数。", font=SMALL_FONT, fill=MUTED)

    FIGURES.mkdir(parents=True, exist_ok=True)
    image.save(FLOW_PATH, optimize=True)


def render_composition(cause_rows: list[dict[str, str]]) -> None:
    by_key = {
        (row["scenario"], int(row["capacity_percent"]), row["cause"]): row
        for row in cause_rows
    }
    image = Image.new("RGB", (2050, 1700), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), "12 个 LRU 场景的 MISS 原因构成", font=TITLE_FONT, fill=FOREGROUND)

    legend_x, legend_y = 60, 100
    for index, cause in enumerate(CAUSES):
        x = legend_x + (index % 3) * 625
        y = legend_y + (index // 3) * 44
        draw.rectangle((x, y + 5, x + 26, y + 27), fill=CAUSE_COLORS[cause])
        draw.text((x + 38, y), CAUSE_LABELS[cause], font=SMALL_FONT, fill=FOREGROUND)

    rows = [(scenario, capacity) for scenario in SCENARIOS for capacity in (1, 3, 10)]

    def panel(top: int, title: str, metric: str) -> None:
        draw.text((60, top), title, font=SUBTITLE_FONT, fill=FOREGROUND)
        left, right = 330, 1970
        chart_top = top + 55
        row_height = 45
        bar_height = 27
        for tick in range(0, 101, 20):
            x = left + (right - left) * tick / 100
            draw.line((x, chart_top - 12, x, chart_top + row_height * len(rows) - 10), fill=GRID, width=1)
            draw.text((x - 13, chart_top + row_height * len(rows) - 4), f"{tick}%", font=TINY_FONT, fill=MUTED)

        for row_index, (scenario, capacity) in enumerate(rows):
            y = chart_top + row_index * row_height
            if row_index in (3, 6, 9):
                draw.line((60, y - 9, right, y - 9), fill=FRAME, width=1)
            label = f"{SHORT_SCENARIOS[scenario]}  {capacity}%"
            draw.text((60, y + 2), label, font=SMALL_FONT, fill=FOREGROUND)
            cursor = left
            for cause in CAUSES:
                share = float(by_key[(scenario, capacity, cause)][metric])
                width = (right - left) * share
                if width > 0:
                    draw.rectangle((cursor, y, cursor + width, y + bar_height), fill=CAUSE_COLORS[cause])
                    if share >= 0.06:
                        text = f"{share:.0%}"
                        box = draw.textbbox((0, 0), text, font=TINY_FONT)
                        text_width = box[2] - box[0]
                        if width >= text_width + 12:
                            draw.text((cursor + width / 2 - text_width / 2, y + 3), text, font=TINY_FONT, fill=(255, 255, 255))
                cursor += width
            draw.rectangle((left, y, right, y + bar_height), outline=FRAME, width=1)
        draw.text((left + 690, chart_top + row_height * len(rows) + 28), "该场景全部 MISS 中的占比", font=SMALL_FONT, fill=MUTED)

    panel(215, "按 I/O 次数归因", "event_share")
    panel(920, "按对象大小代理字节归因", "byte_share")
    image.save(COMPOSITION_PATH, optimize=True)


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"


def fmt_int(value: str | int) -> str:
    return f"{int(value):,}"


def fmt_bytes(value: str | int) -> str:
    number = int(value)
    for divisor, unit in ((1024**5, "PiB"), (1024**4, "TiB"), (1024**3, "GiB"), (1024**2, "MiB")):
        if number >= divisor:
            return f"{number / divisor:,.2f} {unit}"
    return f"{number:,} B"


def fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:,.2f} 小时"
    if seconds >= 60:
        return f"{seconds / 60:,.2f} 分钟"
    return f"{seconds:,.2f} 秒"


def generate_report(data: dict[str, list[dict[str, str]] | dict[str, str]]) -> None:
    summary = data["summary"]
    results = data["results"]
    causes = data["causes"]
    operations = data["operations"]
    waits = data["waits"]
    diagnostics = data["diagnostics"]
    top_rows = data["top"]
    assert isinstance(summary, dict)
    assert all(isinstance(value, list) for value in (results, causes, operations, waits, diagnostics, top_rows))

    result_by_key = {(row["scenario"], int(row["capacity_percent"])): row for row in results}
    cause_by_key = {
        (row["scenario"], int(row["capacity_percent"]), row["cause"]): row
        for row in causes
    }
    op_by_key = {
        (row["scenario"], int(row["capacity_percent"]), row["operational_state"]): row
        for row in operations
    }
    wait_by_key = {
        (row["scenario"], int(row["capacity_percent"]), row["wait_bucket"]): row
        for row in waits
    }
    diag_by_key = {(row["scenario"], int(row["capacity_percent"])): row for row in diagnostics}

    instant3_cold = cause_by_key[("瞬时取回", 3, "cold_start")]
    instant3_capacity = cause_by_key[("瞬时取回", 3, "capacity_eviction")]
    ten3_queue = cause_by_key[("10通道 × 200 MiB/s", 3, "bandwidth_queue")]
    sixty3_transfer = cause_by_key[("60通道 × 200 MiB/s", 3, "bandwidth_transfer")]
    ten3_diag = diag_by_key[("10通道 × 200 MiB/s", 3)]
    sixty3_diag = diag_by_key[("60通道 × 200 MiB/s", 3)]

    report: list[str] = [
        "# 文件夹粒度 LRU MISS 归因分析",
        "",
        "## 1. 结论",
        "",
        f"- **容量本身是瞬时模型的主矛盾。** 3% 容量、瞬时取回下共有 {fmt_int(result_by_key[('瞬时取回', 3)]['miss_events'])} 次 MISS，其中冷启动占 {pct(float(instant3_cold['event_share']))}，容量淘汰后再访问占 {pct(float(instant3_capacity['event_share']))}；按代理字节分别占 {pct(float(instant3_cold['byte_share']))} 与 {pct(float(instant3_capacity['byte_share']))}。",
        f"- **10 通道首先受排队约束。** 3% 容量时，`bandwidth_queue` 占有限模型 MISS 次数的 {pct(float(ten3_queue['event_share']))}、MISS 代理字节的 {pct(float(ten3_queue['byte_share']))}；平均未就绪等待为 {fmt_duration(float(ten3_diag['mean_wait_seconds']))}。",
        f"- **通道增至 60 后，瓶颈由排队转向单对象传输时延。** 3% 容量时，`bandwidth_transfer` 占 MISS 次数的 {pct(float(sixty3_transfer['event_share']))}、代理字节的 {pct(float(sixty3_transfer['byte_share']))}；平均等待降至 {fmt_duration(float(sixty3_diag['mean_wait_seconds']))}。",
        f"- **带宽状态分叉很少。** 3% 容量的 10/30/60 通道场景中，该原因分别只有 {fmt_int(cause_by_key[('10通道 × 200 MiB/s', 3, 'bandwidth_state_divergence')]['cause_events'])}/{fmt_int(cause_by_key[('30通道 × 200 MiB/s', 3, 'bandwidth_state_divergence')]['cause_events'])}/{fmt_int(cause_by_key[('60通道 × 200 MiB/s', 3, 'bandwidth_state_divergence')]['cause_events'])} 次，说明主问题不是两套 LRU 状态偶然分叉，而是对象未及时就绪。",
        "",
        "## 2. 数据与归因口径",
        "",
        f"分析覆盖 `{summary['start_time']}` 至 `{summary['end_time']}` 的 7 份日志，共 {fmt_int(summary['event_count'])} 次访问、{fmt_int(summary['object_count'])} 个文件夹对象。沿用基线的 1%/3%/10% 总容量，以及瞬时取回、10/30/60 通道 × `200 MiB/s` 共 12 个场景。",
        "",
        "对每个容量，先回放一套瞬时 LRU，并保存每个事件是 HIT 还是哪种固有 MISS；随后有限带宽模型在同一事件上与这套结果对照。主原因严格互斥：",
        "",
        "1. `cold_start`：对象不超过容量，且为轨迹内首次观测。",
        "2. `capacity_eviction`：对象曾被访问，但此时不在同容量瞬时 LRU 中。",
        "3. `oversize`：对象自身大于该场景缓存容量；该判断优先于首次观测。",
        "4. `bandwidth_queue`：瞬时模型本应 HIT，但有限模型中的对象仍在中央 FIFO。",
        "5. `bandwidth_transfer`：瞬时模型本应 HIT，但对象仍占用通道传输。",
        "6. `bandwidth_state_divergence`：瞬时模型本应 HIT，但有限模型已无该对象，需要重新触发取回。",
        "",
        "有限带宽 MISS 还同时记录一条操作状态轴：新触发物理取回、合并到排队任务、合并到在途任务。它回答‘这次 MISS 如何被系统处理’，不与六类主原因重复累加。",
        "",
        "## 3. MISS 归因流程",
        "",
        "![LRU MISS 逐事件反事实归因流程](figures/miss_attribution_flow.png)",
        "",
        "每个场景均满足：",
        "",
        r"\[\sum_c N_{miss,c}=N_{miss},\qquad \sum_c B_{miss,c}=B_{request}-B_{hit}\]",
        "",
        "## 4. 12 场景原因构成",
        "",
        "![12 个 LRU 场景的 MISS 原因构成](figures/miss_cause_composition.png)",
        "",
        "| 场景 | 容量 | MISS 次数 | 首要次数原因 | 次数占比 | 首要字节原因 | 字节占比 |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for scenario in SCENARIOS:
        for capacity in (1, 3, 10):
            candidates = [cause_by_key[(scenario, capacity, cause)] for cause in CAUSES]
            top_event = max(candidates, key=lambda row: int(row["cause_events"]))
            top_byte = max(candidates, key=lambda row: int(row["cause_bytes"]))
            result = result_by_key[(scenario, capacity)]
            report.append(
                f"| {scenario} | {capacity}% | {fmt_int(result['miss_events'])} | {CAUSE_LABELS[top_event['cause']]} | "
                f"{pct(float(top_event['event_share']))} | {CAUSE_LABELS[top_byte['cause']]} | {pct(float(top_byte['byte_share']))} |"
            )

    report.extend(
        [
            "",
            "完整原因数值见 [miss_cause_summary.csv](tables/miss_cause_summary.csv)。",
            "",
            "## 5. 3% 容量重点拆解",
            "",
            "### 5.1 主原因",
            "",
            "| 场景 | 原因 | MISS 次数 | 次数占比 | 代理字节 | 字节占比 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in SCENARIOS:
        for cause in CAUSES:
            row = cause_by_key[(scenario, 3, cause)]
            if int(row["cause_events"]) == 0:
                continue
            report.append(
                f"| {scenario} | {CAUSE_LABELS[cause]} | {fmt_int(row['cause_events'])} | {pct(float(row['event_share']))} | "
                f"{fmt_bytes(row['cause_bytes'])} | {pct(float(row['byte_share']))} |"
            )

    report.extend(
        [
            "",
            "### 5.2 有限带宽操作状态",
            "",
            "| 场景 | 新触发取回 | 合并到排队 | 合并到在途 | 物理取回提交数 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in SCENARIOS[1:]:
        result = result_by_key[(scenario, 3)]
        values = [op_by_key[(scenario, 3, code)] for code in OP_LABELS]
        report.append(
            f"| {scenario} | {fmt_int(values[0]['state_events'])} ({pct(float(values[0]['event_share']))}) | "
            f"{fmt_int(values[1]['state_events'])} ({pct(float(values[1]['event_share']))}) | "
            f"{fmt_int(values[2]['state_events'])} ({pct(float(values[2]['event_share']))}) | {fmt_int(result['submitted_transfers'])} |"
        )

    report.extend(
        [
            "",
            "`MISS 次数` 大于 `物理取回提交数` 是正常现象：同一对象排队或传输期间的重复访问被合并到已有任务，但每次访问在对象就绪前仍计为 MISS。完整数据见 [miss_operational_state.csv](tables/miss_operational_state.csv)。",
            "",
            "### 5.3 等待时间分布",
            "",
            "等待时间定义为请求到达时刻到对应对象本次物理取回完成时刻的差；同一在途对象的后续请求会有更短的剩余等待。",
            "",
            "| 等待桶 | 10 通道 | 30 通道 | 60 通道 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for bucket in WAIT_ORDER:
        cells = []
        for scenario in SCENARIOS[1:]:
            row = wait_by_key[(scenario, 3, bucket)]
            cells.append(f"{fmt_int(row['bucket_events'])} ({pct(float(row['event_share']))})")
        report.append(f"| {WAIT_LABELS[bucket]} | " + " | ".join(cells) + " |")

    report.extend(
        [
            "",
            "完整等待桶及字节占比见 [miss_wait_buckets.csv](tables/miss_wait_buckets.csv)。",
            "",
            "## 6. 淘汰、重复取回与合并效率",
            "",
            "| 场景 | 容量 | 淘汰次数 | 物理取回 | 重复取回 | 重复取回占提交 | 平均等待 | 最大等待 | 有限模型独有 HIT |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in SCENARIOS:
        for capacity in (1, 3, 10):
            row = diag_by_key[(scenario, capacity)]
            report.append(
                f"| {scenario} | {capacity}% | {fmt_int(row['evictions'])} | {fmt_int(row['submitted_transfers'])} | "
                f"{fmt_int(row['refetch_transfers'])} | {pct(float(row['refetch_transfer_share']))} | "
                f"{fmt_duration(float(row['mean_wait_seconds']))} | {fmt_duration(float(row['max_wait_seconds']))} | "
                f"{fmt_int(row['counterfactual_finite_only_hits'])} |"
            )

    report.extend(
        [
            "",
            "`有限模型独有 HIT` 表示该事件在有限模型中命中、但同容量瞬时 LRU 在其自身状态中 MISS。它不属于有限模型 MISS，仅用于揭示两套缓存状态可能因完成时刻不同而分叉。完整诊断见 [miss_diagnostics.csv](tables/miss_diagnostics.csv)。",
            "",
            "## 7. 60 通道、3% 容量的高频 MISS 对象",
            "",
            "| 排名 | path_index | 文件夹路径 | 对象大小 | 总访问 | MISS | 主要原因 |",
            "| ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    selected_top = [
        row
        for row in top_rows
        if row["scenario"] == "60通道 × 200 MiB/s" and row["capacity_percent"] == "3"
    ][:10]
    for row in selected_top:
        cause_counts = {cause: int(row[f"{cause}_events"]) for cause in CAUSES}
        primary = max(cause_counts, key=cause_counts.get)
        escaped_path = row["path"].replace("|", r"\|")
        report.append(
            f"| {row['rank']} | {row['path_index']} | `{escaped_path}` | {fmt_bytes(row['size_bytes'])} | "
            f"{fmt_int(row['total_accesses'])} | {fmt_int(row['miss_events'])} | {CAUSE_LABELS[primary]} |"
        )

    report.extend(
        [
            "",
            "所有场景各自的 Top 20 明细见 [miss_top_objects.csv](tables/miss_top_objects.csv)。",
            "",
            "## 8. 可复现文件",
            "",
            "- [simulate_lru.cs](../../pipelines/lru_miss_attribution_analysis/simulate_lru.cs)：12 场景回放与在线 MISS 归因。",
            "- [render_miss_attribution.py](../../pipelines/lru_miss_attribution_analysis/render_miss_attribution.py)：流程图、构成图和本报告。",
            "- [run_analysis.ps1](../../pipelines/lru_miss_attribution_analysis/run_analysis.ps1)：完整运行入口。",
            "- [miss_cause_summary.csv](tables/miss_cause_summary.csv)：六类主原因。",
            "- [miss_operational_state.csv](tables/miss_operational_state.csv)：有限带宽操作状态。",
            "- [miss_wait_buckets.csv](tables/miss_wait_buckets.csv)：等待时间分布。",
            "- [miss_diagnostics.csv](tables/miss_diagnostics.csv)：淘汰、重复取回和等待诊断。",
            "- [miss_top_objects.csv](tables/miss_top_objects.csv)：各场景 Top 20 MISS 对象。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "status": "passed",
        "method": "per-event counterfactual against same-capacity instant LRU",
        "scenario_count": 12,
        "cause_codes": list(CAUSES),
        "cause_event_closure": True,
        "cause_byte_closure": True,
        "outputs": [
            REPORT_PATH.name,
            str(FLOW_PATH.relative_to(ROOT)),
            str(COMPOSITION_PATH.relative_to(ROOT)),
            "tables/miss_cause_summary.csv",
            "tables/miss_operational_state.csv",
            "tables/miss_wait_buckets.csv",
            "tables/miss_diagnostics.csv",
            "tables/miss_top_objects.csv",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    data = load_data()
    causes = data["causes"]
    assert isinstance(causes, list)
    render_flow()
    render_composition(causes)
    generate_report(data)
    print(REPORT_PATH)
    print(FLOW_PATH)
    print(COMPOSITION_PATH)


if __name__ == "__main__":
    main()
