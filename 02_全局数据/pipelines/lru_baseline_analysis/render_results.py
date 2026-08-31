from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PIPELINE_ROOT = Path(__file__).resolve().parent
ROOT = PIPELINE_ROOT.parent.parent / "analysis" / "lru_baseline_analysis"
RESULTS_PATH = ROOT / "tables" / "lru_results.csv"
SUMMARY_PATH = ROOT / "tables" / "dataset_summary.csv"
FIGURE_PATH = ROOT / "figures" / "lru_hit_rates.png"
REPORT_PATH = ROOT / "report.md"
MANIFEST_PATH = ROOT / "manifest.json"

BACKGROUND = (250, 251, 253)
FOREGROUND = (29, 39, 52)
MUTED = (91, 101, 116)
GRID = (218, 224, 232)
FRAME = (157, 168, 183)
SERIES = {
    "瞬时取回": ((48, 108, 170), "瞬时取回"),
    "10通道 × 200 MiB/s": ((213, 94, 0), "10 通道"),
    "30通道 × 200 MiB/s": ((126, 87, 194), "30 通道"),
    "60通道 × 200 MiB/s": ((24, 145, 90), "60 通道"),
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
PANEL_FONT = font(27, True)
AXIS_FONT = font(18)
VALUE_FONT = font(16)
LEGEND_FONT = font(18)


def load_data() -> tuple[dict[str, str], list[dict[str, str]]]:
    with SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        summary = next(csv.DictReader(file))
    with RESULTS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 12:
        raise RuntimeError(f"结果应有 12 行，实际为 {len(rows)}")
    if any(row["validation_status"] != "passed" for row in rows):
        raise RuntimeError("存在未通过校验的模拟结果")
    return summary, rows


def centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, text_font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=text_font, fill=fill)


def draw_panel(draw: ImageDraw.ImageDraw, box, title: str, metric: str, grouped) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0), title, font=PANEL_FONT, fill=FOREGROUND)
    left, right = x0 + 80, x1 - 28
    top, bottom = y0 + 60, y1 - 72
    values = [float(row[metric]) * 100 for rows in grouped.values() for row in rows]
    lower = max(0, math.floor((min(values) - 4) / 5) * 5)
    upper = min(100, math.ceil((max(values) + 4) / 5) * 5)
    if upper <= lower:
        upper = lower + 5
    draw.rectangle((left, top, right, bottom), outline=FRAME, width=2)

    for index in range(6):
        ratio = index / 5
        value = lower + (upper - lower) * ratio
        y = bottom - (bottom - top) * ratio
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{value:.0f}%"
        label_box = draw.textbbox((0, 0), label, font=AXIS_FONT)
        draw.text((left - 11 - (label_box[2] - label_box[0]), y - 10), label, font=AXIS_FONT, fill=MUTED)

    capacities = [1, 3, 10]
    xs = [left + (right - left) * index / 2 for index in range(3)]
    for capacity, x in zip(capacities, xs):
        draw.line((x, bottom, x, bottom + 7), fill=FRAME, width=2)
        centered(draw, x, bottom + 13, f"{capacity}%", AXIS_FONT, MUTED)

    label_offsets = [-31, -10, 22, -32]
    for series_index, (scenario, rows) in enumerate(grouped.items()):
        color = SERIES[scenario][0]
        points = []
        for x, row in zip(xs, rows):
            value = float(row[metric]) * 100
            y = bottom - (bottom - top) * (value - lower) / (upper - lower)
            points.append((x, y, value))
        draw.line([(x, y) for x, y, _ in points], fill=color, width=5, joint="curve")
        for point_index, (x, y, value) in enumerate(points):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
            if point_index == 1:
                centered(draw, x, y + label_offsets[series_index], f"{value:.2f}%", VALUE_FONT, FOREGROUND)

    centered(draw, (left + right) / 2, y1 - 25, "缓存容量 / 对象总容量", AXIS_FONT, FOREGROUND)


def render_chart(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for scenario in SERIES:
        selected = [row for row in rows if row["scenario"] == scenario]
        selected.sort(key=lambda row: int(row["capacity_percent"]))
        if len(selected) != 3:
            raise RuntimeError(f"{scenario} 的容量结果不完整")
        grouped[scenario] = selected

    image = Image.new("RGB", (1900, 820), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((65, 28), "文件夹粒度 LRU 命中率", font=TITLE_FONT, fill=FOREGROUND)
    legend_x = 560
    for scenario, (color, label) in SERIES.items():
        draw.line((legend_x, 58, legend_x + 34, 58), fill=color, width=5)
        draw.ellipse((legend_x + 12, 52, legend_x + 24, 64), fill=color)
        draw.text((legend_x + 44, 45), label, font=LEGEND_FONT, fill=FOREGROUND)
        legend_x += 230 if scenario == "瞬时取回" else 205

    draw_panel(draw, (65, 115, 920, 770), "I/O 次数命中率", "io_hit_rate", grouped)
    draw_panel(draw, (1010, 115, 1865, 770), "字节命中率", "byte_hit_rate", grouped)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_PATH, optimize=True)


def pct(value: float) -> str:
    return f"{value:.2%}"


def pp(value: float) -> str:
    return f"{value * 100:.2f} pp"


def format_bytes(value: int) -> str:
    units = ((1024**5, "PiB"), (1024**4, "TiB"), (1024**3, "GiB"))
    for divisor, unit in units:
        if value >= divisor:
            return f"{value / divisor:,.2f} {unit}"
    return f"{value:,} B"


def generate_report(summary: dict[str, str], rows: list[dict[str, str]]) -> None:
    by_key = {
        (row["scenario"], int(row["capacity_percent"])): row
        for row in rows
    }
    instant_1 = by_key[("瞬时取回", 1)]
    instant_3 = by_key[("瞬时取回", 3)]
    instant_10 = by_key[("瞬时取回", 10)]
    ten_3 = by_key[("10通道 × 200 MiB/s", 3)]
    thirty_3 = by_key[("30通道 × 200 MiB/s", 3)]
    sixty_3 = by_key[("60通道 × 200 MiB/s", 3)]

    event_count = int(summary["event_count"])
    accessed_objects = int(summary["accessed_object_count"])
    total_object_bytes = int(summary["total_object_bytes"])
    total_request_bytes = int(summary["total_request_bytes"])
    first_access_bytes = int(summary["first_access_bytes"])
    io_upper = (event_count - accessed_objects) / event_count
    byte_upper = (total_request_bytes - first_access_bytes) / total_request_bytes

    report: list[str] = [
        "# 文件夹粒度 LRU Baseline 分析",
        "",
        "## 1. 结论",
        "",
        f"- 瞬时取回下，容量从 1% 增至 10% 时，I/O 次数命中率由 {pct(float(instant_1['io_hit_rate']))} 升至 {pct(float(instant_10['io_hit_rate']))}，字节命中率由 {pct(float(instant_1['byte_hit_rate']))} 升至 {pct(float(instant_10['byte_hit_rate']))}。",
        f"- 3% 容量下，10/30/60 通道的 I/O 次数命中率分别为 {pct(float(ten_3['io_hit_rate']))}/{pct(float(thirty_3['io_hit_rate']))}/{pct(float(sixty_3['io_hit_rate']))}，字节命中率分别为 {pct(float(ten_3['byte_hit_rate']))}/{pct(float(thirty_3['byte_hit_rate']))}/{pct(float(sixty_3['byte_hit_rate']))}。",
        f"- 瞬时模型的字节命中率从 1% 到 3% 提升 {pp(float(instant_3['byte_hit_rate']) - float(instant_1['byte_hit_rate']))}，从 3% 到 10% 提升 {pp(float(instant_10['byte_hit_rate']) - float(instant_3['byte_hit_rate']))}。",
        f"- 无预取按需缓存的理论上限为：I/O 次数命中率 {pct(io_upper)}、字节命中率 {pct(byte_upper)}；首次访问必为 miss。",
        "",
        "## 2. 模拟口径",
        "",
        f"分析覆盖 `{summary['start_time']}` 至 `{summary['end_time']}` 的 7 份日志，共 {event_count:,} 次访问、{int(summary['object_count']):,} 个文件夹对象。冷启动，不设置预热期，全部事件参与计分。",
        "",
        "对象大小取 `path_catalog.csv` 的 `total_size_bytes`。缓存容量为：",
        "",
        r"\[",
        r"C_p=\left\lfloor p\sum_f S_f\right\rfloor,\qquad p\in\{1\%,3\%,10\%\}",
        r"\]",
        "",
        "| 容量比例 | 容量字节 | 容量 TiB |",
        "| ---: | ---: | ---: |",
    ]
    for capacity in (1, 3, 10):
        row = by_key[("瞬时取回", capacity)]
        report.append(f"| {capacity}% | {int(row['capacity_bytes']):,} | {float(row['capacity_tib']):.3f} |")

    report.extend(
        [
            "",
            "- 缓存对象为完整文件夹代理；命中后更新为 MRU，miss 对象取回完成后进入缓存，并按 LRU 淘汰至容量约束内。",
            "- 瞬时取回：当前访问记为 miss，对象随后立即进入缓存。",
            "- 有限带宽：10/30/60 个并行通道，每通道 `200 MiB/s`，无额外固定准备时间；单对象传输时间为 `S_f / (200 MiB/s)`。",
            "- 通道满时进入中央 FIFO；同一对象排队或传输期间不重复提交物理传输，但期间访问仍记为 miss；传输完成前绝不命中。",
            "- 日志没有逐次读取字节数，因此字节命中率按命中访问对应的完整对象大小加权，不代表范围读取命中率。",
            "",
            r"\[",
            r"H_{IO}=\frac{N_{hit}}{N_{all}},\qquad H_{byte}=\frac{\sum_i S_{f_i}I_i}{\sum_i S_{f_i}}",
            r"\]",
            "",
            "## 3. 命中率结果",
            "",
            "![LRU I/O 次数与字节命中率](figures/lru_hit_rates.png)",
            "",
            "| 容量 | 瞬时 I/O | 瞬时字节 | 10通道 I/O | 10通道字节 | 30通道 I/O | 30通道字节 | 60通道 I/O | 60通道字节 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for capacity in (1, 3, 10):
        selected = [
            by_key[("瞬时取回", capacity)],
            by_key[("10通道 × 200 MiB/s", capacity)],
            by_key[("30通道 × 200 MiB/s", capacity)],
            by_key[("60通道 × 200 MiB/s", capacity)],
        ]
        values = []
        for row in selected:
            values.extend((pct(float(row["io_hit_rate"])), pct(float(row["byte_hit_rate"]))))
        report.append(f"| {capacity}% | " + " | ".join(values) + " |")

    report.extend(
        [
            "",
            "完整数值见 [lru_results.csv](tables/lru_results.csv)。",
            "",
            "## 4. 有限带宽传输与排队诊断",
            "",
            "| 通道 | 容量 | 物理传输提交数 | 最大排队对象数 | 日志末尾排队 | 日志末尾传输中 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in ("10通道 × 200 MiB/s", "30通道 × 200 MiB/s", "60通道 × 200 MiB/s"):
        for capacity in (1, 3, 10):
            row = by_key[(scenario, capacity)]
            report.append(
                f"| {row['channels']} | {capacity}% | {int(row['submitted_transfers']):,} | "
                f"{int(row['max_queued']):,} | {int(row['queued_at_log_end']):,} | {int(row['inflight_at_log_end']):,} |"
            )

    report.extend(
        [
            "",
            "物理传输提交数小于 miss 次数，是因为同对象在排队或传输期间的重复访问会合并到已有任务；这些访问仍是缓存 miss，不会虚增命中率。",
            "",
            "## 5. 理论上限",
            "",
            "无预取按需缓存中，每个对象的首次访问必为 miss，因此：",
            "",
            r"\[",
            rf"H_{{IO}}^{{max}}=\frac{{{event_count:,}-{accessed_objects:,}}}{{{event_count:,}}}={io_upper * 100:.2f}\%",
            r"\]",
            "",
            r"\[",
            rf"H_{{byte}}^{{max}}=\frac{{{total_request_bytes:,}-{first_access_bytes:,}}}{{{total_request_bytes:,}}}={byte_upper * 100:.2f}\%",
            r"\]",
            "",
            f"对象总容量为 {format_bytes(total_object_bytes)}；按对象大小对每次访问加权后的请求总量为 {format_bytes(total_request_bytes)}。",
            "",
            "## 6. 限制",
            "",
            "- 当前对象是目录/文件夹聚合代理，不是物理文件；`total_size_bytes` 可能高于一次真实回源读取量。",
            "- 日志只有秒级时间戳；同秒事件按日志原始顺序回放。",
            "- `200 MiB/s` 和通道数是给定工程情景，不包含固定寻道、挂载、网络瓶颈、失败重试或预取竞争。",
            "- 请求轨迹为固定开放环，缓存策略不会改变后续请求到达过程。",
            "",
            "## 7. 可复现文件",
            "",
            "- [simulate_lru.cs](../../pipelines/lru_baseline_analysis/simulate_lru.cs)：日志解析、字节容量 LRU 和有限带宽回放。",
            "- [run_analysis.ps1](../../pipelines/lru_baseline_analysis/run_analysis.ps1)：完整运行入口。",
            "- [render_results.py](../../pipelines/lru_baseline_analysis/render_results.py)：图表、报告与清单生成。",
            "- [dataset_summary.csv](tables/dataset_summary.csv)：数据规模与时间范围。",
            "- [lru_results.csv](tables/lru_results.csv)：12 组完整结果。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "status": "passed",
        "capacity_percentages": [1, 3, 10],
        "channel_scenarios": ["instant", 10, 30, 60],
        "bandwidth_bytes_per_second_per_channel": 200 * 1024 * 1024,
        "log_file_count": int(summary["log_file_count"]),
        "event_count": event_count,
        "result_row_count": len(rows),
        "validation": {
            "all_scenarios_passed": True,
            "excluded_raw_file_used": False,
        },
        "outputs": [
            "report.md",
            "figures/lru_hit_rates.png",
            "tables/dataset_summary.csv",
            "tables/lru_results.csv",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    summary, rows = load_data()
    render_chart(rows)
    generate_report(summary, rows)
    print(REPORT_PATH)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
