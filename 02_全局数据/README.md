# 02 全局数据

本目录负责保存原始访问数据、跨模块复用制品、数据处理流水线和分析结果。业务口径以[业务约束](../01_业务与方案/业务约束.md)为准，模块职责与验收要求见[02 全局数据模块方案](../01_业务与方案/设计方案/02_全局数据/模块方案.md)。

## 目录

```text
02_全局数据/
├── raw/          # 原始访问日志，只读
├── artifacts/    # 下游模块可依赖的正式数据制品
├── pipelines/    # 数据处理与分析代码
└── analysis/     # 报告、图表、明细表和运行清单
```

## 当前内容

- **原始访问日志。** `raw/` 保存七份 `access_*.txt`，实际事件范围为 2026-06-17 23:58:59 至 2026-06-24 23:59:49，共 30,549,077 条。
- **全局词表。** `artifacts/vocab/path_catalog.csv` 是目录对象主表，`artifacts/vocab/semantic_vocab.csv` 保存复用语义词表。
- **访问特征分析。** `analysis/access_analysis/` 保存时间流量、峰值、热点集中度、突发和空闲分析，摘要记录七份源日志 SHA256。
- **全局统计。** `analysis/global_statistics/` 保存容量长尾、工作集、跨日复用、热点漂移、访问间隔、语义质量和离线容量潜力，27 项自动校验已通过。
- **LRU 基线。** `analysis/lru_baseline_analysis/` 保存不同容量和通道情景的回放结果，当前 manifest 状态为 `passed`。
- **LRU Miss 归因。** `analysis/lru_miss_attribution_analysis/` 保存逐事件反事实归因结果，当前 manifest 状态为 `passed`。
- **路径语义。** `artifacts/path_sematic/embedding_records.csv` 保存 49,922 条逐目录解析结果，并提供 `embedding_text`、`source_text`、`instance_context`、`observe_date_iso`、`date_ordinal` 和 `has_observe_date`；`semantic_catalog.csv` 保存 2,019 个语义组。

## 使用规则

- `raw/` 只读，不回写清洗结果或分析产物。
- 下游模块优先读取 `artifacts/` 中已发布、可校验的制品；需要完整有序事件流的模块可以按已冻结接口只读访问 `raw/access_*.txt`，但不得读取 `pipelines/` 的中间结果。
- `analysis/` 只用于阅读、审计和复现实验，不作为训练接口。
- 正式制品和分析结果必须附带输入摘要、配置及 manifest；校验失败时停止使用。
- 业务主容量为目录对象总字节数的 3%；其他容量只作为分析对照。

## 运行 LRU 分析

在项目根目录执行：

```powershell
& '.\02_全局数据\pipelines\lru_baseline_analysis\run_analysis.ps1'
& '.\02_全局数据\pipelines\lru_miss_attribution_analysis\run_analysis.ps1'
```

脚本读取 `artifacts/vocab/path_catalog.csv` 和 `raw/access_*.txt`，并更新对应的 `analysis/` 子目录。运行前应确认这些输出不包含需要保留但尚未归档的人工修改。

## 主要入口

- [访问特征分析报告](analysis/access_analysis/总体访问特征分析报告.md)
- [访问特征分析摘要](analysis/access_analysis/analysis_summary.json)
- [数据全局统计综合报告](analysis/global_statistics/数据全局统计综合报告.md)
- [全局统计运行清单](analysis/global_statistics/manifest.json)
- [LRU 基线报告](analysis/lru_baseline_analysis/report.md)
- [LRU 基线运行清单](analysis/lru_baseline_analysis/manifest.json)
- [LRU Miss 归因报告](analysis/lru_miss_attribution_analysis/report.md)
- [LRU Miss 归因运行清单](analysis/lru_miss_attribution_analysis/manifest.json)
- [路径语义流水线说明](pipelines/path_semantics/README.md)
