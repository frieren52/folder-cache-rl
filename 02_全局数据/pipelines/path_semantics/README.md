# 路径语义解析与Embedding输入生成

> 目录规范：本目录只保留流水线代码和开发说明。正式复用数据位于 `../../artifacts/path_semantics/v1/`，阶段审计位于 `../../reports/path_semantics/`。下游代码不得读取步骤运行目录。

## 1. 这个目录用于做什么

本目录提供一套可复现流程，将目录路径聚合数据转换为：

1. 可解释的卫星/平台、仪器、等级、产品、区域、分辨率等结构化字段。
2. 可直接输入BGE等文本Embedding模型的中文语义文本。
3. 语义组到每条具体路径实例的一对多映射。
4. 供路径实例编码器、历史访问模型和预取排序使用的旁路字段。

当前输入数据是目录级聚合路径，不包含目录内具体文件名。因此当前输出是“每个目录路径的输入数据”，不是目录内每个具体文件的输入数据。如果需要一文件一向量，输入中必须包含完整文件名或完整文件路径。

## 2. 目录中最重要的文件

### 执行脚本

| 脚本 | 作用 |
| --- | --- |
| `01_audit_paths.ps1` | 审计输入格式、路径骨架和已有规则 |
| `02_classify_path_tokens.py` | 为每次Token出现生成候选类型、置信度和证据 |
| `03_parse_path_semantics.py` | 使用最终版分支路由和状态机解析路径 |
| `04_build_extension_report.py` | 对比扩展前后的解析覆盖率；只用于规则开发复核 |
| `05_build_embedding_inputs.py` | 将结构化路径转换为Embedding输入和语义目录 |

### 说明文档

| 文档 | 内容 |
| --- | --- |
| `文件路径语义解析_五步骤工作说明.md` | 五个步骤分别输入什么、处理什么、输出什么 |
| `主要输出文件_样例解析与后续训练计划.md` | 输出字段、真实样例和后续训练方案 |

### 最终数据

| 文件 | 规模 | 用途 |
| --- | ---: | --- |
| `../../artifacts/path_semantics/v1/embedding_records.csv.gz` | 49,922行 | 每条具体路径的语义、实例、来源和旁路特征 |
| `../../artifacts/path_semantics/v1/semantic_catalog.csv.gz` | 2,019行 | 去重后的BGE语义输入 |
| `../../artifacts/path_semantics/v1/parsed_paths.csv.gz` | 49,922行 | 全量结构化路径语义字段 |
| `../../reports/path_semantics/05_embedding/step5_summary.json` | 1份 | 自动质量检查 |

## 3. 运行环境

需要：

- Windows PowerShell 5.1或PowerShell 7。
- Python 3.10或更高版本。
- 步骤1—5的数据处理脚本只使用Python标准库，不要求安装PyTorch、Pandas或BGE。
- 真正生成浮点向量时才需要安装Embedding模型相关依赖。

以下命令假设当前工作目录为：

```powershell
D:\桌面\文件夹\转码\HUAWEI\模型方案
```

先指定Python。正常安装Python时可以使用：

```powershell
$Python = "python"
```

当前开发环境也可以使用：

```powershell
$Python = "C:\Users\29543\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
```

## 4. 准备输入文件

输入文本必须使用UTF-8或UTF-8 BOM编码，每个非空行格式为：

```text
访问次数 目录路径 聚合数据大小字节数
```

示例：

```text
4525 /FYDATAOUTSHARE/DATAIOT/FY3/FY3F/GNOSO/L1/IEG/2026/20260522 109864266
```

约束：

- 三个字段使用空白字符分隔。
- 路径中不能包含空格。
- 访问次数和数据大小必须是非负整数。
- 以`#`开头的行作为注释忽略。
- 当前脚本接受有无开头`/`的路径，并统一按`/`拆分。

现有输入文件位于：

```text
recovered_attachments_merged/合并完整数据.txt
```

处理新数据时，建议使用新的输出目录，避免覆盖本次已经验证的结果。

## 5. 按步骤运行

下面以新输入`new_data/paths.txt`和临时输出目录`02_全局数据/processed/work/path_semantics/run_new`为例。临时结果验证通过后，只发布正式制品和报告，不长期保留重型样例文件。

### 步骤1：审计路径和规则

```powershell
powershell -ExecutionPolicy Bypass -File .\02_全局数据\processed\pipelines\path_semantics\01_audit_paths.ps1 `
  -InputFile .\new_data\paths.txt `
  -OutputDir .\02_全局数据\processed\work\path_semantics\run_new\01_audit
```

重点检查：

- `步骤1_路径规则审计报告.md`
- `audit_summary.json`
- `path_shape_summary.csv`
- `malformed_rows.csv`

如果`malformed_rows.csv`不为空，应先修复输入格式。新数据出现大量新骨架时，不要直接假设现有解析器仍然正确。

### 步骤2：识别Token候选类型

```powershell
& $Python .\02_全局数据\processed\pipelines\path_semantics\02_classify_path_tokens.py `
  --input .\new_data\paths.txt `
  --output-dir .\02_全局数据\processed\work\path_semantics\run_new\02_token
```

重点检查：

- `unknown_tokens.csv`
- `ambiguous_tokens.csv`
- `token_catalog.csv`
- `步骤2_Token类型识别报告.md`

`unknown`不等于错误，它表示仅凭当前路径和规则还不能确认业务角色。不要为了消除unknown而给Token强行添加中文释义。

步骤3会直接加载`02_classify_path_tokens.py`中的分类逻辑，因此脚本文件必须和`03_parse_path_semantics.py`保持在同一目录。步骤2生成的CSV主要用于审计，不是步骤3的直接输入。

### 步骤3：解析路径语义

当前`03_parse_path_semantics.py`已经包含步骤4验证后的最终扩展规则。对新数据直接运行：

```powershell
& $Python .\02_全局数据\processed\pipelines\path_semantics\03_parse_path_semantics.py `
  --input .\new_data\paths.txt `
  --output-dir .\02_全局数据\processed\work\path_semantics\run_new\03_parser
```

主要输出：

- `parsed_paths.csv.gz`：全量结构化解析结果。
- `route_summary.csv`：各路径语法数量。
- `field_coverage_by_route.csv`：字段覆盖率。
- `unresolved_extra_tokens.csv`：未解释Token。
- `field_conflicts.csv`：字段冲突。
- `parsed_path_samples.jsonl`：带证据和置信度的样例。
- `step3_summary.json`：本次解析摘要。这里的文件名是脚本历史命名，内容代表当前最终解析器结果。

必须检查：

1. `supported_path_percentage`是否明显下降。
2. `supported_paths_with_conflicts`是否增加。
3. `unresolved_extra_tokens.csv`中是否出现新的高频业务Token。
4. 新增路径骨架是否进入了错误路由。

### 步骤4：规则扩展复核（新数据可选）

步骤4不是生成Embedding输入的必要步骤。它用于在拥有“旧解析器结果”和“扩展解析器结果”时生成覆盖率对比报告。

对当前已完成的数据，可复现报告：

```powershell
& $Python .\02_全局数据\processed\pipelines\path_semantics\04_build_extension_report.py `
  --baseline-dir .\02_全局数据\processed\reports\path_semantics\03_parser_baseline `
  --expanded-dir .\02_全局数据\processed\reports\path_semantics\04_parser_expanded
```

处理全新数据时，如果没有同一批数据的扩展前基线，跳过该命令。应直接人工审阅步骤1—3的报告，再决定是否修改规则并重新运行步骤3。

### 步骤5：生成Embedding输入

```powershell
& $Python .\02_全局数据\processed\pipelines\path_semantics\05_build_embedding_inputs.py `
  --parsed-input .\02_全局数据\processed\work\path_semantics\run_new\03_parser\parsed_paths.csv.gz `
  --output-dir .\02_全局数据\processed\work\path_semantics\run_new\05_embedding
```

必须检查`step5_summary.json`：

- `input_parsed_paths`应等于输入路径数。
- `output_embedding_records`应等于输入路径数。
- `instance_time_leak_checks`应为`passed`。

脚本还会保证每条记录都生成非空`embedding_text`；如果出现无法生成文本或实例时间泄漏，执行会直接失败。

## 6. 如何理解最终两层输出

### 逐路径记录

`embedding_records.csv.gz`保存所有路径：

```text
path
semantic_group_id
embedding_text
semantic_sequence
source_context
instance_context
quality_tier
access_count
total_size_bytes
```

其中：

- `embedding_text`表示文件或目录“是什么”。
- `source_context`表示存储在哪里、通过哪种路径语法解析。
- `instance_context`表示观测日期和实例编号。
- `access_count`表示聚合热度。
- `total_size_bytes`表示预取资源成本。

### 去重语义目录

`semantic_catalog.csv.gz`只保存唯一`semantic_sequence`。

同一产品的不同日期路径可以共享一个`semantic_group_id`：

```text
一个semantic_group_id
├── 20260522路径实例
├── 20260523路径实例
└── 20260524路径实例
```

日期没有参与语义去重，但仍完整保存在逐路径文件的`instance_context`中。

## 7. 质量层的使用方法

| `quality_tier` | 含义 | 建议 |
| --- | --- | --- |
| `supported_clean` | 专用路由，且没有未解释Token | 用作第一版主训练集 |
| `supported_with_unresolved` | 主结构可信，但存在未知业务缩写 | 保留并降低权重，等待代码表确认 |
| `fallback_raw` | 没有专用卫星语法或属于非卫星目录 | 单独分区，不强行作为卫星产品训练 |

当前数据分布：

```text
supported_clean：49,618条
supported_with_unresolved：88条
fallback_raw：216条
```

## 8. 将输出送入BGE

不要对49,922条重复语义逐条编码。应优先读取`semantic_catalog.csv.gz`中的`embedding_text`，只编码2,019个唯一语义，然后通过`semantic_group_id`映射回逐路径记录。

安装真正生成向量所需依赖：

```powershell
pip install sentence-transformers pandas numpy
```

示例：

```python
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

catalog = pd.read_csv(
    "02_全局数据/processed/artifacts/path_semantics/v1/semantic_catalog.csv.gz",
    compression="gzip",
)

model = SentenceTransformer("BAAI/bge-m3")
vectors = model.encode(
    catalog["embedding_text"].fillna("").tolist(),
    batch_size=128,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

np.save("semantic_vectors.npy", vectors)

pd.DataFrame({
    "vector_row": np.arange(len(catalog)),
    "semantic_group_id": catalog["semantic_group_id"],
}).to_csv("semantic_vector_index.csv", index=False, encoding="utf-8-sig")
```

模型名称、查询提示词和池化方式应以所选模型的正式说明为准。首次实验应同时比较原始路径、`semantic_text_strict`和`embedding_text`。

## 9. 生成有区分度的最终路径向量

共享BGE语义向量不能单独区分同一产品的不同日期实例。最终静态路径向量应包含两个部分：

```text
s_i：embedding_text生成的共享语义向量
p_i：source_context + instance_context + 原始路径Token生成的路径实例向量
```

可以不使用MLP，直接加权拼接：

```text
v_static_i = Normalize([√α · s_i ; √(1-α) · p_i])
```

初始可令`α=0.8`。同类文件语义接近，但不同日期、来源和实例不会得到完全相同的最终向量。

正式版本的路径实例向量建议使用字段Embedding、Token/Subword Embedding、层级位置Embedding和日期周期编码组成的小型PathEncoder。详细训练方案见`主要输出文件_样例解析与后续训练计划.md`。

## 10. 与历史访问模型衔接

当前产物是文件或目录的item侧元数据。历史访问模型还需要事件级日志：

```text
访问时间戳
文件或目录ID
用户、节点或会话ID
读取字节数（如果存在）
```

按时间排序后构造：

```text
最近N次访问 → 下一次真实访问路径
```

每条路径最终可以维护：

```text
static_vector：共享语义 + 路径实例
dynamic_vector：长期访问行为
```

检索得分：

```text
score(i)
= λ × cosine(q_static, static_vector_i)
+ (1-λ) × cosine(q_dynamic, dynamic_vector_i)
```

访问次数和文件大小不要拼入BGE文本。访问次数进入行为模型或样本权重，数据大小用于预取成本和最终排序。

## 11. 推荐验证指标

路径解析阶段：

- 专用路由覆盖率。
- 字段冲突数。
- 未解释Token数量和频率。
- 人工抽样准确率。

Embedding和预取阶段：

- Recall@K、HitRate@K、MRR或NDCG。
- 字节命中率。
- 无效预取率。
- 带宽放大率和缓存占用。
- 按卫星、仪器、产品、质量层分组的结果。

访问数据必须按时间划分，不能随机打乱。七天数据可以先采用前5天训练、第6天验证、第7天测试。

## 12. 常见问题

### 为什么是2,019条而不是49,922条？

2,019是唯一静态语义组合数量；49,922条路径全部保留在`embedding_records.csv.gz`中。

### 日期是否被删除？

没有。日期只是不参与共享语义分组，仍保存在`instance_context`、`observe_year`和`observe_date`中。

### 能否区分同一目录中的具体文件？

不能，因为当前输入没有具体文件名。需要先提供完整文件路径，再扩展文件名Token规则。

### unknown是否应该全部消除？

不应该。无法确认含义的Token应保留原值和低置信状态，等待正式业务代码表，而不是生成看似完整但错误的语义。

### 修改规则后需要重新运行哪些步骤？

- 修改Token分类规则：重新运行步骤2、3、5。
- 修改路径路由或状态机：重新运行步骤3、5。
- 只修改Embedding文本模板：只重新运行步骤5。
- 更换BGE模型：无需重新解析路径，只重新生成浮点向量。
