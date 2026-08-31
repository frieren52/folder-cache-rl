# 03 静态语义向量生成

本模块训练并交付静态语义编码模型。输入来自 02 的路径解析结果；全量目录向量生成和向量库构建由后续模块完成。

## 目录

```text
03_静态语义向量生成/
├── config/config.yaml       模型、采样、损失和训练参数
├── data/                    数据准备脚本生成的训练数据与 BGE 缓存
├── scripts/
│   ├── prepare_data.py      切分数据并离线编码 BGE-M3
│   ├── download_model.py    下载固定版本 BGE-M3
│   ├── train.py             训练并保存最佳模型
│   └── evaluate.py          验证并生成评估报告
├── src/
│   ├── data.py              数据准备与批量张量
│   ├── sampling.py          三类三元组采样
│   ├── model.py             双分支模型与冻结 BGE
│   ├── losses.py            全部训练损失
│   ├── training.py          训练、验证共用的单轮循环
│   ├── reporting.py         损失日志保存与曲线绘制
│   └── inference.py         模型加载与正式推理接口
└── outputs/                 正式模型交付目录
```

## 运行

所有命令从仓库根目录通过 Docker 执行：

```powershell
pwsh -NoProfile -File ./dev.ps1 run python /workspace/03_静态语义向量生成/scripts/download_model.py
pwsh -NoProfile -File ./dev.ps1 run python /workspace/03_静态语义向量生成/scripts/prepare_data.py
pwsh -NoProfile -File ./dev.ps1 run python /workspace/03_静态语义向量生成/scripts/train.py
pwsh -NoProfile -File ./dev.ps1 run python /workspace/03_静态语义向量生成/scripts/evaluate.py
```

Docker 默认申请 NVIDIA GPU。数据准备和正式推理在 CUDA 下以 FP16 加载 BGE-M3；投影头训练也会自动使用 CUDA。
训练最多运行 200 轮；验证损失进入平台后自动降低学习率，并由早停结束训练。

官方 `huggingface.co` 在当前容器网络中不可达，但下载脚本会通过可用镜像获取配置中固定的 revision。

模型保存在宿主机的 `03_静态语义向量生成/models/bge-m3`，配置已默认指向该目录。下载完成后再依次运行数据准备、训练和评估。只检查数据切分时可使用 `prepare_data.py --skip-bge-cache`。

## 输入与交付

正式输入：

- `02_全局数据/artifacts/path_sematic/embedding_records.csv`
- `02_全局数据/artifacts/path_sematic/semantic_catalog.csv`

正式交付：

- `outputs/static_encoder.safetensors`
- `outputs/model_config.json`
- `outputs/evaluation_report.json`
- `outputs/loss_history.jsonl`
- `outputs/loss_curves.png`
- `outputs/manifest.json`

推理接口：

```python
from src import StaticSemanticEncoder

encoder = StaticSemanticEncoder.load("outputs", base_model_path="/path/to/bge-m3")
result = encoder.encode(records)
```

`result["path_indices"]` 保持输入顺序，`result["vectors"]` 为逐行 L2 归一化的 `float32 [B, 128]` 数组。
