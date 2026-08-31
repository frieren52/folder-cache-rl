# 04 动态历史向量生成

本模块使用目录最近24小时访问事件训练动态历史编码与需求预测模型。04只交付模型，不生成动态向量文件或向量库。

## 目录

```text
04_动态历史向量生成/
├── config/config.yaml       数据、模型、损失和训练参数
├── data/                    按 dataset_id 保存分片样本与构造报告
├── scripts/
│   ├── prepare_data.py      校验事件并构造训练、验证样本
│   ├── train.py             训练、保存检查点、损失日志和曲线
│   └── evaluate.py          验证最佳检查点并发布正式模型
├── src/
│   ├── data.py              事件解析、特征和 Parquet 加载
│   ├── sampling.py          滚动历史池与条件采样
│   ├── preparation.py       数据构造闭环与报告
│   ├── model.py             TCN、Transformer和预测头
│   ├── losses.py            三项训练损失
│   ├── training.py          训练、验证共用循环
│   ├── reporting.py         损失日志保存与曲线绘制
│   └── inference.py         正式模型加载与推理接口
├── tests/                   特征、采样、模型和小数据闭环测试
└── outputs/
    ├── runs/<run_id>/       可恢复训练过程
    └── releases/<model_id>/ 正式模型制品
```

`requirements.txt` 只引用仓库根依赖文件，与03保持一致。

## 运行

所有命令从仓库根目录通过 Docker 执行：

```powershell
pwsh -NoProfile -File ./dev.ps1 run python /workspace/04_动态历史向量生成/scripts/prepare_data.py --dataset-id <dataset_id>
pwsh -NoProfile -File ./dev.ps1 run python /workspace/04_动态历史向量生成/scripts/train.py --dataset-dir /workspace/04_动态历史向量生成/data/datasets/<dataset_id> --run-id <run_id>
pwsh -NoProfile -File ./dev.ps1 run python /workspace/04_动态历史向量生成/scripts/evaluate.py --dataset-dir /workspace/04_动态历史向量生成/data/datasets/<dataset_id> --run-dir /workspace/04_动态历史向量生成/outputs/runs/<run_id> --model-id <model_id>
```

临时试跑可在训练命令末尾增加`--max-epochs 2`；该值只写入本次run的解析后配置，不修改默认30轮配置。

测试：

```powershell
pwsh -NoProfile -File ./dev.ps1 python -c "import sys,unittest; result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('04_动态历史向量生成/tests', pattern='test_*.py')); sys.exit(not result.wasSuccessful())"
```

## 正式交付

`outputs/releases/<model_id>/` 包含：

- `dynamic_encoder.safetensors`
- `model_config.json`
- `feature_stats.json`
- `evaluation_report.json`
- `loss_history.jsonl`
- `loss_curves.png`
- `manifest.json`

训练每完成一轮都会追加 `loss_history.jsonl`，并自动刷新包含总损失、是否访问损失、首次访问时间损失和次数损失的 `loss_curves.png`。
训练和验证期间每20个微批次或最长60秒输出一次当前batch、完成比例、已处理样本、预计剩余时间、吞吐量、区间/本轮累计分项loss及GPU显存。训练进度同时追加到`training_progress.jsonl`，并原子刷新`live_training_metrics.png`；图中包含四项loss、吞吐量和GPU显存，训练中断后已写入内容仍保留。当前`micro_batch_size=1024`、梯度累积为1。

Parquet加载器直接把整批数据转换为Tensor，不再逐样本复制；删除并重新生成 `data/` 后仍自动使用该实现。数据兼容性只绑定历史、标签、采样、存储和切分配置，调整batch、学习率或进度间隔不会要求重新构造样本。

## 推理

```python
from src import DynamicHistoryEncoder

encoder = DynamicHistoryEncoder.load("outputs/releases/<model_id>")
result = encoder.encode(history_records)
```

`history_records` 为目录编号、带上海时区的快照时刻及此前24小时访问时刻；结果保持输入顺序，返回128维L2归一化动态向量、10档访问时间概率和未来1小时次数点估计。10档依次为9个一小时内首次访问时间档和“未来一小时无访问”档，边界为`[0, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]`秒。

时间边界、概率维度、无访问档索引和预测时域同时写入`model_config.json`的`output_contract`。05/06加载时必须校验该合同；旧8维数据集和检查点不能用于当前接口，需使用新的`dataset_id`、`run_id`和`model_id`重建。
