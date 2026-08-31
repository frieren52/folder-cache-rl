# 05 监督微调

05加载03/04正式模型，构建双路向量库和因果监督样本，训练Actor并在测试集评价自然候选召回。所有命令均从仓库根目录通过Docker执行。

## 运行前提

- `02_全局数据/raw/`中已有访问日志；
- 03正式模型可用；
- `config/config.yaml`指向04正式release，其中包含`manifest.json`、`model_config.json`、`feature_stats.json`和`dynamic_encoder.safetensors`。

04训练目录中的`best.pt`不能直接作为05输入。

## 操作顺序

```powershell
# 环境与单元测试
pwsh -NoProfile -File ./dev.ps1 build
pwsh -NoProfile -File ./dev.ps1 smoke
pwsh -NoProfile -File ./dev.ps1 actor-test

# 构建向量库和监督数据
pwsh -NoProfile -File ./dev.ps1 actor-build-vector-store
pwsh -NoProfile -File ./dev.ps1 actor-prepare-data --data-version actor-v1

# 训练20轮
pwsh -NoProfile -File ./dev.ps1 actor-train --data-version actor-v1 --run-id actor-v1-run1

# 训练完成后只运行一次测试
pwsh -NoProfile -File ./dev.ps1 actor-evaluate --data-version actor-v1 --checkpoint /workspace/05_监督微调/outputs/runs/actor-v1-run1/checkpoints/actor_supervised_final.pt --report-id actor-v1-test1
```

`data-version`、`run-id`和`report-id`应使用新名称，程序默认拒绝覆盖既有产物。

## 数据与输出

6月18日只预热，6月19—23日训练，6月24日测试；不设置验证集、不早停，训练过程不读取测试集。

- `data/vector_store/`：静态向量库和初始历史索引；
- `data/versions/<data-version>/`：监督样本；
- `outputs/runs/<run-id>/`：实时Loss、曲线和检查点；
- `actor_supervised_last.pt`：恢复检查点；
- `actor_supervised_final.pt`：第20轮正式模型；
- `outputs/reports/<report-id>/`：测试损失、召回报告和图像。
