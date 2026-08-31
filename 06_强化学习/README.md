# 06 强化学习

本模块复用 05 的监督 Actor，在有限通道、共享 FIFO 和字节容量 LRU 环境中训练双 Critic，并用 Critic 优势对 Actor 做一次强化更新。默认使用 30 通道、每通道 200 MiB/s、10 秒决策周期和每宏步最多 2 个预取对象。

## 目录

```text
06_强化学习/
├── config/                 环境、策略、Critic 和训练超参数
├── data/versions/          分版本 Replay，不写入 Docker 镜像
├── scripts/
│   ├── build_replay.py     真实访问流回放与 Replay 发布
│   ├── train_critic.py     初始/最终双 Critic 训练
│   ├── train_actor_rl.py   优势排序与 05 损失锚定
│   └── evaluate.py         单策略回放、策略冻结和测试汇总
├── src/                    环境、候选、路由、模型和训练实现
├── tests/                  环境、候选、Replay 与训练单元测试
└── outputs/runs/<run_id>/  检查点、loss 日志、曲线和报告
```

## 环境与配置

所有执行均从仓库根目录通过 Docker 进行：

```powershell
pwsh -NoProfile -File ./dev.ps1 build
pwsh -NoProfile -File ./dev.ps1 smoke
pwsh -NoProfile -File ./dev.ps1 rl-test
```

容器固定使用 Python 3.10，默认自动选择 CUDA。根目录 `requirements.txt` 是唯一第三方依赖来源。原始访问日志以只读方式挂载，Replay、检查点和报告写回宿主机。

四个 YAML 是运行参数的唯一来源。改变决策周期、通道数、带宽、缓存比例、每步预取上限、候选规模、折扣或模型结构后，应使用新的 Replay 版本和 `run_id`，不得复用旧 Critic。

正式运行前必须存在：

- `04_动态历史向量生成/outputs/releases/current/`；
- `05_监督微调/data/vector_store/`；
- `05_监督微调/outputs/ckpt/trained/actor_supervised_final.pt`；
- 05 的 Actor 锚定样本版本。

## 训练顺序

以下命令展示监督 Actor 对应的一轮流程，版本名和运行 ID 由实验自行确定。

1. 生成训练拟合 Replay，并只在该 Replay 上拟合特征统计：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-build-replay --run-id sup-initial-fit --replay-version sup-initial-fit --split train_fit --behavior initial
```

2. 生成固定的训练段内部校准 Replay，强制复用训练拟合统计：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-build-replay --run-id sup-calibration --replay-version sup-calibration --split train_calibration --behavior initial --normalization-from /workspace/06_强化学习/data/versions/sup-initial-fit
```

3. 训练初始 Critic：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-train-critic --run-id sup-critic-initial --stage initial --initial-replay /workspace/06_强化学习/data/versions/sup-initial-fit --calibration-replay /workspace/06_强化学习/data/versions/sup-calibration
```

4. 使用初始 Critic 生成 epsilon-greedy Replay：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-build-replay --run-id sup-epsilon --replay-version sup-epsilon --split train_fit --behavior epsilon --critic-checkpoint /workspace/06_强化学习/outputs/runs/sup-critic-initial/critic_initial/critic_best.pt --normalization-from /workspace/06_强化学习/data/versions/sup-initial-fit
```

5. 混合两类 Replay 继续训练最终 Critic。最终阶段必须显式提供初始 Critic，防止误从随机参数开始：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-train-critic --run-id sup-critic-final --stage final --initial-replay /workspace/06_强化学习/data/versions/sup-initial-fit --epsilon-replay /workspace/06_强化学习/data/versions/sup-epsilon --calibration-replay /workspace/06_强化学习/data/versions/sup-calibration --resume-critic /workspace/06_强化学习/outputs/runs/sup-critic-initial/critic_initial/critic_best.pt
```

6. 使用最终策略轨迹得到的 Q 校准误差 P90 作为 `--advantage-margin`，强化 Actor：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-train-actor --run-id actor-rl-v1 --actor-data-version <05数据版本> --train-replay /workspace/06_强化学习/data/versions/sup-epsilon --calibration-replay /workspace/06_强化学习/data/versions/sup-calibration --critic-checkpoint /workspace/06_强化学习/outputs/runs/sup-critic-final/critic_final/critic_best.pt --advantage-margin <P90>
```

Actor 更新后必须传入新的 `--actor-checkpoint`，重新执行步骤 1～5；旧 Actor 的 Replay 和 Critic 只能保留审计，不能继续用于新 Actor。

## loss 与进度

Critic 和 Actor 在第 1 次更新、固定更新间隔或超过 60 秒时输出一行 JSON，包含：

- 当前 update、loss 分项、学习率和累计耗时；
- CUDA 显存已分配量和保留量；
- Critic 的 Q 均值、TD 误差、固定回报 MAE/P90；
- Actor 的优势三路损失、05 锚定损失和校准损失。

训练目录同时保存 `loss_history.jsonl`、`loss_curves.png` 或 `actor_loss_history.jsonl`、`actor_loss_curves.png`。早停后交付校准指标最优的检查点。

## 策略评价

`evaluate.py` 有两种用法：`--policy` 在真实访问流上运行单个策略并输出指标；`--metrics` 汇总多个指标文件。

训练段内部校准时，分别运行四个策略。其中 `critic` 使用监督 Actor 对应的 Critic，`actor_critic` 使用强化 Actor 对应的重建 Critic：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-evaluate --phase calibration --policy no_prefetch --run-id cal-base --output /workspace/06_强化学习/outputs/reports/cal-no-prefetch.json
pwsh -NoProfile -File ./dev.ps1 rl-evaluate --phase calibration --policy simple_greedy --run-id cal-greedy --output /workspace/06_强化学习/outputs/reports/cal-simple-greedy.json
pwsh -NoProfile -File ./dev.ps1 rl-evaluate --phase calibration --policy critic --run-id cal-critic --critic-checkpoint <监督Actor对应Critic> --output /workspace/06_强化学习/outputs/reports/cal-critic.json
pwsh -NoProfile -File ./dev.ps1 rl-evaluate --phase calibration --policy actor_critic --run-id cal-actor-critic --actor-checkpoint <强化Actor> --critic-checkpoint <强化Actor对应Critic> --output /workspace/06_强化学习/outputs/reports/cal-actor-critic.json
```

通过自然召回保护线后使用 `--actor-recall-ok` 汇总并冻结策略：

```powershell
pwsh -NoProfile -File ./dev.ps1 rl-evaluate --phase calibration --metrics <四个校准指标文件> --actor-recall-ok --output /workspace/06_强化学习/outputs/reports/policy-selection.json
```

冻结后将 `--phase` 改为 `test`，在 6 月 24 日只运行一次；汇总时通过 `--frozen-policy` 指定训练阶段已选策略。10 通道参考评价在单策略回放时增加 `--channel-count 10`，不得用测试结果重新选择策略或调参。

## 时间切分

- 6 月 19 日 00:00 至 6 月 23 日末：训练；
- 训练结束前第 7 至第 1 小时：固定内部校准；
- 每个切分最后 1 小时：停止新预取，只结算在途任务；
- 6 月 24 日：冻结后的单次测试，不参与梯度、早停或策略选择。
