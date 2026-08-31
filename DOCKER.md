# Docker 开发与离线交付

本项目的 Docker 环境以 `linux/amd64` 为当前目标平台，基础系统为 Ubuntu 22.04，固定使用 Python 3.10，并包含 PowerShell、C# 动态编译能力以及现有报告生成所需的 Pillow 和中文字体。

Docker 只是可复现运行环境。源码、依赖清单和数据均独立保留；即使最终服务器不使用 Docker，`requirements.txt` 仍可用于普通 Python 环境或离线 wheel 包安装。

## 构建与验证

所有 Codex 窗口都应使用根目录的统一入口，避免意外调用宿主机 Python：

```powershell
pwsh -NoProfile -File .\dev.ps1 build
pwsh -NoProfile -File .\dev.ps1 smoke
```

打开容器中的 PowerShell：

```powershell
pwsh -NoProfile -File .\dev.ps1 shell
```

容器内的项目路径固定为 `/workspace`。

运行任意 Python 或其他非交互命令：

```powershell
pwsh -NoProfile -File .\dev.ps1 python --version
pwsh -NoProfile -File .\dev.ps1 pip freeze
pwsh -NoProfile -File .\dev.ps1 run bash -lc "id && pwd"
```

根目录的 `AGENTS.md` 要求 Codex 窗口只使用上述入口。Codex 在宿主机编辑由 Compose 挂载的源码，但项目进程、依赖和测试在 Linux 容器中运行。

## 运行现有分析

运行基础 LRU 分析：

```powershell
pwsh -NoProfile -File .\dev.ps1 baseline
```

运行 MISS 归因分析：

```powershell
pwsh -NoProfile -File .\dev.ps1 miss-attribution
```

Compose 会把宿主机项目挂载到 `/workspace`，并把 `02_全局数据/raw` 覆盖为只读挂载。分析结果仍写回宿主机的 `02_全局数据/analysis`。

`dev.ps1` 使用跨进程互斥，防止同一台 Windows 电脑上的多个 Codex 窗口同时构建同一镜像标签或运行这两个共享目录写入任务。只读检查可以并行；未来训练任务必须使用独立运行编号和输出目录。

## 数据与镜像边界

以下内容不会进入镜像：

- `02_全局数据/raw` 原始访问日志；
- 可重新生成的分析、训练和评估输出；
- Python 缓存、虚拟环境和离线导出包。

服务器运行时应分别挂载输入与输出目录。Linux 示例：

```bash
docker run --rm \
  --mount type=bind,src=/srv/folder-cache-rl/raw,dst=/workspace/02_全局数据/raw,readonly \
  --mount type=bind,src=/srv/folder-cache-rl/analysis,dst=/workspace/02_全局数据/analysis \
  folder-cache-rl:dev baseline
```

## 离线服务器交付

在联网开发电脑构建并导出：

```powershell
docker build --platform linux/amd64 -t folder-cache-rl:0.1.0 .
docker save -o folder-cache-rl_0.1.0_linux-amd64.tar folder-cache-rl:0.1.0
Get-FileHash .\folder-cache-rl_0.1.0_linux-amd64.tar -Algorithm SHA256
```

把镜像归档、校验值、运行配置和部署说明通过公司批准的方式传到服务器。服务器有 Docker 时：

```bash
sha256sum folder-cache-rl_0.1.0_linux-amd64.tar
docker load -i folder-cache-rl_0.1.0_linux-amd64.tar
docker image inspect folder-cache-rl:0.1.0
```

生产镜像版本、CUDA、强化学习框架和服务器运行参数将在对应模块代码与服务器规格确定后锁定，避免现在引入尚未使用的大型依赖。Python 当前固定为 3.10；如需调整，必须重建镜像并重新执行冒烟和分析测试。
