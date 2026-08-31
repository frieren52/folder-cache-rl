[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("build", "smoke", "shell", "baseline", "miss-attribution", "actor-test", "actor-build-vector-store", "actor-prepare-data", "actor-train", "actor-evaluate", "rl-test", "rl-build-replay", "rl-train-critic", "rl-train-actor", "rl-evaluate", "python", "pip", "run")]
    [string]$Action = "shell",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ActionArguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-LastExitCode {
    param([string]$Description)

    if ($LASTEXITCODE -ne 0) {
        throw "$Description 失败，退出码：$LASTEXITCODE"
    }
}

function Invoke-WithMutex {
    param(
        [string]$Name,
        [string]$BusyMessage,
        [scriptblock]$Operation
    )

    $mutex = [System.Threading.Mutex]::new($false, "Local\$Name")
    $hasLock = $false
    try {
        try {
            $hasLock = $mutex.WaitOne(0)
        }
        catch [System.Threading.AbandonedMutexException] {
            $hasLock = $true
        }

        if (-not $hasLock) {
            throw $BusyMessage
        }

        & $Operation
    }
    finally {
        if ($hasLock) {
            $mutex.ReleaseMutex()
        }
        $mutex.Dispose()
    }
}

function Invoke-Analysis {
    param([string]$Mode)

    $analysisMutexArguments = @{
        Name = "FolderCacheRlAnalysisWriter"
        BusyMessage = "另一个 Codex 窗口正在运行会写入共享分析目录的任务，请等待其结束后重试。"
        Operation = {
            & docker compose run --rm --no-TTY dev $Mode @ActionArguments
            Assert-LastExitCode "Docker 分析任务 $Mode"
        }
    }
    Invoke-WithMutex @analysisMutexArguments
}

Push-Location -LiteralPath $PSScriptRoot
try {
    switch ($Action) {
        "build" {
            $buildMutexArguments = @{
                Name = "FolderCacheRlImageBuild"
                BusyMessage = "另一个 Codex 窗口正在构建项目镜像，请等待其结束后重试。"
                Operation = {
                    & docker compose build dev
                    Assert-LastExitCode "Docker 镜像构建"
                }
            }
            Invoke-WithMutex @buildMutexArguments
        }
        "smoke" {
            & docker compose run --rm --no-TTY dev smoke @ActionArguments
            Assert-LastExitCode "Docker 冒烟测试"
        }
        "shell" {
            & docker compose run --rm dev
            Assert-LastExitCode "Docker 交互终端"
        }
        "baseline" {
            Invoke-Analysis "baseline"
        }
        "miss-attribution" {
            Invoke-Analysis "miss-attribution"
        }
        "actor-test" {
            & docker compose run --rm --no-TTY dev python -m unittest discover -s "/workspace/05_监督微调/tests" -v @ActionArguments
            Assert-LastExitCode "05单元测试"
        }
        "actor-build-vector-store" {
            & docker compose run --rm --no-TTY dev python "/workspace/05_监督微调/scripts/build_vector_store.py" @ActionArguments
            Assert-LastExitCode "05向量库构建"
        }
        "actor-prepare-data" {
            & docker compose run --rm --no-TTY dev python "/workspace/05_监督微调/scripts/prepare_data.py" @ActionArguments
            Assert-LastExitCode "05监督数据构建"
        }
        "actor-train" {
            & docker compose run --rm --no-TTY dev python "/workspace/05_监督微调/scripts/train.py" @ActionArguments
            Assert-LastExitCode "05监督训练"
        }
        "actor-evaluate" {
            & docker compose run --rm --no-TTY dev python "/workspace/05_监督微调/scripts/evaluate.py" @ActionArguments
            Assert-LastExitCode "05自然候选评价"
        }
        "rl-test" {
            & docker compose run --rm --no-TTY dev python -m unittest discover -s "/workspace/06_强化学习/tests" -v @ActionArguments
            Assert-LastExitCode "06单元测试"
        }
        "rl-build-replay" {
            & docker compose run --rm --no-TTY dev python "/workspace/06_强化学习/scripts/build_replay.py" @ActionArguments
            Assert-LastExitCode "06 Replay构建"
        }
        "rl-train-critic" {
            & docker compose run --rm --no-TTY dev python "/workspace/06_强化学习/scripts/train_critic.py" @ActionArguments
            Assert-LastExitCode "06 Critic训练"
        }
        "rl-train-actor" {
            & docker compose run --rm --no-TTY dev python "/workspace/06_强化学习/scripts/train_actor_rl.py" @ActionArguments
            Assert-LastExitCode "06 Actor强化训练"
        }
        "rl-evaluate" {
            & docker compose run --rm --no-TTY dev python "/workspace/06_强化学习/scripts/evaluate.py" @ActionArguments
            Assert-LastExitCode "06策略评价"
        }
        "python" {
            & docker compose run --rm --no-TTY dev python @ActionArguments
            Assert-LastExitCode "容器 Python 命令"
        }
        "pip" {
            & docker compose run --rm --no-TTY dev python -m pip @ActionArguments
            Assert-LastExitCode "容器 pip 命令"
        }
        "run" {
            if ($ActionArguments.Count -eq 0) {
                throw "run 操作至少需要一个容器内命令。"
            }
            & docker compose run --rm --no-TTY dev @ActionArguments
            Assert-LastExitCode "容器自定义命令"
        }
    }
}
finally {
    Pop-Location
}
