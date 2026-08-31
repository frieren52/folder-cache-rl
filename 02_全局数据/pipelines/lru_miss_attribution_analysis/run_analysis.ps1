$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$sourcePath = Join-Path $PSScriptRoot "simulate_lru.cs"
$rendererPath = Join-Path $PSScriptRoot "render_miss_attribution.py"
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$pythonExe = if (Test-Path -LiteralPath $bundledPython) {
    $bundledPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

Add-Type -Path $sourcePath
[FolderLruAnalysis]::Run($projectRoot)

& $pythonExe $rendererPath
if ($LASTEXITCODE -ne 0) {
    throw "MISS 归因报告生成失败，Python 退出码：$LASTEXITCODE"
}
