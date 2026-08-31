#!/usr/bin/env bash
set -euo pipefail

mode="${1:-shell}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${mode}" in
    shell)
        exec pwsh -NoLogo -NoProfile "$@"
        ;;
    smoke)
        python -c 'import platform; from PIL import ImageFont, __version__; ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18); print(f"Python={platform.python_version()} Pillow={__version__} font=ok")'
        python -c 'import numpy, pandas, pyarrow, safetensors, torch, transformers, yaml; assert torch.cuda.is_available(), "CUDA GPU is required"; print(f"NumPy={numpy.__version__} pandas={pandas.__version__} PyTorch={torch.__version__} CUDA={torch.version.cuda} GPU={torch.cuda.get_device_name(0)} ml-deps=ok")'
        python -c 'from pathlib import Path; files=list(Path("/workspace/03_静态语义向量生成").rglob("*.py")); [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in files]; print(f"static-encoder-syntax={len(files)} files ok")'
        python -c 'from pathlib import Path; files=list(Path("/workspace/04_动态历史向量生成").rglob("*.py")); [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in files]; print(f"history-encoder-syntax={len(files)} files ok")'
        python -c 'from pathlib import Path; files=list(Path("/workspace/05_监督微调").rglob("*.py")); [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in files]; import folder_cache_actor; print(f"actor-syntax={len(files)} files package={folder_cache_actor.__version__} ok")'
        python -c 'from pathlib import Path; files=list(Path("/workspace/06_强化学习").rglob("*.py")); [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in files]; print(f"rl-syntax={len(files)} files ok")'
        pwsh -NoLogo -NoProfile -Command 'Add-Type -Path "/workspace/02_全局数据/pipelines/lru_baseline_analysis/simulate_lru.cs"; Write-Host "PowerShell=$($PSVersionTable.PSVersion) baseline-CSharp=ok"'
        pwsh -NoLogo -NoProfile -Command 'Add-Type -Path "/workspace/02_全局数据/pipelines/lru_miss_attribution_analysis/simulate_lru.cs"; Write-Host "miss-attribution-CSharp=ok"'
        ;;
    baseline)
        exec pwsh -NoLogo -NoProfile -File "/workspace/02_全局数据/pipelines/lru_baseline_analysis/run_analysis.ps1" "$@"
        ;;
    miss-attribution)
        exec pwsh -NoLogo -NoProfile -File "/workspace/02_全局数据/pipelines/lru_miss_attribution_analysis/run_analysis.ps1" "$@"
        ;;
    *)
        exec "${mode}" "$@"
        ;;
esac
