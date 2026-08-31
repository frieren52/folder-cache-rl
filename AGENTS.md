# Repository development rules

## Docker-first execution

- Editing files on the host is allowed. All project execution, dependency installation, compilation, tests, analysis, training, and evaluation must run through the repository Docker environment.
- Use `pwsh -NoProfile -File ./dev.ps1 <action>` from the repository root. Do not run the host `python`, `pip`, `pytest`, PowerShell analysis scripts, or project C#/.NET commands directly.
- For an arbitrary non-interactive command, use `pwsh -NoProfile -File ./dev.ps1 run <command> [arguments...]`.
- Treat `requirements.txt` as the dependency source of truth. Do not make an interactive `pip install` the final state of a change; update the dependency file, rebuild, and run the smoke test.
- The supported Python line is 3.10. The Docker build must fail if the container resolves another Python minor version.

## Standard commands

- Build: `pwsh -NoProfile -File ./dev.ps1 build`
- Environment smoke test: `pwsh -NoProfile -File ./dev.ps1 smoke`
- Container shell: `pwsh -NoProfile -File ./dev.ps1 shell`
- Python command: `pwsh -NoProfile -File ./dev.ps1 python <arguments...>`
- Pip command: `pwsh -NoProfile -File ./dev.ps1 pip <arguments...>`
- Baseline analysis: `pwsh -NoProfile -File ./dev.ps1 baseline`
- MISS attribution analysis: `pwsh -NoProfile -File ./dev.ps1 miss-attribution`

## Data and concurrency

- Never add `02_全局数据/raw` to a Docker image. It is runtime input and must be mounted read-only.
- Generated analysis and training outputs must remain outside the image.
- The baseline and MISS attribution commands write shared output locations. Invoke them only through `dev.ps1`, which prevents two local Codex windows from running these writers concurrently.
- Image builds must also use `dev.ps1`; it serializes builds so concurrent Codex windows cannot race to replace the shared `folder-cache-rl:dev` tag.
- Multiple windows may run read-only checks in parallel. When future experiments write results, give each run a unique output directory or run identifier.

## Verification

- After changing Docker, Python dependencies, the entrypoint, or Linux compatibility code, run `pwsh -NoProfile -File ./dev.ps1 build` followed by `pwsh -NoProfile -File ./dev.ps1 smoke`.
- Changes to an existing analysis path should run its corresponding containerized analysis before completion when the input data is available.
