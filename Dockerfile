ARG BASE_IMAGE=mcr.microsoft.com/powershell:7.5-ubuntu-22.04
FROM ${BASE_IMAGE}

ARG APP_UID=1000
ARG APP_GID=1000
ARG EXPECTED_PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        fonts-noto-cjk \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && test "$(/opt/venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "${EXPECTED_PYTHON_VERSION}"

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --requirement /tmp/requirements.txt

RUN groupadd --gid "${APP_GID}" ubuntu \
    && useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" --shell /bin/bash ubuntu

ENV HOME=/home/ubuntu \
    USERPROFILE=/home/ubuntu

WORKDIR /workspace
COPY --chown=ubuntu:ubuntu . /workspace
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/project-entrypoint

# 05/06共享Actor以正式包安装；第三方依赖仍只由根requirements.txt提供。
RUN python -m pip install --no-cache-dir --no-deps /workspace/05_监督微调

# 原始日志不进入镜像；运行时通过只读挂载提供。
RUN mkdir -p \
        /workspace/02_全局数据/raw \
        /workspace/02_全局数据/analysis/lru_baseline_analysis \
        /workspace/02_全局数据/analysis/lru_miss_attribution_analysis \
    && chown -R ubuntu:ubuntu /workspace /home/ubuntu

USER ubuntu

ENTRYPOINT ["/usr/local/bin/project-entrypoint"]
CMD ["shell"]
