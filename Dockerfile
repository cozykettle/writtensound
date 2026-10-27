# syntax=docker/dockerfile:1.7
#
# One Dockerfile, two explicit final targets:
#   final-standard  CPU + Vulkan (default)
#   final-cuda      CUDA + Vulkan
#
# Engine sources are compiled from pinned tags. Model weights are never baked.

ARG WHISPERCPP_TAG=v1.9.5
ARG NEMOSPEECH_TAG=v0.2.0
ARG CUDA_VERSION=12.8.1
# Local single-GPU builds may override this (for example `75` for a T600).
# The release default keeps the fat image usable across common NVIDIA GPUs.
ARG CUDA_ARCHITECTURES="75;80;86;89;90;120"

# ---------------------------------------------------------------------------
# Standard engine build (CPU + Vulkan)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS engine-build-base

ARG DEBIAN_FRONTEND=noninteractive
# Pipelined or proxied apt downloads can arrive corrupted with an identical
# byte count (apt reports "Hash Sum mismatch"); fetch one request at a time and
# bypass proxy caches in every apt layer.
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true update && apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        glslc \
        libsentencepiece-dev \
        libvulkan-dev \
        ninja-build \
        patch \
        pkg-config \
        python3 \
        spirv-headers \
    && rm -rf /var/lib/apt/lists/*

FROM engine-build-base AS whisper-vulkan-build

ARG WHISPERCPP_TAG
RUN git clone --depth 1 --branch "${WHISPERCPP_TAG}" \
        https://github.com/ggml-org/whisper.cpp /src/whisper.cpp \
    && cmake -S /src/whisper.cpp -B /src/whisper.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_VULKAN=ON \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_TESTS=OFF \
    && cmake --build /src/whisper.cpp/build --target whisper-cli --parallel "$(nproc)" \
    && mkdir -p /out/bin /out/lib \
    && cp /src/whisper.cpp/build/bin/whisper-cli /out/bin/ \
    && find /src/whisper.cpp/build -name '*.so*' -exec cp -a {} /out/lib/ \; \
    && LD_LIBRARY_PATH=/out/lib /out/bin/whisper-cli --help >/dev/null

FROM engine-build-base AS nemo-vulkan-build

ARG NEMOSPEECH_TAG
RUN git clone --depth 1 --branch "${NEMOSPEECH_TAG}" \
        https://github.com/NVIDIA/NeMo-Speech.cpp /src/NeMo-Speech.cpp \
    && cd /src/NeMo-Speech.cpp \
    && git submodule update --init --recursive \
    && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_VULKAN=ON \
        -DNEMO_SPEECH_GGML_PATCHED=ON \
        -DNEMO_SPEECH_BUILD_ASR=ON \
        -DNEMO_SPEECH_BUILD_DIAR=ON \
        -DNEMO_SPEECH_BUILD_TTS=OFF \
        -DNEMO_SPEECH_BUILD_NMT=OFF \
        -DNEMO_SPEECH_BUILD_S2S=OFF \
        -DNEMO_SPEECH_BUILD_CLI=ON \
        -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
        -DNEMO_SPEECH_BUILD_HTTP=OFF \
        -DNEMO_SPEECH_BUILD_GRPC=OFF \
        -DNEMO_SPEECH_BUILD_TESTS=OFF \
        -DNEMO_SPEECH_BUILD_EXAMPLES=OFF \
        -DNEMO_SPEECH_BUILD_TOOLS=OFF \
    && cmake --build build --target nemo-speech --parallel "$(nproc)" \
    && mkdir -p /out/bin /out/lib \
    && cp build/bin/nemo-speech /out/bin/ \
    && find build -name '*.so*' -exec cp -a {} /out/lib/ \; \
    && LD_LIBRARY_PATH=/out/lib /out/bin/nemo-speech --help >/dev/null

# ---------------------------------------------------------------------------
# CUDA engine build (CUDA + Vulkan)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS cuda-engine-build-base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true update && apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        glslc \
        libsentencepiece-dev \
        libvulkan-dev \
        ninja-build \
        patch \
        pkg-config \
        python3 \
        spirv-headers \
    && rm -rf /var/lib/apt/lists/*

FROM cuda-engine-build-base AS whisper-cuda-build

ARG WHISPERCPP_TAG
ARG CUDA_ARCHITECTURES
RUN git clone --depth 1 --branch "${WHISPERCPP_TAG}" \
        https://github.com/ggml-org/whisper.cpp /src/whisper.cpp \
    && cmake -S /src/whisper.cpp -B /src/whisper.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_VULKAN=ON \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_TESTS=OFF \
    && cmake --build /src/whisper.cpp/build --target whisper-cli --parallel "$(nproc)" \
    && mkdir -p /out/bin /out/lib \
    && cp /src/whisper.cpp/build/bin/whisper-cli /out/bin/ \
    && find /src/whisper.cpp/build -name '*.so*' -exec cp -a {} /out/lib/ \; \
    && LD_LIBRARY_PATH=/out/lib /out/bin/whisper-cli --help >/dev/null

FROM cuda-engine-build-base AS nemo-cuda-build

ARG NEMOSPEECH_TAG
ARG CUDA_ARCHITECTURES
RUN git clone --depth 1 --branch "${NEMOSPEECH_TAG}" \
        https://github.com/NVIDIA/NeMo-Speech.cpp /src/NeMo-Speech.cpp \
    && cd /src/NeMo-Speech.cpp \
    && git submodule update --init --recursive \
    && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_VULKAN=ON \
        -DNEMO_SPEECH_GGML_PATCHED=ON \
        -DNEMO_SPEECH_BUILD_ASR=ON \
        -DNEMO_SPEECH_BUILD_DIAR=ON \
        -DNEMO_SPEECH_BUILD_TTS=OFF \
        -DNEMO_SPEECH_BUILD_NMT=OFF \
        -DNEMO_SPEECH_BUILD_S2S=OFF \
        -DNEMO_SPEECH_BUILD_CLI=ON \
        -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
        -DNEMO_SPEECH_BUILD_HTTP=OFF \
        -DNEMO_SPEECH_BUILD_GRPC=OFF \
        -DNEMO_SPEECH_BUILD_TESTS=OFF \
        -DNEMO_SPEECH_BUILD_EXAMPLES=OFF \
        -DNEMO_SPEECH_BUILD_TOOLS=OFF \
    && cmake --build build --target nemo-speech --parallel "$(nproc)" \
    && mkdir -p /out/bin /out/lib \
    && cp build/bin/nemo-speech /out/bin/ \
    && find build -name '*.so*' -exec cp -a {} /out/lib/ \; \
    && LD_LIBRARY_PATH=/out/lib /out/bin/nemo-speech --help >/dev/null

# ---------------------------------------------------------------------------
# Python application layers
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS app-standard

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true update && apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        gosu \
        libgomp1 \
        libpython3.12 \
        libsentencepiece0 \
        libsndfile1 \
        libvulkan1 \
        mesa-vulkan-drivers \
        python3 \
        python3-pip \
        python3-venv \
        vulkan-tools \
    && pip3 install --no-cache-dir --break-system-packages "uv==0.13.5" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --extra engines --no-install-project
COPY backend ./backend
COPY backend/entrypoint.sh ./entrypoint.sh
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --extra engines \
    && sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh \
    && /app/.venv/bin/python -c "import backend" \
    && /app/.venv/bin/python -m backend.engines.qwen3.runtime

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS app-cuda

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true update && apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        gosu \
        libgomp1 \
        libpython3.12 \
        libsentencepiece0 \
        libsndfile1 \
        libvulkan1 \
        mesa-vulkan-drivers \
        python3 \
        python3-pip \
        python3-venv \
        vulkan-tools \
    && pip3 install --no-cache-dir --break-system-packages "uv==0.13.5" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY backend ./backend
COPY backend/entrypoint.sh ./entrypoint.sh
# The committed lock is CPU-only. Install it first, then replace just the torch
# wheels with cu128 builds in the same layer so the CPU wheels are not retained.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --extra engines \
    && uv pip install --python /app/.venv/bin/python \
        --index https://download.pytorch.org/whl/cu128 \
        --index https://pypi.org/simple \
        --reinstall-package torch --reinstall-package torchaudio \
        "torch==2.9.1+cu128" "torchaudio==2.9.1+cu128" \
    && /app/.venv/bin/python -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda" \
    && sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh \
    && /app/.venv/bin/python -c "import backend" \
    && /app/.venv/bin/python -m backend.engines.qwen3.runtime

# ---------------------------------------------------------------------------
# Final targets
# ---------------------------------------------------------------------------
FROM app-standard AS final-standard

ENV WS_DATA_DIR=/data \
    WS_IMAGE_VARIANT=standard \
    WS_LOG_LEVEL=INFO \
    WS_NEMO_SPEECH=/opt/engines/nemo/bin/nemo-speech \
    WS_PORT=8080 \
    WS_WHISPER_CLI=/opt/engines/whispercpp/bin/whisper-cli \
    HF_HOME=/data/models/hf-cache \
    LD_LIBRARY_PATH=/opt/engines/whispercpp/lib:/opt/engines/nemo/lib

COPY --from=whisper-vulkan-build /out /opt/engines/whispercpp
COPY --from=nemo-vulkan-build /out /opt/engines/nemo

RUN mkdir -p /data \
    && ln -sf /opt/engines/whispercpp/bin/whisper-cli /usr/local/bin/whisper-cli \
    && ln -sf /opt/engines/nemo/bin/nemo-speech /usr/local/bin/nemo-speech

WORKDIR /app
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=5s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${WS_PORT}/healthz" || exit 1
CMD ["/app/entrypoint.sh"]

FROM app-cuda AS final-cuda

ENV WS_DATA_DIR=/data \
    WS_IMAGE_VARIANT=cuda \
    WS_LOG_LEVEL=INFO \
    WS_NEMO_SPEECH=/opt/engines/nemo/bin/nemo-speech \
    WS_PORT=8080 \
    WS_WHISPER_CLI=/opt/engines/whispercpp/bin/whisper-cli \
    HF_HOME=/data/models/hf-cache \
    LD_LIBRARY_PATH=/opt/engines/whispercpp/lib:/opt/engines/nemo/lib:/usr/local/cuda/lib64

COPY --from=whisper-cuda-build /out /opt/engines/whispercpp
COPY --from=nemo-cuda-build /out /opt/engines/nemo

RUN mkdir -p /data \
    && ln -sf /opt/engines/whispercpp/bin/whisper-cli /usr/local/bin/whisper-cli \
    && ln -sf /opt/engines/nemo/bin/nemo-speech /usr/local/bin/nemo-speech

WORKDIR /app
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=5s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${WS_PORT}/healthz" || exit 1
CMD ["/app/entrypoint.sh"]

# Default local build is the standard image. Compose selects an explicit target.
FROM final-standard AS final
