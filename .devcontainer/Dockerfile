# ToolScope devcontainer: Python eval harness + CUDA llama-server (aarch64 / DGX Spark).
ARG LLAMA_CPP_TAG=b4897

# ── Build llama-server ──────────────────────────────────────────────────────
FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu22.04 AS llama-builder
ARG LLAMA_CPP_TAG
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggerganov/llama.cpp.git .

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target llama-server -j "$(nproc)" \
    && mkdir -p /opt/llama.cpp/bin \
    && cp -a build/bin/llama-server build/bin/*.so* /opt/llama.cpp/bin/

# ── Dev image ─────────────────────────────────────────────────────────────────
FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG LLAMA_CPP_TAG
ENV DEBIAN_FRONTEND=noninteractive \
    TOOLSCOPE_DEVCONTAINER=1 \
    TOOLSCOPE_INFERENCE_MODE=native \
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_API_KEY=local \
    TOOLSCOPE_MODEL_CACHE=/workspace/eval/local/models \
    LLAMA_CPP_TAG=${LLAMA_CPP_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama-builder /opt/llama.cpp/bin/ /opt/llama.cpp/bin/
ENV PATH=/opt/llama.cpp/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/llama.cpp/bin:${LD_LIBRARY_PATH}
RUN ln -sf /opt/llama.cpp/bin/llama-server /usr/local/bin/llama-server
COPY eval/local/container/entrypoint.sh /usr/local/bin/llama-entrypoint.sh
RUN chmod +x /usr/local/bin/llama-entrypoint.sh

WORKDIR /workspace

# Python deps baked at image build time (avoid ~1.3GB pip on every docker run --rm).
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY eval/requirements.txt ./eval/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -e ".[st,dev]" \
    && python3 -m pip install --no-cache-dir -r eval/requirements.txt

COPY .devcontainer/post-create.sh /usr/local/bin/toolscope-post-create.sh
RUN chmod +x /usr/local/bin/toolscope-post-create.sh

# Marker for serve_model.sh native-mode detection.
RUN mkdir -p /etc/toolscope && echo "1" > /etc/toolscope/devcontainer

CMD ["sleep", "infinity"]
