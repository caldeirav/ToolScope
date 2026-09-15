# ToolScope devcontainer: Python eval harness + CUDA llama-server (aarch64 / DGX Spark GB10).
#
# GB10 (sm_121) requires CUDA 13.x nvcc — CUDA 12.6 cannot target compute_121.
# Recipes: llama.cpp #18022, NVIDIA DGX Spark forums, stelterlab/spark.Dockerfile gist,
# merve/llama.cpp-dgx-spark-gb10-sm121a (HF prebuilt @ 121a-real).
ARG CUDA_VERSION=13.0.2
ARG UBUNTU_VERSION=24.04
ARG LLAMA_CPP_TAG=b6985

# ── Build llama-server ──────────────────────────────────────────────────────
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama-builder
ARG LLAMA_CPP_TAG
ENV DEBIAN_FRONTEND=noninteractive
# Required so cmake/nvcc can link libcuda during image build (no GPU in buildkit).
ENV LD_LIBRARY_PATH=/usr/local/cuda-13/compat:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY eval/local/vendor/llama.cpp /src
COPY eval/local/patches/llama_server_extended_ctx.patch /tmp/llama_server_extended_ctx.patch
RUN patch -p1 -d /src < /tmp/llama_server_extended_ctx.patch

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=121a-real \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    && cmake --build build --target llama-server -j "$(nproc)" \
    && mkdir -p /opt/llama.cpp/bin \
    && cp -a build/bin/llama-server build/bin/*.so* /opt/llama.cpp/bin/

# ── Dev image ─────────────────────────────────────────────────────────────────
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG LLAMA_CPP_TAG
ARG CUDA_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    TOOLSCOPE_DEVCONTAINER=1 \
    TOOLSCOPE_INFERENCE_MODE=native \
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_API_KEY=local \
    TOOLSCOPE_MODEL_CACHE=/workspace/eval/local/models \
    LLAMA_CPP_TAG=${LLAMA_CPP_TAG} \
    CUDA_VERSION=${CUDA_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl ca-certificates libgomp1 \
    cmake build-essential patch \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama-builder /opt/llama.cpp/bin/ /opt/llama.cpp/bin/
ENV PATH=/opt/llama.cpp/bin:${PATH}
# Runtime: llama shared objects only. Host libcuda comes from nvidia-container-toolkit.
# Do not prepend cuda-*/compat here — on current GB10 drivers that causes error 803
# (see llama.cpp #23111). Compat is build-stage only (llama-builder above).
ENV LD_LIBRARY_PATH=/opt/llama.cpp/bin:${LD_LIBRARY_PATH}
RUN ln -sf /opt/llama.cpp/bin/llama-server /usr/local/bin/llama-server
COPY eval/local/scripts/cuda_runtime_env.sh /usr/local/bin/toolscope-cuda-runtime-env.sh
COPY eval/local/container/entrypoint.sh /usr/local/bin/llama-entrypoint.sh
RUN chmod +x /usr/local/bin/toolscope-cuda-runtime-env.sh /usr/local/bin/llama-entrypoint.sh

WORKDIR /workspace

# Python deps baked at image build time (avoid ~1.3GB pip on every docker run --rm).
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY eval/requirements.txt ./eval/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -e ".[st,dev]" \
    && python3 -m pip install --no-cache-dir --break-system-packages -r eval/requirements.txt

COPY .devcontainer/post-create.sh /usr/local/bin/toolscope-post-create.sh
RUN chmod +x /usr/local/bin/toolscope-post-create.sh

# Marker for serve_model.sh native-mode detection.
RUN mkdir -p /etc/toolscope && echo "1" > /etc/toolscope/devcontainer

CMD ["sleep", "infinity"]
