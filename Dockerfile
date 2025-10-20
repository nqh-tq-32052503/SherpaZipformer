# ---- Base: CUDA 12.4 runtime with cuDNN (Ubuntu 22.04) ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# System deps (Python 3.11 + audio libs + git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Create a clean virtualenv
RUN python3.11 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# -------- Python deps (mirror the notebook) --------
# Torch stack (CUDA 12.4 wheels)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Core libs the notebook used
RUN pip install \
    fastapi==0.115.4 uvicorn[standard]==0.32.0 \
    jiwer==3.0.3 sentencepiece tensorboard \
    kaldialign kaldilm kaldifst

# Lhotse (from Git, as in your notebook)
RUN pip install "git+https://github.com/lhotse-speech/lhotse"

# ---- Optional: install k2 from a local wheel (recommended for CUDA 12.4) ----
# Put your wheel in ./wheels before building (example: k2-1.24.4.dev20250715+cuda12.4...cp311-manylinux2014_x86_64.whl)
# If no wheel is provided, this layer will be a no-op.
ARG K2_WHEEL_URL="https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250715+cuda12.4.torch2.6.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
RUN set -eux; \
    curl -L --retry 5 --retry-all-errors --fail \
         -o /tmp/k2.whl \
         "$K2_WHEEL_URL"; \
    pip install /tmp/k2.whl && rm -f /tmp/k2.whl


# ---- App code ----
WORKDIR /app
RUN wget https://huggingface.co/zzasdf/viet_iter3_pseudo_label/resolve/main/exp/pretrained.pt
# (Copy only dependency descriptors first if you later add a requirements.txt to speed caching)
COPY . /app

# Health + ports
EXPOSE 8002

# Helpful defaults for deterministic threading
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ---- Run FastAPI (adjust target if your module/path differs) ----
# Single worker by default (good for GPU). Scale via process manager if needed.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
