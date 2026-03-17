FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    cmake \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Core ML runtime (CUDA 12.1)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      torch==2.6.0 \
      torchvision==0.21.0 \
      torchaudio==2.6.0

RUN python3 -m pip install vllm==0.8.5
RUN python3 -m pip install flash-attn==2.7.3 --no-build-isolation

# DeepSeek OCR source is vendored locally (no git clone in build).
COPY vendor/ /app/vendor/
RUN python3 -m pip install -r /app/vendor/requirements.deepseek.txt

COPY pdf2md_service.py /app/pdf2md_service.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
