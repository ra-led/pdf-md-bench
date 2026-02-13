FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DPSK_WORKDIR=/opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ninja-build \
    cmake \
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
      # --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install vllm==0.8.5
RUN python3 -m pip install flash-attn==2.7.3 --no-build-isolation

# DeepSeek OCR source and dependencies
RUN git clone https://github.com/deepseek-ai/DeepSeek-OCR.git /opt/DeepSeek-OCR
RUN python3 -m pip install -r /opt/DeepSeek-OCR/requirements.txt

COPY requirements.pipeline.txt /app/requirements.pipeline.txt
RUN python3 -m pip install -r /app/requirements.pipeline.txt

COPY openrouter_stream.py /app/openrouter_stream.py
COPY ocr_runner.py /app/ocr_runner.py
COPY pdf_to_characteristics_json.py /app/pdf_to_characteristics_json.py
COPY pdf_to_characteristics_json_tables.py /app/pdf_to_characteristics_json_tables.py
COPY pdf_to_characteristics_json_pages.py /app/pdf_to_characteristics_json_pages.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
