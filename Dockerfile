# ================================================================
# Dockerfile — worker RunPod Serverless utk training Pivot 6
# (scaling Qwen3.5 0.8B/2B/4B, LoRA bf16 / QLoRA via saklar job).
# Build otomatis oleh RunPod GitHub integration (tak perlu Docker lokal).
#
# Base: torch 2.8.0 + CUDA 12.8 (Hopper/H100 OK) -> xformers 0.0.32.post2
# (peta versi di requirements_pivot6.txt). torch TIDAK di-replace:
# paket inti dipasang --no-deps, identik scripts/setup_runpod_p6.sh.
# triton sudah dibawa base image -> pip skip (no upgrade tanpa -U).
# ================================================================
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Layer dependensi terpisah dari kode -> rebuild kode tak mengulang install
RUN pip install -q sentencepiece protobuf "huggingface_hub>=0.34.0" hf_transfer langdetect rouge_score && \
    pip install -q --no-deps unsloth_zoo bitsandbytes accelerate xformers==0.0.32.post2 peft triton unsloth tyro msgspec cut_cross_entropy torchao && \
    pip install -q transformers==4.56.2 && \
    pip install -q --no-deps trl==0.22.2 && \
    pip install -q "datasets>=3.4.1,<4.0.0" && \
    pip install -q runpod

COPY . /app

CMD ["python", "-u", "handler.py"]
