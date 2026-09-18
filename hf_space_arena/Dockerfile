# ================================================================
# Dockerfile.gpu — mode PRODUKSI (t4-medium, $0,60/jam).
#
# JANGAN disunting langsung sebagai `Dockerfile`. Pakai:
#     python scripts/space_mode.py gpu
# yang menyalin berkas ini menjadi `Dockerfile`, menyetel variabel Space, lalu
# mengunggahnya (unggahan itu yang memicu rebuild).
#
# Dua hal yang HARUS berubah bersamaan saat pindah ke GPU, dan kegagalannya senyap
# kalau hanya salah satu:
#   1. wheel llama-cpp-python dari indeks cu124 (wheel CPU tidak akan menyentuh GPU
#      walau GPU_LAYERS diisi — Space tetap jalan, hanya tetap selambat CPU);
#   2. basis image yang memuat pustaka CUDA runtime. python:3.12-slim tidak punya,
#      jadi wheel cu124 gagal dimuat di sana.
#
# Wheel cu124 0.3.35 adalah manylinux_2_35 → butuh glibc >= 2.35, yaitu Ubuntu 22.04.
# Itu sebabnya basisnya bukan image CUDA yang lebih lama.
# ================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Space menjalankan kontainer sebagai uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --only-binary llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
      -r requirements.txt

COPY --chown=user . /app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
