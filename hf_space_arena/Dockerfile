# ================================================================
# Dockerfile — Space Docker "arena penilaian dokter".
#
# Tidak ada torch: model dijalankan llama.cpp lewat GGUF Q4_K_M, transformers dipakai
# HANYA untuk chat template (AutoTokenizer). Image kecil, build beberapa menit.
#
# DUA MODE PERANGKAT KERAS:
#   demo/pengembangan : cpu-upgrade (8 vCPU/32 GB) — $0,03/jam, GPU_LAYERS=0
#   produksi          : t4-medium  (8 vCPU/30 GB + T4 16 GB) — $0,60/jam, GPU_LAYERS=99
#
# Untuk produksi, indeks wheel di bawah HARUS diganti ke cu124 (barisnya sudah
# disiapkan) — wheel CPU tidak akan memakai GPU meski GPU_LAYERS diisi, dan
# kegagalannya senyap: Space tetap jalan, hanya tetap selambat CPU.
# ================================================================
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1

# Space menjalankan kontainer sebagai uid 1000; menulis ke /root tidak diizinkan,
# jadi cache HF harus berada di rumah pengguna itu.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

COPY --chown=user requirements.txt .
# Wheel PREBUILT — jangan biarkan pip mengompilasi llama-cpp-python dari sumber.
# --only-binary bukan optimasi melainkan jaring pengaman: kalau pin di requirements.txt
# menunjuk versi tanpa wheel, pip diam-diam jatuh ke sdist dan mengompilasi llama.cpp
# di image tanpa toolchain — gagal setelah puluhan menit dengan galat compiler yang
# tidak menyebut-nyebut soal versi. Flag ini membuatnya gagal SEKETIKA.
#
# GANTI baris indeks ini saat pindah ke t4-medium:
#   https://abetlen.github.io/llama-cpp-python/whl/cu124
RUN pip install --no-cache-dir --only-binary llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      -r requirements.txt

# Kode + halaman + bank disalin belakangan supaya menyuntingnya tidak mengulang
# layer dependensi. Bank berisi jawaban pra-generate (JSON kecil); bobot GGUF TIDAK
# dipanggang ke image — 9,3 GB akan memperlambat tiap rebuild dan tiap pull.
COPY --chown=user . /app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
