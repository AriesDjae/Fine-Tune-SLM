#!/usr/bin/env bash
# convert_to_gguf.sh — wrapper tipis untuk export_gguf.py (Bagian 4).
# Konversi KEDUA model merged 16-bit -> GGUF Q4_K_M via llama.cpp.
#
# Pakai:  bash scripts/convert_to_gguf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python export_gguf.py --merged_dir outputs/merged/qwen35-2b-medical \
    --model_type qwen --verify

python export_gguf.py --merged_dir outputs/merged/gemma4-e2b-medical \
    --model_type gemma --verify

echo "Selesai. GGUF Q4_K_M ada di outputs/gguf/"
echo "Lanjut: eval ulang terkuantisasi -> 'python eval.py --gguf ... --label *_q4'"
