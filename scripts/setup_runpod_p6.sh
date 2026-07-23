#!/usr/bin/env bash
# ================================================================
# setup_runpod_p6.sh — setup environment training Pivot 6 di RunPod.
# Target: H100/A100 (bf16 LoRA, default notebook) atau GPU 24GB (QLoRA).
# Identik dengan sel Install di notebooks/train_*_p6.ipynb.
#
# Pakai:  bash scripts/setup_runpod_p6.sh
# Lalu:   upload Data/processed_id_final/{train,val,test}.jsonl
#         ke /workspace/processed_id_final/ (dicari otomatis oleh notebook).
# ================================================================
set -euo pipefail

# xformers dipilih sesuai torch bawaan image (JANGAN re-install torch).
XFORMERS=$(python - <<'PY'
import re, torch
v = re.match(r"\d+\.\d+", str(torch.__version__)).group(0)
m = {"2.11":"0.0.35","2.10":"0.0.34","2.9":"0.0.33.post1","2.8":"0.0.32.post2",
     "2.6":"0.0.29.post3","2.5":"0.0.29.post3","2.4":"0.0.27.post2"}
print("xformers==" + m.get(v, "0.0.35"))
PY
)
echo "torch=$(python -c 'import torch;print(torch.__version__)') -> ${XFORMERS}"

pip install -q sentencepiece protobuf "huggingface_hub>=0.34.0" hf_transfer langdetect rouge_score
pip install -q --no-deps unsloth_zoo bitsandbytes accelerate "${XFORMERS}" peft triton unsloth \
    tyro msgspec cut_cross_entropy torchao
pip install -q transformers==5.5.0    # qwen3_5 butuh >=5.x; 5.5.0 = maks didukung unsloth
pip install -q --no-deps trl==0.24.0
pip install -q "datasets>=3.4.1,<4.0.0"

python - <<'PY'
import torch, unsloth, transformers, trl, datasets, xformers
print("VERIFIKASI:")
print("  torch       ", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("  bf16 support:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "-")
for m in (unsloth, transformers, trl, datasets, xformers):
    print(f"  {m.__name__:<12}", m.__version__)
PY
echo "Setup selesai. Buka notebooks/train_{gemma3_4b,llama32_3b,qwen35_0.8b}_p6.ipynb"
