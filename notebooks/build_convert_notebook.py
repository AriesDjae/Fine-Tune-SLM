"""
build_convert_notebook.py — generator notebook Colab untuk CONVERT hasil training P6
(adapter LoRA di HuggingFace) -> model merged 16-bit + GGUF (Q4_K_M dst).

Sumber adapter: repo HF `AriesDjaenuri/p6-hasil`, subfolder per model:
    qwen35_0.8b_p6/adapter  (base unsloth/Qwen3.5-0.8B  = TEKS  -> FastLanguageModel)
    qwen35_2b_p6/adapter    (base unsloth/Qwen3.5-2B    = VLM   -> FastModel, text-only)
    qwen35_4b_p6/adapter    (base unsloth/Qwen3.5-4B    = VLM   -> FastModel, text-only)

Jangan edit .ipynb manual — edit file ini lalu:
    python notebooks/build_convert_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "convert_to_gguf_p6.ipynb"

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": src})


md("""# Convert Adapter P6 → Model Merged 16-bit + GGUF (Colab)

Meng-convert hasil fine-tuning **Qwen3.5 0.8B / 2B / 4B** (Pivot 6) dari **adapter LoRA di HuggingFace**
(`AriesDjaenuri/p6-hasil`) menjadi:
1. **Model merged 16-bit** (format HF standar, siap eval/deploy transformers), dan
2. **GGUF terkuantisasi** (Q4_K_M dst) untuk llama.cpp / on-device (Termux).

**Cara pakai:** Runtime → **GPU** (T4/L4 cukup untuk 0.8B/2B; 4B lebih aman L4/A100).
Jalankan sel berurutan. Pilih model di sel **Konfigurasi**.

> Adapter dilatih **bf16 LoRA** (`LOAD_IN_4BIT=False`). Merge sebaiknya dari base **16-bit**
> (bukan 4-bit) agar tanpa artefak kuantisasi ganda. 0.8B=teks→`FastLanguageModel`;
> 2B/4B=VLM (vision beku, text-only)→`FastModel`. Prompt = ChatML Qwen (scaffold `<think></think>`).""")

md("## 1. Install Unsloth + toolchain (PINNED — sama seperti notebook training) — lalu **Restart jika diminta**")

code("""%%capture
# Install RESMI Unsloth versi PINNED (menyelesaikan konflik torchaudio/_LazyModule/quant_state
# pada model VLM Qwen3.5). JANGAN ubah versi. Sumber: notebook resmi Unsloth Qwen3.5 (Vision).
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install --upgrade transformers
!pip install --no-deps trl==0.22.2
import torch; torch._dynamo.config.recompile_limit = 64""")

md("## 2. Login HuggingFace (untuk baca adapter & — opsional — push hasil)")

code('''# Repo adapter `p6-hasil` publik? Jika ya, login opsional (tetap disarankan agar rate-limit longgar).
# Untuk PUSH hasil (opsional, sel terakhir) login WAJIB.
# Simpan token di Colab: kunci gembok kiri > Secrets > tambah "HF_TOKEN".
import os
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF login OK.")
else:
    print("HF_TOKEN tak diset -> lanjut anonim (cukup jika repo publik; push dinonaktifkan).")''')

md("""## 3. Konfigurasi — **pilih model** di sini

`MODEL_KEY` menentukan subfolder adapter, base model, dan loader. Ubah lalu jalankan.""")

code('''# ====================== PILIH MODEL ======================
MODEL_KEY = "0.8b"          # "0.8b" | "2b" | "4b"

# ====================== SUMBER ADAPTER (HuggingFace) ======================
REPO_ID = "AriesDjaenuri/p6-hasil"
# Peta: key -> (subfolder adapter di repo, base model, loader Unsloth)
REGISTRY = {
    "0.8b": dict(subfolder="qwen35_0.8b_p6/adapter", base="unsloth/Qwen3.5-0.8B", loader="language"),
    "2b":   dict(subfolder="qwen35_2b_p6/adapter",   base="unsloth/Qwen3.5-2B",   loader="vision"),
    "4b":   dict(subfolder="qwen35_4b_p6/adapter",   base="unsloth/Qwen3.5-4B",   loader="vision"),
}
assert MODEL_KEY in REGISTRY, f"MODEL_KEY harus salah satu dari {list(REGISTRY)}"
CFG = REGISTRY[MODEL_KEY]

# ====================== OUTPUT ======================
MAX_SEQ_LENGTH = 1024
OUT_NAME    = f"qwen35-{MODEL_KEY}-medical-id"        # nama model hasil merge
MERGED_DIR  = f"/content/outputs/merged/{OUT_NAME}"
GGUF_DIR    = f"/content/outputs/gguf/{OUT_NAME}"
GGUF_QUANTS = ["q4_k_m", "q8_0"]                      # Q4_K_M=deploy; Q8_0=referensi mutu (trade-off kuantisasi)

# ====================== (opsional) SIMPAN KE DRIVE ======================
SAVE_TO_DRIVE = False
DRIVE_BASE    = "/content/drive/MyDrive/Aries/Fine-Tune SLM for Medical Chatbot/outputs"

# ====================== (opsional) PUSH HASIL KE HF ======================
PUSH_TO_HF        = False
HF_MERGED_REPO    = f"AriesDjaenuri/{OUT_NAME}"        # repo tujuan model merged 16-bit
HF_GGUF_REPO      = f"AriesDjaenuri/{OUT_NAME}-GGUF"   # repo tujuan GGUF

import os
os.makedirs(MERGED_DIR, exist_ok=True); os.makedirs(GGUF_DIR, exist_ok=True)
print(f"MODEL_KEY={MODEL_KEY} | base={CFG['base']} | loader={CFG['loader']}")
print(f"adapter subfolder: {REPO_ID}/{CFG['subfolder']}")
print(f"MERGED_DIR={MERGED_DIR}\\nGGUF_DIR={GGUF_DIR} | quants={GGUF_QUANTS}")''')

md("## 4. Download adapter LoRA dari HF (subfolder saja)")

code('''from huggingface_hub import snapshot_download
DL_ROOT = "/content/hf_adapter"
snapshot_download(repo_id=REPO_ID, allow_patterns=f"{CFG['subfolder']}/*",
                  local_dir=DL_ROOT, token=HF_TOKEN)
ADAPTER_DIR = os.path.join(DL_ROOT, CFG["subfolder"])
print("adapter ->", ADAPTER_DIR)
print("isi:", os.listdir(ADAPTER_DIR))
assert os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")), \\
    "adapter_config.json tak ada -> cek subfolder / MODEL_KEY."''')

md("""## 5. Load base + adapter (16-bit, untuk merge bersih)

Unsloth `from_pretrained` mendeteksi `adapter_config.json` → memuat base (`base_model_name_or_path`)
lalu menempelkan adapter LoRA. `load_in_4bit=False` → merge dari bobot 16-bit (tanpa kuantisasi ganda).""")

code('''import os
# WAJIB sebelum import unsloth: cegah bug NameError('args') di compiled module saat model.generate
# (kena sanity-generate sel 6 & eval Fase 1). Tak berpengaruh ke bobot/merge, hanya matikan kompilasi.
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

if CFG["loader"] == "language":
    from unsloth import FastLanguageModel as Loader   # 0.8B = teks
else:
    from unsloth import FastModel as Loader           # 2B/4B = VLM (text-only)

model, tokenizer = Loader.from_pretrained(
    model_name     = ADAPTER_DIR,     # dir adapter -> Unsloth auto base + adapter
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,            # auto: bf16 (Ampere/Hopper) / fp16 (T4)
    load_in_4bit   = False,           # 16-bit -> merge bersih
)
# Safety: sebagian VLM Qwen3.5 punya config.architectures=None saat text-only -> set eksplisit
if not getattr(model.config, "architectures", None):
    model.config.architectures = ["Qwen3ForCausalLM"]
print("loaded:", CFG["base"], "| dtype:", next(model.parameters()).dtype)''')

md("## 6. Sanity generate (ChatML Qwen, scaffold `<think></think>`) — cek adapter hidup")

code('''try:
    Loader.for_inference(model)
except Exception:
    pass
q = "Apa penyebab umum demam pada anak dan kapan sebaiknya dibawa ke dokter?"
prompt = ("<|im_start|>system\\nYou are a helpful medical assistant.<|im_end|>\\n"
          f"<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n")
inp = tokenizer(text=prompt, return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=200, do_sample=False,
                     no_repeat_ngram_size=3, repetition_penalty=1.1)
print(tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
try:
    Loader.for_training(model)   # kembalikan mode default sebelum save
except Exception:
    pass''')

md("""## 7. Merge → simpan model **16-bit** (format HF)

`save_pretrained_merged(..., save_method="merged_16bit")` menggabungkan adapter ke base lalu
menyimpan model utuh 16-bit (config + tokenizer). Ini "model"-nya (bisa untuk eval `eval.py --loader hf`).""")

code('''model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print("MERGED 16-bit ->", MERGED_DIR)
for f in sorted(os.listdir(MERGED_DIR)):
    p = os.path.join(MERGED_DIR, f)
    print(f"  {f:40s} {os.path.getsize(p)/1e6:8.1f} MB")''')

md("""## 8. Export **GGUF** — jalur A (Unsloth, satu baris)

Cara termudah: Unsloth otomatis clone+build llama.cpp lalu quantize. Jika gagal
(umum utk VLM), lanjut ke **jalur B** (llama.cpp manual di sel berikut).""")

code('''gguf_ok = False
try:
    # Unsloth menerima str atau list quant. Merge sudah dilakukan; ini re-quantize dari model di memori.
    model.save_pretrained_gguf(GGUF_DIR, tokenizer,
                               quantization_method=GGUF_QUANTS if len(GGUF_QUANTS) > 1 else GGUF_QUANTS[0])
    gguf_ok = True
    print("GGUF (jalur A) OK ->", GGUF_DIR)
except TypeError:
    # API lama: nama argumen berbeda
    try:
        model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_type=GGUF_QUANTS[0])
        gguf_ok = True
        print("GGUF (jalur A, API lama) OK ->", GGUF_DIR)
    except Exception as e:
        print("[jalur A gagal]", repr(e)[:200], "\\n-> pakai jalur B (sel berikut).")
except Exception as e:
    print("[jalur A gagal]", repr(e)[:200], "\\n-> pakai jalur B (sel berikut).")

import glob
# Unsloth menyimpan GGUF ke folder ber-suffix "_gguf" -> deteksi lokasi sebenarnya
GGUF_OUT = next((d for d in (GGUF_DIR + "_gguf", GGUF_DIR)
                 if glob.glob(os.path.join(d, "*.gguf"))), GGUF_DIR)
if gguf_ok:
    print("GGUF di:", GGUF_OUT)
    for f in sorted(glob.glob(os.path.join(GGUF_OUT, "*.gguf"))):
        tag = "  <- VISION mmproj (ABAIKAN utk teks)" if "mmproj" in f.lower() else ""
        print(f"  {os.path.basename(f)}  {os.path.getsize(f)/1e9:.2f} GB{tag}")''')

md("""## 9. Export GGUF — jalur B (llama.cpp manual) — **hanya jika jalur A gagal**

Convert `MERGED_DIR` (HF 16-bit) → `f16.gguf` → quantize. Robust untuk arsitektur Qwen3.5
(termasuk VLM: llama.cpp mengekstrak language-model utk teks). Lewati sel ini jika `gguf_ok=True`.""")

code('''if gguf_ok:
    print("jalur A sudah sukses -> lewati jalur B.")
else:
    import subprocess, sys
    LLAMA = "/content/llama.cpp"
    if not os.path.exists(LLAMA):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp", LLAMA], check=True)
    # dependency converter
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(LLAMA, "requirements.txt")], check=True)
    # build (llama-quantize)
    if not os.path.exists(os.path.join(LLAMA, "build/bin/llama-quantize")):
        subprocess.run(["cmake", "-B", f"{LLAMA}/build", "-S", LLAMA,
                        "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"], check=True)
        subprocess.run(["cmake", "--build", f"{LLAMA}/build", "--config", "Release",
                        "-j", "--target", "llama-quantize", "llama-cli"], check=True)

    conv = os.path.join(LLAMA, "convert_hf_to_gguf.py")
    f16 = os.path.join(GGUF_DIR, f"{OUT_NAME}-f16.gguf")
    subprocess.run([sys.executable, conv, MERGED_DIR, "--outfile", f16, "--outtype", "f16"], check=True)

    qbin = os.path.join(LLAMA, "build/bin/llama-quantize")
    for q in GGUF_QUANTS:
        outq = os.path.join(GGUF_DIR, f"{OUT_NAME}-{q.upper()}.gguf")
        subprocess.run([qbin, f16, outq, q.upper()], check=True)
        print("GGUF ->", outq, f"({os.path.getsize(outq)/1e9:.2f} GB)")
    os.remove(f16)   # hemat disk (hapus f16 antara)
    gguf_ok = True
    GGUF_OUT = GGUF_DIR   # jalur B menulis langsung ke GGUF_DIR''')

md("## 10. (Opsional) Verifikasi GGUF dengan llama-cli")

code('''import glob, subprocess, os
_dir = next((d for d in (GGUF_DIR + "_gguf", GGUF_DIR)
             if glob.glob(os.path.join(d, "*.gguf"))), GGUF_DIR)
ggufs = [f for f in glob.glob(os.path.join(_dir, "*.gguf")) if "mmproj" not in f.lower()]
cli = "/content/llama.cpp/build/bin/llama-cli"
if ggufs and os.path.exists(cli):
    q = "Apa penyebab umum demam pada anak dan kapan harus ke dokter?"
    prompt = ("<|im_start|>system\\nYou are a helpful medical assistant.<|im_end|>\\n"
              f"<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n")
    subprocess.run([cli, "-m", ggufs[0], "-p", prompt, "-n", "120", "--temp", "0", "-no-cnv"])
else:
    print("llama-cli tak tersedia (jalur A dipakai) atau tak ada .gguf -> lewati verify.")
    print("GGUF:", ggufs)''')

md("## 11. (Opsional) Simpan hasil ke Google Drive")

code('''if SAVE_TO_DRIVE:
    from google.colab import drive
    if not os.path.exists("/content/drive"):
        drive.mount("/content/drive")
    import shutil
    for src, sub in [(MERGED_DIR, "merged"), (GGUF_DIR, "gguf")]:
        dst = os.path.join(DRIVE_BASE, sub, OUT_NAME)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("copied ->", dst)
else:
    print("SAVE_TO_DRIVE=False -> lewati.")''')

md("## 12. (Opsional) Push model merged + GGUF ke HuggingFace Hub")

code('''if PUSH_TO_HF and HF_TOKEN:
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    # model merged 16-bit
    create_repo(HF_MERGED_REPO, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_MERGED_REPO, token=HF_TOKEN)
    print("pushed merged ->", HF_MERGED_REPO)
    # GGUF (hanya LLM q4_k_m/q8_0; file *mmproj* = vision -> skip utk deploy teks)
    if gguf_ok:
        import glob
        _dir = next((d for d in (GGUF_DIR + "_gguf", GGUF_DIR)
                     if glob.glob(os.path.join(d, "*.gguf"))), GGUF_DIR)
        create_repo(HF_GGUF_REPO, exist_ok=True, token=HF_TOKEN)
        api.upload_folder(folder_path=_dir, repo_id=HF_GGUF_REPO,
                          allow_patterns="*.gguf", ignore_patterns="*mmproj*", token=HF_TOKEN)
        print("pushed GGUF ->", HF_GGUF_REPO, "(dari", _dir, ", mmproj di-skip)")
elif PUSH_TO_HF:
    print("PUSH_TO_HF=True tapi HF_TOKEN kosong -> set Secret HF_TOKEN dulu.")
else:
    print("PUSH_TO_HF=False -> lewati.")''')

md("""---
**Selesai.** Ulangi dari sel **Konfigurasi** dengan `MODEL_KEY` berbeda (`2b`, `4b`) untuk model lain.

- Model merged 16-bit → `MERGED_DIR` (eval: `python eval.py --model <dir> --loader hf`).
- GGUF Q4_K_M → `GGUF_DIR` (on-device/Termux via llama.cpp).
- **Catatan 4B:** patuhi GATE RAM on-device (lihat memory/HANDOFF) sebelum dinyatakan layak deploy Puskesmas.""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# beri id unik tiap sel (nbformat_minor 5)
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, "|", len(cells), "cells")
