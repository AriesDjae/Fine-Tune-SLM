"""
build_train_qwen_qlora.py — generator notebook training QLoRA (T4-ready) untuk model ≤1B.

Menghasilkan TIGA notebook dengan struktur IDENTIK (harness yang sudah terbukti pada
Qwen3.5-0.8B), beda hanya di identitas model + turn-token:
  - train_qwen_qlora.ipynb        (unsloth/Qwen3.5-0.8B    — ChatML <|im_start|>, scaffold <think>)
  - train_gemma3_1b_qlora.ipynb   (unsloth/gemma-3-1b-it   — <start_of_turn>, system di-merge ke user)
  - train_llama32_1b_qlora.ipynb  (unsloth/Llama-3.2-1B-Instruct — header <|start_header_id|>, system native)

JANGAN edit .ipynb manual — edit generator ini lalu jalankan
`python notebooks/build_train_qwen_qlora.py`.

Basis: notebook Qwen0.8B yang sudah BERHASIL (FastLanguageModel, train_on_responses_only,
packing=False, pilot->GATE->full) untuk dataset Pivot-4 `Data/processed_id/` (30003/2997/2998),
dijalankan peneliti di Kaggle/Colab **T4**. Gemma 3 1B & Llama 3.2 1B juga model TEKS ≤1B ->
loader sama (FastLanguageModel).

Perbedaan per-model (diverifikasi dari template chat resmi + notebook Unsloth):
  - Qwen3.5  : ChatML; template sisip `<think>\n\n</think>` kosong walau enable_thinking=False
               -> RESPONSE_PART memuat scaffold itu (loss hanya jawaban, train==infer). System native.
  - Gemma 3  : turn-token `<start_of_turn>role`; template TIDAK punya role `system`
               -> system di-merge ke giliran user pertama (MERGE_SYSTEM_INTO_USER=True). Tanpa scaffold.
  - Llama 3.2: header `<|start_header_id|>role<|end_header_id|>`; role `system` native. Tanpa scaffold.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
_CID = [0]


def _nid():
    _CID[0] += 1
    return f"c{_CID[0]:02d}"


def md(src):
    return {"cell_type": "markdown", "id": _nid(), "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": _nid(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


CELL_INSTALL = r"""
# Install Unsloth + TRL + PEFT + bnb (T4). TANPA %%capture supaya error/konflik install terlihat.
# Kaggle/Colab sudah punya torch CUDA -> --no-deps utk paket inti agar torch tak ke-replace.
# Map xformers per torch (termasuk torch 2.11 Colab-modern). Dep baru Unsloth (tyro/msgspec/
# cut_cross_entropy/torchao) ikut dipasang eksplisit; datasets di-pin <4.0 (load_dataset json stabil).
import os, re, torch
v = re.match(r'\d+\.\d+', str(torch.__version__)).group(0)
xf_map = {'2.11':'0.0.35','2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2','2.5':'0.0.29.post3','2.4':'0.0.27.post2'}
xformers = 'xformers==' + xf_map.get(v, '0.0.35')
if v not in xf_map:
    print(f"[WARN] torch {v} tak ada di map -> default {xformers} (cek bila import gagal).")
print(f"torch={torch.__version__} -> {xformers}")
!pip install -q sentencepiece protobuf "huggingface_hub>=0.34.0" hf_transfer langdetect rouge_score
!pip install -q --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft triton unsloth tyro msgspec cut_cross_entropy torchao
!pip install -q transformers==4.56.2
!pip install -q --no-deps trl==0.22.2
!pip install -q "datasets>=3.4.1,<4.0.0"
import torch; torch._dynamo.config.recompile_limit = 64
print("install selesai — Runtime > Restart, lalu jalankan sel Verifikasi.")
"""

CELL_VERIFY_IMPORTS = r"""
# === Verifikasi (jalankan SETELAH Runtime > Restart, sebelum Konfigurasi) ===
import torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import unsloth, transformers, trl, datasets, xformers
from unsloth import FastLanguageModel
for m in (unsloth, transformers, trl, datasets, xformers):
    print(m.__name__, m.__version__)
print("OK semua import sukses.")
"""

CELL_ENV = r"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # single GPU (T4) — hindari multi-GPU surprise
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
import torch, random, numpy as np, json, glob, math
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""

CELL_CONFIG = r"""
# ====================== IDENTITAS MODEL ======================
MODEL_ID          = "__MODEL_ID__"            # __LOADER_NOTE__
MODEL_ID_FALLBACK = "__MODEL_ID_FALLBACK__"   # cadangan bila id utama gagal

# ====================== RUN MODE ======================
# "pilot" = cek sinyal cepat (subset, ~250 step). "full" = training penuh.
# MULAI dari "pilot", baca GATE, baru ganti ke "full".
RUN_MODE = "pilot"      # "pilot" | "full"

# ====================== HYPERPARAMETER (T4-safe) ======================
MAX_SEQ_LENGTH = 1024            # p99 dataset ~634 -> aman
LOAD_IN_4BIT   = True            # QLoRA
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

LEARNING_RATE = 1e-4             # ikut notebook lama (2e-4 terbukti terlalu agresif)
EPOCHS        = 3                # full-mode; + EarlyStopping(patience=2)
WARMUP_RATIO  = 0.10
WEIGHT_DECAY  = 0.01
PER_DEVICE_BATCH = 2             # T4-safe utk model ~1B 4-bit, seq 1024
GRAD_ACCUM       = 8             # effective batch = 16  (TUNABLE; lihat TROUBLESHOOTING)
MAX_GRAD_NORM    = 1.0           # grad clipping (anti loss meledak)

# ====================== FORMAT TURN (__LABEL_SHORT__) ======================
# __TURN_COMMENT__
INSTRUCTION_PART       = "__INSTRUCTION_PART__"   # penanda giliran user
RESPONSE_PART          = "__RESPONSE_PART__"      # penanda giliran asisten (loss mulai di sini)
EXPECTED_PROMPT_TAIL   = "__EXPECTED_TAIL__"      # ekor prompt inferensi (utk SELF-CHECK decode)
MERGE_SYSTEM_INTO_USER = __MERGE_SYSTEM__         # Gemma: True (template tak punya role system)

# ====================== PILOT SUBSET ======================
PILOT_TRAIN_N, PILOT_VAL_N, PILOT_MAX_STEPS = 1500, 200, 250

# ====================== OUTPUT ======================
ADAPTER_DIR = "__ADAPTER_DIR__"
print(f"RUN_MODE={RUN_MODE} | model={MODEL_ID} | eff_batch={PER_DEVICE_BATCH*GRAD_ACCUM} | seq={MAX_SEQ_LENGTH}")
"""

CELL_PATHS = r"""
# Cari dataset beku (Kaggle input / Colab Drive / lokal). Lihat README_TRAIN.md utk upload.
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(os.path.join(p, "train.jsonl")):
            return p
    return None

CANDIDATES = [
    os.environ.get("DATA_DIR", ""),
    "/kaggle/input/processed-id",                  # Kaggle dataset (rename sesuai uploadmu)
    "/kaggle/input/medical-id-v2/processed_id",
    "/content/drive/MyDrive/Fine-Tune SLM for Medical Chatbot/Data/processed_id",
    "../Data/processed_id", "Data/processed_id",
]
# Colab: mount Drive bila perlu
if any("drive/MyDrive" in c for c in CANDIDATES) and not os.path.exists("/content/drive"):
    try:
        from google.colab import drive; drive.mount("/content/drive")
    except Exception:
        pass
DATA_DIR = _first_existing(CANDIDATES)
assert DATA_DIR, ("Dataset tak ketemu. Set env DATA_DIR atau upload "
                  "train/val/test.jsonl. Lihat README_TRAIN.md.")
os.makedirs(ADAPTER_DIR, exist_ok=True)
print("DATA_DIR =", DATA_DIR)
"""

CELL_LOAD = r"""
from unsloth import FastLanguageModel

def _load(mid):
    return FastLanguageModel.from_pretrained(
        model_name     = mid,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,          # auto: T4 -> fp16 (T4 tak dukung bf16 dgn baik)
        load_in_4bit   = LOAD_IN_4BIT,
    )

try:
    model, tokenizer = _load(MODEL_ID); print("Loaded:", MODEL_ID)
except Exception as e:
    print("Gagal:", repr(e)[:120], "-> fallback", MODEL_ID_FALLBACK)
    model, tokenizer = _load(MODEL_ID_FALLBACK)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("vocab:", len(tokenizer), "| pad:", tokenizer.pad_token)
"""

CELL_PEFT = r"""
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    target_modules             = LORA_TARGET,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",   # hemat VRAM (T4)
    random_state               = SEED,
)
model.print_trainable_parameters()   # CATAT angka ini utk BAB III
"""

CELL_DATA = r"""
from datasets import load_dataset

def to_model_messages(messages):
    # Gemma: template tak punya role 'system' -> merge ke giliran user pertama.
    # Qwen/Llama: system native -> no-op (MERGE_SYSTEM_INTO_USER=False).
    if not MERGE_SYSTEM_INTO_USER:
        return messages
    sys_txt, out, injected = "", [], False
    for m in messages:
        if m["role"] == "system":
            sys_txt = m["content"]; continue
        if m["role"] == "user" and not injected and sys_txt:
            out.append({"role": "user", "content": sys_txt.strip() + "\n\n" + m["content"]})
            injected = True
        else:
            out.append(m)
    if sys_txt and not injected:
        out.insert(0, {"role": "user", "content": sys_txt})
    return out

def render_chat(messages, add_generation_prompt=False):
    msgs = to_model_messages(messages)
    # Qwen: enable_thinking=False (template tetap sisip <think></think> kosong, di-mask via RESPONSE_PART).
    # Gemma/Llama: template tak terima kwarg ini -> TypeError -> fallback tanpa kwarg.
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                   add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                   add_generation_prompt=add_generation_prompt)

def formatting_func(ex):
    return {"text": render_chat(ex["messages"], add_generation_prompt=False)}

train_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "train.jsonl"), split="train")
val_ds   = load_dataset("json", data_files=os.path.join(DATA_DIR, "val.jsonl"),   split="train")

if RUN_MODE == "pilot":
    train_ds = train_ds.shuffle(seed=SEED).select(range(min(PILOT_TRAIN_N, len(train_ds))))
    val_ds   = val_ds.shuffle(seed=SEED).select(range(min(PILOT_VAL_N, len(val_ds))))

train_ds = train_ds.map(formatting_func, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(formatting_func,   remove_columns=val_ds.column_names)
print(f"RUN_MODE={RUN_MODE} | train={len(train_ds)} | val={len(val_ds)}")
"""

CELL_DECODECHECK = r"""
# ===== SELF-CHECK decode (sama logika dgn validate_dataset_for_training.py) =====
# Pastikan: token KONTEN pertama jawaban IKUT dipelajari (tak ke-mask / off-by-one),
# format turn konsisten train==inferensi.
def _find_sub(seq, sub):
    for i in range(len(seq)-len(sub)+1):
        if seq[i:i+len(sub)] == sub: return i
    return -1

_rec   = json.loads(open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8").readline())
_msgs  = _rec["messages"]
_full  = render_chat(_msgs, add_generation_prompt=False)
_prompt= render_chat([m for m in _msgs if m["role"]!="assistant"], add_generation_prompt=True)
_ids   = tokenizer(_full, add_special_tokens=False)["input_ids"]
_marker= tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]
_pos   = _find_sub(_ids, _marker)
assert _pos >= 0, "RESPONSE_PART tak ditemukan di teks ter-render! Cek template/turn-token model."
_start = _pos + len(_marker)
_ans   = next(m["content"] for m in _msgs if m["role"]=="assistant")
_w1    = _ans.split()[0] if _ans.split() else ""
_reco  = tokenizer.decode(_ids[_start:_start+12]).lstrip()
print("first ACTIVE tokens :", [tokenizer.decode([t]) for t in _ids[_start:_start+8]])
print("jawaban kata-1 asli :", repr(_w1), "| rekonstruksi:", repr(_reco[:40]))
print("infer prompt ends   :", repr(_prompt.rstrip()[-40:]))
assert _reco.lower().startswith(_w1[:6].lower()), "OFF-BY-ONE: token jawaban pertama ke-mask/terpotong!"
assert _prompt.rstrip().endswith(EXPECTED_PROMPT_TAIL), \
    f"Inferensi TIDAK berakhir {EXPECTED_PROMPT_TAIL!r} -> train != infer!"
print("OK decode-check: jawaban medis pertama dipelajari, train==inferensi konsisten.")
"""

CELL_TRAIN_CFG = r"""
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from unsloth.chat_templates import train_on_responses_only

USE_BF16 = torch.cuda.is_bf16_supported()   # T4 -> False -> fp16

# cadence & durasi tergantung RUN_MODE
if RUN_MODE == "pilot":
    steps_kw = dict(max_steps=PILOT_MAX_STEPS, num_train_epochs=1,
                    eval_steps=50, save_steps=50)
else:
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (PER_DEVICE_BATCH*GRAD_ACCUM)))
    es = max(10, steps_per_epoch // 10)     # ~10x/epoch
    steps_kw = dict(num_train_epochs=EPOCHS, eval_steps=es, save_steps=es)

cfg = SFTConfig(
    output_dir                  = ADAPTER_DIR,
    per_device_train_batch_size = PER_DEVICE_BATCH,
    per_device_eval_batch_size  = PER_DEVICE_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_ratio                = WARMUP_RATIO,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = "cosine",
    weight_decay                = WEIGHT_DECAY,
    optim                       = "adamw_8bit",
    fp16                        = not USE_BF16,    # T4 -> fp16
    bf16                        = USE_BF16,
    max_grad_norm               = MAX_GRAD_NORM,   # grad clipping
    logging_steps               = 10,
    eval_strategy               = "steps",
    save_strategy               = "steps",
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    seed                        = SEED, data_seed = SEED,
    dataset_text_field          = "text",
    max_length                  = MAX_SEQ_LENGTH,
    packing                     = False,           # WAJIB utk train_on_responses_only
    report_to                   = "none",
    **steps_kw,
)

_cb = [] if RUN_MODE == "pilot" else [EarlyStoppingCallback(3, 0.001)]
_kw = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=cfg, callbacks=_cb)
try:
    trainer = SFTTrainer(processing_class=tokenizer, **_kw)
except TypeError:
    trainer = SFTTrainer(tokenizer=tokenizer, **_kw)

# loss HANYA pada jawaban medis (penanda turn diverifikasi di sel SELF-CHECK)
trainer = train_on_responses_only(trainer,
    instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

# Unsloth otomatis mem-patch cross-entropy (fused) -> hemat VRAM, tanpa compute_loss_func manual.
# Jika muncul OOM di logits, kurangi seq/batch (lihat TROUBLESHOOTING di README_TRAIN.md).
print("trainer siap. RUN_MODE =", RUN_MODE, "| steps_kw =", steps_kw)
"""

CELL_TRAIN_RUN = r"""
# Resume bila checkpoint ada (tahan disconnect Colab/Kaggle).
_ckpts = glob.glob(os.path.join(ADAPTER_DIR, "checkpoint-*"))
_resume = len(_ckpts) > 0
print("resume:", _resume, f"({len(_ckpts)} checkpoint)")
trainer_stats = trainer.train(resume_from_checkpoint=_resume or None)
print("training selesai.")
"""

CELL_GATE = r"""
# ===== GATE (jalankan setelah PILOT). Cetak PASS / STOP. =====
import re as _re
from langdetect import detect, DetectorFactory; DetectorFactory.seed = 42

logs = trainer.state.log_history
losses = [l["loss"] for l in logs if "loss" in l]
def _mean(x): return sum(x)/len(x) if x else float("nan")
k = max(1, len(losses)//5)
loss_start, loss_end = _mean(losses[:k]), _mean(losses[-k:])
loss_max = max(losses) if losses else float("nan")
B2_trend = (loss_end <= loss_start - 0.15)
B2_noexplode = (loss_max <= 2*losses[0]) if losses else False
B1_stable = all(np.isfinite(l) for l in losses) and len(losses) > 0
print(f"[B1] stabil(no NaN/Inf): {B1_stable}")
print(f"[B2] loss {loss_start:.3f} -> {loss_end:.3f} (turun>=0.15: {B2_trend}; tak meledak: {B2_noexplode})")

# B3 generasi: 10 prompt tetap dari test
FastLanguageModel.for_inference(model)
test_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "test.jsonl"), split="train")
gens = []
id_ok = degen = 0
for i in range(10):
    msgs = [m for m in test_ds[i]["messages"] if m["role"] != "assistant"]
    p = render_chat(msgs, add_generation_prompt=True)
    inp = tokenizer(p, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=200, do_sample=False, no_repeat_ngram_size=3,
                         repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
    txt = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    gens.append((msgs[-1]["content"], txt))
    # bahasa
    try:
        if detect(txt[:500]) == "id": id_ok += 1
    except Exception:
        pass
    # degeneratif: >30% 4-gram berulang
    toks = txt.split()
    grams = [tuple(toks[j:j+4]) for j in range(len(toks)-3)]
    rep = (1 - len(set(grams))/len(grams)) if grams else 0
    if rep > 0.30: degen += 1
B3a_lang = id_ok >= 9
B3b_degen = degen == 0
print(f"[B3a] Indonesia {id_ok}/10 (anti language-mixing: {B3a_lang})")
print(f"[B3b] non-degeneratif {10-degen}/10 (no >30% 4-gram berulang: {B3b_degen})")

# simpan generasi
with open("pilot_generations.txt", "w", encoding="utf-8") as f:
    for q, a in gens:
        f.write(f"Q: {q}\nA: {a}\n{'-'*60}\n")
print("contoh 3 generasi:")
for q, a in gens[:3]:
    print("Q:", q[:80]); print("A:", a[:160]); print("-"*40)

PASS = B1_stable and B2_trend and B2_noexplode and B3a_lang and B3b_degen
print("\n" + "="*60)
print("GATE:", "PASS_GREEN -> boleh set RUN_MODE='full'" if PASS
      else "STOP / NEEDS_HUMAN -> lihat TROUBLESHOOTING di README_TRAIN.md")
print("="*60)
FastLanguageModel.for_training(model)
"""

CELL_SAVE = r"""
# Simpan adapter LoRA (kecil) utk inspeksi/arsip. (Merge 16-bit dilakukan saat eval.)
print("best checkpoint:", trainer.state.best_model_checkpoint)
print("best eval_loss :", trainer.state.best_metric)
model.save_pretrained(ADAPTER_DIR); tokenizer.save_pretrained(ADAPTER_DIR)
print("adapter ->", ADAPTER_DIR, "| 10 generasi -> pilot_generations.txt")
"""

CELL_EVAL = r"""
# ===== (PLACEHOLDER) eval token-F1 + ROUGE-L pada val — siap pakai setelah training =====
# EM (MCQA) di-DROP utk Pivot-4 native-only; metrik = token-F1 + ROUGE-L (lihat memory).
!pip install -q rouge-score
from rouge_score import rouge_scorer
import collections

def _f1(pred, ref):
    p, r = pred.split(), ref.split()
    common = collections.Counter(p) & collections.Counter(r)
    ns = sum(common.values())
    if ns == 0 or not p or not r: return 0.0
    prec, rec = ns/len(p), ns/len(r)
    return 2*prec*rec/(prec+rec)

FastLanguageModel.for_inference(model)
val_eval = load_dataset("json", data_files=os.path.join(DATA_DIR, "val.jsonl"), split="train")
N = min(100, len(val_eval))
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
f1s, rls = [], []
for i in range(N):
    msgs = val_eval[i]["messages"]
    ref = next(m["content"] for m in msgs if m["role"]=="assistant")
    p = render_chat([m for m in msgs if m["role"]!="assistant"], add_generation_prompt=True)
    inp = tokenizer(p, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=256, do_sample=False, no_repeat_ngram_size=3,
                         repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
    pred = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    f1s.append(_f1(pred, ref)); rls.append(scorer.score(ref, pred)["rougeL"].fmeasure)
print(f"[EVAL n={N}] token-F1={sum(f1s)/N:.4f} | ROUGE-L={sum(rls)/N:.4f}")
FastLanguageModel.for_training(model)
"""


def make_notebook(M):
    cells = [
        md(f"# Fine-Tuning {M['label']} (QLoRA) — Chatbot Medis ID (Pivot 4)\n\n"
           "Dataset **ID native-only** v2.1-remediated (`Data/processed_id/`, "
           "30003/2997/2998). Toolchain **Unsloth + TRL + PEFT**, QLoRA 4-bit, "
           "dijalankan di **Kaggle/Colab T4 (single GPU)**.\n\n"
           f"> Harness IDENTIK dengan notebook Qwen3.5-0.8B yang sudah berhasil; yang berbeda "
           f"hanya identitas model & turn-token ({M['turn_note']}).\n\n"
           "**Cara pakai:** jalankan sel berurutan. Mulai `RUN_MODE=\"pilot\"` (sel "
           "Konfigurasi) → baca **GATE** → kalau PASS, set `RUN_MODE=\"full\"`, "
           "Restart & run. Detail + troubleshooting di `README_TRAIN.md`.\n\n"
           "> Validasi dataset terpisah: `preprocessing/validate_dataset_for_training.py` "
           "(harus VERDICT=PASS sebelum notebook ini)."),
        md("## 1. Install (Unsloth/TRL/PEFT/bnb) — lalu **Runtime > Restart**\n\n"
           "Tanpa `%%capture` agar konflik versi terlihat. Setelah selesai → "
           "**Runtime > Restart session** lalu lanjut dari sel Verifikasi."),
        code(CELL_INSTALL),
        md("## 1b. Verifikasi import (setelah Restart, sebelum Konfigurasi)"),
        code(CELL_VERIFY_IMPORTS),
        md("## 2. Environment (single GPU, seed)"),
        code(CELL_ENV),
        md("## 3. Konfigurasi — set `RUN_MODE` di sini\n\n"
           "`pilot` = subset 1500/200, ~250 step (cek sinyal). `full` = data penuh "
           f"+ EarlyStopping. Hyperparam T4-safe. Turn-token model ini: {M['turn_note']}."),
        code(CELL_CONFIG),
        md("## 4. Lokasi dataset beku (Kaggle/Colab/lokal)\n\n"
           "Upload `train/val/test.jsonl` sbg Kaggle dataset atau ke Drive — lihat "
           "`README_TRAIN.md`. Sel ini mencari otomatis (atau set env `DATA_DIR`)."),
        code(CELL_PATHS),
        md(f"## 5. Load model 4-bit (FastLanguageModel)\n\n"
           f"{M['label']} = model **teks** ≤1B → `FastLanguageModel` (bukan VLM). "
           "`dtype=None` → T4 auto fp16."),
        code(CELL_LOAD),
        md("## 6. LoRA adapter (QLoRA)\n\n**Catat `print_trainable_parameters()`** (BAB III)."),
        code(CELL_PEFT),
        md(f"## 7. Dataset → teks (chat template {M['short']})\n\n{M['system_note']}"),
        code(CELL_DATA),
        md("## 8. SELF-CHECK decode (WAJIB dilihat sebelum train)\n\n"
           "Membuktikan token **konten pertama jawaban** ikut dipelajari (bukan ke-mask "
           "/ off-by-one) dan format turn membuat **train == inferensi**. "
           "Identik dgn Deliverable 1 (`validate_dataset_for_training.py`)."),
        code(CELL_DECODECHECK),
        md("## 9. Konfigurasi training (SFT + train_on_responses_only)\n\n"
           "`train_on_responses_only` → loss **hanya pada jawaban medis**. `packing=False` "
           "(wajib). T4 → `fp16=True`, `max_grad_norm=1.0`. Unsloth otomatis patch "
           "cross-entropy (tanpa `compute_loss_func` manual)."),
        code(CELL_TRAIN_CFG),
        md("## 10. Train (resume-aware)"),
        code(CELL_TRAIN_RUN),
        md("## 11. GATE (setelah PILOT) — PASS/STOP\n\n"
           "Cek: loss turun (mean 20% akhir ≤ 20% awal − 0.15) & tak meledak/NaN; "
           "10 generasi: ≥9/10 Indonesia (**anti language-mixing** — cacat utama "
           "Gemma3-1B dulu), tidak degeneratif. Kalau **STOP** → "
           "`README_TRAIN.md` › TROUBLESHOOTING."),
        code(CELL_GATE),
        md("## 12. Simpan adapter + generasi"),
        code(CELL_SAVE),
        md("## 13. (Placeholder) Eval token-F1 + ROUGE-L (val)\n\n"
           "EM/MCQA di-DROP (Pivot-4 native-only) → metrik open-ended **token-F1 + "
           "ROUGE-L**. Sel siap pakai setelah training penuh."),
        code(CELL_EVAL),
        md(f"---\n**Selesai.** {M['label']} bagian dari studi comparative model ≤1B "
           "(Qwen3.5-0.8B / Gemma 3 1B / Llama 3.2 1B) — harness/notebook serupa, "
           "evaluasi token-F1 + ROUGE-L per-bahasa."),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU", "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    repl = {
        "__MODEL_ID__": M["model_id"], "__MODEL_ID_FALLBACK__": M["model_id_fallback"],
        "__LOADER_NOTE__": M["loader_note"], "__LABEL_SHORT__": M["short"],
        "__TURN_COMMENT__": M["turn_comment"],
        "__INSTRUCTION_PART__": M["instruction_part"], "__RESPONSE_PART__": M["response_part"],
        "__EXPECTED_TAIL__": M["expected_tail"], "__MERGE_SYSTEM__": M["merge_system"],
        "__ADAPTER_DIR__": M["adapter_dir"],
    }
    for c in cells:
        c["source"] = ["".join(_sub(line, repl)) for line in c["source"]]
    out = NB_DIR / M["fname"]
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    # validasi ringan: JSON valid + tak ada placeholder repl yang tersisa
    raw = out.read_text(encoding="utf-8")
    leftover = [k for k in repl if k in raw]
    assert not leftover, f"Placeholder belum tersubstitusi di {M['fname']}: {leftover}"
    json.loads(raw)
    print("wrote", out)


def _sub(line, repl):
    for k, v in repl.items():
        line = line.replace(k, v)
    return line


# Turn-token & flag per model. Diverifikasi dari template chat resmi + train_on_responses_only Unsloth.
MODELS = [
    dict(short="Qwen3.5", label="Qwen3.5-0.8B", fname="train_qwen_qlora.ipynb",
         model_id="unsloth/Qwen3.5-0.8B",
         model_id_fallback="unsloth/Qwen3.5-0.8B-Instruct",
         loader_note="model TEKS ~0.8B -> FastLanguageModel",
         turn_note="ChatML `<|im_start|>` + scaffold `<think>`",
         turn_comment=("Template Qwen sisip <think>\\n\\n</think> kosong walau enable_thinking=False"
                       " -> RESPONSE_PART memuat scaffold itu agar loss hanya di jawaban (train==infer)."),
         system_note="Qwen mendukung role `system` native (tidak di-merge).",
         instruction_part=r"<|im_start|>user\n",
         response_part=r"<|im_start|>assistant\n<think>\n\n</think>\n\n",
         expected_tail=r"</think>", merge_system="False",
         adapter_dir="checkpoints/qwen_qlora"),
    dict(short="Gemma 3", label="Gemma 3 1B IT", fname="train_gemma3_1b_qlora.ipynb",
         model_id="unsloth/gemma-3-1b-it",
         model_id_fallback="unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
         loader_note="model TEKS 1B -> FastLanguageModel",
         turn_note="turn-token `<start_of_turn>`, system di-merge ke user",
         turn_comment=("Gemma pakai <start_of_turn>role; template TIDAK punya role 'system'"
                       " -> system di-merge ke giliran user pertama. Tanpa scaffold thinking."),
         system_note=("**Gemma tidak punya role `system`** di template → `MERGE_SYSTEM_INTO_USER=True` "
                      "menyisipkan system prompt ke giliran user pertama."),
         instruction_part=r"<start_of_turn>user\n",
         response_part=r"<start_of_turn>model\n",
         expected_tail=r"<start_of_turn>model", merge_system="True",
         adapter_dir="checkpoints/gemma3_1b_qlora"),
    dict(short="Llama 3.2", label="Llama 3.2 1B Instruct", fname="train_llama32_1b_qlora.ipynb",
         model_id="unsloth/Llama-3.2-1B-Instruct",
         model_id_fallback="unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
         loader_note="model TEKS 1B -> FastLanguageModel",
         turn_note="header `<|start_header_id|>`, system native",
         turn_comment=("Llama pakai header <|start_header_id|>role<|end_header_id|>; role 'system'"
                       " native (tidak di-merge). Tanpa scaffold thinking."),
         system_note="Llama mendukung role `system` native (tidak di-merge).",
         instruction_part=r"<|start_header_id|>user<|end_header_id|>\n\n",
         response_part=r"<|start_header_id|>assistant<|end_header_id|>\n\n",
         expected_tail=r"<|start_header_id|>assistant<|end_header_id|>", merge_system="False",
         adapter_dir="checkpoints/llama32_1b_qlora"),
]


def build():
    for M in MODELS:
        make_notebook(M)
    print("done.")


if __name__ == "__main__":
    build()
