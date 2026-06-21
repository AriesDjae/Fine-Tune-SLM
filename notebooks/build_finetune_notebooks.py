"""
build_finetune_notebooks.py

Generator untuk DUA notebook training (struktur identik, sesuai MASTER NOTE Bagian 2):
  - finetune_qwen35_2b.ipynb  (Qwen3.5-2B  : dense ~2B; VLM di-load text-only; system native)
  - finetune_gemma4_e2b.ipynb (Gemma 4 E2B: PLE effective-param dense ~2.3B aktif/5.1B total;
                               VLM di-load text-only; system native (baru di Gemma 4))

Keduanya: Unsloth `FastVisionModel` (KEDUA model multimodal -> di-load lewat loader vision,
tapi LoRA hanya pada layer teks via finetune_vision_layers=False), di Colab (L4), dataset
Data/processed_final/. Fakta model diverifikasi dari notebook/doc resmi Unsloth (Qwen3.5-2B
& Gemma 4 E2B keduanya notebook "Vision"; Gemma 4 = dense+PLE, BUKAN MatFormer elastic).
Dibuat lewat satu builder agar kedua notebook benar-benar paralel (mudah dibandingkan
untuk skripsi). Jalankan: `python notebooks/build_finetune_notebooks.py`.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


_CID = [0]


def _nid():
    _CID[0] += 1
    return f"cell{_CID[0]:02d}"


def md(src):
    return {"cell_type": "markdown", "id": _nid(), "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": _nid(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


# --------------------------------------------------------------------------- #
# Cell-cell BERSAMA (string dengan placeholder __X__ untuk bagian model-spesifik)
# --------------------------------------------------------------------------- #
CELL_INSTALL = r"""
%%capture
# Cell install RESMI Unsloth dengan versi PINNED (menyelesaikan konflik
# torchaudio/_LazyModule/no-quant_state pada model VLM Qwen3.5 & Gemma 4).
# JANGAN ubah versi. Sumber: notebook resmi Unsloth Qwen3.5/Gemma 4 (Vision).
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!pip install torchcodec
import torch; torch._dynamo.config.recompile_limit = 64
"""

CELL_INSTALL_TIMM = r"""
%%capture
# Komponen vision/audio Qwen3.5 & Gemma 4 (di-import walau kita latih TEKS saja).
!pip install --no-deps --upgrade timm
"""

CELL_CONFIG = r"""
import os, glob, torch

# ============ Identitas model (SATU-SATUNYA bagian yang beda antar notebook) ==
MODEL_LABEL            = "__MODEL_LABEL__"
MODEL_ID               = "__MODEL_ID__"            # varian Unsloth (di-4bit-kan saat load)
MODEL_ID_FALLBACK      = "__MODEL_ID_FALLBACK__"   # cadangan
MERGE_SYSTEM_INTO_USER = __MERGE_SYSTEM__          # False utk KEDUA model (system native)
USE_ENABLE_THINKING    = __USE_THINKING__          # kirim enable_thinking=False (thinking OFF)
CHAT_TEMPLATE_NAME     = __CHAT_TEMPLATE__         # Gemma 4: "gemma-4"; Qwen: None (ChatML bawaan)
INSTRUCTION_PART       = "__INSTRUCTION_PART__"    # penanda turn user (train_on_responses_only)
RESPONSE_PART          = "__RESPONSE_PART__"       # penanda turn asisten (loss hanya di jawaban)
OUT_TRAIN              = "__OUT_TRAIN__"
OUT_MERGED             = "__OUT_MERGED__"
OUT_ADAPTER            = "__OUT_ADAPTER__"

# ============ Hyperparameter BERSAMA (justifikasi di sel markdown di atas) =====
MAX_SEQ_LENGTH = 1024          # menampung mayoritas jawaban (p95 token < 1024)
LOAD_IN_4BIT   = True
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
# LoRA dipasang via API model VLM (finetune_*_layers) + target_modules="all-linear".
# Daftar di bawah HANYA untuk breakdown grafik B.3 (mencocokkan nama modul nyata).
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

LEARNING_RATE = 1e-4           # 2e-4 terbukti terlalu agresif (val loss plateau cepat)
EPOCHS        = 5              # + EarlyStopping(patience=2) -> ruang konvergensi tanpa overfit
WARMUP_RATIO  = 0.10
WEIGHT_DECAY  = 0.01
BATCH_SIZE    = 4
GRAD_ACCUM    = 4              # effective batch = 16
SEED          = 42
torch.manual_seed(SEED)

# SMOKE TEST (hemat kuota): True = dry-run 20 step utk pastikan pipeline jalan end-to-end.
# Setelah sukses -> set False, RESTART runtime, jalankan ulang untuk training penuh.
SMOKE_TEST    = False
"""

CELL_DRIVE = r"""
# Mount Drive (Colab). Di luar Colab -> pakai root project lokal.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = "/content/drive/MyDrive/Fine-Tune SLM for Medical Chatbot"
except Exception:
    PROJECT_DIR = os.path.abspath("..")

DATA_DIR    = os.path.join(PROJECT_DIR, "Data", "processed_final")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs", "checkpoints", OUT_TRAIN)
MERGED_DIR  = os.path.join(PROJECT_DIR, "outputs", "merged", OUT_MERGED)
ADAPTER_DIR = os.path.join(PROJECT_DIR, "outputs", "checkpoints", OUT_ADAPTER)
for d in (OUTPUT_DIR, MERGED_DIR, ADAPTER_DIR):
    os.makedirs(d, exist_ok=True)

print("DATA_DIR :", DATA_DIR)
assert os.path.exists(os.path.join(DATA_DIR, "train_final.jsonl")), \
    "train_final.jsonl tak ada -> upload folder Data/processed_final ke Drive dulu."
"""

CELL_LOAD = r"""
# KEDUA model multimodal -> di-load lewat FastVisionModel (pola notebook resmi Unsloth),
# tapi nanti LoRA hanya pada layer TEKS (finetune_vision_layers=False). Untuk Gemma 4,
# objek kedua adalah processor (punya .apply_chat_template + .tokenizer); kita tetap
# menamainya `tokenizer` karena antarmuka apply_chat_template sama.
from unsloth import FastVisionModel

def _load(model_name):
    return FastVisionModel.from_pretrained(
        model_name                 = model_name,
        load_in_4bit               = LOAD_IN_4BIT,
        use_gradient_checkpointing = "unsloth",
        dtype                      = None,
    )

try:
    model, tokenizer = _load(MODEL_ID)
    print("Loaded:", MODEL_ID)
except Exception as e:
    print("Gagal load (", repr(e)[:140], ") -> fallback:", MODEL_ID_FALLBACK)
    model, tokenizer = _load(MODEL_ID_FALLBACK)
"""

CELL_CHATTEMPLATE = r"""
# Pasang chat template yang BENAR sebelum apa pun (Gemma 4 -> "gemma-4").
# Template ikut tersimpan di tokenizer hasil merge -> eval/deploy memakai format IDENTIK.
if CHAT_TEMPLATE_NAME:
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_NAME)
    print("Chat template dipasang:", CHAT_TEMPLATE_NAME)
else:
    print("Pakai chat template bawaan tokenizer (Qwen3.5 = ChatML).")
"""

CELL_TESTFIRST = r"""
# ===== TEST-FIRST (WAJIB sebelum PEFT) : pastikan model utuh, tanpa 'no quant_state' =====
FastVisionModel.for_inference(model)
_msgs = [{"role": "user", "content": "Halo, apa itu demam berdarah?"}]
_inp = tokenizer.apply_chat_template(_msgs, add_generation_prompt=True, tokenize=True,
                                     return_dict=True, return_tensors="pt").to(model.device)
_out = model.generate(**_inp, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(_out[0], skip_special_tokens=True))
# Output masuk akal + TANPA 'no quant_state'/'UNEXPECTED' -> lanjut ke PEFT.
FastVisionModel.for_training(model)
"""

CELL_PEFT = r"""
# LoRA IDENTIK untuk kedua model (validitas perbandingan, Bagian 9 note):
# vision OFF, language/attention/mlp ON, target_modules="all-linear".
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # TEXT-ONLY (matikan menara vision)
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    target_modules = "all-linear",
    random_state   = SEED,
    use_rslora     = False,
    loftq_config   = None,
)
model.print_trainable_parameters()   # CATAT angka NYATA ini untuk skripsi (BAB III)
"""

CELL_PROMPT = r"""
# ===== build_prompt() : SATU fungsi format, dipakai SAMA di train/eval/deploy =====
SYSTEM_DEFAULT = ("You are a helpful medical assistant. Answer patient questions with "
                  "accurate, empathetic responses based on established clinical knowledge. "
                  "Always recommend consulting a healthcare professional.")

def to_model_messages(messages):
    '''KEDUA model (Qwen3.5 & Gemma 4) mendukung role 'system' native -> MERGE_SYSTEM_INTO_USER
    = False, fungsi ini no-op. Dipertahankan untuk model lama tanpa turn system.'''
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
    kw = {"enable_thinking": False} if USE_ENABLE_THINKING else {}
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)
    except TypeError:   # template tak menerima enable_thinking
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

def build_prompt(question, system=SYSTEM_DEFAULT):
    '''Untuk INFERENSI/DEPLOY: prompt sampai giliran assistant (add_generation_prompt).'''
    return render_chat([{"role": "system", "content": system},
                        {"role": "user",   "content": question}],
                       add_generation_prompt=True)
"""

CELL_DATA = r"""
from datasets import load_dataset

def formatting_func(example):
    # render seluruh percakapan (system+user+assistant) jadi satu teks training
    return {"text": render_chat(example["messages"], add_generation_prompt=False)}

train_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "train_final.jsonl"), split="train")
val_ds   = load_dataset("json", data_files=os.path.join(DATA_DIR, "val_final.jsonl"),   split="train")
train_ds = train_ds.map(formatting_func, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(formatting_func,   remove_columns=val_ds.column_names)
print("train:", len(train_ds), "| val:", len(val_ds))
"""

CELL_VERIFY = r"""
# ===== Verifikasi format (WAJIB): token turn benar, system native ada, tanpa thinking =====
SPECIAL = "__SPECIAL__"          # Qwen: <|im_start|> | Gemma 4: <|turn>
print(train_ds[0]["text"][:1400])
print("=" * 80)
for i in range(3):
    t = train_ds[i]["text"]
    assert SPECIAL in t, f"Token turn {SPECIAL} tak ditemukan di sampel {i}!"
    assert INSTRUCTION_PART in t, f"Penanda user {INSTRUCTION_PART!r} tak ada di sampel {i}!"
    assert RESPONSE_PART in t,    f"Penanda asisten {RESPONSE_PART!r} tak ada di sampel {i}!"
    assert "<think>" not in t and "</think>" not in t, f"Ada blok thinking di sampel {i}!"
print(f"OK format terverifikasi (SPECIAL={SPECIAL}, response_part={RESPONSE_PART!r}).")
"""

CELL_TRAIN_CFG = r"""
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
try:
    from unsloth import is_bfloat16_supported
    BF16 = is_bfloat16_supported()
except Exception:
    BF16 = torch.cuda.is_bf16_supported()

cfg = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_ratio                = WARMUP_RATIO,
    num_train_epochs            = EPOCHS,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = "cosine",
    weight_decay                = WEIGHT_DECAY,
    optim                       = "adamw_8bit",
    fp16                        = not BF16,
    bf16                        = BF16,
    logging_steps               = 20,
    eval_strategy               = "steps",     # eval rapat -> kurva data-scaling B.2 mulus
    eval_steps                  = 50,          # PLACEHOLDER; di-override ~10x/epoch di bawah
    save_strategy               = "steps",     # match eval_strategy -> load_best_model_at_end OK
    save_steps                  = 50,          # = eval_steps (round-multiple utk best-model)
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    seed                        = SEED,
    data_seed                   = SEED,
    dataset_text_field          = "text",
    max_seq_length              = MAX_SEQ_LENGTH,
    packing                     = False,       # OFF: wajib agar train_on_responses_only akurat
    report_to                   = "none",
)

_kw = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=cfg,
           callbacks=[EarlyStoppingCallback(early_stopping_patience=2,
                                            early_stopping_threshold=0.001)])
try:                       # TRL baru: processing_class
    trainer = SFTTrainer(processing_class=tokenizer, **_kw)
except TypeError:          # TRL lama: tokenizer
    trainer = SFTTrainer(tokenizer=tokenizer, **_kw)

# train_on_responses_only: loss HANYA pada jawaban asisten (bukan pertanyaan).
# String penanda turn diverifikasi dari notebook/doc resmi Unsloth (lihat config).
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(trainer,
    instruction_part = INSTRUCTION_PART,   # Qwen: <|im_start|>user\n | Gemma 4: <|turn>user\n
    response_part    = RESPONSE_PART)      # Qwen: <|im_start|>assistant\n | Gemma 4: <|turn>model\n

# Cadence eval ~10x/epoch dari dataloader NYATA (packing OFF -> 1 sampel per baris).
import math
STEPS_PER_EPOCH = max(1, math.ceil(len(trainer.get_train_dataloader()) / GRAD_ACCUM))
EVAL_STEPS = max(10, STEPS_PER_EPOCH // 10)
trainer.args.eval_steps = EVAL_STEPS
trainer.args.save_steps = EVAL_STEPS          # tetap = eval_steps (best-model aman)
SAMPLES_PER_STEP = BATCH_SIZE * GRAD_ACCUM    # packing OFF -> tiap step = eff_batch sampel
print(f"steps/epoch~{STEPS_PER_EPOCH} -> eval & save tiap {EVAL_STEPS} step (~10x/epoch); "
      f"{SAMPLES_PER_STEP} sampel/step")

if SMOKE_TEST:                                  # dry-run: hentikan setelah 20 step
    trainer.args.max_steps = 20
    trainer.args.eval_steps = 10
    trainer.args.save_steps = 10
    print("SMOKE_TEST aktif -> max_steps=20 (validasi pipeline, BUKAN training penuh).")
"""

CELL_TRAIN_RUN = r"""
# Resume otomatis jika sesi Colab terputus (checkpoint ada di OUTPUT_DIR).
ckpts = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
resume = len(ckpts) > 0
print("Resume dari checkpoint:", resume, f"({len(ckpts)} ditemukan)")
trainer_stats = trainer.train(resume_from_checkpoint=resume or None)
"""

CELL_MERGE = r"""
# load_best_model_at_end=True -> model di memori = checkpoint val_loss TERENDAH.
print("Best checkpoint :", trainer.state.best_model_checkpoint)
print("Best eval_loss  :", trainer.state.best_metric)

# 1) simpan adapter LoRA (kecil, untuk arsip)
model.save_pretrained(ADAPTER_DIR); tokenizer.save_pretrained(ADAPTER_DIR)
# 2) merge ke 16-bit untuk evaluasi & export GGUF
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print("Adapter ->", ADAPTER_DIR)
print("Merged  ->", MERGED_DIR)
"""

CELL_INFER = r"""
# Uji inferensi cepat (sanity). Pakai build_prompt() yang SAMA dgn training.
FastVisionModel.for_inference(model)
q = "Apa penyebab umum demam pada anak dan kapan sebaiknya dibawa ke dokter?"
inputs = tokenizer(build_prompt(q), return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False, no_repeat_ngram_size=3)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
"""

CELL_PUSH = r"""
# (Opsional) push hasil merge ke HuggingFace Hub. Set HF_TOKEN di Colab Secrets.
from google.colab import userdata
HF_TOKEN = None
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    pass
if HF_TOKEN:
    repo = "USERNAME/" + OUT_MERGED          # ganti USERNAME
    model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
    print("Pushed ->", repo)
else:
    print("HF_TOKEN tak diset -> lewati push (opsional).")
"""


CELL_VIZ_SETUP = r"""
# ===== Visualisasi training (Bagian B note) — dari DATA/MODEL NYATA =====
import matplotlib.pyplot as plt
import numpy as np, os

FIGDIR = os.path.join(PROJECT_DIR, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)
FIG_PREFIX = OUT_MERGED            # prefix nama model -> PNG dua model tak tertukar
logs = trainer.state.log_history   # sumber semua grafik training

def _savefig(fig, name):
    p = os.path.join(FIGDIR, f"{FIG_PREFIX}_{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); print("  ->", p)
"""

CELL_VIZ_LOSS = r"""
# B.1 — Kurva train/val loss
tr = [(l["step"], l["loss"]) for l in logs if "loss" in l]
ev = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]
fig, ax = plt.subplots(figsize=(8, 4))
if tr: ax.plot(*zip(*tr), label="train loss", color="#378ADD")
if ev: ax.plot(*zip(*ev), "o-", label="val loss", color="#BA7517", markersize=3)
ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title(f"Kurva training — {MODEL_LABEL}")
ax.legend(); _savefig(fig, "fig_loss_curve"); plt.show()
if ev:
    best = min(ev, key=lambda x: x[1])
    print(f"[B.1] val_loss terendah {best[1]:.4f} @ step {best[0]}. "
          f"Train < val terus & keduanya turun -> belajar tanpa overfit berat.")
"""

CELL_VIZ_SCALING = r"""
# B.2 — Kurva DATA-SCALING (val_loss vs jumlah sampel dilihat) di EPOCH 1.
# Sumbu-X = SAMPEL training (packing OFF -> tiap step = eff_batch sampel utuh).
ev = [(l["step"] * SAMPLES_PER_STEP, l["eval_loss"]) for l in logs if "eval_loss" in l]
e1_limit = STEPS_PER_EPOCH * SAMPLES_PER_STEP
ev_e1 = [(s, v) for (s, v) in ev if s <= e1_limit]
fig, ax = plt.subplots(figsize=(8, 4))
if ev_e1:
    xs = [s / 1e3 for s, _ in ev_e1]; ys = [v for _, v in ev_e1]
    ax.plot(xs, ys, "o-", color="#185FA5")
ax.set_xlabel("Sampel training yang sudah dilihat (ribu) — epoch 1")
ax.set_ylabel("val loss"); ax.set_title(f"Kurva data-scaling (epoch 1) — {MODEL_LABEL}")
_savefig(fig, "fig_datascaling"); plt.show()
print(f"[B.2] {len(ev_e1)} titik eval di epoch 1.")
print("  Kurva mendatar menjelang akhir epoch 1 -> data tambahan diminishing returns.")
print("  Masih turun tajam -> data lebih banyak kemungkinan masih membantu.")
"""

CELL_VIZ_PARAMS = r"""
# B.3 — Trainable params (LoRA) + ceiling kapasitas, dari MODEL NYATA
from collections import defaultdict
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"[B.3] Trainable (LoRA): {trainable:,} ({trainable/total*100:.2f}% dari {total/1e9:.2f}B)")
for bits in (3.6, 2.0):    # Morris et al. 2025: ~3.6 bit/param (batas ATAS memorisasi)
    print(f"   @ {bits} bit/param -> {trainable*bits/8/1e6:.2f} MB (batas atas, bukan target)")

per_mod = defaultdict(int)
for n, p in model.named_parameters():
    if p.requires_grad:
        for key in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
            if key in n: per_mod[key] += p.numel()
if per_mod:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(per_mod.keys()), [v/1e6 for v in per_mod.values()], color="#534AB7")
    ax.set_xlabel("Trainable params (juta)")
    ax.set_title(f"Parameter trainable LoRA per modul — {MODEL_LABEL}")
    _savefig(fig, "fig_lora_params"); plt.show()
print("  Kapasitas adapter dibatasi rank LoRA (bukan jumlah data) -> tetap berapapun ukuran data.")
"""


def make_notebook(M):
    cells = [
        md(f"# Fine-Tuning: {M['label']}\n\n"
           "Bagian dari skripsi *Chatbot Medis Offline untuk Puskesmas* — perbandingan\n"
           "**Qwen3.5-2B (standard dense)** vs **Gemma 4 E2B (PLE effective-param dense)**\n"
           "pada trade-off akurasi vs efisiensi deployment. Full **Unsloth @ Google Colab (L4)**.\n\n"
           "**Urutan run skripsi:** preprocessing (lokal) → notebook ini (train) →\n"
           "`eval.py` (baseline + finetuned) → `export_gguf.py` → `benchmark_ondevice.py`.\n\n"
           "Dataset: `Data/processed_final/{train,val}_final.jsonl` (upload ke Drive)."),
        md("## 1. Install dependensi (versi PINNED resmi)\n\n"
           "⚠️ **Setelah kedua sel install selesai → `Runtime > Restart session`** sebelum lanjut "
           "(menghindari konflik versi torch/xformers pada model VLM)."),
        code(CELL_INSTALL),
        code(CELL_INSTALL_TIMM),
        md("> **RESTART SESSION DI SINI** (Runtime → Restart session), lalu lanjut dari sel berikut."),
        md("## 2. Konfigurasi\n\n"
           "**Justifikasi hyperparameter (BAB III):** `lr=1e-4` (2e-4 terlalu agresif), "
           "`5 epoch + EarlyStopping(patience=2)` (ruang konvergensi tanpa overfit), "
           "`r=16/alpha=32` (~1–2% trainable params, seimbang kapasitas vs efisiensi QLoRA), "
           "`effective batch=16`. Seed di-fix 42 (reproducibility)."),
        code(CELL_CONFIG),
        md("## 3. Mount Drive & path"),
        code(CELL_DRIVE),
        md(f"## 4. Load model + chat template + TEST-FIRST (Unsloth, 4-bit, text-only)\n\n"
           f"`FastVisionModel` memuat model **multimodal** {M['label']} ({M['note']}); kita latih "
           f"**teks saja** (vision dimatikan di sel LoRA). **TEST-FIRST wajib**: pastikan model utuh "
           f"(tanpa `no quant_state`) sebelum PEFT."),
        code(CELL_LOAD),
        code(CELL_CHATTEMPLATE),
        code(CELL_TESTFIRST),
        md("## 5. LoRA adapter (text-only, IDENTIK kedua model)\n\n"
           "`finetune_vision_layers=False` mematikan menara vision; `target_modules=\"all-linear\"` "
           "+ konfigurasi sama persis untuk kedua model → perbandingan valid (Bagian 9). "
           "**Catat `print_trainable_parameters()`** (angka nyata untuk BAB III)."),
        code(CELL_PEFT),
        md("## 6. `build_prompt()` — fungsi format BERSAMA\n\n"
           "Konsistensi prompt = *silent killer*: format chat HARUS identik di "
           "**training, evaluasi, dan deployment**. Semua tahap memanggil `render_chat()`/"
           "`build_prompt()` ini."),
        code(CELL_PROMPT),
        md("## 7. Dataset → teks (chat template)"),
        code(CELL_DATA),
        md("## 8. Verifikasi format (WAJIB)"),
        code(CELL_VERIFY),
        md("## 9. Training (SFTTrainer + train_on_responses_only + EarlyStopping + resume)\n\n"
           "`train_on_responses_only` → loss **hanya pada jawaban asisten** (string penanda turn "
           "diverifikasi dari notebook/doc resmi Unsloth). Ini mengharuskan `packing=False` "
           "(packing menggabung sampel → batas turn jadi kabur). **Smoke test dulu** (`SMOKE_TEST=True` "
           "di sel Konfigurasi) untuk validasi pipeline 20 step sebelum run penuh 5 epoch.\n\n"
           "> **Catatan loss model multimodal**: nilai loss absolut TIDAK sebanding antar arsitektur "
           "(Qwen vs Gemma 4). Sukses dinilai dari (a) tren loss menurun + (b) metrik downstream "
           "(`eval.py`), BUKAN loss absolut."),
        code(CELL_TRAIN_CFG),
        code(CELL_TRAIN_RUN),
        md("## 10. Best checkpoint & merge 16-bit\n\n"
           "Pastikan yang di-merge = checkpoint **val_loss terendah** (bukan epoch terakhir; "
           "dijamin oleh `load_best_model_at_end=True`)."),
        code(CELL_MERGE),
        md("## 11. Visualisasi training (grafik untuk skripsi)\n\n"
           "Semua grafik dari **data/model NYATA** run ini (bukan ilustrasi). PNG disimpan di "
           "`results/figures/<model>_*.png` (prefix per-model agar tak tertukar) + dijelaskan singkat."),
        code(CELL_VIZ_SETUP),
        md("### 11.1 Kurva train/val loss"),
        code(CELL_VIZ_LOSS),
        md("### 11.2 Kurva data-scaling (epoch 1)\n\n"
           "Sumbu-X = **jumlah sampel** yang sudah dilihat (`packing=False` → tiap step = "
           "eff_batch sampel utuh). Bukti empiris untuk justifikasi ukuran dataset."),
        code(CELL_VIZ_SCALING),
        md("### 11.3 Trainable params LoRA + ceiling kapasitas"),
        code(CELL_VIZ_PARAMS),
        md("## 12. (Opsional) Uji inferensi cepat"),
        code(CELL_INFER),
        md("## 13. (Opsional) Push ke HuggingFace Hub"),
        code(CELL_PUSH),
        md("---\n**Selesai.** Lanjut ke `eval.py` (evaluasi baseline + finetuned, multi-metrik, "
           "per-bahasa) lalu `export_gguf.py` untuk Q4_K_M."),
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
    # substitusi placeholder model-spesifik di seluruh source
    repl = {
        "__MODEL_LABEL__": M["label"], "__MODEL_ID__": M["model_id"],
        "__MODEL_ID_FALLBACK__": M["model_id_fallback"],
        "__MERGE_SYSTEM__": M["merge_system"], "__USE_THINKING__": M["use_thinking"],
        "__CHAT_TEMPLATE__": M["chat_template"], "__SPECIAL__": M["special"],
        "__INSTRUCTION_PART__": M["instruction_part"], "__RESPONSE_PART__": M["response_part"],
        "__OUT_TRAIN__": M["out_train"], "__OUT_MERGED__": M["out_merged"],
        "__OUT_ADAPTER__": M["out_adapter"],
    }
    for c in cells:
        new = []
        for line in c["source"]:
            for k, v in repl.items():
                line = line.replace(k, v)
            new.append(line)
        c["source"] = new
    out = NB_DIR / M["fname"]
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


# Catatan: PEFT call kini SERAGAM (CELL_PEFT, FastVisionModel.get_peft_model untuk
# kedua model) -> tak ada lagi blok QWEN_PEFT/GEMMA_PEFT yang berbeda.
# Fakta model diverifikasi dari notebook resmi Unsloth (Qwen3.5-2B & Gemma 4 E2B = Vision nb;
# turn-token: Qwen ChatML <|im_start|>, Gemma 4 <|turn>; Gemma 4 = dense+PLE, system native).
MODELS = [
    dict(label="Qwen3.5-2B (standard dense, vision OFF, thinking OFF)",
         fname="finetune_qwen35_2b.ipynb",
         model_id="unsloth/Qwen3.5-2B",
         model_id_fallback="unsloth/Qwen3.5-2B-Instruct",
         merge_system="False", use_thinking="True",
         chat_template="None", special="<|im_start|>",
         instruction_part=r"<|im_start|>user\n", response_part=r"<|im_start|>assistant\n",
         out_train="qwen35-2b-train", out_merged="qwen35-2b-medical",
         out_adapter="qwen35-2b-adapter",
         note="dense ~2B; di-load via FastVisionModel, vision OFF; ChatML; thinking OFF"),
    dict(label="Gemma 4 E2B (PLE effective-param dense, text-only)",
         fname="finetune_gemma4_e2b.ipynb",
         model_id="unsloth/gemma-4-E2B-it",
         model_id_fallback="unsloth/gemma-4-E2B",
         merge_system="False", use_thinking="True",
         chat_template='"gemma-4"', special="<|turn>",
         instruction_part=r"<|turn>user\n", response_part=r"<|turn>model\n",
         out_train="gemma4-e2b-train", out_merged="gemma4-e2b-medical",
         out_adapter="gemma4-e2b-adapter",
         note="dense + Per-Layer Embeddings (~2.3B aktif / 5.1B total); system native; thinking OFF"),
]

if __name__ == "__main__":
    for M in MODELS:
        make_notebook(M)
    print("done.")
