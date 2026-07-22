"""
build_train_pivot6.py — generator notebook training untuk STUDI PIVOT 6 (jurnal).

UPDATE 2026-07-23 (keputusan user): roster diganti dari lintas-arsitektur (Gemma-3-4B /
Llama-3.2-3B — DIBATALKAN) menjadi SCALING PARAMETER SATU KELUARGA Qwen3.5 — arsitektur &
tokenizer IDENTIK, sehingga sumbu parameter TERKONTROL (bukan tren observasional):

  - train_qwen35_0.8b_p6.ipynb  (unsloth/Qwen3.5-0.8B — TEKS -> FastLanguageModel)    [lantai range]
  - train_qwen35_2b_p6.ipynb    (unsloth/Qwen3.5-2B   — VLM  -> FastModel, text-only) [titik tengah]
  - train_qwen35_4b_p6.ipynb    (unsloth/Qwen3.5-4B   — VLM  -> FastModel, text-only) [plafon range;
                                 TUNDUK GATE RAM on-device — bagian 7 rencana]

Checkpoint 2B/4B DIVERIFIKASI dari HF unsloth (2026-07-23): keduanya MULTIMODAL (vision
encoder) -> `FastModel` + `finetune_vision_layers=False`; template ChatML + scaffold <think>
IDENTIK dgn 0.8B (enable_thinking=False tetap sisip <think></think> kosong; sel SELF-CHECK
membuktikannya saat run). File lama train_{gemma3_4b,llama32_3b}_p6.ipynb = artefak roster
batal (boleh dihapus bila yakin).

Harness IDENTIK antar model (basis notebook Qwen0.8B yang sudah terbukti). Catatan teknis:
  1. Dataset -> `Data/processed_id_final/` (20000/2947/2952, terbersih) — BUKAN processed_id (30k).
  2. Encode/decode dibuat aman terhadap PROCESSOR (FastModel VLM bisa mengembalikan processor,
     bukan tokenizer murni) via `TOK = getattr(tokenizer,"tokenizer",tokenizer)` + `PAD_ID`.

JANGAN edit .ipynb manual — edit generator ini lalu jalankan
`python notebooks/build_train_pivot6.py`.

UPDATE 2026-07-23 (H100): tambah saklar presisi `LOAD_IN_4BIT` di sel Konfigurasi.
  - False (DEFAULT) = LoRA bf16 murni — utk H100/A100 (VRAM >=40GB). Base 16-bit, tanpa bnb.
  - True            = QLoRA 4-bit    — utk GPU 24GB (RTX4090/L4/A5000), perilaku lama.
  MODEL_ID kini menunjuk checkpoint PLAIN (bukan -unsloth-bnb-4bit); kuantisasi dikontrol
  flag `load_in_4bit` saat load. Fallback = checkpoint pre-quantized bnb (dipaksa 4-bit +
  WARNING bila mode bf16 — darurat saja). Effective batch DIKUNCI 16 di kedua mode
  (komparabilitas hyperparameter): bf16 H100 -> 8x2, QLoRA 24GB -> 2x8.

CATATAN: leg 2B & 4B belum diuji lokal (tak ada GPU) — perhatikan output sel SELF-CHECK
& Load saat run pertama di RunPod/H100 (processor vs tokenizer, pad_token, scaffold <think>).
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
# Install Unsloth + TRL + PEFT + bnb. TANPA %%capture supaya error/konflik install terlihat.
# RunPod/Colab sudah punya torch CUDA -> --no-deps utk paket inti agar torch tak ke-replace.
# Map xformers per torch. Dep baru Unsloth (tyro/msgspec/cut_cross_entropy/torchao) dipasang eksplisit.
import os, re, torch
v = re.match(r'\d+\.\d+', str(torch.__version__)).group(0)
xf_map = {'2.11':'0.0.35','2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2','2.6':'0.0.29.post3','2.5':'0.0.29.post3','2.4':'0.0.27.post2'}
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
from unsloth import __LOADER__
for m in (unsloth, transformers, trl, datasets, xformers):
    print(m.__name__, m.__version__)
print("OK semua import sukses. Loader model ini:", "__LOADER__")
"""

CELL_ENV = r"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # single GPU — hindari multi-GPU surprise
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

# ====================== HYPERPARAMETER ======================
MAX_SEQ_LENGTH = 1024            # p99 antar-model ~931 -> aman

# ---- PRESISI (saklar utama H100 vs GPU kecil) ----
# False (DEFAULT) = LoRA bf16 murni  -> H100/A100, VRAM >=40GB (base 16-bit, tanpa artefak kuantisasi)
# True            = QLoRA 4-bit bnb  -> GPU 24GB (RTX4090/L4/A5000)
LOAD_IN_4BIT   = False

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

LEARNING_RATE = 1e-4             # dikunci (2e-4 terbukti terlalu agresif di studi ini)
EPOCHS        = 3                # full-mode; + EarlyStopping(patience=3)
WARMUP_RATIO  = 0.10
WEIGHT_DECAY  = 0.01
# Effective batch DIKUNCI = 16 di kedua mode (komparabilitas hyperparameter antar-run/antar-model).
if LOAD_IN_4BIT:                 # GPU 24GB: batch kecil + akumulasi. Model 4B: 1x16 bila OOM.
    PER_DEVICE_BATCH, GRAD_ACCUM = 2, 8
else:                            # H100 80GB bf16: batch besar, akumulasi kecil (16x1 juga boleh).
    PER_DEVICE_BATCH, GRAD_ACCUM = 8, 2
MAX_GRAD_NORM    = 1.0           # grad clipping (anti loss meledak)

# ====================== FORMAT TURN (__LABEL_SHORT__) ======================
# __TURN_COMMENT__
INSTRUCTION_PART       = "__INSTRUCTION_PART__"   # penanda giliran user
RESPONSE_PART          = "__RESPONSE_PART__"      # penanda giliran asisten (loss mulai di sini)
EXPECTED_PROMPT_TAIL   = "__EXPECTED_TAIL__"      # ekor prompt inferensi (utk SELF-CHECK decode)
MERGE_SYSTEM_INTO_USER = __MERGE_SYSTEM__         # True hanya utk model tanpa role system (Qwen: False)

# ====================== PILOT SUBSET ======================
PILOT_TRAIN_N, PILOT_VAL_N, PILOT_MAX_STEPS = 1500, 200, 250

# ====================== OUTPUT ======================
ADAPTER_DIR = "__ADAPTER_DIR__"
print(f"RUN_MODE={RUN_MODE} | model={MODEL_ID} | eff_batch={PER_DEVICE_BATCH*GRAD_ACCUM} | seq={MAX_SEQ_LENGTH}")
"""

CELL_PATHS = r"""
# Cari dataset beku (RunPod volume / Kaggle input / Colab Drive / lokal). Lihat README_TRAIN.md.
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(os.path.join(p, "train.jsonl")):
            return p
    return None

CANDIDATES = [
    os.environ.get("DATA_DIR", ""),
    "/workspace/processed_id_final",               # RunPod: unggah ke volume /workspace
    "/workspace/Data/processed_id_final",
    "/kaggle/input/processed-id-final",            # Kaggle dataset (rename sesuai uploadmu)
    "/content/drive/MyDrive/Fine-Tune SLM for Medical Chatbot/Data/processed_id_final",
    "../Data/processed_id_final", "Data/processed_id_final",
]
# Colab: mount Drive bila perlu
if any("drive/MyDrive" in c for c in CANDIDATES) and not os.path.exists("/content/drive"):
    try:
        from google.colab import drive; drive.mount("/content/drive")
    except Exception:
        pass
DATA_DIR = _first_existing(CANDIDATES)
assert DATA_DIR, ("Dataset tak ketemu. Set env DATA_DIR atau upload "
                  "train/val/test.jsonl (processed_id_final). Lihat README_TRAIN.md.")
os.makedirs(ADAPTER_DIR, exist_ok=True)
print("DATA_DIR =", DATA_DIR)
"""

CELL_LOAD = r"""
from unsloth import __LOADER__

def _load(mid):
    # Checkpoint "-bnb-4bit" adalah pre-quantized -> WAJIB load_in_4bit=True walau mode bf16.
    four = LOAD_IN_4BIT or ("bnb-4bit" in mid)
    if four and not LOAD_IN_4BIT:
        print(f"[WARN] {mid} = checkpoint pre-quantized -> dipaksa QLoRA 4-bit (bukan bf16 murni). "
              "Hasil run ini JANGAN dicampur dgn run bf16 di tabel jurnal.")
    return __LOADER__.from_pretrained(
        model_name     = mid,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,          # auto: bf16 di Ampere/Hopper (H100), fp16 di GPU lama
        load_in_4bit   = four,
    )

try:
    model, tokenizer = _load(MODEL_ID); print("Loaded:", MODEL_ID)
except Exception as e:
    print("Gagal:", repr(e)[:160], "-> fallback", MODEL_ID_FALLBACK)
    model, tokenizer = _load(MODEL_ID_FALLBACK)

_q = bool(getattr(model, "is_loaded_in_4bit", False))
print("precision:", "QLoRA 4-bit (bnb)" if _q else f"LoRA {next(model.parameters()).dtype}",
      "| target mode:", "4-bit" if LOAD_IN_4BIT else "bf16")
assert _q == LOAD_IN_4BIT or not LOAD_IN_4BIT, "Mode 4-bit diminta tapi model tidak ter-quantize?"

# VLM (Qwen3.5-2B/4B via FastModel) bisa mengembalikan PROCESSOR, bukan tokenizer murni.
# TOK = tokenizer teks di baliknya utk encode/decode/pad; templating tetap via `tokenizer`
# (processor mendelegasikan apply_chat_template). Utk Qwen0.8B (teks), TOK identik dgn tokenizer.
TOK = getattr(tokenizer, "tokenizer", tokenizer)
if getattr(TOK, "pad_token", None) is None:
    TOK.pad_token = TOK.eos_token
TOK.padding_side = "right"
try:
    tokenizer.padding_side = "right"
except Exception:
    pass
PAD_ID = TOK.pad_token_id
print("loader: __LOADER__ | vocab:", len(TOK), "| pad:", TOK.pad_token,
      "| processor?", TOK is not tokenizer)
"""

CELL_PEFT = r"""
__PEFT_BODY__
model.print_trainable_parameters()   # CATAT angka ini utk tabel jurnal (RQ parameter)
"""

CELL_DATA = r"""
from datasets import load_dataset

def to_model_messages(messages):
    # Utk model tanpa role 'system' -> merge ke giliran user pertama.
    # Qwen3.5 (semua ukuran): system native -> no-op (MERGE_SYSTEM_INTO_USER=False).
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
    # Bila template tak terima kwarg ini -> TypeError -> fallback tanpa kwarg.
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
# format turn konsisten train==inferensi. Encode/decode via TOK (aman utk processor VLM).
def _find_sub(seq, sub):
    for i in range(len(seq)-len(sub)+1):
        if seq[i:i+len(sub)] == sub: return i
    return -1

_rec   = json.loads(open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8").readline())
_msgs  = _rec["messages"]
_full  = render_chat(_msgs, add_generation_prompt=False)
_prompt= render_chat([m for m in _msgs if m["role"]!="assistant"], add_generation_prompt=True)
_ids   = TOK(_full, add_special_tokens=False)["input_ids"]
_marker= TOK(RESPONSE_PART, add_special_tokens=False)["input_ids"]
_pos   = _find_sub(_ids, _marker)
assert _pos >= 0, "RESPONSE_PART tak ditemukan di teks ter-render! Cek template/turn-token model."
_start = _pos + len(_marker)
_ans   = next(m["content"] for m in _msgs if m["role"]=="assistant")
_w1    = _ans.split()[0] if _ans.split() else ""
_reco  = TOK.decode(_ids[_start:_start+12]).lstrip()
print("first ACTIVE tokens :", [TOK.decode([t]) for t in _ids[_start:_start+8]])
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

USE_BF16 = torch.cuda.is_bf16_supported()   # RunPod Ampere+ -> True

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
    fp16                        = not USE_BF16,
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
print("trainer siap. RUN_MODE =", RUN_MODE, "| steps_kw =", steps_kw)
"""

CELL_TRAIN_RUN = r"""
# Resume bila checkpoint ada (tahan disconnect RunPod/Colab).
_ckpts = glob.glob(os.path.join(ADAPTER_DIR, "checkpoint-*"))
_resume = len(_ckpts) > 0
print("resume:", _resume, f"({len(_ckpts)} checkpoint)")
trainer_stats = trainer.train(resume_from_checkpoint=_resume or None)
print("training selesai.")
"""

CELL_GATE = r"""
# ===== GATE (jalankan setelah PILOT). Cetak PASS / STOP. =====
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
__LOADER__.for_inference(model)
test_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "test.jsonl"), split="train")
gens = []
id_ok = degen = 0
for i in range(10):
    msgs = [m for m in test_ds[i]["messages"] if m["role"] != "assistant"]
    p = render_chat(msgs, add_generation_prompt=True)
    inp = TOK(p, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=200, do_sample=False, no_repeat_ngram_size=3,
                         repetition_penalty=1.1, pad_token_id=PAD_ID)
    txt = TOK.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    gens.append((msgs[-1]["content"], txt))
    try:
        if detect(txt[:500]) == "id": id_ok += 1
    except Exception:
        pass
    toks = txt.split()
    grams = [tuple(toks[j:j+4]) for j in range(len(toks)-3)]
    rep = (1 - len(set(grams))/len(grams)) if grams else 0
    if rep > 0.30: degen += 1
B3a_lang = id_ok >= 9
B3b_degen = degen == 0
print(f"[B3a] Indonesia {id_ok}/10 (anti language-mixing: {B3a_lang})")
print(f"[B3b] non-degeneratif {10-degen}/10 (no >30% 4-gram berulang: {B3b_degen})")

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
__LOADER__.for_training(model)
"""

CELL_SAVE = r"""
# Simpan adapter LoRA (kecil) utk inspeksi/arsip. (Merge 16-bit dilakukan saat eval.)
print("best checkpoint:", trainer.state.best_model_checkpoint)
print("best eval_loss :", trainer.state.best_metric)
model.save_pretrained(ADAPTER_DIR)
try:
    tokenizer.save_pretrained(ADAPTER_DIR)
except Exception:
    TOK.save_pretrained(ADAPTER_DIR)
print("adapter ->", ADAPTER_DIR, "| 10 generasi -> pilot_generations.txt")
"""

CELL_EVAL = r"""
# ===== (PLACEHOLDER) eval token-F1 + ROUGE-L pada val — siap pakai setelah training =====
# Metrik utama studi = token-F1 + ROUGE-L (+ BERTScore di eval.py). EM/MCQA di-DROP (native-only).
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

__LOADER__.for_inference(model)
val_eval = load_dataset("json", data_files=os.path.join(DATA_DIR, "val.jsonl"), split="train")
N = min(100, len(val_eval))
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
f1s, rls = [], []
for i in range(N):
    msgs = val_eval[i]["messages"]
    ref = next(m["content"] for m in msgs if m["role"]=="assistant")
    p = render_chat([m for m in msgs if m["role"]!="assistant"], add_generation_prompt=True)
    inp = TOK(p, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=256, do_sample=False, no_repeat_ngram_size=3,
                         repetition_penalty=1.1, pad_token_id=PAD_ID)
    pred = TOK.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    f1s.append(_f1(pred, ref)); rls.append(scorer.score(ref, pred)["rougeL"].fmeasure)
print(f"[EVAL n={N}] token-F1={sum(f1s)/N:.4f} | ROUGE-L={sum(rls)/N:.4f}")
__LOADER__.for_training(model)
"""


def make_notebook(M):
    cells = [
        md(f"# Fine-Tuning {M['label']} (LoRA bf16 / QLoRA) — Chatbot Medis ID (Pivot 6, jurnal)\n\n"
           "Studi **scaling parameter satu keluarga** — Qwen3.5 **0.8B / 2B / 4B** "
           "(arsitektur & tokenizer **identik** → sumbu parameter terkontrol) "
           "untuk QA medis Bahasa Indonesia.\n\n"
           "Dataset **ID native-only terbersih** `Data/processed_id_final/` "
           "(20000/2947/2952, 0 noise/dup/leak). Toolchain **Unsloth + TRL + PEFT**.\n\n"
           "**Presisi via `LOAD_IN_4BIT` (sel Konfigurasi):** `False` (default) = **LoRA bf16** "
           "utk **H100/A100 ≥40GB**; `True` = QLoRA 4-bit utk GPU 24GB. Effective batch dikunci 16 "
           "di kedua mode.\n\n"
           f"> Loader model ini: **{M['loader']}** — {M['load_md']}\n\n"
           f"> Harness IDENTIK antar model; beda hanya identitas + loader + turn-token "
           f"({M['turn_note']}).\n\n"
           "**Cara pakai:** jalankan sel berurutan. Mulai `RUN_MODE=\"pilot\"` → baca **GATE** "
           "→ kalau PASS set `RUN_MODE=\"full\"`, Restart & run. Detail di `README_TRAIN.md`."),
        md("## 1. Install (Unsloth/TRL/PEFT/bnb) — lalu **Runtime > Restart**\n\n"
           "Tanpa `%%capture` agar konflik versi terlihat."),
        code(CELL_INSTALL),
        md("## 1b. Verifikasi import (setelah Restart, sebelum Konfigurasi)"),
        code(CELL_VERIFY_IMPORTS),
        md("## 2. Environment (single GPU, seed)"),
        code(CELL_ENV),
        md("## 3. Konfigurasi — set `RUN_MODE` di sini\n\n"
           "`pilot` = subset 1500/200, ~250 step. `full` = data penuh + EarlyStopping. "
           f"Turn-token model ini: {M['turn_note']}."),
        code(CELL_CONFIG),
        md("## 4. Lokasi dataset beku (RunPod/Kaggle/Colab/lokal)\n\n"
           "Upload `train/val/test.jsonl` (**processed_id_final**) ke volume RunPod `/workspace` "
           "atau Drive — lihat `README_TRAIN.md`. Sel ini mencari otomatis (atau set env `DATA_DIR`)."),
        code(CELL_PATHS),
        md(f"## 5. Load model ({M['loader']}) — presisi mengikuti `LOAD_IN_4BIT`\n\n{M['load_md']}\n\n"
           "> `TOK`/`PAD_ID` disiapkan agar encode/decode aman baik untuk tokenizer murni "
           "(Qwen3.5-0.8B) maupun processor VLM (Qwen3.5-2B/4B)."),
        code(CELL_LOAD),
        md("## 6. LoRA adapter\n\n**Catat `print_trainable_parameters()`** "
           "(dipakai di tabel jurnal — jumlah parameter aktif per model)."),
        code(CELL_PEFT),
        md(f"## 7. Dataset → teks (chat template {M['short']})\n\n{M['system_note']}"),
        code(CELL_DATA),
        md("## 8. SELF-CHECK decode (WAJIB dilihat sebelum train)\n\n"
           "Membuktikan token **konten pertama jawaban** ikut dipelajari (bukan ke-mask/off-by-one) "
           "dan format turn membuat **train == inferensi**."),
        code(CELL_DECODECHECK),
        md("## 9. Konfigurasi training (SFT + train_on_responses_only)\n\n"
           "`train_on_responses_only` → loss **hanya pada jawaban**. `packing=False` (wajib). "
           "RunPod Ampere+ → `bf16=True`. Unsloth otomatis patch cross-entropy."),
        code(CELL_TRAIN_CFG),
        md("## 10. Train (resume-aware)"),
        code(CELL_TRAIN_RUN),
        md("## 11. GATE (setelah PILOT) — PASS/STOP\n\n"
           "Cek: loss turun & tak meledak/NaN; 10 generasi ≥9/10 Indonesia (anti "
           "language-mixing), tidak degeneratif. STOP → `README_TRAIN.md` › TROUBLESHOOTING."),
        code(CELL_GATE),
        md("## 12. Simpan adapter + generasi"),
        code(CELL_SAVE),
        md("## 13. (Placeholder) Eval token-F1 + ROUGE-L (val)\n\n"
           "Eval final lintas-model via `eval.py` (token-F1 + ROUGE-L + BERTScore, per-bahasa). "
           "Sel ini hanya cek cepat pada val setelah training."),
        code(CELL_EVAL),
        md(f"---\n**Selesai.** {M['label']} — bagian studi scaling parameter Qwen3.5 "
           "(0.8B / 2B / 4B, arsitektur & tokenizer identik). Semua dilatih di "
           "`processed_id_final`, dievaluasi dengan protokol identik."),
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
        "__LOADER__": M["loader"], "__PEFT_BODY__": M["peft_body"],
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
    raw = out.read_text(encoding="utf-8")
    leftover = [k for k in repl if k in raw]
    assert not leftover, f"Placeholder belum tersubstitusi di {M['fname']}: {leftover}"
    json.loads(raw)
    print("wrote", out)


def _sub(line, repl):
    # ganti placeholder terpanjang dulu agar __LOADER__ tak menabrak __LOADER_NOTE__
    for k in sorted(repl, key=len, reverse=True):
        line = line.replace(k, repl[k])
    return line


_PEFT_FLM = (
    "model = FastLanguageModel.get_peft_model(\n"
    "    model,\n"
    "    r                          = LORA_R,\n"
    "    target_modules             = LORA_TARGET,\n"
    "    lora_alpha                 = LORA_ALPHA,\n"
    "    lora_dropout               = LORA_DROPOUT,\n"
    "    bias                       = \"none\",\n"
    "    use_gradient_checkpointing = \"unsloth\",\n"
    "    random_state               = SEED,\n"
    ")"
)
# VLM (Qwen3.5-2B/4B): latih TEXT-ONLY (bekukan menara vision). target_modules auto (jangan hardcode).
_PEFT_FASTMODEL = (
    "model = FastModel.get_peft_model(\n"
    "    model,\n"
    "    finetune_vision_layers     = False,   # TEXT-ONLY: bekukan menara vision (model multimodal)\n"
    "    finetune_language_layers   = True,\n"
    "    finetune_attention_modules = True,\n"
    "    finetune_mlp_modules       = True,\n"
    "    r                          = LORA_R,\n"
    "    lora_alpha                 = LORA_ALPHA,\n"
    "    lora_dropout               = LORA_DROPOUT,\n"
    "    bias                       = \"none\",\n"
    "    use_gradient_checkpointing = \"unsloth\",\n"
    "    random_state               = SEED,\n"
    ")"
)


# Turn-token & flag per model. Diverifikasi dari template chat resmi + docs Unsloth (2026-07-21).
MODELS = [
    dict(short="Qwen3.5", label="Qwen3.5-0.8B", fname="train_qwen35_0.8b_p6.ipynb",
         model_id="unsloth/Qwen3.5-0.8B",
         model_id_fallback="unsloth/Qwen3.5-0.8B-Instruct",
         loader="FastLanguageModel", peft_body=_PEFT_FLM,
         loader_note="model TEKS ~0.8B -> FastLanguageModel",
         load_md="Qwen3.5-0.8B = model **teks** → `FastLanguageModel`. `dtype=None` auto-precision.",
         turn_note="ChatML `<|im_start|>` + scaffold `<think>`",
         turn_comment=("Template Qwen sisip <think>\\n\\n</think> kosong walau enable_thinking=False"
                       " -> RESPONSE_PART memuat scaffold itu agar loss hanya di jawaban (train==infer)."),
         system_note="Qwen mendukung role `system` native (tidak di-merge).",
         instruction_part=r"<|im_start|>user\n",
         response_part=r"<|im_start|>assistant\n<think>\n\n</think>\n\n",
         expected_tail=r"</think>", merge_system="False",
         adapter_dir="checkpoints/qwen35_0.8b_p6"),
    dict(short="Qwen3.5-2B", label="Qwen3.5-2B", fname="train_qwen35_2b_p6.ipynb",
         model_id="unsloth/Qwen3.5-2B",
         model_id_fallback="Qwen/Qwen3.5-2B",   # fallback resmi Qwen (ungated)
         loader="FastModel", peft_body=_PEFT_FASTMODEL,
         loader_note="Qwen3.5-2B MULTIMODAL (vision encoder) -> FastModel, text-only (vision beku)",
         load_md=("Qwen3.5-2B adalah **multimodal (text+image)** → dimuat via `FastModel` dan "
                  "dilatih **TEXT-ONLY** (`finetune_vision_layers=False` di sel LoRA). "
                  "Template ChatML + scaffold `<think>` **identik dgn 0.8B**."),
         turn_note="ChatML `<|im_start|>` + scaffold `<think>` (identik 0.8B)",
         turn_comment=("Template Qwen sisip <think>\\n\\n</think> kosong walau enable_thinking=False"
                       " -> RESPONSE_PART memuat scaffold itu agar loss hanya di jawaban (train==infer)."),
         system_note="Qwen mendukung role `system` native (tidak di-merge).",
         instruction_part=r"<|im_start|>user\n",
         response_part=r"<|im_start|>assistant\n<think>\n\n</think>\n\n",
         expected_tail=r"</think>", merge_system="False",
         adapter_dir="checkpoints/qwen35_2b_p6"),
    dict(short="Qwen3.5-4B", label="Qwen3.5-4B", fname="train_qwen35_4b_p6.ipynb",
         model_id="unsloth/Qwen3.5-4B",
         model_id_fallback="Qwen/Qwen3.5-4B",   # fallback resmi Qwen (ungated)
         loader="FastModel", peft_body=_PEFT_FASTMODEL,
         loader_note="Qwen3.5-4B MULTIMODAL (vision encoder) -> FastModel, text-only (vision beku)",
         load_md=("Qwen3.5-4B adalah **multimodal (text+image)** → dimuat via `FastModel` dan "
                  "dilatih **TEXT-ONLY** (`finetune_vision_layers=False` di sel LoRA). "
                  "Template ChatML + scaffold `<think>` **identik dgn 0.8B**.\n\n"
                  "> ⚠️ **4B = plafon range, TUNDUK GATE RAM on-device (bagian 7 rencana):** "
                  "training & eval tetap jalan, tapi kelolosan ke tahap deploy Puskesmas "
                  "ditentukan benchmark RAM Q4_K_M di perangkat uji."),
         turn_note="ChatML `<|im_start|>` + scaffold `<think>` (identik 0.8B)",
         turn_comment=("Template Qwen sisip <think>\\n\\n</think> kosong walau enable_thinking=False"
                       " -> RESPONSE_PART memuat scaffold itu agar loss hanya di jawaban (train==infer)."),
         system_note="Qwen mendukung role `system` native (tidak di-merge).",
         instruction_part=r"<|im_start|>user\n",
         response_part=r"<|im_start|>assistant\n<think>\n\n</think>\n\n",
         expected_tail=r"</think>", merge_system="False",
         adapter_dir="checkpoints/qwen35_4b_p6"),
]


def build():
    for M in MODELS:
        make_notebook(M)
    print("done.")


if __name__ == "__main__":
    build()
