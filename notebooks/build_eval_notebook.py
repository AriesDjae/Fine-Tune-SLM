"""
build_eval_notebook.py — generator notebook Colab SELF-CONTAINED untuk Fase 1 EVAL P6
(baseline vs fine-tuned, 3 model 0.8B/2B/4B). TIDAK memanggil eval.py — semua logika
inline di sel (user minta "di notebook saja").

Metrik: token-F1 + ROUGE-L (+ BERTScore opsional). Protokol fair greedy (do_sample=False).
Loader: 0.8B=FastLanguageModel (teks), 2B/4B=FastModel (VLM text-only). Fix generate:
UNSLOTH_COMPILE_DISABLE=1. Finetuned di-load dari merged HF `AriesDjaenuri/qwen35-<size>-medical-id`.

Jangan edit .ipynb manual — edit file ini lalu:
    python notebooks/build_eval_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "run_eval_p6.ipynb"
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})


md("""# Fase 1 — Eval P6: baseline vs fine-tuned (Colab, self-contained)

Meng-eval **Qwen3.5 0.8B / 2B / 4B** (baseline pre-trained vs fine-tuned) dengan metrik
**token-F1 + ROUGE-L** (+ BERTScore opsional). Semua logika **inline di notebook** — tak
memanggil `eval.py`.

**Cara pakai:** Runtime → **GPU** (T4/L4; 4B lebih aman L4/A100).
1. **Smoke dulu:** sel Konfigurasi `SMOKE_TEST=True`, `MODEL_KEY="0.8b"` → jalankan sel 1-9
   (n=20, ~2-3 mnt). Periksa contoh pred/ref + ETA yang dicetak.
2. **Run asli:** `SMOKE_TEST=False` → ulang sel 3,6,7,8 untuk `0.8b`, lalu `2b`, lalu `4b`.
3. Sel Ringkasan → tabel + delta (RQ1/RQ2).

**Anti-putus:** tiap `CKPT_EVERY` batch, prediksi disimpan ke `CKPT_DIR` **di Drive**
(`MyDrive/p6_eval_ckpt/`). Kalau sesi Colab mati di tengah run 2998, cukup jalankan ulang
sel yang sama — otomatis **RESUME** dari sampel terakhir, tidak mengulang dari nol.
Checkpoint dihapus sendiri begitu run selesai.

> Protokol fair greedy **identik `eval.py`** (`do_sample=False, num_beams=1,
> no_repeat_ngram_size=3, max_new_tokens=256` — TANPA repetition_penalty, agar angka
> langsung sebanding dgn eval Pivot 5), n asli = 2998 (full test set, konsisten dgn eval
> Pivot 5). Fine-tuned di-load dari **merged HF** (hasil Fase 0). Output → `results/<label>.json`.""")

md("## 1. Install (Unsloth pinned + metrik) — lalu **Restart jika diminta**")

code("""%%capture
import os, re
# WAJIB sebelum import unsloth (cegah NameError('args') saat generate)
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install --upgrade transformers
!pip install -q rouge-score bert-score langdetect
import torch; torch._dynamo.config.recompile_limit = 64""")

md("## 2. Login HuggingFace (baca merged model finetuned)")

code('''import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login; login(token=HF_TOKEN)
    print("HF login OK.")
else:
    print("HF_TOKEN tak diset -> cukup jika repo merged publik.")''')

md("""## 3. Konfigurasi — pilih model & parameter eval

**Urutan yang disarankan:** `SMOKE_TEST=True` + `MODEL_KEY="0.8b"` dulu (n=20, ~2-3 menit)
untuk memastikan loader/template/metrik jalan → baru `SMOKE_TEST=False` untuk run asli
(0.8b → 2b → 4b). Hasil smoke diberi label ber-suffix `_smoke` supaya **tidak mencemari**
tabel Ringkasan.""")

code('''# ====================== PILIH MODEL ======================
MODEL_KEY = "0.8b"          # "0.8b" | "2b" | "4b"

# ====================== SMOKE TEST ======================
# True  = validasi pipeline cepat (n=20, label ber-suffix _smoke, tak masuk tabel Ringkasan)
# False = run asli utk naskah (n = N_EVAL_FULL)
SMOKE_TEST  = True
N_EVAL_FULL = 2998          # run asli: 2998 (full test set) — konsisten dgn eval Pivot 5

# key -> (base pre-trained, finetuned merged HF, loader)
REGISTRY = {
    "0.8b": dict(base="unsloth/Qwen3.5-0.8B", ft="AriesDjaenuri/qwen35-0.8b-medical-id", loader="language"),
    "2b":   dict(base="unsloth/Qwen3.5-2B",   ft="AriesDjaenuri/qwen35-2b-medical-id",   loader="vision"),
    "4b":   dict(base="unsloth/Qwen3.5-4B",   ft="AriesDjaenuri/qwen35-4b-medical-id",   loader="vision"),
}
assert MODEL_KEY in REGISTRY, f"MODEL_KEY harus {list(REGISTRY)}"
CFG = REGISTRY[MODEL_KEY]

# ====================== PARAMETER EVAL ======================
N_EVAL         = 20 if SMOKE_TEST else N_EVAL_FULL   # protokol fair greedy
LABEL_SUFFIX   = "_smoke" if SMOKE_TEST else ""
MAX_NEW_TOKENS = 256
MAX_SEQ_LENGTH = 1024
BATCH_SIZE     = 16           # batched generate (spt eval.py). Turunkan bila OOM: 8 utk 4B/T4
CKPT_EVERY     = 5            # simpan checkpoint tiap N batch (anti-putus di run panjang)
LOAD_IN_4BIT   = False        # 16-bit (Bagian 3). True utk GPU kecil (mis. 4B di <16GB)
DO_BERTSCORE   = True         # BERTScore F1 (semantik). Set False kalau mau cepat
BERT_MODEL     = "bert-base-multilingual-cased"
POST_CLEAN     = True         # buang sapaan residual (konsisten train==eval==deploy)
SHOW_SAMPLES   = 2 if SMOKE_TEST else 0   # cetak contoh pred/ref utk diperiksa mata

RESULTS_DIR = "/content/results"
import os; os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"MODE={'SMOKE' if SMOKE_TEST else 'RUN ASLI'} | MODEL_KEY={MODEL_KEY} | loader={CFG['loader']}")
print(f"  base = {CFG['base']}")
print(f"  ft   = {CFG['ft']}")
print(f"N_EVAL={N_EVAL} | 4bit={LOAD_IN_4BIT} | BERTScore={DO_BERTSCORE} | label suffix='{LABEL_SUFFIX}'")
if SMOKE_TEST:
    print("\\n>> SMOKE: cek 3 hal — (1) kedua model ter-load, (2) pred bukan kosong/ngawur,")
    print(">>        (3) ETA full run masuk akal. Lalu set SMOKE_TEST=False & ulang dari sel ini.")''')

md("""## 4. Test set (`Data/processed_id/test.jsonl`, 2998)

Cari otomatis di Drive / lokal (set env `DATA_DIR` bila beda). Colab: mount Drive bila perlu.""")

code('''import os
from datasets import load_dataset

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(os.path.join(p, "test.jsonl")):
            return p
    return None

CANDIDATES = [
    os.environ.get("DATA_DIR", ""),
    "/content/drive/MyDrive/Aries/Fine-Tune SLM for Medical Chatbot/Data/processed_id",
    "../Data/processed_id", "Data/processed_id",
]
if any("drive/MyDrive" in c for c in CANDIDATES) and not os.path.exists("/content/drive"):
    try:
        from google.colab import drive; drive.mount("/content/drive")
    except Exception:
        pass
DATA_DIR = _first_existing(CANDIDATES)
assert DATA_DIR, "test.jsonl tak ketemu. Set env DATA_DIR atau upload ke Drive."
test_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "test.jsonl"), split="train")
print("DATA_DIR =", DATA_DIR, "| test =", len(test_ds))

# Checkpoint WAJIB di Drive: /content ikut TERHAPUS kalau runtime Colab mati,
# jadi checkpoint di /content tak melindungi dari risiko utama (sesi putus).
CKPT_DIR = RESULTS_DIR
if os.path.isdir("/content/drive/MyDrive"):
    CKPT_DIR = "/content/drive/MyDrive/p6_eval_ckpt"
    os.makedirs(CKPT_DIR, exist_ok=True)
else:
    print("!! Drive tak ter-mount -> checkpoint hanya di /content (hilang bila runtime mati)")
print("CKPT_DIR =", CKPT_DIR)''')

md("## 5. Helper: chat template, cleaner sapaan, metrik")

code('''import re, collections
from rouge_score import rouge_scorer

_GREETING = [
    re.compile(r"^@\\w+[\\s,:]*"),
    re.compile(r"^(?:hai|halo|alo|hi|hello|hallo|hey)\\b[\\s,!.:;-]*", re.I),
    re.compile(r"^(?:salam)\\b[\\s,!.:;-]*(?:sehat|sejahtera|kenal)?[\\s,!.:;-]*", re.I),
    re.compile(r"^(?:selamat\\s+(?:pagi|siang|sore|malam|datang))[\\s,!.:;-]*\\w*[\\s,!.:;-]*", re.I),
    re.compile(r"^(?:hi|hello|hey)\\s+there\\b[\\s,!.:;-]*", re.I),
    re.compile(r"^(?:terima\\s*kasih|terimakasih|thanks|thank\\s*you)[\\s,!.]*(?:atas|telah|for)?[^.!?]*[.!?]\\s*", re.I),
    re.compile(r"^(?:dear)\\b[\\s,!.:;-]*\\w*[\\s,!.:;-]*", re.I),
]
def clean_greeting(text, min_keep=15):
    original = (text or "").strip(); t = original
    for _ in range(4):
        prev = t
        for pat in _GREETING: t = pat.sub("", t).strip()
        if t == prev: break
    if t and t[0].islower(): t = t[0].upper() + t[1:]
    return t if len(t.strip()) >= min_keep else original

def token_f1(pred, ref):
    p, r = pred.split(), ref.split()
    common = collections.Counter(p) & collections.Counter(r)
    ns = sum(common.values())
    if ns == 0 or not p or not r: return 0.0
    prec, rec = ns/len(p), ns/len(r)
    return 2*prec*rec/(prec+rec)

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def render_chat(tok, messages, add_generation_prompt=False):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                   add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False,
                   add_generation_prompt=add_generation_prompt)
print("helper siap.")''')

md("""## 6. Fungsi eval satu model (load → generate → metrik → simpan → bebaskan VRAM)""")

code('''import json, gc, os, time, torch

# ---------- checkpoint: run panjang tak boleh hangus krn sesi Colab putus ----------
def _ckpt_path(label): return f"{CKPT_DIR}/{label}.ckpt.json"

def _ckpt_load(label, model_name, n):
    """Kembalikan preds MENTAH yg sudah selesai (list kosong bila tak ada/beda config)."""
    fp = _ckpt_path(label)
    if not os.path.exists(fp): return []
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        print(f"  checkpoint {fp} korup -> diabaikan"); return []
    if (d.get("model"), d.get("n"), d.get("batch_size"), d.get("max_new_tokens")) != \\
       (model_name, n, BATCH_SIZE, MAX_NEW_TOKENS):
        print("  checkpoint ada tapi BEDA konfigurasi -> diabaikan (mulai dari 0)"); return []
    preds = d.get("preds", [])
    if preds: print(f"  RESUME: {len(preds)}/{n} sampel sudah ada di checkpoint")
    return preds

def _ckpt_save(label, model_name, n, preds):
    fp = _ckpt_path(label); tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "n": n, "batch_size": BATCH_SIZE,
                   "max_new_tokens": MAX_NEW_TOKENS, "preds": preds}, f, ensure_ascii=False)
    os.replace(tmp, fp)   # atomik: file lama tetap utuh bila mati saat menulis


def eval_model(model_name, label):
    n = min(N_EVAL, len(test_ds))
    refs = [next(m["content"] for m in test_ds[i]["messages"] if m["role"] == "assistant")
            for i in range(n)]
    print(f"\\n=== [{label}] {model_name} (n={n}) ===")
    preds = _ckpt_load(label, model_name, n)   # MENTAH (clean_greeting baru di akhir)

    dt, n_new = 0.0, 0
    if len(preds) < n:
        if CFG["loader"] == "language":
            from unsloth import FastLanguageModel as L
        else:
            from unsloth import FastModel as L
        model, tok = L.from_pretrained(model_name, max_seq_length=MAX_SEQ_LENGTH,
                                       load_in_4bit=LOAD_IN_4BIT, dtype=None)
        if not getattr(model.config, "architectures", None):
            model.config.architectures = ["Qwen3ForCausalLM"]
        L.for_inference(model)
        # batched generate perlu LEFT padding (decoder-only) — identik eval.py:220-222
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        prompts = [render_chat(tok, [m for m in test_ds[i]["messages"] if m["role"] != "assistant"],
                               add_generation_prompt=True) for i in range(n)]

        start = len(preds)          # lanjut dari sampel terakhir yg tersimpan
        t0, n_new = time.time(), n - start
        for bi, i in enumerate(range(start, n, BATCH_SIZE)):
            batch = prompts[i:i + BATCH_SIZE]
            enc = tok(text=batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                # protokol IDENTIK eval.py (fair greedy Pivot 5): tanpa repetition_penalty
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     num_beams=1, no_repeat_ngram_size=3,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            preds.extend(p.strip() for p in tok.batch_decode(gen, skip_special_tokens=True))
            done = min(i + BATCH_SIZE, n)
            if (bi + 1) % CKPT_EVERY == 0 or done >= n:
                _ckpt_save(label, model_name, n, preds)
            print(f"  {done}/{n}")
        dt = time.time() - t0
        del model; gc.collect(); torch.cuda.empty_cache()
    else:
        print("  semua sampel sudah ada -> generate DILEWATI (model tak perlu di-load)")

    assert len(preds) == n, f"preds {len(preds)} != n {n}"
    if POST_CLEAN: preds = [clean_greeting(p) for p in preds]

    for j in range(min(SHOW_SAMPLES, len(preds))):
        print(f"\\n--- contoh #{j+1} ---\\n[PRED] {preds[j][:400]}\\n[REF ] {refs[j][:400]}")
    n_empty = sum(1 for p in preds if len(p.strip()) < 5)
    if n_empty: print(f"\\n!! {n_empty}/{len(preds)} prediksi (nyaris) KOSONG -> cek chat template/loader")

    f1 = sum(token_f1(p, r) for p, r in zip(preds, refs)) / len(preds)
    rl = sum(_scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(preds)
    res = {"label": label, "model": model_name, "n": len(preds),
           "token_f1": round(f1, 4), "rougeL": round(rl, 4),
           "quant": "4bit" if LOAD_IN_4BIT else "16bit", "batch_size": BATCH_SIZE,
           "gen": f"greedy,num_beams=1,no_repeat_ngram=3,max_new={MAX_NEW_TOKENS}"}
    if n_new:
        res["sec_per_sample"] = round(dt / n_new, 2)
        print(f"\\nwaktu: {dt/60:.1f} mnt utk {n_new} sampel baru ({dt/n_new:.2f} s/sampel)"
              f" -> ETA {N_EVAL_FULL} sampel = {dt/n_new*N_EVAL_FULL/60:.0f} mnt/model")
    else:
        print("\\n(semua dari checkpoint — waktu generate tak diukur)")
    if DO_BERTSCORE:
        from bert_score import score as bscore
        _, _, Fb = bscore(preds, refs, lang="id", model_type=BERT_MODEL, verbose=False)
        res["bertscore_f1"] = round(float(Fb.mean()), 4)

    payload = {**res, "samples": [{"pred": p, "ref": r} for p, r in zip(preds[:5], refs[:5])]}
    outs = [f"{RESULTS_DIR}/{label}.json"]
    if CKPT_DIR != RESULTS_DIR: outs.append(f"{CKPT_DIR}/{label}.json")   # salinan aman di Drive
    for fp in outs:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    if os.path.exists(_ckpt_path(label)):
        # JANGAN hapus: checkpoint = SEMUA prediksi mentah. Disimpan permanen utk
        # analisis faktual (topic-drift / LLM-judge) tanpa perlu generate ulang.
        os.replace(_ckpt_path(label), f"{CKPT_DIR}/{label}.preds.json")
    print("HASIL:", res)
    return res
print("eval_model() siap.")''')

md("## 7. Eval BASELINE (pre-trained)")
code('''res_base = eval_model(CFG["base"], f"qwen{MODEL_KEY}_baseline{LABEL_SUFFIX}")''')

md("## 8. Eval FINE-TUNED (merged HF)")
code('''res_ft = eval_model(CFG["ft"], f"qwen{MODEL_KEY}_finetuned{LABEL_SUFFIX}")''')

md("## 9. Delta model ini (RQ1)")
code('''def _d(a, b): return round(b - a, 4)
print(f"=== {MODEL_KEY}: baseline -> finetuned (delta) ===")
for k in ["token_f1", "rougeL", "bertscore_f1"]:
    if k in res_base and k in res_ft:
        print(f"  {k:14s}: {res_base[k]:.4f} -> {res_ft[k]:.4f}  (Δ {_d(res_base[k], res_ft[k]):+.4f})")
if SMOKE_TEST:
    print("\\n[SMOKE] Angka di atas n=20 -> TIDAK dipakai di naskah, hanya bukti pipeline hidup.")
    print("        Kalau pred wajar & ETA oke: set SMOKE_TEST=False di sel 3, jalankan ulang sel 3,6,7,8.")
else:
    print("\\nUlangi dari sel Konfigurasi dgn MODEL_KEY berbeda (2b, 4b), lalu jalankan sel Ringkasan.")''')

md("""## 10. Ringkasan semua model (jalankan setelah 0.8b/2b/4b selesai)

Membaca semua `results/*.json` → tabel + delta (bukti **RQ1** peningkatan & **RQ2** scaling).""")

code('''import json, glob, os
rows = []
for fp in sorted(glob.glob(f"{RESULTS_DIR}/*.json")):
    if fp.endswith("_smoke.json"): continue          # hasil smoke tak masuk tabel naskah
    d = json.load(open(fp, encoding="utf-8"))
    rows.append(d)
by_label = {d["label"]: d for d in rows}

hdr = f'{"model":10s} {"kondisi":10s} {"tok-F1":>8s} {"ROUGE-L":>8s} {"BERTScore":>10s}'
print(hdr); print("-" * len(hdr))
for key in ["0.8b", "2b", "4b"]:
    for cond in ["baseline", "finetuned"]:
        d = by_label.get(f"qwen{key}_{cond}")
        if d:
            print(f'{key:10s} {cond:10s} {d.get("token_f1",0):8.4f} {d.get("rougeL",0):8.4f} '
                  f'{d.get("bertscore_f1",0):10.4f}')
    b, f = by_label.get(f"qwen{key}_baseline"), by_label.get(f"qwen{key}_finetuned")
    if b and f:
        print(f'{key:10s} {"Δ (ft-base)":10s} {f["token_f1"]-b["token_f1"]:+8.4f} '
              f'{f["rougeL"]-b["rougeL"]:+8.4f} '
              f'{f.get("bertscore_f1",0)-b.get("bertscore_f1",0):+10.4f}')
    print()
print("Target paper: ROUGE-L finetuned mendekati/di atas 0.30 (lihat catatan RQ1).")''')

md("""---
**Selesai.** Simpan `results/*.json` (bisa copy ke Drive utk arsip). Angka ini = bukti **RQ1**
(peningkatan vs baseline) dan **RQ2** (scaling 0.8B→2B→4B) di naskah.

> Untuk **trade-off kuantisasi** (16-bit vs Q4_K_M) dan **on-device** (RAM/tok/s), itu Fase 2
> (eval GGUF via llama.cpp + benchmark CPU) — terpisah.""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, "|", len(cells), "cells")
