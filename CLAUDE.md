# Fine-Tune SLM for Medical Chatbot

> ## 🟢 KONDISI AKTIF = PIVOT 5 (REVISI DOSEN) — baca `HANDOFF_PIVOT4.md` (SUMBER KEBENARAN)
> **Objektif baru: SATU model `Qwen3.5-0.8B`, baseline (pre-trained) vs fine-tuned (QLoRA).**
> BUKAN lagi perbandingan antar-model. Data `Data/processed_id/` 30003/2997/2998, packing=False +
> train_on_responses_only, LR 1e-4. **Qwen3.5-0.8B = model TEKS (FastLanguageModel), BUKAN multimodal.**
> Gemma 3 1B & Llama 3.2 1B **di-drop** (notebook diarsipkan ke `notebooks/notebook-lawas/`).
> **Training Qwen0.8B SUDAH SELESAI** (3 epoch/5628 step, eval loss 2.74→2.075; adapter di Drive;
> kurva `results/qwen35_0_8b_trainer_state.json`). NEXT = eval baseline vs fine-tuned (token-F1+ROUGE-L).
> Detail + koreksi asumsi usang: **`HANDOFF_PIVOT4.md`** (juga di Drive, sinkron dgn Claude browser).

## Project Overview
Fine-tuning Small Language Model (SLM) untuk Medical Chatbot yang mampu menjelaskan keluhan pasien secara detail.
**Patokan lengkap ada di `WORKFLOW.md`** — CLAUDE.md ini adalah ringkasan referensi cepat.

> ### ⚠️ PIVOT 3 (2026-06-14, di-update untuk Gemma 4) — lihat `README.md` sumber kebenaran pipeline
> Dari 3 model kecil (Qwen3.5-0.8B / Llama-3.2-1B / Gemma3-1B) → **2 model ~2B** untuk
> studi **comparative trade-off akurasi vs efisiensi deployment**. KEDUA model multimodal,
> di-load lewat `FastVisionModel` tetapi dilatih **TEXT-ONLY** (`finetune_vision_layers=False`):
> - **Qwen3.5-2B** (standard dense, `unsloth/Qwen3.5-2B`, ChatML `<|im_start|>`, thinking OFF, system native)
> - **Gemma 4 E2B** (PLE effective-param dense ~2.3B aktif/5.1B total, `unsloth/gemma-4-E2B-it`,
>   turn-token `<|turn>…<turn|>`, `get_chat_template("gemma-4")`, system native, thinking OFF)
>
> **RQ1** peningkatan vs baseline · **RQ2** trade-off **standard dense vs PLE effective-param dense**
> (RAM: memuat 5.1B embedding vs komputasi ~2.3B — apakah caching PLE meredam RAM? + latency/size) ·
> **RQ3** kelayakan deploy offline Puskesmas.
>
> Pipeline (Bagian 1-5 master note): `preprocessing/preprocess_dataset.py` → `Data/processed_final/{train,val,test}_final.jsonl`
> → `notebooks/finetune_{qwen35_2b,gemma4_e2b}.ipynb` → `eval.py` (baseline×4 + GGUF) → `export_gguf.py` (Q4_K_M) → `benchmark_ondevice.py`.
> Format prompt tunggal di `chat_format.py` (train=eval=deploy). Notebook lama (00-03) + `Data/processed_shared/` = artefak Pivot-2 (di-superseded).
> Fakta model (loader/turn-token/arsitektur) diverifikasi dari notebook & doc resmi Unsloth (2026-06).

---

## Dataset

### Data Lokal (`Data/`)
| File | Isi | Jumlah | Status |
|------|-----|--------|--------|
| `SimpleTabulation-ICD-11-MMS-en.xlsx` | Klasifikasi ICD-11 MMS | 35,664 berkode | sudah ada |
| `hidr_indicators.xlsx` | Indikator kesehatan WHO | 4,038 baris | sudah ada |
| `indonesia-medical-qna/qna.csv` | QnA pasien-dokter Alodokter | 681k baris | sudah ada |
| `training13b.json` + `13B[1-4]_golden.json` | BioASQ biomedical QA | 5,729 | sudah ada |
| `indonesia-bioner/` | NER biomedis Indonesia | — | **skip** (bukan QA) |

### Data yang Perlu Disiapkan
| File | Cara | Estimasi |
|------|------|---------|
| `data/bioasq/BioASQ-train-factoid-4b.json` | Download dari bioasq.org (perlu registrasi) | ~1,000 QA |
| `data/ppk_kemenkes/ppk_qa_pairs.json` | Buat manual dari PDF PPK Kemenkes | ~300–500 QA |

### HuggingFace Datasets (`Data/data.py`)
| Dataset | HF ID | Estimasi QA |
|---------|-------|------------|
| PubMedQA | `qiaojin/PubMedQA` | ~1,000 |
| MedQuAD | `lavita/MedQuAD` | ~47,000 |
| WikiDoc Medical | `medalpaca/medical_meadow_wikidoc` | ~10,000 |
| MedMCQA | `openlifescienceai/medmcqa` (5k sample) | ~5,000 |
| MedFit | `mlx-community/medfit-dataset` | cek runtime |
| ChatDoctor | `lavita/medical-qa-datasets` (chatdoctor_healthcaremagic) | cek runtime |

### Dataset Gabungan (hasil `preprocessing/build_dataset.py`)
Sumber: BioASQ, Indonesia QnA (Alodokter), ICD-11, HIDR, PubMedQA, MedQuAD, WikiDoc, MedMCQA, MedFit, ChatDoctor.
Setiap sample punya field `"source"` (dipakai untuk reduksi stratified di Notebook 00).

| File | Jumlah |
|------|--------|
| `Data/processed/train.jsonl` | 103,857 |
| `Data/processed/val.jsonl` | 12,982 |
| `Data/processed/test.jsonl` | 12,982 |

Cek distribusi bahasa & sumber: `python preprocessing/analyze_language_distribution.py`

### Dataset Final untuk Training (Pivot 3 — `preprocessing/preprocess_dataset.py`) ★ DIPAKAI
Pipeline cleaning Bagian 1 dari `Data/processed/` (FULL, punya `source`): dedup/anti-leakage →
clean greeting (clause-based) → quality → buang noise `icd11`/`hidr` → balance bahasa → reduksi stratified.

| File | Jumlah |
|------|--------|
| `Data/processed_final/train_final.jsonl` | 20,000 (ID 29.6%) |
| `Data/processed_final/val_final.jsonl` | 1,500 |
| `Data/processed_final/test_final.jsonl` | 10,109 (untuk eval) |

> Residual sapaan ~1.95% (limitasi terdokumentasi). Statistik per-langkah dicetak saat run (untuk tabel BAB III).

### Dataset Tereduksi LAMA (Pivot-2, superseded — `notebooks/00_analisis_dataset.ipynb`)
Karena kuota GPU/Colab terbatas, training memakai subset stratified-by-source dari `train.jsonl`/`val.jsonl`.
Disimpan di `Data/processed_shared/` dan dipakai bersama oleh ketiga notebook fine-tuning:

| File | Jumlah |
|------|--------|
| `Data/processed_shared/train_reduced.jsonl` | 20,000 |
| `Data/processed_shared/val_reduced.jsonl` | 1,500 |
| `Data/processed_shared/training_config_recommended.json` | config (`max_seq_length=640`, dll) |
| `Data/processed_shared/token_length_distribution.png` | histogram panjang token |

> Catatan: ada juga folder `processed/` di root (duplikat lama, belum dibersihkan) — yang dipakai notebook adalah `Data/processed/` dan `Data/processed_shared/`.

---

## Format Dataset
ChatML format:
```
<|im_start|>system
You are a helpful medical assistant. Answer questions accurately based on clinical guidelines and ICD-11 classification.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>
```

> Pivot-3 (Gemma 4): format di atas (ChatML) untuk **Qwen3.5-2B**. **Gemma 4** memakai
> turn-token `<|turn>role … <turn|>` via `get_chat_template("gemma-4")` dan **mendukung role
> `system` native** (beda dari Gemma 3n lama) → `chat_format.py` TIDAK lagi menggabungkan
> `system` ke user (`merge_system=False` kedua model). System prompt bervariasi per sumber
> data (empati / MCQA / biomedis) — ada di dataset, dipertahankan saat training. **Thinking
> OFF** kedua model (`enable_thinking=False`). `train_on_responses_only` → loss hanya di jawaban.

---

## Models & Konfigurasi (Pivot 3 — Qwen3.5-2B standard dense vs Gemma 4 E2B PLE effective-param dense)

> **Pivot 1**: Qwen2.5-2B/SmolLM3-3B/Gemma3-4B → model lebih kecil (keterbatasan GPU).
> **Pivot 2**: QLoRA manual → **Unsloth full di Colab**.
> **Pivot 3 (2026-06-14)**: 3 model kecil → **2 model ~2B** untuk studi *comparative
> trade-off akurasi vs efisiensi deployment*. Semula Gemma 3n (elastic/MatFormer); di-update
> ke **Gemma 4 E2B** (dense + Per-Layer Embeddings) — RQ2 = standard dense vs PLE
> effective-param dense. KEDUA model VLM → `FastVisionModel`, dilatih TEXT-ONLY. Colab L4.

| Urutan | Notebook | Model (Unsloth, + fallback) | Loader | Sifat |
|--------|----------|------------------------------|--------|-------|
| 1 | `notebooks/finetune_qwen35_2b.ipynb` | `unsloth/Qwen3.5-2B` (fallback `unsloth/Qwen3.5-2B-Instruct`) | `FastVisionModel` (vision OFF) | standard dense ~2B; ChatML; thinking OFF; system native |
| 2 | `notebooks/finetune_gemma4_e2b.ipynb` | `unsloth/gemma-4-E2B-it` (fallback `unsloth/gemma-4-E2B`) | `FastVisionModel` (vision OFF) | dense + PLE (~2.3B aktif/5.1B total); turn-token `<\|turn>`; thinking OFF; system native |

Notebook di-generate oleh `notebooks/build_finetune_notebooks.py` — **jangan edit `.ipynb`
manual**; edit generator lalu `python notebooks/build_finetune_notebooks.py`.

**Konfigurasi bersama (kedua notebook IDENTIK, sumber kebenaran prompt: `chat_format.py`)**:
- Loading: `FastVisionModel.from_pretrained(load_in_4bit=True, dtype=None, use_gradient_checkpointing='unsloth')` + TEST-FIRST
- LoRA IDENTIK kedua model: `FastVisionModel.get_peft_model(finetune_vision_layers=False,
  finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True,
  target_modules='all-linear')` — catat `print_trainable_parameters()` (angka nyata utk BAB III)
- `max_seq_length`: 1024 | `lora_r`: 16 | `lora_alpha`: 32 | `lora_dropout`: 0.05
- `batch_size`: 4 | `gradient_accumulation`: 4 (eff batch 16)
- `learning_rate`: **1e-4** (2e-4 terlalu agresif) | `epochs`: 5 + `EarlyStopping(patience=2, threshold=0.001)`
- `lr_scheduler`: cosine | `optim`: adamw_8bit | `weight_decay`: 0.01 | `warmup_ratio`: 0.10 | `seed`: 42
- `train_on_responses_only` (loss hanya di jawaban) → **`packing=False`** (wajib agar batas turn akurat)
- `eval_strategy='steps'` + `save_strategy='steps'`, cadence ~10x/epoch (match → `load_best_model_at_end=True`, metric `eval_loss`)
- Smoke test `SMOKE_TEST=True` (20 step) sebelum run penuh 5 epoch (hemat kuota Colab)
- Training data: `Data/processed_final/{train_final,val_final}.jsonl` (20,000 / 1,500)
- Merge best-checkpoint: `model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method='merged_16bit')`
- Resume otomatis dari checkpoint; install Colab: `pip install unsloth unsloth_zoo timm`

---

## Struktur File
```
├── chat_format.py                     # ★ SUMBER KEBENARAN format prompt (train=eval=deploy)
├── eval.py                            # Bagian 3-4: multi-metrik, baseline×4, per-bahasa, GGUF
├── export_gguf.py                     # Bagian 4: merged 16-bit → GGUF Q4_K_M (llama.cpp)
├── benchmark_ondevice.py              # Bagian 5: RAM load/infer, latency, arsitektur (CPU)
├── README.md                          # ★ panduan pipeline Pivot-3 + urutan run lengkap
├── catatan.txt                        # ringkasan perubahan + to-do
├── Data/
│   ├── data.py                        # download HF datasets
│   ├── processed/                     # output build_dataset.py (FULL, ADA kolom `source`)
│   │   └── train/val/test.jsonl  (103,857 / 12,982 / 12,983)
│   ├── processed_final/               # ★ output preprocess_dataset.py — DIPAKAI TRAINING
│   │   └── train/val/test_final.jsonl  (20,000 / 1,500 / 10,109)
│   └── processed_shared/              # artefak Pivot-2 (superseded)
├── notebooks/
│   ├── build_finetune_notebooks.py    # generator 2 notebook training
│   ├── finetune_qwen35_2b.ipynb       # ★ JALANKAN DULU (Qwen3.5-2B standard dense)
│   ├── finetune_gemma4_e2b.ipynb      # ★ ke-2 (Gemma 4 E2B PLE effective-param dense)
│   └── 00_…/01_…/02_…/03_…ipynb        # artefak Pivot-2 (superseded)
├── outputs/
│   ├── checkpoints/  {qwen35-2b-,gemma4-e2b-}{train,adapter}/
│   ├── merged/       qwen35-2b-medical/  +  gemma4-e2b-medical/
│   └── gguf/         *-Q4_K_M.gguf
├── preprocessing/
│   ├── preprocess_dataset.py          # ★ Bagian 1: cleaning → Data/processed_final/
│   ├── build_dataset.py               # gabung sumber → Data/processed/
│   ├── process_*.py                   # per-sumber (bioasq/icd11/chatdoctor/…)
│   └── analyze_language_distribution.py
├── results/  results_bench/           # output JSON eval.py & benchmark_ondevice.py
└── scripts/convert_to_gguf.sh
```

---

## Evaluasi (Bagian 3 — `eval.py`, protokol fair greedy, n=200, test_final.jsonl)
4 evaluasi protokol identik: {qwen, gemma} × {baseline, finetuned}; lalu diulang pada GGUF Q4_K_M.
`python eval.py --summarize results` → tabel + delta (finetuned − baseline) untuk RQ1.

| Bucket | Metrik | Acuan |
|--------|--------|-------|
| MCQA (medmcqa) | Exact-Match huruf opsi | naik vs baseline |
| Yes/No (pubmedqa) | Exact-Match ternormalisasi | naik vs baseline |
| Open-ended (chatdoctor/alodokter/…) | ROUGE-L / ROUGE-1 / token-F1 | ROUGE-L > 0.30 |
| Per-bahasa | metrik dipisah **ID vs EN** | subset ID paling relevan Puskesmas |

On-device (`benchmark_ondevice.py`, CPU-only): peak RAM **load vs infer**, tok/s, ukuran file
→ tabel trade-off **standard dense vs PLE effective-param dense** (inti RQ2/RQ3): fokus apakah
Gemma 4 memuat ~5.1B embedding ke RAM walau komputasi ~2.3B, dan apakah caching PLE meredamnya.

---

## Progress

### Phase 0 — Environment
- [x] `.env` venv dibuat (lokal Windows)
- [ ] `torch.cuda.is_available()` = True — perlu diverifikasi ulang
- [ ] `pip install -r requirements.txt` selesai

### Phase 1 — Data
- [x] `Data/SimpleTabulation-ICD-11-MMS-en.xlsx` — sudah ada
- [x] `Data/hidr_indicators.xlsx` — sudah ada
- [x] `Data/training13b.json` + `13B[1-4]_golden.json` — sudah ada
- [x] `Data/indonesia-medical-qna/qna.csv` — sudah ada
- [x] `python Data/data.py` — HuggingFace datasets (PubMedQA, MedQuAD, WikiDoc, MedMCQA, MedFit, ChatDoctor) — selesai
- [ ] `data/bioasq/BioASQ-train-factoid-4b.json` — tidak dipakai (skip, BioASQ lokal sudah cukup)
- [ ] `data/ppk_kemenkes/ppk_qa_pairs.json` — belum dibuat (opsional, belum diprioritaskan)

### Phase 2 — Dataset Preparation
- [x] `preprocessing/build_dataset.py` — gabung 10 sumber, cleaning + dedup + cap + filter, split 80/10/10
- [x] `Data/processed/{train,val,test}.jsonl` tersimpan (103,857 / 12,982 / 12,982), tiap sample punya field `source`
- [x] `preprocessing/analyze_language_distribution.py` — cek distribusi bahasa & sumber
- [x] `notebooks/00_analisis_dataset.ipynb` — **SELESAI**: analisis panjang token (max_seq_length=640),
      kapasitas model vs ukuran data, reduksi stratified → `Data/processed_shared/{train_reduced,val_reduced}.jsonl`
      (20,000 / 1,500) + `training_config_recommended.json`

### Phase 2.5 — Preprocessing Final (Pivot 3) ✅
- [x] `preprocessing/preprocess_dataset.py` dijalankan → `Data/processed_final/{train,val,test}_final.jsonl`
      (20,000 / 1,500 / 10,109). Cleaner sapaan clause-based (residual ~1.95%, tanpa content-loss);
      buang noise `icd11`/`hidr`; 0 dedup/leakage (split sudah bersih).

### Phase 3 — Training (Pivot 3: Qwen3.5-2B standard dense + Gemma 4 E2B PLE effective-param dense)
- [x] Verifikasi model dari notebook/doc RESMI Unsloth (2026-06): Qwen3.5-2B & Gemma 4 E2B = VLM
      (`FastVisionModel`); turn-token Qwen ChatML `<|im_start|>`, Gemma 4 `<|turn>`; Gemma 4 dense+PLE
      (BUKAN MatFormer), **system native** kedua model → `merge_system=False`.
- [x] 2 notebook di-generate (`build_finetune_notebooks.py`), nbformat-valid + syntax-checked
      (`finetune_qwen35_2b.ipynb`, `finetune_gemma4_e2b.ipynb`).
- [x] `chat_format.py` (format prompt tunggal) di-update untuk Gemma 4; config Bagian 2 (lr 1e-4,
      5 epoch + EarlyStopping, save/eval='steps' ~10x/epoch match → `load_best_model_at_end`).
- [ ] **`finetune_qwen35_2b.ipynb`** — jalankan DULU di Colab L4 (SMOKE_TEST → run penuh). Belum dijalankan.
- [ ] **`finetune_gemma4_e2b.ipynb`** — ke-2. Belum dijalankan.

> **`packing=False`** (wajib utk `train_on_responses_only` — loss hanya di jawaban). Best-checkpoint
> dijamin `load_best_model_at_end=True` (yang di-merge = val_loss terendah).
> **Loss multimodal TIDAK sebanding antar arsitektur** (Qwen vs Gemma 4) — nilai sukses dari tren
> turun + metrik downstream (`eval.py`), bukan loss absolut.

### Phase 4 — Evaluasi (`eval.py`)
- [ ] Baseline + finetuned (×4) 16-bit, `--summarize results` → delta (RQ1).
- [ ] Multi-metrik (MCQA EM / yes-no EM / ROUGE-L) + per-bahasa ID/EN; pilih model terbaik (trade-off).

### Phase 5 — GGUF Q4_K_M + On-device (`export_gguf.py`, `benchmark_ondevice.py`)
- [ ] Export Q4_K_M (×2) via llama.cpp; eval ulang Q4 (`eval.py --gguf`) → trade-off kuantisasi.
- [ ] Benchmark CPU: peak RAM **load vs infer**, tok/s → tabel trade-off (RQ2/RQ3), perangkat uji SAMA.

---

## Catatan Lanjutan (Next Steps)
1. **Upload ke Drive** `MyDrive/Fine-Tune SLM for Medical Chatbot/` termasuk `Data/processed_final/`
   dan `chat_format.py` (gitignored — wajib upload manual).
2. Colab **Runtime → GPU L4**. `HF_TOKEN` opsional (model `unsloth/*` ungated).
3. Jalankan `finetune_qwen35_2b.ipynb` dulu, lalu `finetune_gemma4_e2b.ipynb`.
4. Eval: 4 perintah `eval.py` (baseline+finetuned ×2) → `eval.py --summarize results`. Detail di `README.md`/`catatan.txt`.
5. Export GGUF + benchmark on-device. (Opsional) tambah data PPK Kemenkes (Bagian 1.7);
   bersihkan artefak Pivot-2 (notebook 00-03, `Data/processed_shared/`, root `processed/`) bila yakin tak dipakai.
