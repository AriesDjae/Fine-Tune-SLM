# HANDOFF / SUMBER KEBENARAN — Pivot 5 (REVISI DOSEN: single-model, baseline vs fine-tuned)

> **Tujuan dokumen ini:** satu acuan tunggal yang dibaca **kedua** asisten (Claude Code di IDE
> dan Claude di browser/claude.ai) supaya tidak saling bertentangan. Jika ada konflik antara
> dokumen lain dan file ini, **file ini yang menang** untuk hal "kondisi proyek saat ini".
>
> _Update terakhir: 2026-06-21 — **Pivot 5 aktif** (revisi dosen). Menggantikan Pivot 4 (3 model komparatif)._

---

## 1. PIVOT AKTIF = PIVOT 5 (1 model: Qwen3.5-0.8B, baseline vs fine-tuned)

- **Revisi dosen:** objektif **BUKAN lagi perbandingan antar-model**. Sekarang **satu model**
  (`Qwen3.5-0.8B`) dibandingkan dengan **dirinya sendiri**: **baseline pre-trained vs setelah fine-tuning**
  (QLoRA pada dataset medis Indonesia).
- Use-case tetap: chatbot medis **offline** (Puskesmas / HP Android via Termux), QLoRA 4-bit di **Kaggle/Colab T4**.
- Bahasa: **Indonesia native-only** (sumber `indonesia_qna` / Alodokter). Tidak ada translasi.
- Metrik eval: **token-F1 + ROUGE-L** (Exact-Match / MCQA tetap **DI-DROP**).

> **Gemma 3 1B & Llama 3.2 1B = DI-DROP dari studi** (revisi dosen). Notebook-nya **diarsipkan** ke
> `notebooks/notebook-lawas/` sebagai jejak eksplorasi (tidak dihapus, untuk sidang). Pivot 3 (2 model
> ~2B, FastVisionModel) juga superseded — notebook `finetune_qwen35_2b.ipynb` & `finetune_gemma4_e2b.ipynb`
> ikut diarsipkan.

---

## 2. DATASET (beku, read-only)

- **`Data/processed_id/{train,val,test}.jsonl` = 30003 / 2997 / 2998** (v2.1-remediated, Strict Clean v2).
- Skema tiap baris: `messages` (`system` → `user` → `assistant`) + `domain, type, source, source_lang, translated, quality_score`.
- System prompt seragam: *"Anda adalah asisten informasi kesehatan berbahasa Indonesia…"*.
- **MAX_SEQ_LENGTH = 1024** — data-driven: p99 ≈ 634 token, hanya **7/30003 (0,023%)** baris > 1024.
- Validasi: `preprocessing/validate_dataset_for_training.py` → **VERDICT=PASS** (cross-split dedup 0, decode-check OK).

---

## 3. MODEL & NOTEBOOK (1 model aktif)

| Notebook | Model | Loader | Turn-token | System |
|---|---|---|---|---|
| `notebooks/train_qwen_qlora.ipynb` (generator) | `unsloth/Qwen3.5-0.8B` | **FastLanguageModel** | ChatML `<\|im_start\|>` + scaffold `<think>` | native |
| `notebooks/03_finetune_qwen3-5_0.8B_RUN.ipynb` | — versi yang **BENAR-BENAR dijalankan** di Colab (hasil ditarik dari Drive, output di-strip) | — | — | — |

> Generator `notebooks/build_train_qwen_qlora.py` masih bisa mem-build 3 model, tapi **hanya Qwen0.8B
> yang dipakai**. **JANGAN edit `.ipynb` manual** — edit generator lalu regenerate. Panduan: `notebooks/README_TRAIN.md`.

Alur notebook: `RUN_MODE="pilot"` → **GATE** (loss turun + ≥9/10 Indonesia + non-degeneratif) → kalau PASS set `RUN_MODE="full"`.

---

## 4. HYPERPARAMETER KANONIK

- QLoRA 4-bit; `r=16, alpha=32, dropout=0.05`; target 7 proj (`q,k,v,o,gate,up,down`).
- `LEARNING_RATE=1e-4` (cosine, warmup 0.10, weight_decay 0.01, adamw_8bit, max_grad_norm 1.0).
- `batch=2 × grad_accum=8` (eff 16); `EPOCHS=3` + **EarlyStopping(patience=3)**; `seed=42`.
- **`packing=False`** + **`train_on_responses_only`** (loss HANYA di jawaban).
- `load_best_model_at_end=True`, eval/save = "steps" (~10x/epoch).

---

## 5. ⚠️ KOREKSI ASUMSI USANG

| Klaim usang (SALAH) | Yang BENAR (Pivot 5) |
|---|---|
| "Studi membandingkan 3 model ≤1B / 2 model ~2B" | **Satu model: Qwen3.5-0.8B baseline vs fine-tuned.** Gemma/Llama di-drop (revisi dosen). |
| "Qwen3.5-0.8B multimodal / VL Processor" | **0.8B = model TEKS → FastLanguageModel.** Yang VLM itu varian 2B (Pivot 3). |
| Data `processed_shared` / `processed_final` | **`processed_id/`** (30003/2997/2998). `processed_shared`/`processed_final` = artefak Pivot-2/3. |
| `packing=True` / `LEARNING_RATE=2e-4` | **`packing=False`** + **`1e-4`**. |
| Notebook target `01/02/03_finetune_*.ipynb` lama | Aktif = **`train_qwen_qlora.ipynb`** (+ run-version `03_finetune_qwen3-5_0.8B_RUN.ipynb`). Sisanya di `notebook-lawas/`. |
| Tanpa scaffold `<think>` untuk Qwen | Qwen3.5 menyisipkan `<think></think>` walau thinking OFF → `RESPONSE_PART` Qwen memuat scaffold itu (kalau tidak, train ≠ infer). |

---

## 6. STATUS & NEXT

- ✅ Dataset `processed_id` final + validasi PASS.
- ✅ Notebook training Qwen0.8B siap + **SUDAH DIJALANKAN** (lihat hasil di bawah).
- ✅ **TRAINING SELESAI (Qwen3.5-0.8B, QLoRA, 3 epoch / 5628 step):**
  - train loss **3.25 → 1.94**; eval loss **2.74 → 2.075** (best = step terakhir 5628, turun monoton → belum overfitting).
  - Artefak di Drive: `MyDrive/Aries/Fine-Tune SLM for Medical Chatbot/outputs/checkpoints/qwen35-0.8b-train/`
    (`adapter_model.safetensors` 25,6MB + config + `checkpoint-5628/`). Kurva loss: `results/qwen35_0_8b_trainer_state.json`.
- ✅ **`eval.py` SUDAH DISESUAIKAN ke Pivot 5** (2026-06-21) & divalidasi di test set nyata (2998/2998 → open/id):
  loader **FastLanguageModel** (0.8B teks, auto-load base+adapter); default `--test_file
  Data/processed_id/test.jsonl`, `--model_type qwen`, `--max_seq_length 1024`; metrik **token-F1 + ROUGE-L**.
- ⏭️ **NEXT (yang diminta dosen — BELUM dijalankan): eval baseline vs fine-tuned.** (butuh GPU/unsloth → Colab/Kaggle/`.venv-gpu`)
  1. Tarik adapter dari Drive ke `outputs/` lokal (atau mount Drive di Colab).
  2. `python eval.py --model unsloth/Qwen3.5-0.8B --label qwen08_baseline` lalu
     `--model outputs/qwen35-0.8b-medical --label qwen08_finetuned` (boleh dir adapter) → `--summarize results`.
  3. Export GGUF Q4_K_M → `benchmark_ondevice.py` (RAM/latency/ukuran) untuk kelayakan offline.

> Jika kamu (Claude mana pun) hendak memberi instruksi yang bertentangan dengan tabel di Bagian 5,
> **berhenti dan konfirmasi ke user** dulu — kemungkinan besar itu knowledge lama.
