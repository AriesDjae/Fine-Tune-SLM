# HANDOFF / SUMBER KEBENARAN — Pivot 4 (sinkronisasi Claude Code ⇄ Claude browser)

> **Tujuan dokumen ini:** satu acuan tunggal yang dibaca **kedua** asisten (Claude Code di IDE
> dan Claude di browser/claude.ai) supaya tidak saling bertentangan. Jika ada konflik antara
> dokumen lain dan file ini, **file ini yang menang** untuk hal "kondisi proyek saat ini".
>
> _Update terakhir: 2026-06-18 — Pivot 4 aktif. Menggantikan deskripsi Pivot 3 di `CLAUDE.md`._

---

## 1. PIVOT AKTIF = PIVOT 4 (≤1B, Indonesia-only)

- Studi **comparative 3 model ≤1B** untuk chatbot medis **offline** (Puskesmas/Termux Android), QLoRA 4-bit di **Kaggle/Colab T4**.
- Bahasa: **Indonesia native-only** (sumber `indonesia_qna` / Alodokter). Tidak ada translasi.
- Metrik eval: **token-F1 + ROUGE-L** (Exact-Match / MCQA **DI-DROP** untuk Pivot 4).

> Pivot 3 (2 model ~2B, FastVisionModel, `processed_final`) **sudah superseded**. Notebook `finetune_qwen35_2b.ipynb` & `finetune_gemma4_e2b.ipynb` = artefak Pivot 3.

---

## 2. DATASET (beku, read-only)

- **`Data/processed_id/{train,val,test}.jsonl` = 30003 / 2997 / 2998** (v2.1-remediated, Strict Clean v2).
- Skema tiap baris: `messages` (`system` → `user` → `assistant`) + `domain, type, source, source_lang, translated, quality_score`.
- System prompt seragam: *"Anda adalah asisten informasi kesehatan berbahasa Indonesia…"*.
- **MAX_SEQ_LENGTH = 1024** — data-driven: p99 ≈ 634 token, hanya **7/30003 (0,023%)** baris > 1024.
- Validasi: `preprocessing/validate_dataset_for_training.py` → **VERDICT=PASS** (cross-split dedup 0, decode-check OK).

---

## 3. MODEL & NOTEBOOK (3 notebook, harness IDENTIK)

Di-generate oleh **`notebooks/build_train_qwen_qlora.py`** (multi-model). **JANGAN edit `.ipynb` manual** — edit generator lalu `python notebooks/build_train_qwen_qlora.py`.

| Notebook | Model | Loader | Turn-token | System |
|---|---|---|---|---|
| `notebooks/train_qwen_qlora.ipynb` | `unsloth/Qwen3.5-0.8B` | **FastLanguageModel** | ChatML `<\|im_start\|>` + scaffold `<think>` | native |
| `notebooks/train_gemma3_1b_qlora.ipynb` | `unsloth/gemma-3-1b-it` | **FastLanguageModel** | `<start_of_turn>` | **di-merge ke user** (Gemma tak punya role system) |
| `notebooks/train_llama32_1b_qlora.ipynb` | `unsloth/Llama-3.2-1B-Instruct` | **FastLanguageModel** | header `<\|start_header_id\|>` | native |

Urutan run: **Qwen0.8B dulu** (baseline sehat) → Gemma 3 1B → Llama 3.2 1B. Alur tiap notebook: `RUN_MODE="pilot"` → **GATE** (loss turun + ≥9/10 Indonesia + non-degeneratif) → kalau PASS set `RUN_MODE="full"`. Panduan: `notebooks/README_TRAIN.md`.

---

## 4. HYPERPARAMETER KANONIK (sama di 3 notebook — controlled variable)

- QLoRA 4-bit; `r=16, alpha=32, dropout=0.05`; target 7 proj (`q,k,v,o,gate,up,down`).
- `LEARNING_RATE=1e-4` (cosine, warmup 0.10, weight_decay 0.01, adamw_8bit, max_grad_norm 1.0).
- `batch=2 × grad_accum=8` (eff 16); `EPOCHS=3` + **EarlyStopping(patience=3)**; `seed=42`.
- **`packing=False`** + **`train_on_responses_only`** (loss HANYA di jawaban).
- `load_best_model_at_end=True`, eval/save = "steps" (~10x/epoch).

---

## 5. ⚠️ KOREKSI ASUMSI USANG (yang sering muncul dari dokumen/NOTE lama)

Ini penyebab utama tabrakan knowledge. **Yang BENAR (Pivot 4):**

| Klaim usang (SALAH untuk Pivot 4) | Yang BENAR |
|---|---|
| "Qwen3.5-0.8B multimodal / VL Processor, perlu override AutoTokenizer" | **0.8B = model TEKS → FastLanguageModel.** Yang VLM itu varian **2B** (Pivot 3), bukan 0.8B. |
| Data `processed_shared/train_reduced.jsonl` | **`processed_id/`** (30003/2997/2998). `processed_shared` & `processed_final` = artefak Pivot-2/3. |
| `packing=True` | **`packing=False`** — wajib agar `train_on_responses_only` akurat. |
| `LEARNING_RATE=2e-4` | **`1e-4`** (2e-4 terbukti terlalu agresif). |
| Notebook target `01/02/03_finetune_*.ipynb` | Notebook aktif = **`train_{qwen,gemma3_1b,llama32_1b}_qlora.ipynb`**. `01/02/03` = lawas/superseded. |
| System prompt ICD-11 | System prompt empati Indonesia (sudah ada di data `processed_id`). |
| Tanpa scaffold `<think>` untuk Qwen | Qwen3.5 menyisipkan `<think></think>` walau thinking OFF → `RESPONSE_PART` Qwen memuat scaffold itu (kalau tidak, train ≠ infer). |

---

## 6. STATUS & NEXT

- ✅ Dataset `processed_id` final + validasi PASS.
- ✅ 3 notebook training siap (harness Pivot-4 + pengerasan install Colab-modern + sel verifikasi import).
- ⏭️ Peneliti: upload `Data/processed_id/` (3 `.jsonl`) + notebook ke Kaggle/Colab → jalankan **pilot → GATE → full** (Qwen dulu).
- ⏭️ Setelah training: eval token-F1 + ROUGE-L per-bahasa → export GGUF Q4_K_M → benchmark on-device (RAM/latency).

> Jika kamu (Claude mana pun) hendak memberi instruksi yang bertentangan dengan tabel di Bagian 5,
> **berhenti dan konfirmasi ke user** dulu — kemungkinan besar itu knowledge lama.
