# Fine-Tune SLM for Medical Chatbot — Qwen3.5-0.8B: **baseline vs fine-tuned**

Skripsi UII — **Chatbot Medis Offline untuk Puskesmas**. Setelah **revisi dosen**, objektifnya
**BUKAN lagi perbandingan antar-model**, melainkan **satu model** (`Qwen3.5-0.8B`) yang dibandingkan
dengan **dirinya sendiri**: kondisi **baseline (pre-trained)** vs **setelah fine-tuning** (QLoRA pada
dataset medis berbahasa Indonesia).

| | Baseline | Fine-tuned |
|---|---|---|
| Model | `unsloth/Qwen3.5-0.8B` (pre-trained, apa adanya) | base + adapter QLoRA hasil training |
| Loader | `FastLanguageModel` (model **teks**, bukan VLM) | sama |
| Data latih | — | `Data/processed_id/` (Indonesia native, 30003/2997/2998) |
| Chat format | ChatML `<\|im_start\|>` + scaffold `<think></think>` | identik (train = eval = deploy) |

> **Gemma 3 1B & Llama 3.2 1B di-DROP** (revisi dosen) — notebook-nya diarsipkan di
> `notebooks/notebook-lawas/` sebagai jejak eksplorasi. Pivot 3 (2 model ~2B, `FastVisionModel`)
> juga superseded & diarsipkan.

## Pertanyaan Penelitian (revisi)
- **RQ1** — Apakah fine-tuning QLoRA **meningkatkan kualitas jawaban** Qwen3.5-0.8B (token-F1, ROUGE-L)
  dibanding baseline pre-trained, khususnya pada subset **Indonesia**?
- **RQ2** — Apakah model hasil fine-tune **layak deploy offline** di Puskesmas/HP (ukuran GGUF Q4_K_M,
  peak RAM load vs infer, latency tok/s) dengan keterbatasan resource?

---

## Status terkini (2026-06-21)

- ✅ **Dataset final** `Data/processed_id/{train,val,test}.jsonl` (30003/2997/2998), VERDICT=PASS.
- ✅ **Training SELESAI** — Qwen3.5-0.8B, QLoRA 4-bit, 3 epoch / 5628 step (Colab):
  - train loss **3.25 → 1.94**; eval loss **2.74 → 2.075** (best = step terakhir, turun monoton).
  - Kurva loss tersimpan: `results/qwen35_0_8b_trainer_state.json`.
  - Adapter di Drive: `MyDrive/Aries/Fine-Tune SLM for Medical Chatbot/outputs/checkpoints/qwen35-0.8b-train/`
    (`adapter_model.safetensors` 25,6MB + `checkpoint-5628/`). **Tarik ke `outputs/` lokal untuk eval** (gitignored).
- ⏭️ **BELUM**: eval baseline vs fine-tuned (token-F1 + ROUGE-L), export GGUF Q4_K_M, benchmark on-device.

---

## Pipeline & artefak

```
preprocessing/  ─► Data/processed_id/{train,val,test}.jsonl   (LOKAL, CPU, gratis)
        │
        └─► notebooks/train_qwen_qlora.ipynb  ─► adapter QLoRA (COLAB/KAGGLE T4)   ✅ SELESAI
                │
        eval.py (baseline vs fine-tuned, 16-bit) ─► results/*.json ─► --summarize   (RQ1)   ⏭️
                │
        export_gguf.py ─► outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf              ⏭️
                │
        benchmark_ondevice.py (CPU-only) ─► results_bench/*.json ─► --summarize     (RQ2)   ⏭️
```

| File | Fungsi |
|------|--------|
| `preprocessing/strict_clean_v2.py` + `build_id_dataset.py` | bangun dataset Indonesia-native → `Data/processed_id/` |
| `preprocessing/validate_dataset_for_training.py` | verifikasi mutu file final (dedup/leakage/decode) → VERDICT |
| `preprocessing/visualize_dataset.py` | grafik dataset (sumber, bahasa, token, noise, funnel) → `results/figures/*.png` |
| `chat_format.py` | **sumber kebenaran** format prompt + `clean_greeting()` — identik train = eval = deploy |
| `notebooks/train_qwen_qlora.ipynb` | training Unsloth `FastLanguageModel` + `train_on_responses_only` (di-generate `build_train_qwen_qlora.py`) |
| `notebooks/03_finetune_qwen3-5_0.8B_RUN.ipynb` | versi notebook yang **benar-benar dijalankan** di Colab (output di-strip) |
| `eval.py` | multi-metrik (token-F1, ROUGE-L), baseline vs fine-tuned, per-bahasa, 16-bit & GGUF |
| `export_gguf.py` | adapter/merged 16-bit → GGUF **Q4_K_M** (llama.cpp) |
| `benchmark_ondevice.py` | RAM load vs infer, throughput, ukuran file (CPU-only) |

---

## Urutan menjalankan

### 0 — Preprocessing (LOKAL, gratis) — *sudah selesai*
```bash
python preprocessing/validate_dataset_for_training.py     # VERDICT=PASS
python preprocessing/visualize_dataset.py                 # -> results/figures/*.png
```

### 1 — Training (COLAB/KAGGLE T4) — *sudah selesai*
Buka `notebooks/train_qwen_qlora.ipynb`, set `RUN_MODE="pilot"` → cek GATE → `RUN_MODE="full"`.
Hasil: adapter QLoRA (di Drive). `HF_TOKEN` opsional (`unsloth/*` ungated).

### 2 — Evaluasi 16-bit (RQ1: baseline vs fine-tuned) — *langkah berikutnya*
```bash
# Catatan: default --test_file eval.py masih menunjuk processed_final lama → override ke processed_id.
python eval.py --model unsloth/Qwen3.5-0.8B  --model_type qwen --label qwen08_baseline  \
               --test_file Data/processed_id/test.jsonl
python eval.py --model outputs/qwen35-0.8b-medical --model_type qwen --label qwen08_finetuned \
               --test_file Data/processed_id/test.jsonl
python eval.py --summarize results          # tabel + delta (finetuned − baseline), per-bahasa ID/EN
```
> Model fine-tuned = adapter dari Drive. Untuk eval, **merge dulu** ke 16-bit
> (`model.save_pretrained_merged("outputs/qwen35-0.8b-medical", tokenizer, save_method="merged_16bit")`)
> atau load base + adapter.

### 3 — Export GGUF Q4_K_M + eval terkuantisasi
```bash
python export_gguf.py --merged_dir outputs/qwen35-0.8b-medical --model_type qwen --verify
python eval.py --gguf outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf \
               --model outputs/qwen35-0.8b-medical --model_type qwen --label qwen08_q4 \
               --test_file Data/processed_id/test.jsonl
```

### 4 — Benchmark on-device (CPU-only → RQ2)
```bash
python benchmark_ondevice.py --gguf outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf --model_type qwen --label qwen08_q4
python benchmark_ondevice.py --summarize results_bench
```
> Jalankan di **perangkat uji yang sama** & **sebutkan spesifikasinya** di metodologi.

---

## Catatan & risiko
- **Konsistensi prompt** = *silent killer*: train, eval, deploy memakai format dari `chat_format.py`
  (notebook meng-inline logika identik). Qwen3.5 menyisipkan `<think></think>` walau thinking OFF →
  `RESPONSE_PART` memuat scaffold itu (kalau tidak, train ≠ infer).
- **`packing=False`** wajib karena `train_on_responses_only` (loss hanya di jawaban) butuh batas turn akurat.
- **`load_best_model_at_end=True`** → yang dipakai = eval_loss terendah (di sini = checkpoint-5628).
- **Keterbatasan (BAB V):** model 0.8B tidak mencapai akurasi klinis; halusinasi faktual pasti ada;
  metrik otomatis (F1/ROUGE) tak menangkap kebenaran medis; single-run. Ini **temuan valid** yang
  memperkuat argumen "butuh RAG = future work", bukan kegagalan.
- **`eval.py` default `--test_file`** masih `Data/processed_final/test_final.jsonl` (artefak Pivot-3) →
  selalu override `--test_file Data/processed_id/test.jsonl`, atau update default-nya di kode.
