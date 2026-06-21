# README_TRAIN — Fine-Tune model ≤1B (QLoRA) di Kaggle/Colab T4

Panduan menjalankan notebook training QLoRA. Harness IDENTIK untuk **tiga model ≤1B**
(beda hanya MODEL_ID + turn-token), di-generate satu generator
`notebooks/build_train_qwen_qlora.py` (**jangan edit `.ipynb` manual** — edit generator
lalu `python notebooks/build_train_qwen_qlora.py`):

| Notebook | Model | Turn-token / catatan |
|---|---|---|
| `train_qwen_qlora.ipynb` | `unsloth/Qwen3.5-0.8B` | ChatML `<\|im_start\|>` + scaffold `<think>`; system native |
| `train_gemma3_1b_qlora.ipynb` | `unsloth/gemma-3-1b-it` | `<start_of_turn>`; **system di-merge ke user** (template Gemma tak punya role system) |
| `train_llama32_1b_qlora.ipynb` | `unsloth/Llama-3.2-1B-Instruct` | header `<\|start_header_id\|>`; system native |

Urutan disarankan: **Qwen0.8B dulu** (baseline sehat) → lalu Gemma 3 1B & Llama 3.2 1B.
Claude Code hanya **membangun** notebook + memvalidasi dataset; **peneliti yang
menjalankan training** (non-lokal, GPU T4).

Dataset beku: `Data/processed_id/{train,val,test}.jsonl` = **30003 / 2997 / 2998**,
ID native-only (Alodokter), v2.1-remediated. **Read-only — jangan diubah.**

---

## 0. Pra-syarat: validasi dataset (sudah PASS)

Sebelum training, dataset divalidasi oleh `preprocessing/validate_dataset_for_training.py`
→ **VERDICT: PASS** (row count, field, cross-split dedup 0/0, decode-check tokenisasi,
panjang token). Output: `results/dataset_validation_YYYYMMDD.json`.

Decode-check membuktikan: token **konten pertama jawaban** (mis. "Timbulnya") **ikut
dipelajari** (tidak ke-mask / off-by-one), dan format **train == inferensi** konsisten.
Hanya **7 record (0.023%)** train > 1024 token (ter-truncate ringan, bukan blocker).

---

## 1. Upload dataset ke Kaggle / Colab

File JSONL ~36k baris kecil (≈75 MB total) → **tidak perlu Git LFS**.

**Kaggle (disarankan):**
1. *Datasets → New Dataset* → upload `train.jsonl`, `val.jsonl`, `test.jsonl`
   (folder `processed_id`).
2. *Add Data* ke notebook → muncul di `/kaggle/input/<nama-dataset>/`.
3. Cek row-count cocok setelah upload:
   ```python
   import os
   d="/kaggle/input/<nama-dataset>"
   for s in ["train","val","test"]:
       print(s, sum(1 for _ in open(f"{d}/{s}.jsonl")))   # 30003 / 2997 / 2998
   ```
4. Kalau path beda dari kandidat di Sel 4, set sebelum run:
   `os.environ["DATA_DIR"]="/kaggle/input/<nama-dataset>"`.

**Colab:** upload folder `Data/processed_id/` ke
`MyDrive/Fine-Tune SLM for Medical Chatbot/Data/processed_id/` (Sel 4 mount Drive otomatis).

---

## 2. Runtime

- **GPU T4, single GPU** (Sel 2 set `CUDA_VISIBLE_DEVICES=0`).
- Kaggle: *Settings → Accelerator → GPU T4 x1* (jangan T4 x2 — single GPU saja).
- Colab: *Runtime → Change runtime type → T4 GPU*.

---

## 3. Urutan run

1. **Sel 1 (Install)** → jika ada peringatan konflik torch/xformers,
   *Restart session* lalu lanjut dari **Sel 3 (Konfigurasi)**.
2. **Sel 3:** pastikan `RUN_MODE = "pilot"` (default).
3. Jalankan Sel 2 → 10 berurutan. **Wajib lihat Sel 8 (SELF-CHECK decode)** —
   harus cetak `OK decode-check ...` (kalau assert gagal → STOP, ada masalah
   template/masking).
4. **Sel 11 (GATE):** baca verdict.
   - **PASS_GREEN** → buka Sel 3, set `RUN_MODE = "full"`, *Restart & Run All*.
   - **STOP / NEEDS_HUMAN** → lihat **TROUBLESHOOTING** di bawah; ubah **satu**
     hal, ulangi pilot.
5. Mode **full**: training penuh + EarlyStopping(patience=3) + simpan checkpoint
   berkala (resume otomatis bila terputus).
6. **Sel 12** simpan adapter (`checkpoints/qwen_qlora/`) + `pilot_generations.txt`.
7. **Sel 13** (opsional) eval token-F1 + ROUGE-L pada val.

> **Pilot dulu, baru full** — hemat kuota GPU & menangkap bug tokenisasi/OOM sebelum
> commit run mahal.

---

## 4. Konfigurasi (T4-safe) — ringkas

| Param | Nilai | Catatan |
|---|---|---|
| loader | `FastLanguageModel` | Qwen3.5-0.8B = model **teks** (bukan VLM) |
| `MODEL_ID` | `unsloth/Qwen3.5-0.8B` | fallback `...-0.8B-Instruct` |
| QLoRA | 4-bit, `r=16/α=32/drop=0.05` | `target_modules` 7 proj |
| `MAX_SEQ_LENGTH` | 1024 | p99 dataset ~634 |
| dtype | fp16 (T4) | `bf16` T4 lemah → `fp16=True` otomatis |
| grad checkpointing | ON (`"unsloth"`) | hemat VRAM |
| `max_grad_norm` | 1.0 | grad clipping |
| batch / accum | 2 / 8 (eff **16**) | TUNABLE |
| LR / scheduler | `1e-4` / cosine | ikut notebook lama |
| epoch (full) | 3 + EarlyStopping | pilot: `max_steps=250` |
| `packing` | **False** | wajib utk `train_on_responses_only` |

**Catatan template Qwen3.5 (penting):** walau `enable_thinking=False`, template
menyisipkan blok kosong `<think>\n\n</think>` setelah `assistant`. Karena itu
`RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"` — scaffold ikut
ter-mask sehingga **loss hanya pada jawaban medis** dan **prompt inferensi konsisten**
(juga berakhir `</think>`). Jangan pakai `assert "<think>" not in text` (akan gagal).

**Vocab Qwen ~248k (besar):** Unsloth otomatis mem-patch cross-entropy (fused/efisien)
→ **tidak perlu `compute_loss_func` manual** (tak ada di notebook lama pun). Kalau muncul
OOM tepat di langkah logits/loss, kecilkan `MAX_SEQ_LENGTH`/batch (lihat bawah).

---

## 5. TROUBLESHOOTING (decision tree)

Ubah **satu** hal per iterasi, ulangi pilot, catat alasannya.

**OOM / out-of-memory**
1. `PER_DEVICE_BATCH` 2 → 1 (naikkan `GRAD_ACCUM` 8 → 16 agar eff batch tetap 16).
2. `MAX_SEQ_LENGTH` 1024 → 768.
3. Pastikan grad checkpointing `"unsloth"` aktif (Sel 6).
> Riwayat proyek: OOM T4 sembuh via single-GPU + turunkan seq-len.

**Loss NaN / meledak (> 2× loss awal)**
1. `LEARNING_RATE` → /3 (mis. `1e-4` → `3.3e-5`).
2. Cek konsistensi dtype: T4 harus `fp16=True` (bukan bf16).
3. Pastikan `max_grad_norm=1.0` (grad clipping) aktif.

**Loss datar / plateau (gagal B2, tanpa NaN)**
1. Naikkan LR moderat (×1.5–2) **atau** tambah step dalam budget.
2. Kalau tetap datar → **curigai label masking** → cek ulang **Sel 8
   (SELF-CHECK decode)**; kalau token jawaban pertama ke-mask / off-by-one → STOP,
   ini bug tokenisasi (butuh mata manusia, jangan paksa lanjut).

**Generasi degeneratif (B3b — >30% 4-gram berulang / ngaco)**
1. Turunkan LR sedikit / tambah step.
2. Saat `generate`, naikkan `repetition_penalty` (mis. 1.1 → 1.2) atau
   `no_repeat_ngram_size`.

**Language mixing (B3a gagal — output campur Inggris/Melayu)**
> Data native-only mestinya **tidak** mixing → indikasi **template/base-model**, bukan
> hyperparam. Lakukan **satu** cek: pastikan chat template & special token Qwen benar
> (Sel 7/8: ada `<|im_start|>`, `enable_thinking=False`, scaffold `</think>`). Kalau
> **masih** mixing setelah itu → **STOP, NEEDS_HUMAN**. Jangan bakar GPU berulang.

---

## 6. Output yang dihasilkan

| Artefak | Isi |
|---|---|
| `checkpoints/qwen_qlora/` | adapter LoRA + tokenizer |
| `pilot_generations.txt` | 10 prompt + output (cek bahasa/kualitas) |
| `trainer.state.log_history` | sumber kurva loss (BAB III) |

Setelah Qwen jadi **baseline sehat**, jalankan Gemma 3 1B IT & Llama 3.2 1B Instruct —
notebook-nya **sudah dibuat** (`train_gemma3_1b_qlora.ipynb`, `train_llama32_1b_qlora.ipynb`)
dengan harness identik; `MODEL_ID` + `RESPONSE_PART`/turn-token per model sudah disetel di
generator. Cukup jalankan pilot → GATE → full seperti Qwen.
