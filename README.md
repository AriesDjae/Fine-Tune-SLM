# Fine-Tune SLM for Medical Chatbot — Qwen3.5-2B (standard dense) vs Gemma 4 E2B (PLE effective-param dense)

Skripsi UII — **Chatbot Medis Offline untuk Puskesmas**. Membandingkan DUA model
~2B dengan strategi parameter berbeda pada **trade-off akurasi vs efisiensi deployment**
(komputasi setara ~2B, total parameter yang dimuat ke RAM berbeda):

| | Model A | Model B |
|---|---|---|
| Model | **Qwen3.5-2B** | **Gemma 4 E2B** |
| Arsitektur | standard dense (~2B aktif = total) | **dense + Per-Layer Embeddings (PLE)** (~5.1B total, ~2.3B aktif) |
| Loading | `FastVisionModel` (VLM, vision OFF), thinking OFF | `FastVisionModel` (VLM, vision OFF), thinking OFF |
| Role `system` | native | native (baru di Gemma 4) |
| Chat format | ChatML `<\|im_start\|>` (template bawaan) | turn-token `<\|turn>…<turn\|>` via `get_chat_template("gemma-4")` |
| HF id (base/finetune) | `unsloth/Qwen3.5-2B` | `unsloth/gemma-4-E2B-it` |

> **Inti kontribusi (RQ2/RQ3):** Gemma 4 E2B mungkin lebih akurat — yang diukur: apakah PLE
> membuatnya **lebih berat di RAM** (tabel embedding total ~5.1B tetap dimuat walau komputasi ~2.3B)
> atau justru **caching PLE meredamnya**, dan apakah Qwen3.5-2B standard dense lebih layak untuk HP RAM 8 GB.
> KEDUA model multimodal → di-load via `FastVisionModel` tapi dilatih **TEXT-ONLY** (`finetune_vision_layers=False`).
> Fakta loader/turn-token/arsitektur diverifikasi dari notebook & doc resmi Unsloth (2026-06).

## Pertanyaan Penelitian
- **RQ1** — Seberapa besar peningkatan akurasi (EM/F1/ROUGE-L) tiap model setelah QLoRA **dibanding baseline** pre-trained?
- **RQ2** — Bagaimana **trade-off** akurasi vs efisiensi (ukuran, RAM load, RAM infer, inference time) antara **standard dense** vs **PLE effective-param dense**?
- **RQ3** — Model mana paling **layak deploy offline** di Puskesmas, mempertimbangkan akurasi **dan** keterbatasan resource?

---

## Pipeline & artefak

```
preprocess_dataset.py ─► Data/processed_final/{train,val,test}_final.jsonl   (LOKAL, CPU, gratis)
        │
        ├─► finetune_qwen35_2b.ipynb   ─► outputs/merged/qwen35-2b-medical/   (COLAB L4)
        └─► finetune_gemma4_e2b.ipynb  ─► outputs/merged/gemma4-e2b-medical/  (COLAB L4)
                │
        eval.py (baseline + finetuned, ×4, 16-bit) ─► results/*.json ─► --summarize  (RQ1)
                │
        export_gguf.py (×2) ─► outputs/gguf/*-Q4_K_M.gguf
                │
        eval.py --gguf (×2, Q4_K_M) ─► results/*_q4.json        (trade-off kuantisasi, Bagian 4)
                │
        benchmark_ondevice.py (×2, CPU-only) ─► results_bench/*.json ─► --summarize  (RQ2/RQ3)
```

| File | Bagian | Fungsi |
|------|--------|--------|
| `preprocessing/preprocess_dataset.py` | 1 | exact-dedup/anti-leakage → **near-dup MinHash LSH (1.2b)** → clean greeting (+ audit over-strip) → quality → ICD/HIDR noise → balance bahasa → reduksi stratified |
| `preprocessing/inspect_noise.py` | 1 | alat inspeksi noise ITERATIF (CPU/lokal); `--simulate` = cek over-stripping |
| `preprocessing/verify_final.py` | 1 | verifikasi mutu file final: dedup/leakage, proporsi bahasa, over-strip → VERDICT |
| `preprocessing/visualize_dataset.py` | 1 | grafik dataset (Bagian A): source, bahasa, panjang token, noise, funnel → `results/figures/*.png` |
| `chat_format.py` | — | **sumber kebenaran** (1) format prompt + (2) `clean_greeting()` post-processing — keduanya identik di train = eval = deploy |
| `notebooks/finetune_{qwen35_2b,gemma4_e2b}.ipynb` | 2 | training Unsloth `FastVisionModel` text-only + `train_on_responses_only` (di-generate oleh `build_finetune_notebooks.py`); meng-**inline** logika format `chat_format.py` (Colab standalone); **grafik training (Bagian B)** loss/data-scaling/LoRA-params → `results/figures/<model>_*.png` |
| `eval.py` | 3, 4 | multi-metrik (MCQA EM / yes-no EM / ROUGE-L+ROUGE-1+F1), baseline×4, per-bahasa, 16-bit & GGUF; **post-clean output** (Task 4, matikan via `--no_postclean`) |
| `export_gguf.py` | 4 | merged 16-bit → GGUF **Q4_K_M** (llama.cpp) |
| `benchmark_ondevice.py` | 5 | RAM load vs infer, throughput, dokumentasi arsitektur (CPU-only) |

---

## Urutan menjalankan

### 0 — Preprocessing (LOKAL, gratis)
```bash
pip install datasketch matplotlib                  # near-dup MinHash LSH (1.2b) + grafik
python preprocessing/preprocess_dataset.py         # -> Data/processed_final/{train,val,test}_final.jsonl (+ results/preprocess_stats.json)
# (opsional) cek noise + over-stripping, verifikasi mutu, dan grafik dataset:
python preprocessing/inspect_noise.py Data/processed_final/train_final.jsonl --simulate
python preprocessing/verify_final.py               # VERDICT dedup/leakage/bahasa/over-strip
python preprocessing/visualize_dataset.py          # -> results/figures/*.png (Bagian A)
```
Lalu **upload seluruh project ke Google Drive** (`MyDrive/Fine-Tune SLM for Medical Chatbot/`),
termasuk `Data/processed_final/` (gitignored — wajib upload manual) dan `chat_format.py`.

### 1–2 — Training (COLAB, **Runtime → GPU L4**)
Buka tiap notebook di Colab, jalankan dari atas. Resume otomatis jika sesi terputus.
`HF_TOKEN` **opsional** (model `unsloth/*` ungated) — set di Colab Secrets hanya untuk push ke Hub.
Output: adapter LoRA + `outputs/merged/<model>-medical/` (16-bit, best-checkpoint).

### 3 — Evaluasi 16-bit (RQ1: baseline vs finetuned)
```bash
python eval.py --model unsloth/Qwen3.5-2B               --model_type qwen  --label qwen_baseline
python eval.py --model outputs/merged/qwen35-2b-medical --model_type qwen  --label qwen_finetuned
python eval.py --model unsloth/gemma-4-E2B-it           --model_type gemma --label gemma_baseline
python eval.py --model outputs/merged/gemma4-e2b-medical --model_type gemma --label gemma_finetuned
python eval.py --summarize results        # tabel + delta (finetuned − baseline)
```

### 4 — Export GGUF Q4_K_M + eval terkuantisasi
```bash
python export_gguf.py --merged_dir outputs/merged/qwen35-2b-medical  --model_type qwen  --verify
python export_gguf.py --merged_dir outputs/merged/gemma4-e2b-medical --model_type gemma --verify
# eval ulang Q4 (protokol identik):
python eval.py --gguf outputs/gguf/qwen35-2b-medical-Q4_K_M.gguf  --model outputs/merged/qwen35-2b-medical  --model_type qwen  --label qwen_q4
python eval.py --gguf outputs/gguf/gemma4-e2b-medical-Q4_K_M.gguf --model outputs/merged/gemma4-e2b-medical --model_type gemma --label gemma_q4
python eval.py --summarize results
```

### 5 — Benchmark on-device (CPU-only → RQ2/RQ3)
```bash
python benchmark_ondevice.py --gguf outputs/gguf/qwen35-2b-medical-Q4_K_M.gguf  --model_type qwen  --label qwen_q4
python benchmark_ondevice.py --gguf outputs/gguf/gemma4-e2b-medical-Q4_K_M.gguf --model_type gemma --label gemma_q4
python benchmark_ondevice.py --summarize results_bench
```
> Jalankan benchmark di **perangkat uji yang sama** (CPU, RAM identik) untuk kedua model,
> dan **sebutkan spesifikasi perangkat** di metodologi.

---

## Estimasi compute (Colab L4, ~65 unit — kasar, verifikasi di run pertama)
| Tahap | GPU | Estimasi |
|-------|-----|----------|
| Preprocessing | — (CPU lokal) | ~1 menit, 0 unit |
| Train Qwen3.5-2B (20k × ≤5 epoch, eff.batch 16) | L4 | ~2–4 jam |
| Train Gemma 4 E2B | L4 | ~2.5–5 jam (total ~5.1B param dimuat) |
| Eval 16-bit ×4 (n=200) | L4 | ~15–30 mnt total |
| Export GGUF ×2 | CPU/L4 | ~10–20 mnt |
| Eval Q4 ×2 + benchmark ×2 | CPU | ~30–60 mnt |

Hemat unit: debug dataset di CPU runtime (0 unit); GPU hanya saat train & eval 16-bit;
install library sekali di awal sesi.

## Dependensi tambahan (di luar `requirements.txt` dasar)
- **Preprocessing (lokal):** `pip install datasketch` (near-dup MinHash LSH, langkah 1.2b).
- **Colab (training):** sel install RESMI Unsloth dengan **versi PINNED** (`transformers==4.56.2`,
  `trl==0.22.2`, dll) + `timm` — lihat sel 1 tiap notebook. **RESTART session** setelah install.
- **Eval/benchmark GGUF:** `pip install llama-cpp-python psutil`.
- **Export GGUF:** `git`+`cmake` (llama.cpp di-clone & di-build otomatis oleh `export_gguf.py`).

## Catatan & risiko
- **Konsistensi prompt** = *silent killer*: training, eval, dan deploy memakai format dari
  `chat_format.py` (notebook meng-inline logika identik). Jangan ubah salah satu saja.
- **Cleaning noise (lever ROI tertinggi, 0 unit GPU):** exact-dedup melaporkan 0 duplikat,
  tapi near-dup MinHash LSH (1.2b) menangkap **118** near-duplicate — **29 di antaranya leakage
  train↔eval** berbasis parafrase yang tak terdeteksi hash exact. Residual sapaan turun dari
  ~1,95% → **~0,07%** (train), tanpa over-stripping konten medis (diverifikasi audit + `inspect_noise --simulate`).
  `clean_greeting()` juga diterapkan pada **output** model saat eval & deploy (jaring pengaman Task 4).
- **`packing=False`**: wajib karena `train_on_responses_only` (loss hanya pada jawaban asisten) butuh
  batas turn yang akurat — packing menggabung sampel sehingga batas turn kabur.
- **Loss multimodal tak sebanding antar arsitektur** (Qwen vs Gemma 4): nilai sukses dari tren loss
  menurun + metrik downstream (`eval.py`), BUKAN nilai loss absolut. Catat ini di BAB IV.
- **TEST-FIRST + SMOKE_TEST**: cek model utuh sebelum PEFT, lalu dry-run 20 step sebelum run penuh (hemat kuota).
- **Best checkpoint**: `load_best_model_at_end=True` → yang di-merge = val_loss terendah (bukan epoch terakhir).
- **Keterbatasan** (untuk BAB V): model 2B tidak mencapai akurasi klinis; halusinasi faktual pasti ada;
  residual sapaan ~0,07% di data (kasus sulit, ditambah post-processing output);
  metrik otomatis tak menangkap kebenaran medis; single-run per model.
  Lihat **Bagian 8 (Log Kelemahan & Limitasi)** di master note.
```
