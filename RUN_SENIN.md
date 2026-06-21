# RUN SENIN — Qwen3.5-0.8B: baseline vs fine-tuned (Pivot 5)

Objektif TUNGGAL: **satu** model `Qwen3.5-0.8B`, baseline (pre-trained) vs fine-tuned (QLoRA).
Test set: `Data/processed_id/test.jsonl` (**2998** sampel → `--n_eval 3000` = full).
Metrik open-ended: **token-F1 & ROUGE-L** (target ROUGE-L ≥ 0.30).

Artefak lokal sudah aman & terverifikasi:
- adapter `outputs/checkpoints/qwen35-0.8b-train/` (25.5 MB) + `checkpoint-5628` (best, eval_loss 2.0754)
- kurva loss `results/figures/qwen35_0_8b_loss_curve.png`

---

> ⚡ **CARA TERMUDAH: buka `notebooks/run_eval_pivot5.ipynb` di Colab** — notebook itu
> mount Drive, install dep, lalu memanggil semua skrip di bawah secara berurutan.
> Langkah 1–6 di bawah = isi sel-sel notebook tsb (untuk referensi manual).

## 0. Lingkungan (Colab L4/T4 — sama spt training)
```bash
pip install unsloth unsloth_zoo bert-score   # bert-score = metrik semantik (opsional)
# untuk eval GGUF & benchmark:
pip install llama-cpp-python
```
> Lokal: `.venv-gpu` punya CUDA tapi BELUM ada unsloth/peft/transformers. Merge & eval
> 16-bit paling mulus di Colab. (Eval GGUF/benchmark bisa CPU lokal jika llama.cpp dibuild.)
>
> **BERTScore**: tambah flag `--bertscore` di tiap perintah eval (pakai
> `bert-base-multilingual-cased`, relevan utk teks Indonesia). Tanpa flag → hanya token-F1 & ROUGE-L.

## 1. Eval BASELINE (16-bit) — untuk delta
```bash
python eval.py --model unsloth/Qwen3.5-0.8B \
               --label qwen08_baseline --n_eval 3000 --bertscore
```

## 2. Eval FINE-TUNED 16-bit — referensi quantization gap
Adapter dimuat langsung (tanpa merge), Unsloth auto-load base+adapter:
```bash
python eval.py --model outputs/checkpoints/qwen35-0.8b-train \
               --label qwen08_finetuned --n_eval 3000 --bertscore
```

## 3. Merge adapter → 16-bit (WAJIB sebelum GGUF)
```bash
python merge_adapter.py \
    --adapter outputs/checkpoints/qwen35-0.8b-train \
    --out     outputs/merged/qwen35-0.8b-medical
```

## 4. Export GGUF Q4_K_M
```bash
python export_gguf.py --merged_dir outputs/merged/qwen35-0.8b-medical --verify
# hasil: outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf
```
> Simpan `.gguf` + `outputs/merged/` ke Drive.

## 5. Eval FINE-TUNED Q4_K_M — angka deployment jujur
```bash
python eval.py --gguf outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf \
               --model outputs/merged/qwen35-0.8b-medical \
               --label qwen08_finetuned_q4 --loader gguf --n_eval 3000 --bertscore
```

## 6. Ringkas → tabel delta + quantization gap + cek ROUGE-L ≥ 0.30
```bash
python eval.py --summarize results
```
Output mencetak:
- Tabel per-run (overall / lang:id / lang:en / open) = `tokenF1 / rougeL`
- **PENINGKATAN** (finetuned − baseline) → RQ1
- **QUANTIZATION GAP** (Q4 − 16bit) → tabel trade-off kuantisasi
- **CEK TARGET ROUGE-L ≥ 0.30** → OK / BELUM per run

---

## Bahan bimbingan (checklist)
- [x] Kurva loss train/val → `results/figures/qwen35_0_8b_loss_curve.png`
- [ ] Tabel delta base vs fine-tuned ← langkah 6
- [ ] Tabel quantization gap 16-bit vs Q4 ← langkah 6
- [ ] Status ROUGE-L ≥ 0.30 ← langkah 6
- [ ] (opsional) benchmark HP: tokens/sec + RAM, Qwen Q4

## Catatan label (penting)
Delta otomatis butuh pasangan `X_baseline` ↔ `X_finetuned` (prefix sama).
Quantization gap butuh `X_finetuned` ↔ `X_finetuned_q4` (suffix `_q4`).
