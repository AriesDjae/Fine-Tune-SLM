# Handoff — Task C2 (eval Q4_K_M + quantization gap)

Tanggal: 2026-06-22 · dari Claude Code (lokal) ke peneliti / Colab.
Status Task A, B, C1, C3 = **SELESAI** lokal. Hanya **C2** yang ditunda ke Colab/Linux
(alasan: `llama-cpp-python` tak bisa di-build di Windows lokal — tanpa compiler; dan
wheel prebuilt kemungkinan terlalu lama utk arsitektur GatedDeltaNet Qwen3.5).

---

## 1. Hasil yang SUDAH ada (jangan diulang)

Eval 16-bit FULL test set `Data/processed_id/test.jsonl` (n=2998, greedy, +BERTScore),
tersimpan di `results/eval_pivot5/`:

| model | token-F1 | ROUGE-L | BERTScore |
|-------|---------:|--------:|----------:|
| `qwen_baseline`  (zero-shot)        | 0.261 | 0.118 | 0.659 |
| `qwen_finetuned` (QLoRA, base+adapter) | 0.306 | 0.163 | 0.698 |
| **delta (finetuned − baseline)**    | **+0.045** | **+0.045** | **+0.039** |

- Lval (best `eval_loss`, checkpoint-5628) = **2.0754** (dari `results/qwen35_0_8b_trainer_state.json`).
- ROUGE-L target >0.30 **belum tercapai** (finetuned 0.163), tapi uplift konsisten di 3 metrik.
- Semua sampel ID (dataset Indonesia-only; `lang:en` kosong = wajar).

## 2. Artefak GGUF yang SUDAH dibuat lokal (Task B, load-tested)

- merged 16-bit : `outputs/merged/qwen35-0.8b-medical/` (1.5 GB)
- f16 GGUF      : `outputs/gguf/qwen35-0.8b-medical-f16.gguf`
- **Q4_K_M GGUF**: `outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf` (529 MB) — lolos uji-load
  (llama-completion.exe b9756, output ID koheren).

> **GOTCHA penting kalau convert ulang:** default `convert_hf_to_gguf.py` membundel
> MTP head Qwen3.5 (block_count=25, `nextn_predict_layers=1`) → runtime error
> `missing tensor 'blk.24.attn_norm.weight'`. **WAJIB pakai flag `--no-mtp`** (MTP hanya
> utk speculative draft, tak dibutuhkan) agar block_count=24 & model load bersih.
> Converter mendukung arsitektur `Qwen3_5ForCausalLM` (llama.cpp master, `conversion/qwen.py`).

## 3. Yang harus dikerjakan di Colab/Linux (C2)

Upload ke Drive proyek: `Q4_K_M.gguf` + sebuah dir tokenizer (cukup
`outputs/checkpoints/qwen35-0.8b-train/` yang punya `tokenizer.json` + `chat_template.jinja`,
atau `outputs/merged/qwen35-0.8b-medical/`) + `eval.py`, `chat_format.py`,
`Data/processed_id/test.jsonl`, dan folder `results/eval_pivot5/` (berisi
`qwen_finetuned.json` agar gap bisa dipasangkan).

```bash
pip install -q llama-cpp-python   # di Linux GPU build mulus (CUDA: CMAKE_ARGS="-DGGML_CUDA=on")

# Eval Q4_K_M — PROTOKOL IDENTIK dgn run 16-bit (seed 42, n_eval 3000, greedy, processed_id):
python eval.py --gguf outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf \
               --model outputs/checkpoints/qwen35-0.8b-train \
               --label qwen_finetuned_q4 --loader gguf \
               --test_file Data/processed_id/test.jsonl \
               --n_eval 3000 --bertscore \
               --out results/eval_pivot5/qwen_finetuned_q4.json

# Ringkas → tabel + delta uplift + QUANTIZATION GAP (Q4 − 16bit):
python eval.py --summarize results/eval_pivot5
```

- Label **harus** `qwen_finetuned_q4` agar `summarize` memasangkannya dgn `qwen_finetuned`
  (16-bit) → baris "QUANTIZATION GAP (Q4_K_M − 16bit)".
- Jaga protokol identik: `--n_eval 3000`, `--seed 42` (default), `Data/processed_id/test.jsonl`.
- Catatan fairness: backend GGUF (llama.cpp) vs HF mungkin beda kecil dlm decoding; eval.py
  sudah pakai greedy (temp 0, top_k 1) di kedua sisi. Laporkan gap apa adanya.

## 4. Alternatif lokal (kalau tak mau Colab)

Binari prebuilt b9756 sudah ada di `./llama_bin/` (CPU x64) dan llama.cpp master di
`./llama.cpp/`. Untuk eval Q4 lokal perlu menambah jalur "gguf via CLI" ke `eval.py`
(panggil `llama_bin/llama-completion.exe`) — CPU-only, lambat utk 2998 sampel (pakai subset).
Belum dikerjakan sesuai keputusan peneliti (pilih Colab).
