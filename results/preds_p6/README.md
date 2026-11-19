# Prediksi mentah eval Pivot 6 (`preds/`, 6 file, ~20 MB)

Isi folder `preds/` **tidak di-track git** (ikut konvensi repo: artefak berat di HF/Drive,
git hanya kode + catatan + hasil ringan). Arsip resminya ada di HuggingFace:

```
AriesDjaenuri/p6-hasil  →  preds/<label>.preds.json
```

Repo HF itu dibatasi aksesnya; set `HF_TOKEN` berisi token baca dari penulis, lalu tarik ulang:

```python
from huggingface_hub import hf_hub_download
for label in ["qwen0.8b_baseline", "qwen0.8b_finetuned",
              "qwen2b_baseline",   "qwen2b_finetuned",
              "qwen4b_baseline",   "qwen4b_finetuned"]:
    hf_hub_download("AriesDjaenuri/p6-hasil", f"preds/{label}.preds.json",
                    local_dir="results/preds_p6")
```

## Skema file

```jsonc
{
  "model": "AriesDjaenuri/qwen35-0.8b-medical-id",  // atau base utk baseline
  "n": 2998,
  "batch_size": 16,
  "max_new_tokens": 256,
  "preds": ["<teks jawaban model>", ...]            // 2998 string, MENTAH (pra-clean_greeting)
}
```

## Cara memasangkan dengan pertanyaan/referensi

`preds[i]` sejajar dengan **baris ke-i** `Data/processed_id/test.jsonl` (2998 baris).
Eval **tidak** melakukan shuffle dan `N_EVAL = 2998` = seluruh test set, jadi urutannya
identik dengan urutan file. Tiap baris test punya `messages` = `system` → `user` → `assistant`;
`user` = pertanyaan, `assistant` = jawaban referensi (dokter Alodokter).

## Catatan penting

- **Definisi token-F1 untuk naskah = `eval.py`** (`\w+` + `.lower()`), BUKAN `pred.split()`
  yang dipakai notebook eval. Perbedaannya ~0,067 absolut; ROUGE-L & BERTScore tidak terpengaruh.
- Prediksi disimpan **pra-`clean_greeting`**; re-apply cleaner bila ingin mereproduksi angka
  yang dilaporkan.
