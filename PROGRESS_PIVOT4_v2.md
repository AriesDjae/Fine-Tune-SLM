# PROGRESS — Pivot 4 Dataset (Strict Clean v2)
_Update terakhir: 2026-06-17 (sesi siang) — **STRICT CLEAN v2 SELESAI & QC BERSIH**._

> ## ✅ STATUS: SELESAI sampai QC (2026-06-17)
> Dataset final `Data/processed_id/{train,val,test}.jsonl` = **30.004 / 2.998 / 2.998**.
> **Verifier independen: VERDICT BERSIH — 0,0% sisa SEMUA kategori** (NER-person, B1 nama/dr/greeting/
> leaked, B2-B6) di ketiga split (`results/verify_v2_20260617.json`). **MAX_SEQ_LENGTH = 1024**.
> Artefak: DATA_CARD (bagian "Strict Clean v2"), spot_check_test_30.txt, results/figures/fig_v2_*.png.
> **NEXT (nanti):** peneliti ACC spot-check 30 sampel → baru wire ke notebook training (3 model ≤1B).
>
> Detail teknis durable ada di memory `pivot4-id-native.md`. Bagian di bawah = catatan proses (historis).

Ringkasan: bangun dataset medis **Indonesia-only, native (indonesia_qna/Alodokter)**, target
**train 30K / val 3K / test 3K**, untuk fine-tune 3 model ≤1B (Gemma 3 1B IT, Qwen3.5 0.8B,
Llama 3.2 1B). EM di-drop → eval **F1 + ROUGE-L**. Semua CPU, seed=42, TANPA re-embedding.

---

## ✅ SUDAH SELESAI
1. **Audit** (`preprocessing/audit_dataset.py` → `results/audit_dataset.json`): premis "ID <5%" SALAH;
   audit fasttext/langdetect membuktikan **ID 23–30%**, `indonesia_qna`=99.98–100% id, pool native abundant.
2. **Pipeline v1** (`preprocessing/build_id_dataset.py`, 12-gerbang + e5 relevance) — pernah menghasilkan 36K,
   TAPI 36K final itu **tertimpa** saat run `--debug` (bug, sudah diperbaiki). Yang SELAMAT & dipakai sebagai
   bahan baku v2: **`Data/processed_id/_intermediate/06_scored.jsonl` = 393,734 sampel** (deduped + ber-komposit,
   TANPA embedding) + `review_borderline.jsonl` (20,483).
3. **Audit manual 30 sampel oleh peneliti** menemukan ~50% artefak di output v1 (sign-off "dr. Nama" 30%,
   ekor artikel 7%, titled name 11%, placeholder [NAMA], emoji 3%, kata-nyambung 7%, dll). Verifier v1
   "0 leaks" ternyata PALSU (circular: pakai pola yang sama dgn stripper).
4. **`preprocessing/strict_clean_v2.py`** dibangun (B1–B6 + NER cahya IndoBERT + B7 + re-select, baca dari
   06_scored, no re-embed). Unit-test cleaning LULUS (semua contoh audit bersih, guard tak over-strip).
   **Bug debug-menimpa-produksi SUDAH diperbaiki** (debug → tulis `_intermediate/*_debug.jsonl`).
5. **`preprocessing/verify_dataset.py`** (verifier INDEPENDEN: NER + pola B1–B6, lapor % sisa) — dibuat, belum dijalankan.
6. **`preprocessing/check_token_lengths.py`** (panjang token per-3-model → MAX_SEQ_LENGTH) — dibuat, belum dijalankan.
7. **Dependency terpasang**: `fasttext-wheel` + `Data/lid.176.bin`(131MB), `sentence-transformers`+`torch 2.12 CPU`,
   `ftfy`, `spacy`(tak dipakai—96% FP), model NER `cahya/bert-base-indonesian-NER`(ter-download).

---

## ⚠️ STATE SAAT INI (PENTING)
- **`Data/processed_id/{train,val,test}.jsonl` SAAT INI TIDAK VALID** (16.964/1.696/1.696) — sisa run yang
  pool-nya keburu ketimpa (cuma 21K). **JANGAN dipakai.** Akan ditimpa run v2 penuh besok.
- **Bahan baku valid = `_intermediate/06_scored.jsonl` (393.734).** Jangan dihapus.
- Run v2 penuh kemarin **di-kill** karena NER CPU terlalu lambat.

## 🔑 TEMUAN TEKNIS (jangan diulang dari nol)
- **NER cahya di CPU ≈ 31 doc/dtk.** 120K doc ≈ ~64 mnt. → di v2 sudah dibatasi: **pre-cap 44K + NER pada ANSWER saja**
  (≈44K doc ≈ ~24 mnt). Pertanyaan ditangani PII-regex + verifier.
- cahya NER **false-positive** pada token gaul (`dok`,`sya`,`trims`,`ass`) → filter ketat: **score≥0.90, len≥4, stoplist**.
- spaCy `xx_ent_wiki_sm` **tidak berguna** (96% FP). 
- fasttext lid **mencampur id/ms** → terima `{id, ms}` sebagai Indonesia.
- GPU RTX 4050 nganggur TAPI torch=CPU-only; reinstall CUDA **berisiko** (torch 2.12 versi ganjil, tak ada wheel cu cocok) → tetap CPU.

---

## ⏭️ TASK BESOK (urut; perintah siap pakai)
1. **Jalankan strict clean v2 penuh** (≈25 mnt, background):
   ```
   python preprocessing/strict_clean_v2.py
   ```
   → tulis `Data/processed_id/{train,val,test}.jsonl` (target 30002/2999/2999) + `dataset_stats.json`
   + `results/strict_clean_v2_log_YYYYMMDD.json`. Cek funnel: pool 393K → clean → precap 44K → NER → 36K.
2. **Verifier independen** (≈40 mnt, NER pada 36K×2 — pertimbangkan jalankan background):
   ```
   python preprocessing/verify_dataset.py
   ```
   → target **0%** sisa semua kategori B1–B5 + NER PERSON. Kalau masih ada sisa, perketat pola di
   `strict_clean_v2.py` lalu ulang langkah 1.
3. **Cek panjang token** → tetapkan MAX_SEQ_LENGTH (sama utk 3 model):
   ```
   python preprocessing/check_token_lengths.py
   ```
   → CATATAN: tokenizer Gemma/Llama mungkin gated; script coba mirror `unsloth/*` dulu. Kalau gagal,
   `huggingface-cli login` / set HF_TOKEN. Lapor median/p90/p95/p99/max + jumlah truncate.
4. **Cetak 30 sampel test BARU** (seed=42) untuk dibaca peneliti:
   ```
   python - <<'PY'
   import json,random
   t=[json.loads(l) for l in open("Data/processed_id/test.jsonl",encoding="utf-8")]
   out=[]
   for i,o in enumerate(random.Random(42).sample(t,30),1):
       out+=["="*100,f"[{i}/30] domain={o['domain']} score={o.get('quality_score')}",
             "Q: "+o["messages"][1]["content"],"","A: "+o["messages"][2]["content"],""]
   open("Data/processed_id/spot_check_test_30.txt","w",encoding="utf-8").write("\n".join(out))
   print("written Data/processed_id/spot_check_test_30.txt")
   PY
   ```
5. **Update `Data/processed_id/DATA_CARD.txt`**: tambah bagian "Strict cleaning v2" (temuan audit manual,
   perbaikan NER + tail/timestamp stripping, hasil verifier, hasil panjang token + MAX_SEQ_LENGTH terpilih).
6. **TAHAN** wire ke notebook training sampai peneliti ACC spot-check v2.

### Opsional (kalau NER CPU masih terlalu lama)
- Kecilkan `precap` di `strict_clean_v2.py` (var `precap`, kini 44000) — tapi jaga ≥ ~42K agar tetap dapat 36K.
- Untuk verifier, kalau 36K×2 kelamaan, sampling subset BESAR (mis. 10K) + catat sbg keterbatasan
  (idealnya tetap SELURUH sesuai brief).
