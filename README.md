# Fine-Tune SLM untuk Chatbot Medis Berbahasa Indonesia

Studi **scaling satu keluarga**: `Qwen3.5` ukuran **0.8B / 2B / 4B**, masing-masing dibandingkan
dengan dirinya sendiri — **baseline (pra-latih)** vs **fine-tuned (LoRA bf16)** pada QnA
pasien–dokter berbahasa Indonesia. Target luaran: jurnal Scopus Q4 + preprint arXiv.

**Sasaran deploy = HP milik masyarakat umum di wilayah tanpa jaringan**, bukan fasilitas
kesehatan. Tidak ada tenaga kesehatan yang menyaring keluaran model, jadi sistem diposisikan
sebagai **informasi dan triase awal, bukan diagnosis**, dan seluruh naskah dibingkai sebagai
**studi kelayakan**.

> Dokumen orientasi lengkap (keputusan, angka, gotcha operasional) ada di `CLAUDE.md`.
> Pivot 1–5 diarsipkan di `arsip/`.

---

## Pertanyaan penelitian

| | Pertanyaan | Status |
|---|---|---|
| RQ1 | Apakah fine-tuning memperbaiki mutu jawaban dibanding baseline? | ✅ selesai — signifikan di tiga ukuran |
| RQ2 | Bagaimana trade-off mutu vs ukuran model? | ✅ selesai |
| RQ3 | Layakkah deploy offline di HP kelas bawah (RAM 3–4 GB)? | ❌ belum — `benchmark_ondevice.py` belum pernah dijalankan |

---

## Hasil utama

Evaluasi kanonik `eval.py` pada n=2998 (`Data/processed_id/test.jsonl`), paired bootstrap 5000×.

| fine-tuned | token-F1 | ROUGE-L | BERTScore |
|---|---|---|---|
| 0.8B | .2965 | .1554 | .6940 |
| 2B | .2956 | .1522 | .6910 |
| **4B** | **.3096** | **.1630** | **.6992** |

Keenam delta (tiga ukuran × baseline/fine-tuned) signifikan, p≈0. **4B unggul, tetapi hanya
~4–5% relatif dengan ~5× parameter.** Bukan bukti plafon tugas: ketiganya dilatih pada 20K
contoh yang sama, jadi bisa saja kelaparan data.

**Plafon manusia.** Kesepakatan antar-dokter pada data yang sama hanya ROUGE-L **0,145** /
token-F1 **0,285** (n=908 pasangan). Ketiga model fine-tuned melampauinya, ketiga baseline di
bawahnya — metrik overlap sudah jenuh untuk tugas ini, dan itulah alasan evaluasi manusia
menjadi keharusan, bukan pelengkap.

**Temuan terkuat (LLM-judge Tier 2, n=200, `gpt-4o`, buta & berpasangan):** fine-tuning
**menolong model kecil dan merusak model besar** — Δ rate "Aman" **+16,5 pp** di 0.8B tetapi
**−17,5 pp** di 2B dan 4B. Arahnya berlawanan dengan metrik overlap. Mekanismenya terukur:
fine-tuning menukar anjuran tegas "segera ke IGD" dengan "konsultasikan ke dokter" (4B: 17,9%
→ 2,1%), dan menghapus disclaimer dari 68–93% menjadi ~0%. Ablation `--strip-disclaimer`
menunjukkan temuan 4B bertahan (−15,5 pp).

---

## Data

| Berkas | Isi | Jumlah |
|---|---|---|
| `Data/processed_id_final/{train,val,test}.jsonl` | data latih (hasil reduksi + audit per baris) | 20000 / 2947 / 2952 |
| `Data/processed_id/test.jsonl` | test set evaluasi (`eval.py`, LLM-judge) | 2998 |
| `Data/processed/` | gabungan mentah 10 sumber (provenance) | 103.857 / 12.982 / 12.982 |

Sumber utama QnA Alodokter, DOI `10.57967/hf/8356`; legalitas ditutup lewat due diligence dan
Pasal 44 UU Hak Cipta. **`Data/` tidak masuk git.**

Karakter data yang menentukan temuan di atas: jawaban referensi memakai register dokter→pasien,
hanya **8,8%** memuat eskalasi eksplisit, **1,3%** menyebut IGD, dan **0,0%** memuat disclaimer.
Dataset ini tidak pernah berisi supervisi keselamatan.

---

## Model & artefak

Ketiganya dilatih di RunPod serverless H100 (`serverless/`, runbook `RUN_SERVERLESS.md`) dengan
hyperparameter **seragam** (LR 1e-4, r=16/α=32, 3 epoch, eff batch 16, MAX_SEQ 1024) — limitasi
yang disebut eksplisit di naskah.

| | eval loss terbaik | checkpoint |
|---|---|---|
| 0.8B | 2.0830 | step 3750 (3 epoch penuh) |
| 2B | 1.9196 | step 2500 (EarlyStopping, ~2,3 epoch) |
| 4B | 1.7723 | step 2500 (EarlyStopping, ~2,3 epoch) |

Adapter, merged 16-bit, dan GGUF (Q4_K_M + Q8_0) ada di HuggingFace `AriesDjaenuri/*`;
ringkasan hasil di `AriesDjaenuri/p6-hasil` bisa dibaca tanpa token.

---

## Evaluasi dokter (Tier 3) — arena penilaian

`hf_space_arena/` adalah satu Space Docker yang memuat halaman penilaian, inferensi, dan
penyimpanan sekaligus. Dokter mengetik satu pertanyaan, menerima **enam jawaban berlabel A–F**
(tiga ukuran × baseline/fine-tuned) dalam urutan yang diacak di server, lalu menilainya dengan
rubrik yang **identik** dengan rubrik Tier 2 — supaya kesepakatan dokter↔GPT-4o bisa dihitung.
Peta label ke nama model tidak ada di dalam Space; tirai dibuka saat analisis.

Dua sumber jawaban, satu protokol (GGUF Q4_K_M, greedy, `seed=42`, pemblokir 3-gram,
`max_tokens=1024`, `n_ctx=4096`):

- **Bank** — jawaban pra-generate di laptop (`scripts/pragenerate_arena.py`), disajikan seketika.
  Sah karena decoding deterministik: jawaban live akan identik.
- **Live** — pertanyaan di luar bank dijalankan di dalam Space.

Penilaian ditulis ke dataset privat `AriesDjaenuri/p6-penilaian-dokter` (`ratings/*.json`,
log sesi di `logs/*.jsonl`).

### Menjalankan Space

```bash
python webapp/uji_konsistensi.py          # gerbang anti-hanyut, WAJIB sebelum deploy
python scripts/space_mode.py status       # hardware, indeks wheel, GPU_LAYERS
python scripts/space_mode.py gpu          # produksi: t4-medium  ($0,60/jam)
python scripts/space_mode.py cpu          # demo:     cpu-upgrade ($0,03/jam)
```

Mengganti hardware di Settings **tanpa** mengganti indeks wheel llama.cpp gagal secara senyap:
wheel CPU menerima `n_gpu_layers=99` tanpa protes lalu tetap menghitung di CPU — tarif GPU,
kecepatan CPU. Karena itu peralihan dilakukan lewat satu perintah, dan langkah hardware-nya
(satu-satunya yang manual) disebutkan di keluaran perintah itu.

Terukur pada satu pertanyaan live, enam model berurutan:

| | total | per model |
|---|---|---|
| `cpu-upgrade` (8 vCPU) | 519 dtk | 47–137 dtk |
| `t4-medium` (T4 16 GB) | **76 dtk** | 1,2–13,3 dtk |

Operator perlu memasang Secret **`HF_TOKEN`** (scope write; tanpa ini penilaian tidak bisa
disimpan dan halaman mengatakannya sejak awal) dan **`KODE_AKSES`**. Job penilaian hidup di
memori proses, jadi **jangan mengganti mode atau menyentuh Settings saat ada sesi berjalan** —
restart menggugurkan jawaban yang sedang dikerjakan.

---

## Struktur berkas

```
├── CLAUDE.md · README.md · RUN_SERVERLESS.md   # orientasi + runbook training
├── skripsi/ · artikel/                         # draf skripsi (ID) + naskah jurnal (EN)
├── eval.py                                     # eval kanonik (token-F1/ROUGE-L/BERTScore)
├── chat_format.py                              # sumber kebenaran format prompt
├── ukur_disclaimer.py                          # rate disclaimer/eskalasi (offline, gratis)
├── llm_judge_tier2.py · judge_run.py           # Tier 2: rubrik, ablation, kappa
├── redflag/ · redflag_annotate.py              # lapisan keselamatan rule-based
├── human_eval/                                 # rubrik dokter, gold set, bank pertanyaan
├── arena/ · hf_space_arena/ · scripts/space_mode.py   # arena penilaian dokter (Tier 3)
├── benchmark_ondevice.py                       # ⚠ masih artefak Pivot 3 — perlu ditulis ulang
├── serverless/ · handler.py · Dockerfile       # harness training H100
├── notebooks/ · preprocessing/ · inference/
├── results/                                    # semua hasil (JSON ringkas ter-track)
└── arsip/                                      # pivot 1–5
```

Gitignored: `Data/`, `outputs/`, `results/**/*.jsonl`, `*.key`, berkas bank jawaban
(`webapp/bank/`, `hf_space_arena/bank/`, `human_eval/bank_pertanyaan.csv` — teks Alodokter
verbatim, aturannya sama dengan `Data/`).

---

## Sisa pekerjaan, urut dampak

1. **RQ3 — benchmark on-device.** Gratis, dampak terbesar, dan naskah belum menjawab
   pertanyaannya sendiri. `llama-cpp-python` sudah terpasang; `benchmark_ondevice.py` perlu
   ditulis ulang untuk tiga ukuran Qwen, idealnya diulang di HP 3–4 GB lewat Termux.
2. **Penulisan naskah** — faktor tunggal terbesar di tier Q4.
3. **Evaluasi dokter (Tier 3)** n=10–20 lewat arena, plus 400 baris anotasi red-flag.
4. Penilai kedua lintas-galur (butuh OpenRouter/Anthropic berbayar; Gemini tier gratis tidak
   layak, 20 permintaan/hari).
5. Replikasi seed untuk anomali 2B.

**Limitasi yang wajib disebut:** data latih 20K · hyperparameter seragam tanpa tuning per
ukuran · anggaran epoch tidak sama antar ukuran · satu penilai dan satu seed · ROUGE-L absolut
masih di bawah target 0,30.

---

## Catatan operasional

- `UNSLOTH_COMPILE_DISABLE=1` wajib di-set **sebelum** `import unsloth` untuk generate/eval.
- `save_pretrained_gguf(dir)` menulis ke folder ber-suffix `_gguf` dan ikut menghasilkan berkas
  `*mmproj*` (proyektor vision) — abaikan untuk deploy teks.
- Konfigurasi 0.8B tetap multimodal (`Qwen3_5ForConditionalGeneration` + `vision_config`), jadi
  warning bobot `UNEXPECTED` saat load = vision tower + kepala MTP; jinak.
- Konsol Windows cp1252: skrip yang mencetak `Δ` atau `→` memaksa stdout UTF-8, kalau tidak ia
  mati setelah pekerjaan terkirim dan tampak seperti kegagalan.
