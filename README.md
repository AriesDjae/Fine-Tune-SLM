# Fine-Tune SLM untuk Chatbot Medis Berbahasa Indonesia

Studi **scaling satu keluarga**: `Qwen3.5` ukuran **0.8B / 2B / 4B**, masing-masing dibandingkan
dengan dirinya sendiri — **baseline (pra-latih)** vs **fine-tuned (LoRA bf16)** pada QnA
pasien–dokter berbahasa Indonesia. Repo ini berisi kode, prompt sistem, rubrik penilai, dan
skrip analisis untuk naskah jurnal terkait.

> [!WARNING]
> **Prototipe riset — bukan untuk penggunaan klinis.** Model dan kode di repo ini dibuat untuk
> studi kelayakan. Jawabannya bisa keliru, tidak lengkap, atau berbahaya, dan tidak pernah
> disaring tenaga kesehatan. Jangan memakainya untuk diagnosis, pengobatan, atau keputusan
> medis apa pun. Dalam keadaan darurat, segera hubungi fasilitas kesehatan terdekat atau 119.
>
> *Research prototype — not for clinical use. Do not use it for diagnosis, treatment, or any
> medical decision.*

**Sasaran deploy = HP milik masyarakat umum di wilayah tanpa jaringan**, bukan fasilitas
kesehatan. Tidak ada tenaga kesehatan yang menyaring keluaran model, jadi sistem diposisikan
sebagai **informasi dan triase awal, bukan diagnosis**, dan seluruh naskah dibingkai sebagai
**studi kelayakan**.

### Yang tidak ada di repo ini

- **Bobot model** (adapter LoRA, merged 16-bit, GGUF): diarsipkan di Hugging Face
  `AriesDjaenuri/*`.
- **Jawaban model untuk 2.998 soal tes**: diarsipkan di Hugging Face `AriesDjaenuri/p6-hasil`
  (lihat `results/preds_p6/README.md`).
- **Dataset**: teks QnA Alodokter, tersedia dari sumber aslinya (DOI `10.57967/hf/8356`).
- **Data penilaian dokter (Tier 3)**: disimpan privat untuk melindungi identitas penilai. Repo
  hanya memuat ringkasan agregat.

---

## Untuk tester

> ⚠ **Ini bukan alat diagnosis.** Jangan memakai jawabannya untuk keputusan medis nyata, dan
> jangan memasukkan data kesehatan pribadi yang sungguhan saat menguji.

### 1. Akses

Repo model di Hugging Face `AriesDjaenuri/*` dibatasi aksesnya (alasan etis: model medis tanpa
penyaring tenaga kesehatan). Minta **token baca** ke penulis, cukup untuk tiga repo `*-GGUF`.
Token itu untuk Anda sendiri; jangan dibagikan atau di-commit.

### 2. Menjalankan chatbot di laptop (CPU, offline setelah model terunduh)

```bash
git clone https://github.com/AriesDjae/Fine-Tune-SLM.git
cd Fine-Tune-SLM
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install numpy huggingface_hub
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

export HF_TOKEN=hf_xxx                 # Windows PowerShell: $env:HF_TOKEN = "hf_xxx"
hf download AriesDjaenuri/qwen35-0.8b-medical-id-GGUF Qwen3.5-0.8B.Q4_K_M.gguf --local-dir outputs/gguf

python demo_chat.py --ukuran 0.8b      # ketik pertanyaan; 'keluar' untuk berhenti
```

Kalau `pip install llama-cpp-python` mulai mengompilasi dari sumber, artinya belum ada wheel
untuk versi Python Anda; di Windows kompilasi itu butuh Visual Studio Build Tools.

| ukuran | repo GGUF | berkas | ukuran berkas | puncak RAM (laptop) |
|---|---|---|---|---|
| 0.8B | `AriesDjaenuri/qwen35-0.8b-medical-id-GGUF` | `Qwen3.5-0.8B.Q4_K_M.gguf` | 0,54 GB | 0,99 GB |
| 2B | `AriesDjaenuri/qwen35-2b-medical-id-GGUF` | `Qwen3.5-2B.Q4_K_M.gguf` | 1,31 GB | 2,17 GB |
| 4B | `AriesDjaenuri/qwen35-4b-medical-id-GGUF` | `Qwen3.5-4B.Q4_K_M.gguf` | 2,78 GB | 4,78 GB |

`demo_chat.py` memakai jalur yang sama dengan benchmark naskah: prompt yang sama dengan saat
training, pengaman pengulangan DRY+ (`pengaman_decoding.py`), dan lapisan red-flag (`redflag/`).
Opsi `--contoh N` membaca `pertanyaan_bench.json`, yang **tidak** ada di git (berisi teks
Alodokter apa adanya), jadi tester cukup mengetik pertanyaan sendiri.

### 3. Menjalankan di HP Android (Termux)

Ikuti `RUN_TERMUX.md`. Karena repo HF kini privat, tambahkan
`-H "Authorization: Bearer $HF_TOKEN"` pada setiap perintah `curl` ke `huggingface.co` di
runbook itu (termasuk unduhan `tools/bench_hp.zip`). Mulai dari 0.8B; 4B butuh HP RAM ≥6 GB
dan sekitar tiga menit per jawaban.

### 4. Melaporkan temuan

Buka **GitHub Issue**, satu temuan per issue. Sertakan ukuran model, pertanyaan, jawaban
lengkap (jalankan dengan `--simpan hasil.json` supaya bisa ditempel), dan apa yang salah.
Yang paling dicari:

- menyarankan tindakan berisiko tanpa peringatan;
- gagal menganjurkan ke IGD untuk tanda bahaya (sesak, nyeri dada, perdarahan, kejang, dsb.);
- informasi medis yang keliru atau menyesatkan bagi orang awam;
- jawaban yang mengulang kalimat, terpotong, atau kata yang rusak.

---

## Pertanyaan penelitian

| | Pertanyaan | Status |
|---|---|---|
| RQ1 | Apakah fine-tuning memperbaiki mutu jawaban dibanding baseline? | ✅ selesai — signifikan di tiga ukuran |
| RQ2 | Bagaimana trade-off mutu vs ukuran model? | ✅ selesai |
| RQ3 | Layakkah deploy offline di HP kelas bawah (RAM 3–4 GB)? | ✅ selesai — 0.8B dan 2B lolos, 4B tidak |

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

**Plafon manusia** (`plafon_manusia.py` → `results/plafon_manusia.json`). Kesepakatan
antar-dokter pada data yang sama hanya token-F1 **0,2972** [0,2916; 0,3028] / ROUGE-L
**0,1496** / BERTScore **0,6867** (n=1.093 pasangan). Ketiga model fine-tuned mencapai plafon
itu (pada token-F1 hanya 4B yang melampauinya), seluruh baseline di bawahnya — metrik overlap
sudah jenuh untuk tugas ini, dan itulah alasan evaluasi manusia menjadi keharusan.

**Temuan terkuat (LLM-judge Tier 2, n=200, `gpt-4o`, buta & berpasangan):** fine-tuning
**menolong model kecil dan merusak model besar** — Δ rate "Aman" **+16,5 pp** di 0.8B tetapi
**−17,5 pp** di 2B dan 4B. Arahnya berlawanan dengan metrik overlap. Mekanismenya terukur:
fine-tuning menukar anjuran tegas "segera ke IGD" dengan "konsultasikan ke dokter" (4B: 17,9%
→ 2,1%), dan menghapus disclaimer dari 68–93% menjadi ~0%. Ablation `--strip-disclaimer`
menunjukkan temuan 4B bertahan (−15,5 pp).

**On-device (RQ3)** — HP POCO X3 GT, `llama.cpp` GGUF Q4_K_M, 4 thread, pengaman DRY+
(`results_bench_hp/`):

| | puncak RAM | latensi per jawaban | tok/dtk |
|---|---|---|---|
| 0.8B | 0,75 GB | 45,6 dtk | 8,0 |
| 2B | 1,50 GB | 87,5 dtk | 3,1 |
| 4B | 2,99 GB | 187 dtk | 1,3 |

0.8B dan 2B muat di HP kelas 3–4 GB; 4B baru muat di 6 GB dan terlalu lambat. Kuantisasi
Q4_K_M tidak menurunkan token-F1/ROUGE-L/BERTScore di ukuran mana pun dibanding 16-bit
(n=400 per ukuran, `results/quant_gap/ringkasan.json`).

---

## Data

| Berkas | Isi | Jumlah |
|---|---|---|
| `Data/processed_id_final/{train,val,test}.jsonl` | data latih (hasil reduksi + audit per baris) | 20000 / 2947 / 2952 |
| `Data/processed_id/test.jsonl` | test set evaluasi (`eval.py`, LLM-judge) | 2998 |
| `Data/processed/` | gabungan mentah 10 sumber (provenance) | 103.857 / 12.982 / 12.982 |

Sumber utama QnA Alodokter, DOI `10.57967/hf/8356`; legalitas ditutup lewat due diligence dan
Pasal 44 UU Hak Cipta. **`Data/` tidak masuk git.**

Karakter data yang menentukan temuan di atas: jawaban referensi memakai register dokter→pasien.
Dari 20.000 jawaban latih (`ukur_disclaimer.py`), hanya **5,3%** memuat eskalasi segera,
**1,3%** menyebut IGD, dan **0,2%** memuat disclaimer. Dataset ini hampir tidak berisi supervisi
keselamatan.

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

Adapter, merged 16-bit, dan GGUF (Q4_K_M + Q8_0) ada di Hugging Face `AriesDjaenuri/*`,
begitu pula jawaban keenam konfigurasi untuk 2.998 soal tes (`AriesDjaenuri/p6-hasil`). Tidak
ada bobot model maupun berkas jawaban lengkap di repo GitHub ini.

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
log sesi di `logs/*.jsonl`). Data penilaian ini **tidak dibagikan** untuk melindungi identitas
penilai; repo hanya memuat ringkasan agregat (`results/tier3/ringkasan.json`).

### Menjalankan Space

```bash
python scripts/uji_konsistensi.py         # gerbang anti-hanyut, WAJIB sebelum deploy
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
├── README.md · LICENSE · RUN_SERVERLESS.md     # orientasi + runbook training
├── RUN_TERMUX.md                               # runbook benchmark/demo di HP Android
├── eval.py                                     # eval kanonik (token-F1/ROUGE-L/BERTScore)
├── chat_format.py                              # sumber kebenaran format prompt
├── ukur_disclaimer.py                          # rate disclaimer/eskalasi (offline, gratis)
├── llm_judge_tier2.py · judge_run.py           # Tier 2: rubrik, ablation, kappa
├── redflag/ · redflag_annotate.py              # lapisan keselamatan rule-based
├── human_eval/                                 # rubrik dokter + indeks sampel (tanpa teks soal)
├── arena/ · hf_space_arena/ · scripts/space_mode.py   # arena penilaian dokter (Tier 3)
├── benchmark_ondevice.py · demo_chat.py        # RQ3 on-device (HP/laptop) + peraga chat
├── pengaman_decoding.py                        # pengaman pengulangan DRY+ untuk deploy
├── serverless/ · handler.py · Dockerfile       # harness training H100
├── notebooks/ · preprocessing/
└── results/ · results_bench/ · results_bench_hp/   # hasil ringkas (JSON, tanpa teks soal)
```

Tidak di-track (lihat `.gitignore`): `Data/`, bobot model, `results/**/*.jsonl`, jawaban model
lengkap, data penilaian dokter, `*.key`, dan semua berkas yang memuat teks Alodokter verbatim
(bank soal arena, `human_eval/bank_pertanyaan.csv`, daftar soal red-flag dan benchmark). Skrip
yang membutuhkannya membuat ulang berkas itu dari dataset sumber.

---

## Keterbatasan

Data latih 20K · hyperparameter seragam tanpa tuning per ukuran · anggaran epoch tidak sama
antar ukuran · satu penilai LLM dan satu seed · ROUGE-L absolut masih di bawah 0,30.

---

## Lisensi

Kode dirilis dengan lisensi [MIT](LICENSE). Lisensi ini tidak mencakup dataset sumber maupun
bobot model; masing-masing mengikuti ketentuan sumbernya.

---

## Catatan operasional

- `UNSLOTH_COMPILE_DISABLE=1` wajib di-set **sebelum** `import unsloth` untuk generate/eval.
- `save_pretrained_gguf(dir)` menulis ke folder ber-suffix `_gguf` dan ikut menghasilkan berkas
  `*mmproj*` (proyektor vision) — abaikan untuk deploy teks.
- Konfigurasi 0.8B tetap multimodal (`Qwen3_5ForConditionalGeneration` + `vision_config`), jadi
  warning bobot `UNEXPECTED` saat load = vision tower + kepala MTP; jinak.
- Konsol Windows cp1252: skrip yang mencetak `Δ` atau `→` memaksa stdout UTF-8, kalau tidak ia
  mati setelah pekerjaan terkirim dan tampak seperti kegagalan.
