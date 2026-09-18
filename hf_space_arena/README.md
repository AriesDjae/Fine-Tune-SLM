---
title: Arena Penilaian Dokter
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Penilaian buta 6 model SLM medis oleh dokter
---

# Arena penilaian dokter — Pivot 6

Dokter mengetik satu pertanyaan kesehatan, mendapat **enam jawaban berlabel A–F**, lalu
menilainya. Label tidak pernah dipetakan ke nama model di dalam Space ini; tirai dibuka
belakangan saat analisis.

Menggantikan pasangan Vercel + RunPod: halamannya berkas yang sama (`public/index.html`),
kontrak API-nya sama, hanya saja inferensi dan penyimpanan kini di satu tempat.

## Dua sumber jawaban, satu protokol

| | |
|---|---|
| **Bank** | Jawaban pra-generate di laptop peneliti, di `bank/<versi_protokol>/<hash>.json`. Disajikan seketika, tanpa inferensi. Sah karena decoding deterministik — jawaban live akan identik. |
| **Live** | Pertanyaan di luar bank dijalankan di sini, enam model berurutan. |

Protokol: GGUF **Q4_K_M**, greedy, `seed=42`, pemblokir **3-gram** keras, `max_tokens=1024`,
`n_ctx=4096`. Kalau pemblokir n-gram tidak bisa aktif, job **gagal** — tidak diteruskan
diam-diam dengan jawaban yang tidak sebanding dengan naskah.

## Perangkat keras

| Mode | Flavor | Biaya | `GPU_LAYERS` |
|---|---|---|---|
| Demo / pengembangan | `cpu-upgrade` 8 vCPU / 32 GB | $0,03/jam | `0` |
| Produksi (sesi dokter) | `t4-medium` 8 vCPU / 30 GB + T4 16 GB | $0,60/jam | `99` |

Di CPU, jawaban live memakan beberapa menit per model — karena itu bank ada. Di T4,
keenam GGUF (9,3 GB) muat sekaligus di VRAM 16 GB.

**Saat pindah ke T4, indeks wheel di `Dockerfile` harus diganti ke `cu124`.** Wheel CPU
tidak akan memakai GPU walau `GPU_LAYERS` diisi, dan kegagalannya senyap.

## Yang harus dipasang operator

- Secret **`HF_TOKEN`** (scope write) — tanpa ini penilaian tidak bisa disimpan dan
  halaman akan mengatakannya sejak awal.
- Variabel opsional: `GPU_LAYERS`, `MAKS_DIMUAT`, `N_CTX`, `N_THREADS`.

Penilaian ditulis ke dataset privat `AriesDjaenuri/p6-penilaian-dokter`
(`ratings/*.json`, log sesi di `logs/*.jsonl`).
