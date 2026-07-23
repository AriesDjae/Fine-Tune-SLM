# RUN_SERVERLESS.md — Runbook Training Pivot 6 di RunPod SERVERLESS

> Pengganti `RUN_H100.md` (asumsi pod/JupyterLab) karena akses yang diberikan = **Serverless**.
> Studi tetap sama: **scaling Qwen3.5 0.8B / 2B / 4B**, LoRA **bf16**, dataset `processed_id_final`.
> Prinsip serverless: tidak ada JupyterLab — kode dibungkus Docker image (dibangun otomatis
> dari repo GitHub ini), training dikirim sebagai **job** via API, GPU nyala hanya selama job
> jalan (bayar per detik pemakaian, mati sendiri — tidak ada risiko "lupa stop pod").

**File harness:** `handler.py` (root — pemindai RunPod hanya cek root) + `serverless/train_p6.py` (konversi 1:1 dari
notebook `train_*_p6.ipynb`; SELF-CHECK, GATE, resume, hyperparameter tak diubah) + `Dockerfile`.

---

## 0. Pilih arsitektur penyimpanan (tanya dosen 1 kalimat)

| | **Plan A — Network Volume** (disarankan) | **Plan B — tanpa volume (HF Hub)** |
|---|---|---|
| Biaya ekstra | ~$3.5/bulan (50 GB) di akun team | Gratis (repo privat HF milikmu) |
| Dataset masuk | upload sekali via S3 API | upload sekali ke HF dataset privat |
| Checkpoint | persisten → **auto-resume kalau worker mati** | hilang kalau job gagal di tengah (ulang dari 0) |
| Cache model base | persisten (hemat waktu cold start) | terunduh ulang tiap worker baru (~menit) |
| Ambil hasil | download via S3 API | otomatis ke-push ke repo HF privat |

Kode **auto-deteksi**: kalau volume terpasang dipakai, kalau tidak jatuh ke mode HF Hub.
Kalau dosen tidak keberatan biaya ~$3.5/bulan → Plan A. Kalau mau nol tambahan → Plan B.

## 1. Persiapan satu kali

### Plan B (tanpa volume) — dari laptop
1. Buat token **write** di huggingface.co/settings/tokens.
2. ```powershell
   $env:HF_TOKEN = "hf_..."
   python scripts/upload_dataset_hf.py --repo <username-hf>/processed-id-final
   ```
3. Catat: `HF_DATA_REPO=<username-hf>/processed-id-final`, `HF_OUT_REPO=<username-hf>/p6-hasil`
   (repo hasil dibuat otomatis oleh worker, privat).

### Plan A (network volume) — di console + laptop
1. Console → **Storage → New Network Volume** → pilih **datacenter yang punya H100** → 50 GB.
2. Console → **Settings → S3 API Keys → Create** → catat access key (`user_...`) + secret (`rps_...`)
   — secret cuma tampil SEKALI.
3. Upload dataset dari laptop (`pip install awscli` sekali):
   ```powershell
   aws configure   # isi access key + secret; region & format kosongkan
   aws s3 cp --region <DC> --endpoint-url https://s3api-<DC>.runpod.io/ `
     "Data\processed_id_final\train.jsonl" s3://<VOLUME_ID>/processed_id_final/train.jsonl
   # ulangi utk val.jsonl dan test.jsonl   (<DC> mis. eu-cz-1; VOLUME_ID di halaman volume)
   ```

## 2. Buat endpoint (sekali, di console — GitHub integration)

**Serverless → New Endpoint → Import Git Repository** → authorize GitHub → pilih repo
`AriesDjae/Fine-Tune-SLM`, branch `main`, Dockerfile di root (default). RunPod build image
otomatis (±15–25 menit; tidak perlu Docker di laptop).

Konfigurasi endpoint (bagian yang WAJIB diubah dari default ditandai ⚠️):

| Setting | Nilai |
|---|---|
| GPU | **H100 PRO 80 GB** (boleh tambah prioritas-2: A100 80GB) |
| Max Workers | **1** ⚠️ (job antri berurutan — sesuai desain B0→B1→B2, cegah dobel biaya) |
| Active Workers | 0 (default) |
| Idle Timeout | 5–60 s |
| **Execution Timeout** | **86400 s (24 jam)** ⚠️⚠️ — default 600 s akan MEMBUNUH training! |
| FlashBoot | on |
| Network Volume | (Plan A) pilih volume-mu — Advanced → Network Volumes |
| Env vars | (Plan B) `HF_TOKEN`, `HF_DATA_REPO`, `HF_OUT_REPO` · (Plan A) tidak perlu |

> Update kode setelah endpoint dibuat: push ke GitHub **lalu buat Release baru** di repo
> (RunPod rebuild dari release, bukan tiap push).

## 3. Jalankan training — URUTAN WAJIB B0 → B1 → B2 (dari laptop)

```powershell
$env:RUNPOD_API_KEY     = "rpa_..."      # Settings -> API Keys (buat bila belum)
$env:RUNPOD_ENDPOINT_ID = "..."          # dari halaman endpoint
```

| Urutan | Perintah pilot | Lalu (bila GATE `PASS_GREEN`) | Estimasi full (H100) |
|---|---|---|---|
| **B0** 0.8B | `python scripts/submit_job.py --model 0.8b --mode pilot` | `... --model 0.8b --mode full` | ~1 jam |
| B1 2B | `python scripts/submit_job.py --model 2b --mode pilot` | `... --model 2b --mode full` | ~2–3 jam |
| B2 4B | `python scripts/submit_job.py --model 4b --mode pilot` | `... --model 4b --mode full` | ~4–6 jam |

- Script otomatis memantau (poll 30 s) dan mencetak **verdict GATE** di akhir job pilot.
  `STOP` → JANGAN lanjut full; baca `pilot_generations.txt` + `traceback` di output job.
- `Ctrl+C` aman — job tetap jalan di cloud. Lanjutkan pantau:
  `python scripts/submit_job.py --status <JOB_ID>`.
- Log real-time lengkap: console → endpoint → **Workers → Logs** (SELF-CHECK, loss per 10
  step, warning `pre-quantized`, dsb. — khusus B1/B2 perhatikan `processor? True`).
- Mode full otomatis + quick-eval (token-F1 + ROUGE-L, n=100 val) di akhir → ada di output job.

## 4. Ambil hasil ke laptop

- **Plan B**: semua otomatis di `https://huggingface.co/<username>/p6-hasil`
  (folder `qwen35_*_p6/adapter` + `results`: `log_history.json`, `trainer_state_best.json`,
  `pilot_generations.txt`, `run_summary.json`).
- **Plan A**: `aws s3 cp --recursive ... s3://<VOLUME_ID>/results/ .\results_p6\`
  (adapter di `s3://<VOLUME_ID>/checkpoints/`). Yang kecil saja dulu; merged/GGUF belakangan.

## 5. Setelah ketiga model selesai (Fase C)

Eval penuh `eval.py` base-vs-finetuned ×3 + merge 16-bit butuh sesi GPU interaktif ATAU
job mode tambahan — **belum ada di harness serverless** (sengaja, fokus dulu training).
Opsi: (a) minta pod sekali utk eval (lihat `RUN_H100.md` bag. 5), (b) tambah mode
`eval`/`merge` di `serverless/train_p6.py` (bilang ke Claude), (c) eval 0.8B/2B lokal
seperti Pivot 5 (`--loader hf`) — 4B tetap tak muat di RTX 4050 6 GB.

---

## Troubleshooting

| Gejala | Aksi |
|---|---|
| Build GitHub gagal | Buka log build di console; umumnya versi pip — laporkan lognya |
| Job `TIMED_OUT` cepat | Execution Timeout masih 600 s default → naikkan ke 86400 (bag. 2) |
| Output `error: Dataset tak ketemu` | Plan A: cek upload S3 ke folder `processed_id_final/` persis; Plan B: cek env `HF_DATA_REPO`+`HF_TOKEN` di endpoint |
| `ASSERT: SELF-CHECK GAGAL` (B1/B2) | Template 2B/4B beda dugaan — JANGAN train; simpan output job, bandingkan `RESPONSE_PART` vs tail ter-render di pesan error |
| Output ada `forced_4bit_fallback: true` | Checkpoint utama gagal → fallback pre-quantized bnb; run ini JANGAN masuk tabel bf16 jurnal |
| GATE `STOP` | Baca `pilot_generations.txt` + `notebooks/README_TRAIN.md` › TROUBLESHOOTING |
| OOM (harusnya tidak di H100 80GB) | Job ulang dgn `"load_in_4bit": true` HANYA utk diagnosa; utk tabel bf16 tetap harus bf16 |
| Worker mati di tengah full run | Plan A: submit ulang job yang sama → auto-resume dari checkpoint volume. Plan B: mulai dari 0 (alasan utama pilih Plan A) |
| `warning: TANPA volume & TANPA HF_OUT_REPO` | Hasil akan hilang! Set env `HF_OUT_REPO`+`HF_TOKEN` di endpoint sebelum job berikutnya |

## Estimasi biaya (H100 PRO $0.00116/dtk ≈ $4.2/jam)

pilot ×3 (~15 mnt each) ≈ $3 · full B0+B1+B2 (~8 jam) ≈ $34 · **total ≈ $37–40**
(+$3.5/bln volume bila Plan A). Tidak ada biaya saat tidak ada job.
