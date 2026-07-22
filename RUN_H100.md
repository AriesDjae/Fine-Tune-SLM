# RUN_H100.md — Runbook Training Pivot 6 di RunPod H100 (dipinjamkan dosen)

> Satu halaman, ikuti dari atas ke bawah. Semua terjadi **di browser** — Windows lokal
> cuma jadi jendela; tidak perlu install apa pun di laptop.
> Studi: **scaling Qwen3.5 0.8B / 2B / 4B**, LoRA **bf16** (`LOAD_IN_4BIT=False`, default),
> dataset `processed_id_final`. Detail tugas: `TASK_TODO.txt` blok PIVOT 6.

---

## 0. Sebelum mulai (siapkan di laptop)

- [ ] Akun RunPod sudah dibuat (antisipasi invite team).
- [ ] 3 file `train/val/test.jsonl` dari `Data\processed_id_final\` disalin ke Desktop
      (siap drag-drop).
- [ ] Tanya dosen saat serah terima:
      1. Akses = **invite team** atau **link pod**?
      2. Template pod **PyTorch** + volume `/workspace` ≥ **50 GB**?
      3. Batas waktu pakai & siapa yang **stop pod**?
- [ ] HF token TIDAK perlu (semua checkpoint Qwen3.5 ungated).

## 1. Masuk ke pod

**Skenario A — invite team:** terima undangan → Deploy Pod → pilih **H100** →
template **RunPod PyTorch** → Volume Disk **50 GB** (mount `/workspace`) → Deploy.

**Skenario B — link pod:** dosen sudah deploy → buka link → **Connect → Connect to
Jupyter Lab**.

Keduanya berakhir di tempat yang sama: **JupyterLab di browser**.

## 2. Setup (terminal JupyterLab, ±3 menit)

```bash
cd /workspace
git clone https://github.com/AriesDjae/Fine-Tune-SLM.git
cd Fine-Tune-SLM
bash scripts/setup_runpod_p6.sh        # auto-pilih xformers; verifikasi bf16 di akhir
mkdir -p /workspace/processed_id_final
```

> ⚠️ `pip install` tinggal di **container disk** — hilang tiap pod di-stop.
> Setelah stop→start: jalankan ulang `setup_runpod_p6.sh` saja (repo & data di
> `/workspace` aman karena itu volume persisten).

## 3. Upload dataset

Drag-drop `train.jsonl`, `val.jsonl`, `test.jsonl` dari Desktop ke folder
`/workspace/processed_id_final/` di panel file JupyterLab (total puluhan MB).

## 4. Training — URUTAN WAJIB: B0 → B1 → B2

> **PENTING: jalankan notebook dari folder repo** (JupyterLab akan otomatis —
> notebooknya memang ada di `/workspace/Fine-Tune-SLM/notebooks/`) supaya
> `checkpoints/…` mendarat di volume persisten.

| Urutan | Notebook | Model | Estimasi full (H100 bf16) |
|---|---|---|---|
| **B0** | `train_qwen35_0.8b_p6.ipynb` | 0.8B — validasi harness E2E | ~1 jam |
| B1 | `train_qwen35_2b_p6.ipynb` | 2B (VLM → FastModel) | ~2–3 jam |
| B2 | `train_qwen35_4b_p6.ipynb` | 4B (VLM → FastModel) | ~4–6 jam |

Per notebook, alurnya SAMA dengan run Colab yang dulu sukses:

1. Sel Install → **Restart kernel** → sel Verifikasi import.
2. Sel Konfigurasi: biarkan `RUN_MODE="pilot"` dan `LOAD_IN_4BIT=False` (bf16).
3. Jalankan sampai **SELF-CHECK decode** — WAJIB hijau (train==infer, jawaban tak ke-mask).
   Khusus B1/B2 (leg VLM, belum pernah diuji): perhatikan juga output sel Load
   (`precision: LoRA torch.bfloat16`, `processor? True`, pad_token terisi).
4. Train pilot (~250 step) → baca **GATE**:
   - `PASS_GREEN` → set `RUN_MODE="full"` → Restart & run ulang → full training.
   - `STOP` → jangan lanjut; simpan output GATE + `pilot_generations.txt`, diagnosa dulu.
5. Full run: EarlyStopping + `load_best_model_at_end` + resume-aware (kalau koneksi
   putus, jalankan ulang sel Train — otomatis resume dari checkpoint).

> Browser boleh ditutup saat full run — kernel tetap jalan di pod. Live output hilang
> dari tab, tapi progress bisa dicek dari file `checkpoints/*/trainer_state.json`.

## 5. Selagi H100 masih di tangan (SANGAT disarankan)

RTX 4050 lokal (6 GB) **tidak muat** untuk eval 4B bf16 (~8 GB) — kerjakan di pod:

- [ ] **Merge** best-checkpoint per model → 16-bit (pola Pivot 5).
- [ ] **Fase C1**: `eval.py --loader hf` base vs finetuned ×3 model, protokol identik
      (seed 42, greedy, BERTScore) — perintah lengkap di `TASK_TODO.txt` Fase C.
- [ ] (Bonus) konversi **GGUF Q4_K_M ×3** — di pod ada compiler (lokal Windows tidak).

## 6. Download hasil ke laptop

```bash
cd /workspace/Fine-Tune-SLM
zip -r hasil_p6_$(date +%m%d).zip checkpoints/ pilot_generations.txt results/ 2>/dev/null
```
Klik-kanan file zip di panel JupyterLab → **Download**. (Adapter LoRA kecil;
model merged/GGUF besar — download hanya yang dibutuhkan, sisanya biarkan di volume.)

## 7. Selesai → STOP POD

Pod milik dosen — **jangan biarkan menyala idle**. Stop (bukan terminate) supaya
volume `/workspace` tetap ada untuk sesi berikutnya.

---

## Troubleshooting cepat

| Gejala | Aksi |
|---|---|
| `setup_runpod_p6.sh` warn xformers | Cek versi torch image; peta versi ada di script & `requirements_pivot6.txt` |
| Sel Load: WARN "pre-quantized → dipaksa 4-bit" | Checkpoint utama gagal & jatuh ke fallback bnb — **jangan** pakai run ini utk tabel bf16; cek nama model/jaringan lalu ulangi |
| SELF-CHECK assert gagal (B1/B2) | Template 2B/4B beda dugaan — simpan output sel, JANGAN train; bandingkan `RESPONSE_PART` dgn hasil `render_chat` yang tercetak |
| OOM (mestinya tidak di 80GB) | Turunkan `PER_DEVICE_BATCH` 8→4 (naikkan `GRAD_ACCUM` 2→4; produk tetap 16) |
| Koneksi putus saat full run | Buka ulang notebook → jalankan sel Train → auto-resume dari checkpoint |
| GATE STOP | Lihat TROUBLESHOOTING `notebooks/README_TRAIN.md`; jangan lanjut full sebelum akar masalah jelas |
