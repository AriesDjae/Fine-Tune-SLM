# Status pengukuran RQ3 di HP

Perangkat: Xiaomi POCO X3 GT (`chopin`, 21061110AG), Android 13 (API 33, build
TP1A.220624.014), MediaTek Dimensity 1100 (MT6891Z/CZA), 4× Cortex-A55 @2,0 GHz +
4× Cortex-A78 @2,6 GHz, MemTotal 7.656.624 kB (7,8 GB), zRAM 6,3 GB,
kernel 4.14.186. Benchmark memakai `--threads 4`.
Rincian: `spek_perangkat.txt`.

## Yang sudah diukur

| Model | Protokol | Puncak RAM | Latensi | tok/dtk | Berkas di repo |
|---|---|---:|---:|---:|---|
| 0.8B | token-3 | 0,76 GB | 32,6 dtk | 7,6 | `bench_0.8b.json` |
| 2B | token-3 | 1,50 GB | 82,3 dtk | 3,1 | `bench_2b.json` |
| 4B | token-3 | ? | ? | ? | **angka belum disalin ke laptop** |

Ketiganya **selesai tanpa dimatikan Android**, termasuk 4B — ini sendiri sudah
temuan, karena naskah sebelumnya hanya punya hitungan ekstrapolasi untuk 4B.

## Yang belum

1. **Angka 4B belum masuk repo.** Ada di HP: `~/bench/results_bench_hp/bench_4b.json`.
   Salin ringkasannya (buang array `jawaban`) dengan cuplikan di `RUN_TERMUX.md` §7.
2. **Ulang ketiganya dengan DRY+.** Semua angka HP di atas memakai pemblokir 3-gram,
   sementara laptop memakai DRY+. Puncak RAM tidak terpengaruh, tapi latensi dan
   throughput HP jadi sedikit optimistis (di laptop DRY+ 8% lebih lambat pada 0.8B).
   Naskah sudah menyebut ketidaksamaan ini apa adanya, jadi ini bukan penghalang
   publikasi — hanya perapian.
3. **Quantization gap** Q4_K_M vs 16-bit: belum diukur sama sekali.

## Sebelum run berikutnya

```bash
cd ~/bench
curl -L -o bench_hp.zip https://huggingface.co/AriesDjaenuri/p6-hasil/resolve/main/tools/bench_hp.zip
ls -l bench_hp.zip          # 25045 byte = versi dengan simpan-parsial
unzip -o bench_hp.zip
python - <<'PY'
import sys
if sys.platform == "android": sys.platform = "linux"
import llama_cpp; print("OK", llama_cpp.__version__)
PY
```

Kalau `llama_cpp` gagal: Termux memperbarui Python ke 3.14.6, jadi modul lama bisa
tidak terbaca. Pasang ulang dengan
`CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF" pip install --force-reinstall --no-cache-dir llama-cpp-python`.

Perintah run:

```bash
termux-wake-lock
python benchmark_ondevice.py --gguf ~/models/Qwen3.5-<UKURAN>.Q4_K_M.gguf \
  --label <0.8b|2b|4b> --decoding dry_plus --pertanyaan pertanyaan_bench.json \
  --out_dir results_bench_hp --threads 4
```

Reboot HP sebelum 4B; MemFree sempat tercatat 250 MB saja. Beri jeda antar-model
supaya throughput tidak tercemar panas.

## Catatan isi berkas

`bench_*.json` di HP menyimpan teks pertanyaan dan jawaban apa adanya, termasuk
keluhan pasien Alodokter verbatim. Salinan di repo ini sengaja **tanpa** array
`jawaban`. Berkas utuh biarkan di HP atau di penyimpanan pribadi.
