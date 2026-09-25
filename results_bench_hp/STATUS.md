# Status pengukuran RQ3 di HP

Perangkat: Xiaomi POCO X3 GT (`chopin`, 21061110AG), Android 13 (API 33, build
TP1A.220624.014), MediaTek Dimensity 1100 (MT6891Z/CZA), 4× Cortex-A55 @2,0 GHz +
4× Cortex-A78 @2,6 GHz, MemTotal 7.656.624 kB (7,8 GB), zRAM 6,3 GB,
kernel 4.14.186. Benchmark memakai `--threads 4`.
Rincian: `spek_perangkat.txt`.

## Yang sudah diukur — RQ3 SELESAI

Ketiga ukuran memakai protokol yang sama dengan laptop (`dry_plus`).

| Model | Protokol | Puncak RAM | Latensi | tok/dtk | Token rata-rata | Diukur |
|---|---|---:|---:|---:|---:|---|
| 0.8B | dry_plus | 0,75 GB | 45,6 dtk | 8,0 | 310 | 2026-09-25 |
| 2B | dry_plus | 1,50 GB | 87,5 dtk | 3,1 | 268 | 2026-09-25 |
| 4B | dry_plus | 2,99 GB | 187,0 dtk | 1,3 | 249 | 2026-09-20 |

Ketiganya **selesai tanpa dimatikan Android**. Puncak RAM di HP lebih rendah
daripada laptop di semua ukuran, dan selisihnya MELEBAR seiring model: 25% (0.8B),
31% (2B), 37% (4B). 4B lolos gerbang HP 6 GB; yang menyingkirkannya dari kelas
sasaran adalah 187 detik per jawaban.

**Run lama 0.8B/2B (pemblokir 3-gram, 2026-09-20) sudah digantikan.** RAM-nya sama
(0,76 / 1,50 GB). Latensi 0.8B naik 32,6 → 45,6 dtk karena jawaban DRY+ lebih panjang
(241 → 310 token), bukan karena lebih lambat (7,6 → 8,0 tok/dtk); 2B 82,3 → 87,5 dtk.

*Quantization gap* Q4_K_M vs 16-bit juga sudah diukur (laptop, `results/quant_gap/`).

## Arsip: perintah yang dipakai untuk run DRY+ 0.8B/2B


Sisa: ulang 0.8B dan 2B dengan `--decoding dry_plus` (4B sudah). Tempel apa adanya.

```bash
cd ~/bench
curl -L -o bench_hp.zip https://huggingface.co/AriesDjaenuri/p6-hasil/resolve/main/tools/bench_hp.zip
ls -l bench_hp.zip          # 27396 byte = versi dengan demo_chat.py
unzip -o bench_hp.zip

python - <<'CEK'
import sys
if sys.platform == "android": sys.platform = "linux"
import llama_cpp; print("OK", llama_cpp.__version__)
CEK

termux-wake-lock

python benchmark_ondevice.py --gguf ~/models/Qwen3.5-0.8B.Q4_K_M.gguf \
  --label 0.8b --decoding dry_plus --pertanyaan pertanyaan_bench.json \
  --out_dir results_bench_hp --threads 4
cp results_bench_hp/bench_0.8b.json ~/storage/downloads/results_bench_hp/

# beri jeda supaya HP dingin, lalu:
python benchmark_ondevice.py --gguf ~/models/Qwen3.5-2B.Q4_K_M.gguf \
  --label 2b --decoding dry_plus --pertanyaan pertanyaan_bench.json \
  --out_dir results_bench_hp --threads 4
cp results_bench_hp/bench_2b.json ~/storage/downloads/results_bench_hp/
```

Lalu tempel ringkasannya (tanpa teks pasien) ke Claude:

```bash
cd ~/bench && python - <<'RINGKAS'
import json, glob
for f in sorted(glob.glob("results_bench_hp/bench_[02]*.json")):
    d = json.load(open(f, encoding="utf-8")); d.pop("jawaban", None)
    print(json.dumps(d, indent=1, ensure_ascii=False))
RINGKAS
```

Yang harus dicek di hasilnya: `"decoding"` harus berbunyi `dry_plus`. Kalau yang
muncul justru `no_repeat_ngram_size`, zip-nya belum terbarui.

Kalau ada `bench_*_parsial.json` tersisa sesudah run, artinya run itu mati di
tengah jalan. Isinya tetap berguna, jangan dihapus.

Model GGUF sebaiknya dipindah ke `~/storage/downloads/models/` supaya selamat
kalau Termux di-reset lagi.

---

## Demo langsung (peraga, bukan pengukuran)

`demo_chat.py` (ada di zip sejak 2026-09-24) menampilkan jawaban kata demi kata dengan
DRY+ dan lapisan red-flag. Angkanya tidak dipakai naskah.

```bash
cd ~/bench && termux-wake-lock
python demo_chat.py --ukuran 0.8b --contoh 3 --simpan demo_jawaban.json
```

Contoh ke-2 dan ke-3 memicu banner red-flag. Sesudah contoh habis, ketik pertanyaan sendiri;
`keluar` untuk berhenti. Nyalakan mode pesawat supaya terlihat offline.

## Sebelum run berikutnya

```bash
cd ~/bench
curl -L -o bench_hp.zip https://huggingface.co/AriesDjaenuri/p6-hasil/resolve/main/tools/bench_hp.zip
ls -l bench_hp.zip          # 27396 byte = versi dengan demo_chat.py
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
