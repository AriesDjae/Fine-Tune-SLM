# Menjalankan benchmark RQ3 di HP lewat Termux

Tujuan: mengulang `benchmark_ondevice.py` pada perangkat sasaran sesungguhnya (HP RAM 3–4 GB),
bukan pada CPU laptop yang selama ini dipakai sebagai proksi. Yang diukur sama persis: puncak
RAM saat memuat dan saat inferensi, latensi, *throughput*, dan *overhead* lapisan red-flag.

**Syarat agar angkanya sebanding:** pertanyaan ujinya harus **sama persis** dengan run laptop.
Karena itu daftar 12 pertanyaan diekspor dari laptop dan disalin ke HP, bukan diacak ulang di
sana. `Data/` sendiri tidak boleh ikut ke HP (aturan lisensi yang sama dengan bank jawaban).

---

## 0. Di laptop, sekali saja

```bash
python benchmark_ondevice.py --simpan-pertanyaan results_bench/pertanyaan_bench.json
```

Berkas itu **tidak masuk git** (berisi teks pertanyaan Alodokter apa adanya). Salin ke HP lewat
kabel USB, Syncthing, atau unggah sementara ke penyimpanan pribadi Anda.

---

## 1. Pasang Termux

Ambil dari **F-Droid** (`https://f-droid.org/packages/com.termux/`), **bukan** dari Play Store —
versi Play Store sudah lama tidak diperbarui dan paketnya gagal dipasang.

```bash
pkg update && pkg upgrade -y
pkg install -y python clang cmake git libopenblas python-numpy
# psutil TIDAK perlu (ditolak di Android); huggingface_hub juga tidak — pakai curl
termux-setup-storage        # supaya ~/storage/downloads bisa dibaca
termux-wake-lock            # cegah Android mematikan proses saat layar mati
```

Ruang yang dibutuhkan: sekitar 1,5 GB untuk paket + 0,6 GB untuk model 0,8B.

> **`pip install psutil` GAGAL di Termux** dengan pesan `platform android is not supported`
> (psutil 7.x menolak `sys.platform == "android"` yang dilaporkan Python 3.13+). Itu tidak
> menghalangi: `benchmark_ondevice.py` membaca `/proc/self/status` dan `/proc/meminfo` bila
> psutil tidak ada, yaitu sumber angka yang sama dengan yang dipakai psutil di Linux.

---

## 2. Pasang mesin inferensi

### Jalur A — `llama-cpp-python` (dipakai harness; angkanya langsung sebanding)

```bash
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF" pip install llama-cpp-python
```

Kompilasi di HP memakan **20–60 menit** dan panas; biarkan tersambung ke charger. `GGML_NATIVE=OFF`
mencegah kompiler memakai flag `-march` yang sering gagal di ARM.

> **`llama_cpp` bisa gagal saat dipakai walau pemasangannya sukses**, dengan
> `RuntimeError: Unsupported platform`. Sebabnya sama dengan psutil: Python 3.13+ di Termux
> melaporkan `sys.platform == "android"`, sedangkan `llama_cpp` hanya mengenal
> win32/darwin/linux. `benchmark_ondevice.py` meluruskannya sendiri sebelum mengimpor
> pustaka itu, jadi pastikan Anda memakai versi skrip terbaru.

### Jalur B — biner `llama.cpp` (cadangan bila Jalur A gagal)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_NATIVE=OFF && cmake --build build -j4
```

Jalur B memberi `build/bin/llama-bench` (token/detik) tetapi **tidak** memberi puncak RAM dengan
cara yang sama seperti harness. Kalau terpaksa memakainya, catat RAM lewat
`grep VmHWM /proc/<pid>/status` dan sebutkan di naskah bahwa alat ukurnya berbeda.

---

## 3. Ambil model dari HuggingFace

Ketiga repo GGUF **publik**, jadi tidak perlu token.

```bash
mkdir -p ~/models && cd ~/models
hf download AriesDjaenuri/qwen35-0.8b-medical-id-GGUF Qwen3.5-0.8B.Q4_K_M.gguf --local-dir .
# opsional, hanya kalau ingin membuktikan kegagalannya di HP:
hf download AriesDjaenuri/qwen35-2b-medical-id-GGUF   Qwen3.5-2B.Q4_K_M.gguf   --local-dir .
```

Kalau `hf` tidak dikenali, pakai `curl`:

```bash
curl -L -o Qwen3.5-0.8B.Q4_K_M.gguf \
  https://huggingface.co/AriesDjaenuri/qwen35-0.8b-medical-id-GGUF/resolve/main/Qwen3.5-0.8B.Q4_K_M.gguf
```

---

## 4. Ambil kode dan daftar pertanyaan

Daftar pertanyaan benchmark berisi teks Alodokter verbatim, jadi tidak ada di repo GitHub.
Paket kecil berisi skrip + modul `redflag/` + daftar pertanyaan ada di repo HuggingFace
`AriesDjaenuri/p6-hasil`, yang aksesnya dibatasi; set `HF_TOKEN` berisi token baca dari
penulis:

```bash
mkdir -p ~/bench && cd ~/bench
curl -L -H "Authorization: Bearer $HF_TOKEN" -o bench_hp.zip \
  https://huggingface.co/AriesDjaenuri/p6-hasil/resolve/main/tools/bench_hp.zip
unzip bench_hp.zip
```

Perbarui paket itu dari laptop bila skripnya berubah:

```bash
python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="bench_hp.zip", path_in_repo="tools/bench_hp.zip",
                    repo_id="AriesDjaenuri/p6-hasil", repo_type="model")
PY
```

Alternatif bila ingin tetap lewat git: buat *fine-grained token* baca-saja lalu
`git clone -b naskah-artikel-p6 https://<TOKEN>@github.com/AriesDjae/Fine-Tune-SLM.git`.

---

## 5. Jalankan

```bash
python benchmark_ondevice.py \
  --gguf ~/models/Qwen3.5-0.8B.Q4_K_M.gguf \
  --label 0.8b \
  --pertanyaan pertanyaan_bench.json \
  --out_dir results_bench_hp

python benchmark_ondevice.py --summarize results_bench_hp
```

Tambahkan `--threads 4` bila HP-nya 4 inti, atau `--threads 8` untuk 8 inti. Nilai bawaannya
setengah jumlah inti yang terlihat, dan di Android jumlah inti yang terlihat sering lebih besar
dari yang benar-benar bisa dipakai terus-menerus.

Salin `results_bench_hp/bench_0.8b.json` kembali ke laptop, lalu laporkan berdampingan dengan
angka laptop di § RQ3 naskah dan Tabel 4.14 BAB IV.

---

## 6. Catat spesifikasi perangkat

Metodologi wajib menyebut perangkat ujinya. Jalankan di HP:

```bash
sh spek_hp.sh > spek_hp.txt
cat spek_hp.txt
```

`spek_hp.sh` ikut di dalam `bench_hp.zip`. Ia hanya membaca `getprop` dan `/proc`,
jadi keluarannya aman ditempel ke naskah. Yang dipakai di naskah: model HP, versi
Android, SoC, jumlah inti, dan **MemTotal** (bukan angka RAM pemasaran — Android
selalu melaporkan lebih kecil karena sebagian dipakai firmware).

---

## 7. Tarik hasilnya ke laptop

Hasil hanya ada di HP sampai Anda menyalinnya. Dua cara:

**a. Ringkasan angka saja** (tanpa teks pasien, aman ditempel ke chat/naskah):

```bash
python - <<'PY'
import json, glob
for f in sorted(glob.glob("results_bench_hp/bench_*.json")):
    d = json.load(open(f, encoding="utf-8"))
    d.pop("jawaban", None)
    print(json.dumps(d, indent=1, ensure_ascii=False))
PY
```

**b. Berkas utuh** (termasuk teks jawaban, untuk arsip):

```bash
termux-setup-storage          # sekali saja
cp -r results_bench_hp ~/storage/downloads/
```

Lalu ambil dari folder Download HP lewat kabel USB atau penyimpanan awan Anda.

> **Awas isi berkasnya.** Sejak commit `8b20c76`, `bench_*.json` menyimpan teks
> pertanyaan dan jawaban apa adanya — termasuk keluhan pasien Alodokter verbatim.
> `results_bench/pertanyaan_bench.json` sengaja di-gitignore karena alasan yang
> sama, jadi pertimbangkan apakah `bench_*.json` versi HP pantas ikut di-commit,
> atau cukup ringkasannya.

---

## Yang harus dicatat apa adanya

- **Spesifikasi HP**: merek, tipe, RAM total, chipset, versi Android. Ini wajib disebut di
  metodologi; tanpa itu angkanya tidak bisa ditafsirkan.
- **Kalau model 2B atau 4B dimatikan Android** di tengah jalan, itu **hasil**, bukan kegagalan
  percobaan: catat pada langkah mana ia mati dan berapa RAM total perangkat. Gerbang RAM di
  naskah memang memperkirakan keduanya gagal di kelas 3–4 GB.
- **Throughput di HP pasti lebih rendah** daripada laptop; yang menentukan kelayakan adalah
  puncak RAM, dan itu sifat model serta panjang konteks, bukan sifat perangkat.
- Suhu perangkat memengaruhi *throughput* (*thermal throttling*). Jalankan dua kali dengan jeda
  dan laporkan keduanya bila selisihnya besar.

> Catatan kejujuran: runbook ini disusun dari dokumentasi Termux dan llama.cpp, **belum pernah
> dijalankan di HP mana pun oleh penulis skrip**. Langkah yang paling mungkin bermasalah adalah
> kompilasi `llama-cpp-python` (Jalur A).
