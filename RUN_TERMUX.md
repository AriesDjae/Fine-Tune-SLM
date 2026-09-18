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
pip install psutil huggingface_hub
termux-setup-storage        # supaya ~/storage/downloads bisa dibaca
termux-wake-lock            # cegah Android mematikan proses saat layar mati
```

Ruang yang dibutuhkan: sekitar 1,5 GB untuk paket + 0,6 GB untuk model 0,8B.

---

## 2. Pasang mesin inferensi

### Jalur A — `llama-cpp-python` (dipakai harness; angkanya langsung sebanding)

```bash
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF" pip install llama-cpp-python
```

Kompilasi di HP memakan **20–60 menit** dan panas; biarkan tersambung ke charger. `GGML_NATIVE=OFF`
mencegah kompiler memakai flag `-march` yang sering gagal di ARM.

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

```bash
cd ~ && git clone -b naskah-artikel-p6 https://github.com/AriesDjae/Fine-Tune-SLM.git
cd Fine-Tune-SLM
cp ~/storage/downloads/pertanyaan_bench.json .      # hasil langkah 0
```

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
