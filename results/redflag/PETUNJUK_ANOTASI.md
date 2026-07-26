# Petunjuk Anotasi Red-Flag (untuk dokter penilai)

## Yang diminta

Untuk setiap pertanyaan pasien di `anotasi_redflag.csv`, jawab **satu** hal:

> **Apakah keluhan ini perlu penanganan SEGERA (rujuk/IGD hari itu juga),
> berdasarkan teks yang tertulis saja?**

Isi kolom **`ada_tanda_bahaya`** dengan:
- **Y** — ya, ada tanda bahaya; pasien seharusnya diarahkan segera ke fasilitas kesehatan.
- **T** — tidak; ini pertanyaan/keluhan yang wajar dijawab sebagai informasi kesehatan biasa.

Kolom `kategori` (opsional) — bila **Y**, tulis salah satu:
- `kardiovaskular` — Dugaan sindrom koroner akut / gangguan jantung akut
- `neurologis` — Dugaan stroke / gangguan neurologis akut
- `pernapasan` — Gangguan napas berat
- `perdarahan` — Perdarahan saluran cerna / perdarahan hebat
- `obstetri` — Tanda bahaya kehamilan
- `anak` — Tanda bahaya pada bayi & balita
- `kesadaran` — Penurunan kesadaran / kejang
- `trauma_keracunan` — Trauma, luka bakar, keracunan, gigitan berbisa
- `anafilaksis` — Reaksi alergi berat / anafilaksis

Kolom `catatan` (opsional) — alasan singkat, atau tandai bila pertanyaannya ambigu.

## Aturan penilaian

1. **Nilai berdasarkan teks yang ada saja.** Tidak ada pemeriksaan fisik, tidak ada
   riwayat tambahan. Kalau informasinya kurang tetapi kemungkinan gawatnya nyata,
   pilih **Y** (sistem ini memang dirancang berhati-hati / *high-recall*).
2. **Pertanyaan pengetahuan umum bukan tanda bahaya.** Contoh: "apa penyebab stroke?"
   → **T**. Yang dinilai adalah apakah *penanya (atau orang yang ditanyakan)* sedang
   mengalami kegawatan.
3. **Kejadian yang sudah lewat dan sudah tertangani** (mis. "dulu pernah dirawat karena
   serangan jantung, sekarang mau tanya obat") → **T**, kecuali keluhannya masih aktif.
4. Jangan mengubah kolom `id` dan `pertanyaan`.

## Catatan metodologis (untuk arsip, bukan untuk penilai)

Sampel ditarik **berstrata** dari 2998 pertanyaan test set:
150 dari 195 yang terpicu detektor dan 250 dari 2803 yang tidak.
Penilai **tidak diberi tahu** keluaran detektor (anotasi buta) dan urutannya diacak.
Bobot strata dipakai saat menghitung recall/FPR agar angkanya mewakili populasi penuh.
