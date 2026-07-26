# Sumber Pedoman Daftar Red-Flag (dasar sitasi naskah)

Dokumen ini adalah **dasar sitasi** untuk setiap pola gejala di `redflag/rules.py`.
Aturan main: **tidak ada satu pun pola yang boleh masuk tanpa baris di tabel ini.**
Setiap aturan di `rules.py` membawa field `source` yang merujuk ke kunci di bawah.

Semua sumber di bawah **sudah diverifikasi langsung dari dokumennya** (bukan dari ingatan
model, bukan dari ringkasan pihak ketiga). Tanggal verifikasi: **2026-07-26**.

---

## S1 — KMK PPK FKTP (kekuatan sitasi: TERTINGGI, regulasi)

> Keputusan Menteri Kesehatan Republik Indonesia Nomor **HK.01.07/MENKES/1186/2022**
> tentang **Panduan Praktik Klinis bagi Dokter di Fasilitas Pelayanan Kesehatan Tingkat
> Pertama**. Jakarta: Kementerian Kesehatan RI; ditetapkan 31 Mei 2022.
> Diubah dengan KMK Nomor **HK.01.07/MENKES/1936/2022** (2 Desember 2022).

Salinan resmi: `jdih.kemkes.go.id` (footer tiap halaman PDF).
Mirror yang dipakai saat verifikasi:
<https://puskesmaslabuapi-dikes.lombokbaratkab.go.id/media/Kepmenkes_nomor_HK.01.07_MENKES_1186_2022_-_PPK_Dokter_Mei_2022.pdf>
(1379 hal.) dan <https://diskes.badungkab.go.id/storage/diskes/file/KMK%201936.2022%20ttg%20PPK.pdf> (300 hal.).

| Kunci | Entri PPK | Halaman | Kutipan yang dipakai |
|---|---|---|---|
| `S1-angina` | Angina Pektoris Stabil (ICD-10 I20.9, Tingkat Kemampuan 3B) | 269–271 | "nyeri dada di daerah sternum atau di bawah sternum (substernal…), atau dada sebelah kiri dan kadang-kadang **menjalar ke lengan kiri**, dapat menjalar ke **punggung, rahang, leher, atau ke lengan kanan**" · "seperti tertekan benda berat, atau seperti diperas" · "**Bila nyeri dada berlangsung lebih dari 20 menit, mungkin pasien mengalami sindrom koroner akut**" · "Nyeri dada bisa disertai **keringat dingin, mual, muntah, sesak dan pucat**" |
| `S1-gi-bleed` | Perdarahan Saluran Cerna (Tingkat Kemampuan 3B) | 154–158 | "keluhan **muntah darah berwarna hitam seperti bubuk kopi (hematemesis)** atau **buang air besar berwarna hitam seperti ter atau aspal (melena)**" · Komplikasi: "**Syok hipovolemia**, Aspirasi pneumonia, Gagal ginjal akut, Anemia karena perdarahan" |
| `S1-kejang-demam` | Kejang Demam (ICD-10 R56.0, Tingkat Kemampuan 4A) | 389–395 | "bangkitan kejang yang terjadi pada kenaikan suhu tubuh (suhu rektal > 38 °C)" |
| `S1-anafilaksis` | Anafilaktik (ICD-10 T78.2, Tingkat Kemampuan 4A) | ~72–77 | "reaksi hipersensitifitas generalisata atau sistemik yang beronset cepat, serius, dan mengancam… dapat menimbulkan syok yang disebut sebagai **syok anafilaktik**. Syok anafilaktik membutuhkan pertolongan cepat" |
| `S1-luka-bakar` | Luka Bakar | ~634–638 | "kerusakan kulit karena suhu yang ekstrim, misalnya air panas, api, sengatan listrik, bahan kimia, radiasi" |
| `S1-keracunan` | Keracunan Makanan (Tingkat Kemampuan 4A) | ~114 | entri PPK keracunan makanan |

---

## S2 — MTBS Kemenkes (kekuatan sitasi: TINGGI, pedoman nasional + WHO/IDAI)

> Kementerian Kesehatan RI. **Buku Bagan Manajemen Terpadu Balita Sakit (MTBS)**.
> Jakarta: Kemenkes RI; 2015. "Sesuai rekomendasi WHO 2014, kebijakan lintas program
> terkait dan protokol UKK IDAI tahun 2015" (halaman pengesahan, 31 Desember 2015).

Salinan: WHO Global Health Observatory / MCA policy documents —
<https://platform.who.int/docs/default-source/mca-documents/policy-documents/guideline/IDN-CH-20-01-GUIDELINE-2015-ind-Chart-Book-of-Integrated-Management-of-Sick-Toddler.pdf>

| Kunci | Bagian | Hal. | Kutipan verbatim |
|---|---|---|---|
| `S2-balita-danger` | MEMERIKSA TANDA BAHAYA UMUM (balita 2 bulan – 5 tahun) → klasifikasi **PENYAKIT SANGAT BERAT**, tindakan **RUJUK SEGERA** | 01 | "Tidak bisa minum atau menyusu · Memuntahkan semua makanan dan/atau minuman · Pernah atau sedang mengalami kejang · Gelisah · Letargis atau tidak sadar · Ada stridor · Tampak biru (sianosis) · Ujung tangan dan kaki pucat dan dingin" |
| `S2-pneumonia-berat` | Batuk atau sukar bernapas → **PNEUMONIA BERAT**, "RUJUK SEGERA" | 02 | "Tarikan dinding dada ke dalam ATAU Saturasi Oksigen <90%". Napas cepat: "2 bulan – <12 bulan: 50 kali atau lebih per menit; 12 bulan – <5 tahun: 40 kali atau lebih per menit" |
| `S2-diare-berat` | Diare → **DIARE DEHIDRASI BERAT**, "RUJUK SEGERA" | 03 | "Letargis atau tidak sadar · Mata cekung · Tidak bisa minum atau malas minum · Cubitan kulit perut kembali sangat lambat"; "Ada darah dalam tinja" → DISENTRI |
| `S2-bayi-muda` | Bayi muda < 2 bulan → **PENYAKIT SANGAT BERAT ATAU INFEKSI BAKTERI BERAT**, "RUJUK SEGERA" | 35 | "Tidak mau minum atau memuntahkan semua · Riwayat kejang · Bayi bergerak hanya ketika distimulasi atau tidak bergerak sama sekali · Napas cepat (≥ 60 kali/menit) · Napas lambat (< 40 kali/menit) · Tarikan dinding dada ke dalam yang sangat kuat · Suhu tubuh ≥ 37,5 °C · Suhu tubuh < 36,5 °C · Mata bernanah banyak · Pusar kemerahan meluas sampai ke dinding perut >1 cm/bernanah". Kotak atas: "Jika bayi muda ditemukan dalam kondisi **kejang atau henti napas**, segera lakukan tindakan/pengobatan sebelum melakukan penilaian dan **RUJUK SEGERA**" |

---

## S3 — Kampanye resmi Kemenkes "SeGeRa Ke RS" (stroke)

> Kementerian Kesehatan RI. *SeGeRa Ke RS Untuk Cegah Faktor Risiko Stroke*.
> Terbit 28 Oktober 2023. <https://kemkes.go.id/id/segera-ke-rs-untuk-cegah-faktor-risiko-stroke>
> Versi Direktorat Promosi Kesehatan: <https://ayosehat.kemkes.go.id/kenali-gejala-stroke-dengan-segera-ke-rs>

| Kunci | Akronim | Kutipan |
|---|---|---|
| `S3-stroke` | **Se·Ge·Ra Ke R·S** | "**Se** = Senyum tidak simetris · **Ge** = Gerak separuh anggota tubuh melemah tiba-tiba · **Ra** = Bicara pelo · **Ke** = Kebas separuh tubuh · **R** = Rabun atau pandangan mata kabur tiba-tiba · **S** = Sakit kepala hebat yang muncul tiba-tiba" · "periode emas… sekitar **4,5 jam**" |

---

## S4 — Tanda bahaya kehamilan (Kemenkes)

> Kementerian Kesehatan RI, Direktorat Promosi Kesehatan. *Tanda Bahaya Kehamilan yang
> Harus Diketahui Oleh Ibu Hamil*. <https://ayosehat.kemkes.go.id/tanda-bahaya-kehamilan-yang-harus-diketahui-oleh-ibu-hamil>
> Pendamping (dokumen resmi, lebih kuat untuk sitasi):
> Kemenkes RI. **Buku Kesehatan Ibu dan Anak (Buku KIA)**, revisi 2024.
> <https://kesprimkom.kemkes.go.id/assets/uploads/contents/others/Buku_KIA_2024.pdf>

| Kunci | Tanda bahaya |
|---|---|
| `S4-hamil` | "Muntah Berlebihan · Demam Tinggi · **Gerakan Janin Berkurang** · Pembengkakan (kaki, tangan, wajah) **disertai** pusing kepala, nyeri ulu hati, kejang dan pandangan kabur · **Pendarahan** · Pecah Ketuban Dini" |

---

## Catatan kejujuran metodologis (WAJIB masuk naskah)

1. **Kekuatan sitasi tidak seragam.** S1 (KMK) dan S2 (MTBS) adalah dokumen
   regulatif/pedoman nasional. S3 dan S4 (ayosehat/kemkes.go.id) adalah materi
   **promosi kesehatan** resmi Kemenkes — sah dikutip sebagai pesan resmi kepada
   masyarakat, tetapi **bukan** pedoman klinis. Untuk S4, sitasi utama sebaiknya
   Buku KIA; halaman ayosehat dipakai sebagai penguat perumusan awam.
2. **Pedoman ini ditulis untuk nakes, dipakai untuk awam.** S1/S2 mengandaikan
   pemeriksaan fisik (tarikan dinding dada, turgor kulit, saturasi). Dalam sistem ini
   deteksi hanya bisa memakai **teks keluhan yang ditulis pengguna**, jadi sebagian
   tanda hanya tertangkap bila pengguna kebetulan mendeskripsikannya. Ini
   **limitasi yang harus dinyatakan**, bukan disembunyikan.
3. **Ambang numerik tidak dipakai sebagai pemicu.** Napas ≥50×/menit atau suhu <36,5 °C
   menuntut pengukuran; sistem hanya menangkap padanan awamnya ("napas cepat",
   "napas tersengal", "badan dingin"). Konsekuensinya recall untuk kategori ini lebih rendah.
4. **Sistem ini tidak mendiagnosis.** Keluaran red-flag adalah **anjuran eskalasi**
   ("segera ke fasilitas kesehatan"), bukan klasifikasi penyakit. Klasifikasi
   MTBS (mis. "PENYAKIT SANGAT BERAT") dipakai hanya sebagai **dasar pemilihan gejala**,
   tidak pernah ditampilkan ke pengguna.
