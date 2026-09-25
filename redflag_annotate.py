"""
redflag_annotate.py — set anotasi & evaluasi lapisan RED-FLAG (terpisah dari metrik mutu).

Lapisan red-flag TIDAK boleh dinilai dengan ROUGE-L/token-F1: ia bukan soal kemiripan teks,
melainkan soal *apakah kegawatan tertangkap*. Karena itu evaluasinya berdiri sendiri:
**recall** (berapa persen keluhan gawat yang tertangkap) dan **false-positive rate**
(berapa persen keluhan biasa yang kena banner). Setelan sistem HIGH-RECALL, jadi FPR yang
agak tinggi adalah konsekuensi yang disengaja — dan harus dilaporkan apa adanya.

Alur pemakaian:

    # 1) lihat perilaku detektor di seluruh test set
    python redflag_annotate.py --stats

    # 2) bikin berkas anotasi (BUTA: dokter tak melihat keluaran detektor)
    python redflag_annotate.py --build --n-terpicu 150 --n-tidak 250

    # 3) tulis daftar pertanyaan untuk "Tugas 2" di Space arena, lalu deploy Space.
    #    File itu diunggah ke dataset privat. Dokter menilai di web yang sama dengan
    #    penilaian jawaban (Tier 3). SAMPLING DUA TAHAP: 20 terpicu + semua yang tidak
    #    terpicu tapi memuat kata waspada + 21 sisanya (±60), maks. 30 per dokter,
    #    maks. 2 dokter per pertanyaan (yang belum dinilai siapa pun didahulukan).
    python redflag_annotate.py --ke-space

    # 4) hitung recall / FPR / presisi + CI bootstrap + kesepakatan antar-dokter,
    #    dari label yang ditulis Space ke dataset (redflag/*.json)
    python redflag_annotate.py --score

BOLEH BERHENTI DI TENGAH. Bobot strata dihitung dari pertanyaan yang BENAR-BENAR sudah
dinilai, bukan dari rencana 150/250, jadi anotasi yang belum selesai tetap menghasilkan
angka yang sah (hanya CI-nya lebih lebar). Space menyajikan pertanyaan menurut urutan
acak, sehingga yang sudah dinilai tetap merupakan sampel acak dari kedua strata.

PENARIKAN SAMPEL BERSTRATA. Kalau 400 pertanyaan diambil acak murni, hanya ~26 yang
terpicu sehingga recall tak bisa diperkirakan dengan layak. Maka sampel dibagi dua strata
(terpicu / tidak terpicu) dengan laju ambil berbeda, lalu tiap sampel diberi BOBOT
= (ukuran strata / jumlah sampel strata) supaya angka akhir tetap mewakili populasi.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter

from redflag import KATEGORI, detect

TEST_FILE = os.path.join("Data", "processed_id", "test.jsonl")
OUT_DIR = os.path.join("results", "redflag")
FILE_ANOTASI = os.path.join(OUT_DIR, "anotasi_redflag.csv")
FILE_KUNCI = os.path.join(OUT_DIR, "anotasi_redflag_kunci.csv")
FILE_PETUNJUK = os.path.join(OUT_DIR, "PETUNJUK_ANOTASI.md")
FILE_HASIL = os.path.join(OUT_DIR, "hasil_redflag.json")
# Daftar pertanyaan untuk Tugas 2 di Space arena (TANPA kunci/strata). Diunggah ke dataset
# PRIVAT, bukan ke repo Space: Space-nya publik, isinya keluhan pasien dari test set.
FILE_SPACE = os.path.join(OUT_DIR, "redflag_soal.json")
# Tempat Space menulis label dokter (redflag/*.json).
HF_DATASET = "AriesDjaenuri/p6-penilaian-dokter"
# Kelompok tahap 2 (bukan untuk penilai): id -> kelompok, dan ukuran populasi tiap kelompok.
FILE_TAHAP2 = os.path.join(OUT_DIR, "tahap2_kunci.json")

# SAMPLING DUA TAHAP. Tahap 1 = 400 pertanyaan anotasi (berstrata menurut detektor).
# Tahap 2 = yang dinilai dokter, berstrata menurut detektor x daftar KATA WASPADA di bawah.
# Daftar ini jauh lebih longgar daripada aturan detektor: ia menandai gejala waspada apa
# pun, bukan kombinasi yang ketat. Pertanyaan yang tidak memicu detektor tetapi memuat
# kata waspada adalah tempat kasus gawat yang lolos (FN) paling mungkin berada, jadi
# kelompok itu diambil dengan peluang tertinggi. Daftar ini HANYA mengarahkan sampel;
# estimasinya tetap tak bias karena tiap label dokter diberi bobot kebalikan peluangnya.
# Mengubah daftar ini mengubah kelompok -> jalankan ulang --ke-space sebelum ada label.
KATA_WASPADA = re.compile(
    r"sesak|susah napas|sulit bernapas|nyeri dada|dada (sakit|nyeri|sesak|terasa)|pingsan"
    r"|tidak sadar|tak sadar|kejang|stroke|lumpuh|mati rasa|pelo|bicara (cadel|pelo)"
    r"|muntah darah|batuk darah|bab (hitam|berdarah|darah)|berak (hitam|darah)|pendarahan"
    r"|perdarahan|darah (banyak|terus)|keluar darah|flek|hamil.*(nyeri|darah|kontraksi)"
    r"|ketuban|bayi.*(demam|kuning|lemas|tidak mau)|demam (tinggi|40|39)|panas tinggi"
    r"|lemas sekali|sangat lemas|kecelakaan|jatuh|benturan|patah|luka bakar|tersiram"
    r"|digigit|gigitan|keracunan|overdosis|minum obat banyak"
    r"|bengkak (wajah|bibir|lidah|tenggorokan)|biduran.*sesak|alergi berat|bunuh diri"
    r"|ingin mati|sakit kepala (hebat|parah|sekali)|nyeri (hebat|sekali|luar biasa)"
    r"|tidak bisa (bangun|jalan|bicara|kencing)")


def kelompok(pertanyaan: str) -> str:
    if detect(pertanyaan).terpicu:
        return "terpicu"
    return "tidak_waspada" if KATA_WASPADA.search(pertanyaan.lower()) else "tidak_biasa"
SEED = 42


# --------------------------------------------------------------------------------------
def muat_pertanyaan(test_file: str) -> list[str]:
    soal = []
    with open(test_file, encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if not baris:
                continue
            rec = json.loads(baris)
            user = [m["content"] for m in rec.get("messages", []) if m.get("role") == "user"]
            soal.append(user[0] if user else "")
    return soal


def stats(test_file: str) -> None:
    soal = muat_pertanyaan(test_file)
    kat, aturan = Counter(), Counter()
    n_terpicu = 0
    for q in soal:
        h = detect(q)
        if h.terpicu:
            n_terpicu += 1
            kat.update(h.kategori)
            aturan.update(h.rule_ids)

    print(f"Test set        : {test_file}")
    print(f"Jumlah pertanyaan: {len(soal)}")
    print(f"Terpicu red-flag : {n_terpicu} ({n_terpicu / len(soal) * 100:.1f}%)\n")
    print(f"{'KATEGORI':22s} {'n':>5s}  {'%':>6s}  keterangan")
    for k, v in kat.most_common():
        print(f"{k:22s} {v:5d}  {v / len(soal) * 100:5.1f}%  {KATEGORI[k]}")
    print(f"\n{'ATURAN':42s} {'n':>5s}")
    for k, v in aturan.most_common():
        print(f"{k:42s} {v:5d}")


# --------------------------------------------------------------------------------------
def build(test_file: str, n_terpicu: int, n_tidak: int) -> None:
    soal = muat_pertanyaan(test_file)
    hasil = [(i, q, detect(q)) for i, q in enumerate(soal)]

    terpicu = [(i, q, h) for i, q, h in hasil if h.terpicu]
    tidak = [(i, q, h) for i, q, h in hasil if not h.terpicu]

    rng = random.Random(SEED)
    n_terpicu = min(n_terpicu, len(terpicu))
    n_tidak = min(n_tidak, len(tidak))
    sampel_a = rng.sample(terpicu, n_terpicu)
    sampel_b = rng.sample(tidak, n_tidak)

    bobot_a = len(terpicu) / n_terpicu
    bobot_b = len(tidak) / n_tidak

    baris = [(i, q, h, "terpicu", bobot_a) for i, q, h in sampel_a]
    baris += [(i, q, h, "tidak_terpicu", bobot_b) for i, q, h in sampel_b]
    rng.shuffle(baris)  # acak supaya strata tak bisa ditebak dari urutan

    os.makedirs(OUT_DIR, exist_ok=True)

    # (a) berkas untuk dokter — BUTA: tidak memuat keluaran detektor sama sekali
    with open(FILE_ANOTASI, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pertanyaan", "ada_tanda_bahaya", "kategori", "catatan"])
        for idx, (i, q, _h, _s, _b) in enumerate(baris, 1):
            w.writerow([f"RF{idx:04d}", " ".join(q.split()), "", "", ""])

    # (b) kunci — dipakai saat --score, JANGAN diberikan ke penilai
    with open(FILE_KUNCI, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "idx_test", "stratum", "bobot", "terpicu", "kategori_detektor", "aturan", "sumber"])
        for idx, (i, _q, h, stratum, bobot) in enumerate(baris, 1):
            w.writerow([
                f"RF{idx:04d}", i, stratum, f"{bobot:.6f}", int(h.terpicu),
                "|".join(h.kategori), "|".join(h.rule_ids), "|".join(h.sumber),
            ])

    tulis_petunjuk(len(terpicu), len(tidak), n_terpicu, n_tidak)

    print(f"Populasi   : {len(soal)} pertanyaan ({len(terpicu)} terpicu / {len(tidak)} tidak)")
    print(f"Sampel     : {len(baris)} ({n_terpicu} + {n_tidak}); bobot {bobot_a:.2f} / {bobot_b:.2f}")
    print(f"Untuk dokter -> {FILE_ANOTASI}")
    print(f"Kunci        -> {FILE_KUNCI}  (JANGAN dibagikan ke penilai)")
    print(f"Petunjuk     -> {FILE_PETUNJUK}")


def tulis_petunjuk(n_pop_a: int, n_pop_b: int, n_a: int, n_b: int) -> None:
    isi = f"""# Petunjuk Anotasi Red-Flag (untuk dokter penilai)

## Yang diminta

Untuk setiap pertanyaan pasien di `anotasi_redflag.csv`, jawab **satu** hal:

> **Apakah keluhan ini perlu penanganan SEGERA (rujuk/IGD hari itu juga),
> berdasarkan teks yang tertulis saja?**

Isi kolom **`ada_tanda_bahaya`** dengan:
- **Y** — ya, ada tanda bahaya; pasien seharusnya diarahkan segera ke fasilitas kesehatan.
- **T** — tidak; ini pertanyaan/keluhan yang wajar dijawab sebagai informasi kesehatan biasa.

Kolom `kategori` (opsional) — bila **Y**, tulis salah satu:
{chr(10).join(f"- `{k}` — {v}" for k, v in KATEGORI.items())}

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

Sampel ditarik **berstrata** dari {n_pop_a + n_pop_b} pertanyaan test set:
{n_a} dari {n_pop_a} yang terpicu detektor dan {n_b} dari {n_pop_b} yang tidak.
Penilai **tidak diberi tahu** keluaran detektor (anotasi buta) dan urutannya diacak.
Bobot strata dipakai saat menghitung recall/FPR agar angkanya mewakili populasi penuh.
"""
    with open(FILE_PETUNJUK, "w", encoding="utf-8") as f:
        f.write(isi)


# --------------------------------------------------------------------------------------
def ke_space(test_file: str, n_terpicu: int, n_waspada: int, n_biasa: int) -> None:
    """Tulis daftar pertanyaan Tugas 2 (sampel tahap 2) untuk Space arena.

    Dari 400 pertanyaan tahap 1, diambil acak per kelompok: `n_terpicu` yang terpicu
    detektor, `n_waspada` yang tidak terpicu tetapi memuat kata waspada (default: semua),
    dan `n_biasa` sisanya. Karena 400 itu sendiri acak di dalam strata detektor, ambilan
    acak dari dalamnya tetap acak di dalam kelompok yang lebih halus ini. Ukuran populasi
    tiap kelompok dihitung pada SELURUH test set, untuk bobot di --score.

    Setiap pertanyaan boleh dinilai sampai dua dokter; Space mengisi dulu yang belum
    dinilai siapa pun. Berkas untuk Space memuat HANYA id dan teks — penilaian buta.
    """
    with open(FILE_KUNCI, encoding="utf-8") as f:
        kunci = {r["id"]: r for r in csv.DictReader(f)}
    with open(FILE_ANOTASI, encoding="utf-8-sig") as f:
        teks = {r["id"]: r["pertanyaan"] for r in csv.DictReader(f)}

    populasi = Counter(kelompok(q) for q in muat_pertanyaan(test_file))
    per_kel: dict[str, list[str]] = {}
    for i in kunci:
        per_kel.setdefault(kelompok(teks[i]), []).append(i)

    rng = random.Random(SEED + 1)
    urutan, id_kel = [], {}
    for kel, n in (("terpicu", n_terpicu), ("tidak_waspada", n_waspada), ("tidak_biasa", n_biasa)):
        ids = sorted(per_kel.get(kel, []))
        rng.shuffle(ids)
        ambil = ids if n < 0 else ids[:n]
        urutan += ambil
        id_kel.update({i: kel for i in ambil})
        print(f"  {kel:14s} populasi {populasi[kel]:5d} | di tahap 1 {len(ids):3d} | diambil {len(ambil):3d}")
    rng.shuffle(urutan)  # kelompok tak bisa ditebak dari urutan

    with open(FILE_TAHAP2, "w", encoding="utf-8") as f:
        json.dump({"populasi": dict(populasi), "kelompok": id_kel}, f, ensure_ascii=False, indent=1)

    isi = {
        "dibuat": "redflag_annotate.py --ke-space",
        "kategori": KATEGORI,
        "soal": [{"id": i, "pertanyaan": teks[i], "target": 2} for i in urutan],
    }
    with open(FILE_SPACE, "w", encoding="utf-8") as f:
        json.dump(isi, f, ensure_ascii=False, indent=1)
    print(f"{len(urutan)} pertanyaan, maks. 2 dokter per pertanyaan -> {FILE_SPACE}")
    print(f"Kelompok & populasi (JANGAN ke Space) -> {FILE_TAHAP2}")

    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=FILE_SPACE, path_in_repo="redflag_soal.json",
                        repo_id=HF_DATASET, repo_type="dataset",
                        commit_message="daftar pertanyaan Tugas 2 (tanda bahaya)")
    print(f"Diunggah -> {HF_DATASET}/redflag_soal.json (restart Space supaya termuat)")


def label_dari_hf() -> dict[str, list[tuple[str, str, str]]]:
    """id -> [(penilai, label, kategori)] dari berkas `redflag/*.json` di dataset penilaian."""
    from huggingface_hub import snapshot_download
    folder = snapshot_download(HF_DATASET, repo_type="dataset", allow_patterns=["redflag/*.json"])
    hasil: dict[str, list[tuple[str, str, str]]] = {}
    d = os.path.join(folder, "redflag")
    for nama in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        rek = json.load(open(os.path.join(d, nama), encoding="utf-8"))
        if rek.get("uji"):
            continue  # baris uji fungsional sistem
        for lab in rek["label"]:
            hasil.setdefault(lab["id"], []).append((rek["email"], lab["label"], lab.get("kategori", "")))
    return hasil


def label_dari_csv(path: str) -> dict[str, list[tuple[str, str, str]]]:
    """Jalur lama: satu penilai mengisi anotasi_redflag.csv langsung."""
    hasil: dict[str, list[tuple[str, str, str]]] = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            label = (r.get("ada_tanda_bahaya") or "").strip().upper()
            if label in {"Y", "T"}:
                hasil[r["id"]] = [("csv", label, (r.get("kategori") or "").strip())]
    return hasil


def _kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _bootstrap(per_strata: dict[str, list[tuple[int, int]]], populasi: dict[str, float],
               n_iter: int = 5000) -> dict:
    """CI 95% untuk recall & FPR lewat bootstrap BERSTRATA: tiap strata diambil ulang
    di dalam dirinya sendiri, lalu diberi bobot populasi/terisi seperti estimasi utamanya.

    per_strata: stratum -> [(label_dokter, terpicu_detektor)]
    """
    rng = random.Random(SEED)
    recalls, fprs = [], []
    for _ in range(n_iter):
        tp = fn = fp = tn = 0.0
        for s, isi in per_strata.items():
            b = populasi[s] / len(isi)
            for _ in range(len(isi)):
                y, p = isi[rng.randrange(len(isi))]
                if y and p:
                    tp += b
                elif y:
                    fn += b
                elif p:
                    fp += b
                else:
                    tn += b
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        if fp + tn > 0:
            fprs.append(fp / (fp + tn))
    recalls.sort()
    fprs.sort()

    def ci(v: list[float]) -> list[float | None]:
        if not v:
            return [None, None]
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]

    return {"recall_ci95": ci(recalls), "fpr_ci95": ci(fprs)}


def score(dari_csv: bool) -> None:
    if not os.path.exists(FILE_ANOTASI) or not os.path.exists(FILE_KUNCI):
        raise SystemExit(f"Belum ada {FILE_ANOTASI} / {FILE_KUNCI}. Jalankan --build dulu.")

    with open(FILE_KUNCI, encoding="utf-8") as f:
        kunci = {r["id"]: r for r in csv.DictReader(f)}

    # Ukuran populasi tiap strata = bobot rencana × jumlah sampel rencana (195 / 2803).
    # Strata untuk pembobotan: kelompok tahap 2 (detektor x kata waspada) bila ada, kalau
    # tidak strata tahap 1. Ukuran populasi strata tahap 1 = bobot rencana x jumlah sampel.
    if os.path.exists(FILE_TAHAP2):
        with open(FILE_TAHAP2, encoding="utf-8") as f:
            t2 = json.load(f)
        populasi = {k: float(v) for k, v in t2["populasi"].items()}
        strata_id = t2["kelompok"]
    else:
        populasi = {}
        for k in kunci.values():
            populasi[k["stratum"]] = populasi.get(k["stratum"], 0) + float(k["bobot"])
        strata_id = {i: k["stratum"] for i, k in kunci.items()}

    if dari_csv:
        berkas = FILE_ANOTASI
        label_per_id = label_dari_csv(FILE_ANOTASI)
    else:
        berkas = f"{HF_DATASET}/redflag/*.json"
        label_per_id = label_dari_hf()
    # (penilai, label, kategori) -> buang penilai; satu penilai dihitung sekali per id.
    per_id: dict[str, list[tuple[str, str]]] = {}
    for i, v in label_per_id.items():
        unik = {pen: (lab, kat) for pen, lab, kat in v}
        per_id[i] = list(unik.values())

    # Kesepakatan antar-dokter pada pertanyaan yang dinilai dua dokter berbeda.
    ganda = {i: v[:2] for i, v in per_id.items() if len(v) >= 2}
    a = [int(v[0][0] == "Y") for v in ganda.values()]
    b = [int(v[1][0] == "Y") for v in ganda.values()]
    kesepakatan = {
        "n": len(ganda),
        "setuju": sum(x == y for x, y in zip(a, b)),
        "kappa": _kappa(a, b),
        "id_berbeda": sorted(i for i, v in ganda.items() if v[0][0] != v[1][0]),
    }

    per_strata: dict[str, list[tuple[int, int]]] = {}
    per_kategori: dict[str, list[int]] = {}
    for i, v in per_id.items():
        k = kunci.get(i)
        if k is None:
            continue
        # Dua dokter berbeda pendapat -> Y. Sama dengan aturan 1 petunjuk ("kalau ragu,
        # pilih Y"); sistemnya dirancang high-recall, jadi keraguan dihitung sebagai bahaya.
        y = int(any(lab == "Y" for lab, _ in v))
        p = int(k["terpicu"])
        if i not in strata_id:
            continue  # dinilai tetapi di luar sampel tahap 2 (mis. dari daftar lama)
        per_strata.setdefault(strata_id[i], []).append((y, p))
        if y:
            kat = next((kt for lab, kt in v if lab == "Y" and kt), "") or "(tak diisi)"
            per_kategori.setdefault(kat, []).append(p)

    if not per_strata:
        raise SystemExit("Belum ada baris yang dianotasi (kolom ada_tanda_bahaya masih kosong).")
    kurang = sorted(set(populasi) - set(per_strata))
    if kurang:
        raise SystemExit(f"Strata {kurang} belum punya satu pun baris terisi — belum bisa dihitung.")

    # Bobot dari baris yang BENAR-BENAR terisi, bukan dari rencana 150/250.
    tp = fn = fp = tn = 0.0
    for s, isi in per_strata.items():
        w = populasi[s] / len(isi)
        tp += w * sum(y and p for y, p in isi)
        fn += w * sum(y and not p for y, p in isi)
        fp += w * sum(p and not y for y, p in isi)
        tn += w * sum(not y and not p for y, p in isi)

    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    presisi = tp / (tp + fp) if tp + fp else None
    prevalensi = (tp + fn) / (tp + fn + fp + tn)

    n_dianotasi = sum(len(v) for v in per_strata.values())
    ci = _bootstrap(per_strata, populasi)
    hasil = {
        "berkas": berkas,
        "n_dianotasi": n_dianotasi,
        "n_belum_dianotasi": len(strata_id) - n_dianotasi,
        "per_strata": {s: {"populasi": round(populasi[s]), "terisi": len(v),
                           "label_Y": sum(y for y, _ in v)} for s, v in sorted(per_strata.items())},
        "kesepakatan_antar_dokter": kesepakatan,
        "terbobot": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "recall": recall,
        "false_positive_rate": fpr,
        "presisi": presisi,
        "prevalensi_tanda_bahaya": prevalensi,
        **ci,
        "recall_per_kategori": {
            k: {"n": len(v), "recall": sum(v) / len(v)} for k, v in sorted(per_kategori.items())
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(FILE_HASIL, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)

    def pct(x):
        return "-" if x is None else f"{x * 100:.1f}%"

    print(f"Sumber label     : {berkas}")
    print(f"Dianotasi        : {n_dianotasi} dari {len(strata_id)} pertanyaan sampel")
    for s, v in hasil["per_strata"].items():
        print(f"  {s:14s} terisi {v['terisi']:3d}  (populasi {v['populasi']}, label Y {v['label_Y']})")
    if kesepakatan["n"]:
        kp = kesepakatan["kappa"]
        print(f"Kesepakatan      : {kesepakatan['setuju']}/{kesepakatan['n']} baris sama, "
              f"kappa {'-' if kp is None else f'{kp:.2f}'}; berbeda -> dihitung Y")
    print(f"Prevalensi       : {pct(prevalensi)} pertanyaan test set memuat tanda bahaya (terbobot)")
    print()
    print(f"RECALL           : {pct(recall)}  CI95 [{pct(ci['recall_ci95'][0])}, {pct(ci['recall_ci95'][1])}]")
    print(f"FALSE-POS RATE   : {pct(fpr)}  CI95 [{pct(ci['fpr_ci95'][0])}, {pct(ci['fpr_ci95'][1])}]")
    print(f"Presisi          : {pct(presisi)}   (rendah = konsekuensi setelan high-recall)")
    if hasil["recall_per_kategori"]:
        print("\nRecall per kategori (menurut label dokter):")
        for k, v in hasil["recall_per_kategori"].items():
            print(f"  {k:22s} n={v['n']:3d}  {pct(v['recall'])}")
    print(f"\nDisimpan -> {FILE_HASIL}")


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="statistik detektor di seluruh test set")
    ap.add_argument("--build", action="store_true", help="bikin berkas anotasi buta + kunci")
    ap.add_argument("--ke-space", action="store_true",
                    help="tulis daftar pertanyaan Tugas 2 untuk Space arena")
    ap.add_argument("--score", action="store_true", help="hitung recall/FPR dari anotasi terisi")
    ap.add_argument("--dari-csv", action="store_true",
                    help="--score dari anotasi_redflag.csv, bukan dari dataset HF")
    ap.add_argument("--space-terpicu", type=int, default=20,
                    help="--ke-space: pertanyaan dari kelompok TERPICU detektor")
    ap.add_argument("--space-waspada", type=int, default=-1,
                    help="--ke-space: tidak terpicu tapi ada kata waspada (-1 = semua)")
    ap.add_argument("--space-biasa", type=int, default=21,
                    help="--ke-space: tidak terpicu dan tanpa kata waspada")
    ap.add_argument("--test-file", default=TEST_FILE)
    ap.add_argument("--n-terpicu", type=int, default=150, help="jumlah sampel dari strata TERPICU")
    ap.add_argument("--n-tidak", type=int, default=250, help="jumlah sampel dari strata TIDAK terpicu")
    args = ap.parse_args()

    if args.stats:
        stats(args.test_file)
    elif args.build:
        build(args.test_file, args.n_terpicu, args.n_tidak)
    elif args.ke_space:
        ke_space(args.test_file, args.space_terpicu, args.space_waspada, args.space_biasa)
    elif args.score:
        score(args.dari_csv)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
