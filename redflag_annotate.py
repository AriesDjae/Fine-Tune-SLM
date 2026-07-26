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

    # 3) dokter mengisi kolom `ada_tanda_bahaya` (Y/T) di
    #    results/redflag/anotasi_redflag.csv  (jangan ubah kolom `id`)

    # 4) hitung recall / FPR / presisi + selang kepercayaan bootstrap
    python redflag_annotate.py --score

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
from collections import Counter

from redflag import KATEGORI, detect

TEST_FILE = os.path.join("Data", "processed_id", "test.jsonl")
OUT_DIR = os.path.join("results", "redflag")
FILE_ANOTASI = os.path.join(OUT_DIR, "anotasi_redflag.csv")
FILE_KUNCI = os.path.join(OUT_DIR, "anotasi_redflag_kunci.csv")
FILE_PETUNJUK = os.path.join(OUT_DIR, "PETUNJUK_ANOTASI.md")
FILE_HASIL = os.path.join(OUT_DIR, "hasil_redflag.json")
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
def _bootstrap(pasangan: list[tuple[int, int, float]], n_iter: int = 5000) -> dict:
    """CI 95% untuk recall & FPR lewat bootstrap berstrata.

    pasangan: (label_dokter, terpicu_detektor, bobot)
    """
    rng = random.Random(SEED)
    recalls, fprs = [], []
    n = len(pasangan)
    for _ in range(n_iter):
        contoh = [pasangan[rng.randrange(n)] for _ in range(n)]
        tp = sum(b for y, p, b in contoh if y == 1 and p == 1)
        fn = sum(b for y, p, b in contoh if y == 1 and p == 0)
        fp = sum(b for y, p, b in contoh if y == 0 and p == 1)
        tn = sum(b for y, p, b in contoh if y == 0 and p == 0)
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


def score() -> None:
    if not os.path.exists(FILE_ANOTASI) or not os.path.exists(FILE_KUNCI):
        raise SystemExit(f"Belum ada {FILE_ANOTASI} / {FILE_KUNCI}. Jalankan --build dulu.")

    with open(FILE_KUNCI, encoding="utf-8") as f:
        kunci = {r["id"]: r for r in csv.DictReader(f)}

    with open(FILE_ANOTASI, encoding="utf-8-sig") as f:
        anotasi = list(csv.DictReader(f))

    pasangan: list[tuple[int, int, float]] = []
    per_kategori: dict[str, list[int]] = {}
    kosong = 0
    for row in anotasi:
        label = (row.get("ada_tanda_bahaya") or "").strip().upper()
        if label not in {"Y", "T"}:
            kosong += 1
            continue
        k = kunci.get(row["id"])
        if k is None:
            continue
        y = 1 if label == "Y" else 0
        p = int(k["terpicu"])
        bobot = float(k["bobot"])
        pasangan.append((y, p, bobot))
        if y == 1:
            kat = (row.get("kategori") or "").strip() or "(tak diisi)"
            per_kategori.setdefault(kat, []).append(p)

    if not pasangan:
        raise SystemExit("Belum ada baris yang dianotasi (kolom ada_tanda_bahaya masih kosong).")

    tp = sum(b for y, p, b in pasangan if y == 1 and p == 1)
    fn = sum(b for y, p, b in pasangan if y == 1 and p == 0)
    fp = sum(b for y, p, b in pasangan if y == 0 and p == 1)
    tn = sum(b for y, p, b in pasangan if y == 0 and p == 0)

    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    presisi = tp / (tp + fp) if tp + fp else None
    prevalensi = (tp + fn) / (tp + fn + fp + tn)

    ci = _bootstrap(pasangan)
    hasil = {
        "n_dianotasi": len(pasangan),
        "n_belum_dianotasi": kosong,
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

    print(f"Dianotasi        : {len(pasangan)} (belum diisi: {kosong})")
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
    ap.add_argument("--score", action="store_true", help="hitung recall/FPR dari anotasi terisi")
    ap.add_argument("--test-file", default=TEST_FILE)
    ap.add_argument("--n-terpicu", type=int, default=150, help="jumlah sampel dari strata TERPICU")
    ap.add_argument("--n-tidak", type=int, default=250, help="jumlah sampel dari strata TIDAK terpicu")
    args = ap.parse_args()

    if args.stats:
        stats(args.test_file)
    elif args.build:
        build(args.test_file, args.n_terpicu, args.n_tidak)
    elif args.score:
        score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
