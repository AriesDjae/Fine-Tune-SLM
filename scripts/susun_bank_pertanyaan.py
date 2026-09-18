"""
susun_bank_pertanyaan.py — menyusun bank pertanyaan Tier 3 (arena penilaian dokter).

Rancangan: `webapp/RENCANA_BANK.md` § 4. Ringkasnya, bank berisi 40 pertanyaan:

    10  gold set          human_eval/gold_set.csv            (punya jawaban referensi)
    24  test set "umum"   dari 200 indeks sampel Tier 2      (supaya κ dokter↔GPT-4o hidup)
     6  test set red-flag dari 200 indeks yang sama          (dimensi keselamatan)

Kenapa 30-nya HARUS dari sampel Tier 2 dan bukan acak dari 2998: pasangan
(pertanyaan × model) yang sama sudah dinilai GPT-4o di Tier 2, jadi kesepakatan dokter↔judge
bisa dihitung. Sampelnya direproduksi persis seperti llm_judge_tier2.py:

    random.Random(42).sample(range(2998), 200)

(sudah diverifikasi identik dengan idx di results/llm_judge/tier2_*.jsonl untuk ketiga ukuran).

DUA KELUARAN, disengaja:

    human_eval/bank_pertanyaan.csv   teks pertanyaan — TIDAK di-track git
    human_eval/bank_indeks.json      id + indeks + kategori — di-track git

Pembagian itu mengikuti sikap legal repo: `Data/` gitignored karena isinya QnA Alodokter
verbatim, dan 30 pertanyaan verbatim di berkas ter-track akan membocorkan sikap itu lewat
pintu belakang. Indeks bukan konten, jadi ia yang menjadi catatan reproduksi.

Jalankan:  python scripts/susun_bank_pertanyaan.py [--tulis]
Tanpa --tulis hanya mencetak ringkasan (aman dijalankan berulang).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redflag.detector import detect          # noqa: E402

TEST = os.path.join("Data", "processed_id", "test.jsonl")
GOLD = os.path.join("human_eval", "gold_set.csv")
KELUAR_CSV = os.path.join("human_eval", "bank_pertanyaan.csv")
KELUAR_IDX = os.path.join("human_eval", "bank_indeks.json")

SEED_TIER2 = 42          # HARUS sama dengan llm_judge_tier2.SEED
N_TIER2 = 200
N_UMUM = 24
N_REDFLAG = 6

# Batas panjang pertanyaan yang boleh masuk bank. Bukan selera:
#  - < 60 karakter  : sering pertanyaan satu baris tanpa konteks klinis, jawabannya
#                     jadi ceramah umum yang tidak membedakan keenam model.
#  - > 450 karakter : dokter membaca 6 jawaban untuk tiap pertanyaan; keluhan yang
#                     sepanjang setengah layar membuat satu sesi tidak selesai.
MIN_PANJANG = 60
MAKS_PANJANG = 450


def muat_test() -> list[str]:
    with open(TEST, encoding="utf-8") as f:
        return [json.loads(b)["messages"][1]["content"].strip() for b in f]


def muat_gold() -> list[dict]:
    with open(GOLD, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def susun() -> list[dict]:
    soal = muat_test()
    idxs = random.Random(SEED_TIER2).sample(range(len(soal)), N_TIER2)

    umum, merah = [], []
    for i in idxs:
        q = soal[i]
        if not (MIN_PANJANG <= len(q) <= MAKS_PANJANG):
            continue
        hasil = detect(q)
        (merah if hasil.terpicu else umum).append((i, q, hasil.kategori))

    # Urutan sampel Tier 2 dipertahankan apa adanya — memilih "yang paling menarik"
    # dengan mata sendiri akan menyeludupkan preferensi peneliti ke dalam bank.
    dipilih_umum = umum[:N_UMUM]

    # Untuk red-flag, urutan mentah saja tidak cukup: ia bisa memberi 2 trauma dan 0
    # kardiovaskular. Ambil satu per kategori dulu (tetap dalam urutan sampel), baru
    # sisanya — jadi keenam kategori yang paling berbahaya terwakili tanpa ada satu pun
    # pertanyaan yang dipilih karena isinya.
    dipilih_merah, sudah = [], set()
    for i, q, kat in merah:
        if kat[0] not in sudah:
            dipilih_merah.append((i, q, kat))
            sudah.add(kat[0])
    for i, q, kat in merah:
        if len(dipilih_merah) >= N_REDFLAG:
            break
        if (i, q, kat) not in dipilih_merah:
            dipilih_merah.append((i, q, kat))
    dipilih_merah = dipilih_merah[:N_REDFLAG]

    baris = []
    for n, g in enumerate(muat_gold(), 1):
        baris.append({"id": f"G{n:03d}", "sumber": "gold_set", "idx_test": "",
                      "kategori": g["kategori"], "pertanyaan": g["pertanyaan"],
                      "catatan": g["catatan_keamanan"]})
    for n, (i, q, _) in enumerate(dipilih_umum, 1):
        baris.append({"id": f"T{n:02d}", "sumber": "test_tier2", "idx_test": i,
                      "kategori": "umum", "pertanyaan": q, "catatan": ""})
    for n, (i, q, kat) in enumerate(dipilih_merah, 1):
        baris.append({"id": f"R{n:02d}", "sumber": "test_tier2", "idx_test": i,
                      "kategori": "redflag:" + "|".join(kat), "pertanyaan": q,
                      "catatan": "wajib dinilai dimensi keamanan"})
    return baris, len(merah), len(umum)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tulis", action="store_true", help="tulis kedua berkas keluaran")
    a = ap.parse_args()

    baris, n_merah, n_umum = susun()
    print(f"kandidat setelah saring panjang {MIN_PANJANG}-{MAKS_PANJANG}: "
          f"{n_umum} umum + {n_merah} red-flag (dari {N_TIER2} sampel Tier 2)")
    if n_merah < N_REDFLAG or n_umum < N_UMUM:
        print("KURANG kandidat — longgarkan batas panjang atau turunkan target.")
        return 1

    print(f"bank tersusun: {len(baris)} pertanyaan "
          f"({sum(b['sumber'] == 'gold_set' for b in baris)} gold, "
          f"{N_UMUM} umum, {N_REDFLAG} red-flag)")
    for b in baris:
        print(f"  {b['id']:5s} {str(b['idx_test']):>5s}  {b['kategori'][:28]:28s} "
              f"{b['pertanyaan'][:70]}")

    if not a.tulis:
        print("\n(kering — jalankan ulang dengan --tulis untuk menyimpan)")
        return 0

    with open(KELUAR_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "sumber", "idx_test", "kategori",
                                          "pertanyaan", "catatan"])
        w.writeheader()
        w.writerows(baris)

    # Catatan reproduksi yang boleh masuk git: indeks + kategori, tanpa teks sumber.
    with open(KELUAR_IDX, "w", encoding="utf-8") as f:
        json.dump({"seed_tier2": SEED_TIER2, "n_sampel_tier2": N_TIER2,
                   "saring_panjang": [MIN_PANJANG, MAKS_PANJANG],
                   "bank": [{k: b[k] for k in ("id", "sumber", "idx_test", "kategori")}
                            for b in baris]},
                  f, ensure_ascii=False, indent=2)
    print(f"\ntertulis: {KELUAR_CSV} (gitignored) + {KELUAR_IDX} (ter-track)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
