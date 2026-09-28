"""
susun_validasi_tier3.py — menyusun 100 pasangan validasi dokter (Tier 3) untuk Space.

Tujuan: dokter menilai PERSIS pasangan yang sudah dinilai GPT-4o di Tier 2, dengan rubrik
yang sama (Aman / Akurat / Menyesatkan + jawaban mana yang lebih baik), supaya kesepakatan
dokter↔GPT-4o sah dilaporkan di naskah.

Kenapa tidak memakai bank arena (Tugas 1 lama): jawaban di bank itu dibangkitkan ulang dari
GGUF Q4_K_M dengan batas 1024 token, sedangkan GPT-4o menilai jawaban 16-bit dengan batas
256 token (results/preds_p6). Teksnya berbeda, jadi kesepakatan di atas bank itu tidak
membandingkan hal yang sama.

Aturan penyusunan (semua berseed, tidak ada pilihan tangan):
  - sumber   : 200 butir Tier 2 (random.Random(42).sample(range(2998), 200)), satu ukuran
               model per pertanyaan, jadi 100 pertanyaan BERBEDA;
  - ukuran   : 34 / 33 / 33 untuk 0.8B / 2B / 4B;
  - red-flag : semua pertanyaan red-flag di sampel Tier 2 ikut, disebar merata antar-ukuran;
  - teks     : jawaban A/B = teks preds yang disodorkan ke GPT-4o, urutan A/B = urutan yang
               dilihat GPT-4o (kunci di tier2_<ukuran>.jsonl); referensi dokter TIDAK
               ditampilkan, sama seperti Tier 2;
  - ganda    : 20 pasangan ber-target 2 (dinilai dua dokter berbeda) untuk kesepakatan
               antar-dokter. Total slot 120 = 4 dokter × 30.

Keluaran:
  human_eval/validasi_indeks.json   id → idx_test, ukuran, kunci A/B, target (DI-TRACK git;
                                    tanpa teks, sama seperti bank_indeks.json)
  human_eval/validasi_soal.json     teks pertanyaan + jawaban A/B, TANPA ukuran/kunci
                                    (TIDAK di-track; diunggah ke dataset privat)

Jalankan:  python scripts/susun_validasi_tier3.py            # ringkasan saja
           python scripts/susun_validasi_tier3.py --tulis    # tulis kedua berkas
           python scripts/susun_validasi_tier3.py --unggah   # tulis + unggah ke dataset privat
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AKAR)
os.chdir(AKAR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm_judge_tier2 import SEED as SEED_TIER2, TEST_FILE, muat_preds, muat_test  # noqa: E402
from redflag.detector import detect  # noqa: E402

UKURAN = ["0.8b", "2b", "4b"]
JATAH = {"0.8b": 34, "2b": 33, "4b": 33}
N_GANDA = 20
SEED_VALIDASI = 2026
HF_DATASET = "AriesDjaenuri/p6-penilaian-dokter"
KELUAR_IDX = os.path.join("human_eval", "validasi_indeks.json")
KELUAR_SOAL = os.path.join("human_eval", "validasi_soal.json")


def muat_tier2(u: str) -> dict[int, dict]:
    path = os.path.join("results", "llm_judge", f"tier2_{u}.jsonl")
    with open(path, encoding="utf-8") as f:
        return {r["idx"]: r for r in map(json.loads, f) if r.get("nilai")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tulis", action="store_true")
    ap.add_argument("--unggah", action="store_true", help="tulis lalu unggah validasi_soal.json")
    a = ap.parse_args()

    test = muat_test(TEST_FILE)
    tier2 = {u: muat_tier2(u) for u in UKURAN}
    sampel = random.Random(SEED_TIER2).sample(range(len(test)), 200)
    # Hanya butir yang punya penilaian GPT-4o di KETIGA ukuran, supaya ukuran mana pun
    # yang jatuh ke sebuah pertanyaan, pasangan pembandingnya ada.
    sah = [i for i in sampel if all(i in tier2[u] for u in UKURAN)]
    merah = [i for i in sah if detect(test[i]["pertanyaan"]).terpicu]
    umum = [i for i in sah if i not in merah]
    print(f"butir Tier 2 lengkap di 3 ukuran: {len(sah)} (red-flag {len(merah)})")

    rng = random.Random(SEED_VALIDASI)
    rng.shuffle(merah)
    rng.shuffle(umum)
    butuh = sum(JATAH.values())
    urutan = merah + umum[:butuh - len(merah)]

    # Bagi ke ukuran bergiliran; red-flag lebih dulu sehingga tersebar merata.
    sisa = dict(JATAH)
    tugas = []
    k = 0
    for i in urutan:
        while sisa[UKURAN[k % 3]] == 0:
            k += 1
        u = UKURAN[k % 3]
        sisa[u] -= 1
        tugas.append((i, u))
        k += 1

    ganda = set(rng.sample(range(len(tugas)), N_GANDA))
    preds = {u: (muat_preds(f"qwen{u}_baseline"), muat_preds(f"qwen{u}_finetuned")) for u in UKURAN}

    indeks, soal = [], []
    for n, (i, u) in enumerate(tugas):
        kunci = tier2[u][i]["kunci"]  # {"A": "baseline"/"finetuned", "B": ...}
        base, ft = preds[u][0][i], preds[u][1][i]
        teks = {"baseline": base, "finetuned": ft}
        sid = f"V{n + 1:03d}"
        target = 2 if n in ganda else 1
        indeks.append({"id": sid, "idx_test": i, "ukuran": u, "kunci": kunci, "target": target,
                       "redflag": i in merah})
        soal.append({"id": sid, "pertanyaan": test[i]["pertanyaan"].strip(),
                     "A": (teks[kunci["A"]] or "(kosong)").strip(),
                     "B": (teks[kunci["B"]] or "(kosong)").strip(),
                     "target": target})

    # Urutan pembagian ke dokter diacak, supaya dokter pertama tidak mendapat semua 0.8B.
    urut = list(range(len(soal)))
    rng.shuffle(urut)
    soal = [soal[j] for j in urut]

    from collections import Counter
    print("per ukuran :", dict(Counter(x["ukuran"] for x in indeks)))
    print("red-flag   :", dict(Counter(x["ukuran"] for x in indeks if x["redflag"])))
    print("ganda      :", dict(Counter(x["ukuran"] for x in indeks if x["target"] == 2)),
          f"→ total slot {sum(x['target'] for x in indeks)}")
    kata = sum(len((s["A"] + " " + s["B"]).split()) for s in soal) / len(soal)
    print(f"rata-rata kata jawaban per pasangan: {kata:.0f}")

    if not (a.tulis or a.unggah):
        print("\n(ringkasan saja — pakai --tulis atau --unggah)")
        return
    meta = {"seed_tier2": SEED_TIER2, "seed_validasi": SEED_VALIDASI, "jatah": JATAH,
            "n_ganda": N_GANDA, "sumber_teks": "results/preds_p6 (sama dengan Tier 2)",
            "referensi_ditampilkan": False}
    with open(KELUAR_IDX, "w", encoding="utf-8") as f:
        json.dump({**meta, "pasangan": indeks}, f, ensure_ascii=False, indent=1)
    with open(KELUAR_SOAL, "w", encoding="utf-8") as f:
        json.dump({"versi": "tier3-v1", "soal": soal}, f, ensure_ascii=False, indent=1)
    print(f"tertulis: {KELUAR_IDX}, {KELUAR_SOAL}")

    if a.unggah:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=KELUAR_SOAL, path_in_repo="validasi_soal.json",
                            repo_id=HF_DATASET, repo_type="dataset",
                            commit_message="validasi Tier 3: 100 pasangan (teks Tier 2)")
        print(f"diunggah -> {HF_DATASET}/validasi_soal.json (restart Space supaya termuat)")


if __name__ == "__main__":
    main()
