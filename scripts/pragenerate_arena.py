"""
pragenerate_arena.py — membuat bank jawaban Tier 3 di laptop, sekali jalan.

Rancangan: `arsip/pivot6_jalur-lama/webapp/RENCANA_BANK.md`. Masukan
`human_eval/bank_pertanyaan.csv` (disusun `scripts/susun_bank_pertanyaan.py`),
keluaran `hf_space_arena/bank/<versi>/<hash>.json`
yang ikut ter-deploy bersama fungsi Vercel.

Kenapa ini sah dan bukan "curang": decoding arena greedy + `seed=42` + `no_repeat_ngram=3`,
jadi jawaban keenam model DETERMINISTIK. Jawaban yang dibuat semalam sebelumnya identik
dengan yang akan dihasilkan server saat itu juga — pra-generate hanya memindahkan kapan
komputasinya terjadi, bukan apa hasilnya.

    python scripts/pragenerate_arena.py               # seluruh bank, bisa dilanjut
    python scripts/pragenerate_arena.py --batas 2     # 2 pertanyaan pertama (uji cepat)
    python scripts/pragenerate_arena.py --label "Model D"   # satu model saja
    python scripts/pragenerate_arena.py --ringkas     # cuma laporkan isi bank saat ini

BERJALAN PER MODEL, bukan per pertanyaan: memuat satu GGUF lalu menjawab 40 pertanyaan
jauh lebih murah daripada menukar bobot 40 kali. Tiap jawaban langsung ditulis ke
berkasnya, jadi mati listrik di tengah jalan tidak mengulang dari nol.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys

_AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _AKAR)

from arena import mesin                                      # noqa: E402
from arena.mesin import LABELS, PARAMS_GENERASI              # noqa: E402

BANK_CSV = os.path.join(_AKAR, "human_eval", "bank_pertanyaan.csv")
KELUAR = os.path.join(_AKAR, "hf_space_arena", "bank")


def muat_bank() -> list[dict]:
    if not os.path.exists(BANK_CSV):
        raise SystemExit(f"Tidak ada {BANK_CSV} — jalankan dulu "
                         "`python scripts/susun_bank_pertanyaan.py --tulis`")
    with open(BANK_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jalur(versi: str, h: str) -> str:
    return os.path.join(KELUAR, versi, f"{h}.json")


def baca(versi: str, h: str) -> dict | None:
    p = jalur(versi, h)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tulis(versi: str, rek: dict) -> None:
    p = jalur(versi, rek["hash"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rek, f, ensure_ascii=False, indent=1)


def rekaman_baru(baris: dict, versi: str) -> dict:
    return {
        "id": baris["id"],
        "pertanyaan": baris["pertanyaan"],
        "hash": mesin.hash_pertanyaan(baris["pertanyaan"]),
        "kategori": baris["kategori"],
        "sumber": baris["sumber"],
        "idx_test": baris.get("idx_test", ""),
        "versi_protokol": versi,
        # Dicatat supaya pelanggaran "separuh bank di CPU, separuh di CUDA" kelihatan
        # saat analisis — llama.cpp bisa berbeda tipis antar-backend.
        "mesin": "lokal-cpu",
        "params_dipakai": {k: v for k, v in PARAMS_GENERASI.items()},
        "jawaban": {},
    }


BENCH = os.path.join(_AKAR, "results", "bench_cpu_lokal.json")


def tulis_metrik() -> None:
    """Simpan metrik kecepatan/memori per label — hasil sampingan gratis dari run bank.

    Bukan jawaban RQ3: perangkatnya laptop, bukan HP 3-4 GB, dan labelnya masih buta
    (pemetaan ke ukuran model ada di arena/config.py). Tapi ini satu-satunya angka
    llama.cpp CPU yang dimiliki repo, dan ia diukur pada jalur yang PERSIS sama dengan
    artefak deploy (GGUF Q4_K_M, n_ctx & prompt yang sama).
    """
    if not mesin.metrik:
        return
    import platform
    os.makedirs(os.path.dirname(BENCH), exist_ok=True)
    with open(BENCH, "w", encoding="utf-8") as f:
        json.dump({
            "catatan": "hasil sampingan pragenerate_arena.py — laptop peneliti, CPU-only, "
                       "label masih buta. BUKAN pengganti uji HP 3-4 GB (RQ3).",
            "waktu": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "mesin": {"platform": platform.platform(), "prosesor": platform.processor(),
                      "python": platform.python_version()},
            "params": {k: v for k, v in PARAMS_GENERASI.items() if k != "system_prompt"},
            "per_label": mesin.metrik,
        }, f, ensure_ascii=False, indent=2)


def ringkas(versi: str) -> int:
    bank = muat_bank()
    lengkap = potong = 0
    for b in bank:
        rek = baca(versi, mesin.hash_pertanyaan(b["pertanyaan"]))
        ada = len(rek["jawaban"]) if rek else 0
        potong += sum(1 for j in (rek or {}).get("jawaban", {}).values() if j["terpotong"])
        lengkap += ada == len(LABELS)
        tanda = "OK " if ada == len(LABELS) else f"{ada}/{len(LABELS)}"
        print(f"  {b['id']:5s} {tanda:5s} {b['pertanyaan'][:60]}")
    print(f"\nversi protokol : {versi}")
    print(f"lengkap        : {lengkap}/{len(bank)} pertanyaan")
    print(f"terpotong      : {potong} jawaban dari {lengkap * len(LABELS)} yang selesai")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batas", type=int, default=0, help="hanya N pertanyaan pertama")
    ap.add_argument("--label", action="append", help="hanya label ini (boleh berulang)")
    ap.add_argument("--ringkas", action="store_true", help="laporkan isi bank, jangan generate")
    ap.add_argument("--ulang", action="store_true", help="timpa jawaban yang sudah ada")
    a = ap.parse_args()

    versi = mesin.versi_protokol()
    if a.ringkas:
        return ringkas(versi)

    # GERBANG. Tanpa pemblokir n-gram, jawabannya tidak sebanding dengan protokol naskah
    # dan bank yang dihasilkan tidak layak dinilai dokter — jadi berhenti di sini, jangan
    # hasilkan 240 jawaban yang harus dibuang.
    proc = mesin.prosesor(PARAMS_GENERASI["no_repeat_ngram"])
    if proc is None:
        print("BERHENTI: pemblokir n-gram tidak bisa aktif (LogitsProcessorList tidak ada "
              "di versi llama-cpp-python ini). Ganti versinya dulu.")
        return 1

    bank = muat_bank()
    if a.batas:
        bank = bank[:a.batas]
    labels = a.label or LABELS

    print(f"bank {len(bank)} pertanyaan x {len(labels)} model, versi {versi}")
    print(f"keluaran {os.path.join(KELUAR, versi)}\n")

    t_mulai = datetime.datetime.now()
    for label in labels:
        perlu = []
        for b in bank:
            rek = baca(versi, mesin.hash_pertanyaan(b["pertanyaan"]))
            if a.ulang or rek is None or label not in rek["jawaban"]:
                perlu.append(b)
        if not perlu:
            print(f"{label}: sudah lengkap, dilewati")
            continue

        print(f"{label}: {len(perlu)} pertanyaan")
        mesin.muat(label)
        for n, b in enumerate(perlu, 1):
            h = mesin.hash_pertanyaan(b["pertanyaan"])
            rek = baca(versi, h) or rekaman_baru(b, versi)
            hasil = mesin.jawab(label, b["pertanyaan"], proc=proc)
            rek["jawaban"][label] = hasil
            rek["dibuat"] = datetime.datetime.now(datetime.timezone.utc) \
                .isoformat(timespec="seconds")
            rek["pemblokir_ngram"] = True
            tulis(versi, rek)
            print(f"  [{n:2d}/{len(perlu)}] {b['id']:5s} {hasil['detik']:6.1f}s "
                  f"{hasil['token'] or 0:4d} tok"
                  f"{'  TERPOTONG' if hasil['terpotong'] else ''}")
        mesin.bongkar(label)
        tulis_metrik()

    print(f"\nselesai dalam {datetime.datetime.now() - t_mulai}")
    return ringkas(versi)


if __name__ == "__main__":
    raise SystemExit(main())
