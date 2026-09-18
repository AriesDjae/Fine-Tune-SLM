"""analisis_tier2_subset.py — membedah 200 penilaian Tier 2 yang SUDAH ada.

Reviewer mengajukan tiga keberatan yang semuanya bisa dijawab tanpa satu pun panggilan
API baru, karena `results/llm_judge/tier2_<ukuran>.jsonl` menyimpan penilaian PER BUTIR
lengkap dengan `idx` (baris ke-i test.jsonl), peta A/B ke kondisi, dan alasan penilai:

1. **Populasi berbeda.** Cakupan eskalasi dihitung pada 195 butir red-flag di seluruh
   himpunan uji, sedangkan laju "Aman" dihitung pada 200 butir acak. Skrip ini menghitung
   berapa butir red-flag yang benar-benar masuk sampel 200 itu, lalu melaporkan laju Aman
   terpisah untuk subset terpicu dan tak terpicu.
2. **Konfound pemenggalan di Tingkat 2.** Jawaban baseline terpenggal jauh lebih sering
   (~90%) daripada fine-tuned. Skrip ini menghitung ulang Δ Aman HANYA pada butir yang
   kedua jawabannya selesai utuh — definisi `utuh()` disalin dari analisis_truncation.py.
3. **Mekanisme belum terbukti.** Butir yang berubah Aman -> Tidak aman dikumpulkan beserta
   alasan penilai, sehingga dapat dikategorikan manual. Kategorisasi otomatis di sini hanya
   pengelompokan kata kunci untuk mempercepat pembacaan, BUKAN hasil yang dilaporkan.

    python analisis_tier2_subset.py            # -> results/llm_judge/subset_tier2.json
    python analisis_tier2_subset.py --alasan   # cetak alasan butir yang berubah

Angka yang dihasilkan bersifat deskriptif atas sampel yang sama; tidak ada penilaian baru.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

# Konsol Windows cp1252 menolak Δ; dipaksa UTF-8 seperti skrip judge.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from redflag.detector import detect  # noqa: E402

TEST_FILE = os.path.join("Data", "processed_id", "test.jsonl")
PREDS_DIR = os.path.join("results", "preds_p6", "preds")
JUDGE_DIR = os.path.join("results", "llm_judge")
OUT = os.path.join(JUDGE_DIR, "subset_tier2.json")
UKURAN = ("0.8b", "2b", "4b")
SEED = 42
N_BOOT = 5000

# Disalin dari analisis_truncation.py supaya definisi "terpotong" persis sama.
_SELESAI = re.compile(r"[.!?:)\]\"']\s*$")


def utuh(teks: str) -> bool:
    return bool(_SELESAI.search((teks or "").strip()))


def muat_test() -> tuple[list[str], list[str]]:
    tanya, acuan = [], []
    with open(TEST_FILE, encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if not baris:
                continue
            msg = json.loads(baris)["messages"]
            tanya.append(next(m["content"] for m in msg if m["role"] == "user"))
            acuan.append(next(m["content"] for m in msg if m["role"] == "assistant"))
    return tanya, acuan


def muat_preds(label: str) -> list[str]:
    with open(os.path.join(PREDS_DIR, f"{label}.preds.json"), encoding="utf-8") as f:
        return json.load(f)["preds"]


def muat_judge(ukuran: str) -> list[dict]:
    butir = []
    with open(os.path.join(JUDGE_DIR, f"tier2_{ukuran}.jsonl"), encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if baris:
                butir.append(json.loads(baris))
    return butir


def nilai_per_kondisi(rec: dict) -> dict:
    """Balik peta A/B menjadi {'baseline': {...}, 'finetuned': {...}}."""
    keluar = {}
    for slot, kondisi in rec["kunci"].items():
        keluar[kondisi] = rec["nilai"][slot]
    return keluar


def boot_delta(pasangan: list[tuple[int, int]], n_boot: int = N_BOOT) -> tuple[float, list[float]]:
    """Δ proporsi (ft − base) berpasangan, dengan SK 95% bootstrap pada butir yang sama."""
    if not pasangan:
        return float("nan"), [float("nan"), float("nan")]
    n = len(pasangan)
    delta = sum(f - b for b, f in pasangan) / n
    rng = random.Random(SEED)
    sampel = []
    for _ in range(n_boot):
        pilih = [pasangan[rng.randrange(n)] for _ in range(n)]
        sampel.append(sum(f - b for b, f in pilih) / n)
    sampel.sort()
    lo = sampel[int(0.025 * n_boot)]
    hi = sampel[int(0.975 * n_boot) - 1]
    return delta, [lo, hi]


def ringkas(pasangan: list[tuple[int, int]]) -> dict:
    n = len(pasangan)
    if n == 0:
        return {"n": 0}
    base = sum(b for b, _ in pasangan) / n
    ft = sum(f for _, f in pasangan) / n
    delta, ci = boot_delta(pasangan)
    return {
        "n": n,
        "aman_baseline": round(base, 4),
        "aman_finetuned": round(ft, 4),
        "delta_pp": round(delta * 100, 2),
        "ci95_pp": [round(ci[0] * 100, 2), round(ci[1] * 100, 2)],
        "signifikan": not (ci[0] <= 0 <= ci[1]),
    }


# Pengelompokan kata kunci untuk mempercepat pembacaan alasan penilai. BUKAN hasil akhir:
# kategori yang dilaporkan harus dari pembacaan manual.
_KATEGORI = {
    "tidak_menganjurkan_dokter": ("tidak menyarankan", "tidak menganjurkan", "tanpa menyarankan",
                                  "tidak merujuk", "tidak menyebut dokter"),
    "informasi_salah": ("salah", "keliru", "tidak akurat", "tidak benar", "menyesatkan"),
    "saran_obat": ("obat", "dosis", "antibiotik", "konsumsi"),
    "klaim_diagnosis": ("diagnosis", "memastikan", "dipastikan"),
    "tidak_lengkap": ("tidak lengkap", "kurang lengkap", "terlalu umum", "tidak menjawab"),
    "teks_rusak": ("berulang", "tidak koheren", "tidak nyambung", "acak", "tidak jelas"),
}


def kategori_kasar(alasan: str) -> list[str]:
    t = (alasan or "").lower()
    return [k for k, kata in _KATEGORI.items() if any(w in t for w in kata)] or ["lainnya"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alasan", action="store_true", help="cetak alasan butir yang berubah")
    a = ap.parse_args()

    tanya, _ = muat_test()
    terpicu = [bool(detect(q).terpicu) for q in tanya]

    hasil = {
        "catatan": "dihitung ulang dari penilaian Tier 2 yang sudah ada; tidak ada panggilan judge baru",
        "n_test": len(tanya),
        "n_terpicu_seluruh_uji": sum(terpicu),
        "seed_sampel": SEED,
        "per_ukuran": {},
    }

    for s in UKURAN:
        butir = muat_judge(s)
        base_preds = muat_preds(f"qwen{s}_baseline")
        ft_preds = muat_preds(f"qwen{s}_finetuned")

        semua, rf, non_rf, utuh_dua = [], [], [], []
        berubah = []  # aman -> tidak aman setelah fine-tuning
        pulih = []    # tidak aman -> aman
        for rec in butir:
            i = rec["idx"]
            v = nilai_per_kondisi(rec)
            b = 1 if v["baseline"]["aman"] == "ya" else 0
            f = 1 if v["finetuned"]["aman"] == "ya" else 0
            semua.append((b, f))
            (rf if terpicu[i] else non_rf).append((b, f))
            if utuh(base_preds[i]) and utuh(ft_preds[i]):
                utuh_dua.append((b, f))
            if b == 1 and f == 0:
                berubah.append({"idx": i, "terpicu": terpicu[i],
                                "alasan": v["finetuned"].get("alasan", ""),
                                "kategori_kasar": kategori_kasar(v["finetuned"].get("alasan", "")),
                                "terpotong_ft": not utuh(ft_preds[i])})
            if b == 0 and f == 1:
                pulih.append({"idx": i, "terpicu": terpicu[i],
                              "alasan": v["finetuned"].get("alasan", "")})

        hitung = {}
        for b in berubah:
            for k in b["kategori_kasar"]:
                hitung[k] = hitung.get(k, 0) + 1

        hasil["per_ukuran"][s] = {
            "n_sampel": len(semua),
            "n_terpicu_dalam_sampel": len(rf),
            "porsi_terpicu_dalam_sampel": round(len(rf) / len(semua), 4) if semua else None,
            "seluruh_sampel": ringkas(semua),
            "subset_terpicu": ringkas(rf),
            "subset_tak_terpicu": ringkas(non_rf),
            "subset_kedua_jawaban_utuh": ringkas(utuh_dua),
            "aman_ke_tidak_aman": {"n": len(berubah),
                                   "n_terpicu": sum(1 for x in berubah if x["terpicu"]),
                                   "n_terpotong_ft": sum(1 for x in berubah if x["terpotong_ft"]),
                                   "kategori_kasar": dict(sorted(hitung.items(),
                                                                 key=lambda kv: -kv[1]))},
            "tidak_aman_ke_aman": {"n": len(pulih),
                                   "n_terpicu": sum(1 for x in pulih if x["terpicu"])},
            "butir_berubah": berubah,
        }

        print(f"\n=== {s} ===")
        r = hasil["per_ukuran"][s]
        print(f"  red-flag dalam sampel : {r['n_terpicu_dalam_sampel']}/{r['n_sampel']}")
        for nama in ("seluruh_sampel", "subset_terpicu", "subset_tak_terpicu",
                     "subset_kedua_jawaban_utuh"):
            x = r[nama]
            if x.get("n"):
                print(f"  {nama:28s} n={x['n']:3d}  base {x['aman_baseline']:.3f} -> "
                      f"ft {x['aman_finetuned']:.3f}  Δ {x['delta_pp']:+.1f} pp "
                      f"[{x['ci95_pp'][0]:+.1f}; {x['ci95_pp'][1]:+.1f}]"
                      f"{'' if x['signifikan'] else '  (tidak signifikan)'}")
        ab = r["aman_ke_tidak_aman"]
        print(f"  aman -> tidak aman    : {ab['n']} butir "
              f"({ab['n_terpicu']} di antaranya red-flag, {ab['n_terpotong_ft']} jawaban ft terpotong)")
        print(f"  kategori kasar        : {ab['kategori_kasar']}")
        print(f"  tidak aman -> aman    : {r['tidak_aman_ke_aman']['n']} butir")

        if a.alasan:
            for x in berubah:
                tanda = "RF" if x["terpicu"] else "  "
                print(f"    [{tanda}] idx {x['idx']:4d}: {x['alasan'][:150]}")

    os.makedirs(JUDGE_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=1)
    print(f"\nDitulis: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
