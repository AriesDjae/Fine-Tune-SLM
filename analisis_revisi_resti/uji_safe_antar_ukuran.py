"""uji_safe_antar_ukuran.py — selisih laju Safe ANTAR ukuran pada 200 butir Tier 2 yang sama.

Pertanyaan Putaran 8 no. 1: apakah 0.8B fine-tuned dinilai Safe lebih sering daripada
(a) 2B fine-tuned dan (b) 2B baseline? Ke-200 butir sama di semua ukuran (seed tetap), jadi
pasangan per idx sah. Metode sama dengan llm_judge_tier2._bootstrap_delta: paired bootstrap
5000x, random.Random(42), CI persentil 2,5–97,5. Ditambah McNemar eksak sebagai cek kedua.

Catatan: kedua pembanding berasal dari panggilan penilai yang BERBEDA (tiap panggilan menilai
satu pasangan baseline–fine-tuned dari satu ukuran), jadi ini perbandingan lintas panggilan.

    python analisis_revisi_resti/uji_safe_antar_ukuran.py  -> analisis_revisi_resti/uji_safe_antar_ukuran.json
"""
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / "results" / "llm_judge"
SEED, N_ITER = 42, 5000


def muat(ukuran):
    out = {}
    for baris in open(J / f"tier2_{ukuran}.jsonl", encoding="utf-8"):
        r = json.loads(baris)
        aman = {}
        for slot, kondisi in r["kunci"].items():
            aman[kondisi] = int(r["nilai"][slot]["aman"] == "ya")
        out[r["idx"]] = aman
    return out


def bootstrap(pasangan):
    rng = random.Random(SEED)
    m = len(pasangan)
    d = []
    for _ in range(N_ITER):
        sa = sb = 0
        for _ in range(m):
            a, b = pasangan[rng.randrange(m)]
            sa += a
            sb += b
        d.append((sa - sb) / m)
    d.sort()
    return [round(d[int(0.025 * N_ITER)], 4), round(d[int(0.975 * N_ITER) - 1], 4)]


def mcnemar(pasangan):
    n01 = sum(1 for a, b in pasangan if a == 1 and b == 0)
    n10 = sum(1 for a, b in pasangan if a == 0 and b == 1)
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"hanya_kiri_aman": n01, "hanya_kanan_aman": n10, "p_eksak": round(p, 4)}


def main():
    d08, d2 = muat("0.8b"), muat("2b")
    idx = sorted(set(d08) & set(d2))
    assert len(idx) == 200, len(idx)
    hasil = {"n": len(idx), "metode": "paired bootstrap 5000x seed 42 + McNemar eksak"}
    for nama, kanan in [("0.8b_ft_vs_2b_ft", "finetuned"), ("0.8b_ft_vs_2b_base", "baseline")]:
        pas = [(d08[i]["finetuned"], d2[i][kanan]) for i in idx]
        kiri = sum(a for a, _ in pas) / len(pas)
        kan = sum(b for _, b in pas) / len(pas)
        hasil[nama] = {"safe_kiri": kiri, "safe_kanan": kan, "delta_pp": round((kiri - kan) * 100, 1),
                       "ci95_pp": [x * 100 for x in bootstrap(pas)], "mcnemar": mcnemar(pas)}
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hasil, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
