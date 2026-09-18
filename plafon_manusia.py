"""plafon_manusia.py — hitung ulang plafon kesepakatan antar-dokter, dan lengkapi BERTScore.

KENAPA ADA. Angka plafon (ROUGE-L 0,145 / token-F1 0,285 pada 908 pasangan) dipakai di
naskah sebagai acuan penafsiran, tetapi prosedurnya tidak pernah tersimpan sebagai skrip —
hanya hasilnya yang tertulis di `results/plot_hasil_p6.py`. Reviewer yang ingin memeriksanya
tidak punya yang bisa dijalankan. Skrip ini menutup celah itu sekaligus menambahkan
BERTScore antar-dokter, yang sebelumnya dilaporkan sebagai keterbatasan padahal murah.

PROSEDUR (eksplisit, supaya bisa diperiksa):

1. Korpus mentah `Data/indonesia-medical-qna/qna.csv` dikelompokkan per pertanyaan
   (`question_clean`, dinormalkan spasi + huruf kecil).
2. Dari tiap kelompok diambil jawaban dari **dokter yang berbeda** (`answer_by`); satu
   pertanyaan menyumbang paling banyak satu pasangan, supaya pertanyaan yang dijawab banyak
   dokter tidak mendominasi.
3. Pasangan dengan kedua jawaban identik dibuang.
4. Metrik dihitung dengan implementasi yang PERSIS sama seperti `eval.py` (`token_f1`,
   `rougeL`, `clean_greeting`), lalu BERTScore dengan konfigurasi `hitung_bertscore_p6.py`.

Karena metriknya simetris hanya untuk token-F1, ROUGE-L dihitung dua arah lalu dirata-rata.

    python plafon_manusia.py                 # tanpa BERTScore (cepat)
    python plafon_manusia.py --bertscore     # + BERTScore (butuh unduh model)

Menulis `results/plafon_manusia.json`.
"""

from __future__ import annotations

import argparse
import csv
import collections
import json
import os
import random
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_format import clean_greeting  # noqa: E402
from eval import rougeL, token_f1  # noqa: E402

QNA = os.path.join("Data", "indonesia-medical-qna", "qna.csv")
OUT = os.path.join("results", "plafon_manusia.json")
BERTSCORE_MODEL = "bert-base-multilingual-cased"
SEED = 42

csv.field_size_limit(10 ** 9)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def bangun_pasangan() -> list[tuple[str, str, str]]:
    """(pertanyaan, jawaban dokter 1, jawaban dokter 2) — satu pasangan per pertanyaan."""
    per_tanya: dict[str, dict[str, str]] = collections.defaultdict(dict)
    with open(QNA, encoding="utf-8", newline="") as f:
        for baris in csv.DictReader(f):
            q = norm(baris.get("question_clean") or baris.get("question"))
            a = (baris.get("answer_clean") or "").strip()
            d = (baris.get("answer_by") or "").strip()
            if q and a and d:
                per_tanya[q].setdefault(d, a)

    pasangan = []
    for q, per_dokter in per_tanya.items():
        if len(per_dokter) < 2:
            continue
        (_, a1), (_, a2) = list(per_dokter.items())[:2]
        if norm(a1) == norm(a2):
            continue
        pasangan.append((q, clean_greeting(a1), clean_greeting(a2)))
    return pasangan


def bootstrap(nilai: list[float], n_iter: int = 5000) -> list[float]:
    rng = random.Random(SEED)
    n = len(nilai)
    sampel = sorted(sum(nilai[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_iter))
    return [sampel[int(0.025 * n_iter)], sampel[int(0.975 * n_iter) - 1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bertscore", action="store_true", help="ikut hitung BERTScore (lebih lambat)")
    a = ap.parse_args()

    if not os.path.exists(QNA):
        raise SystemExit(f"Tidak ada {QNA}")

    pasangan = bangun_pasangan()
    print(f"Pasangan jawaban dokter berbeda: {len(pasangan)}")

    f1 = [token_f1(p[1], p[2]) for p in pasangan]
    # ROUGE-L tidak simetris (recall dihitung terhadap acuan), jadi dua arah lalu dirata-rata.
    rl = [(rougeL(p[1], p[2]) + rougeL(p[2], p[1])) / 2 for p in pasangan]

    hasil = {
        "sumber": QNA,
        "n_pasangan": len(pasangan),
        "prosedur": "satu pasangan per pertanyaan, dokter berbeda (answer_by), "
                    "jawaban identik dibuang, clean_greeting diterapkan seperti eval.py",
        "token_f1": {"mean": round(sum(f1) / len(f1), 4), "ci95": [round(x, 4) for x in bootstrap(f1)]},
        "rougeL": {"mean": round(sum(rl) / len(rl), 4), "ci95": [round(x, 4) for x in bootstrap(rl)],
                   "catatan": "rata-rata dua arah karena ROUGE-L tidak simetris"},
    }
    print(f"  token-F1 : {hasil['token_f1']['mean']:.4f}  SK95 {hasil['token_f1']['ci95']}")
    print(f"  ROUGE-L  : {hasil['rougeL']['mean']:.4f}  SK95 {hasil['rougeL']['ci95']}")

    if a.bertscore:
        from bert_score import score as _bs
        print(f"\nBERTScore ({BERTSCORE_MODEL}) pada {len(pasangan)} pasangan...")
        _, _, F = _bs([p[1] for p in pasangan], [p[2] for p in pasangan],
                      model_type=BERTSCORE_MODEL, lang="id", verbose=False, batch_size=32)
        nilai = [float(x) for x in F]
        hasil["bertscore_f1"] = {"model": BERTSCORE_MODEL,
                                 "mean": round(sum(nilai) / len(nilai), 4),
                                 "ci95": [round(x, 4) for x in bootstrap(nilai)]}
        print(f"  BERTScore: {hasil['bertscore_f1']['mean']:.4f}  "
              f"SK95 {hasil['bertscore_f1']['ci95']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=1)
    print(f"\nDitulis: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
