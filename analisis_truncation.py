"""
analisis_truncation.py — apakah Δ RQ1 sekadar artefak batas panjang generasi?

KENAPA ADA. Prediksi P6 dibangkitkan dengan `max_new_tokens=256`. Ternyata **89,9-91,2%
jawaban BASELINE terpotong** di tengah kalimat, sementara fine-tuned hanya 39,1-63,1%.
Referensi dokter bahkan mencapai 307 kata — jauh di atas plafon itu. Karena truncation
memotong isi, ia menekan recall dan karenanya token-F1/ROUGE-L; dan karena baseline jauh
lebih sering terpotong, **baseline dihukum lebih berat**. Maka muncul keberatan yang sah:

    "Δ RQ1 Anda sebagian hanyalah artefak anggaran generasi, bukan mutu model."

Konfound ini BELUM pernah diuji — ablation format sebelumnya (buang markdown/preamble)
tidak menyentuhnya, sebab membuang penanda tidak mengembalikan teks yang hilang.

CARA UJINYA TANPA MEMBANGKITKAN ULANG: hitung ulang metrik hanya pada item yang **kedua
kondisinya selesai utuh** (subset bersih). Di subset itu tak ada kondisi yang dirugikan
batas panjang. Kalau Δ bertahan → keberatan gugur; kalau runtuh → konfoundnya nyata dan
harus dilaporkan apa adanya.

Perhatikan asimetri arah: truncation merugikan BASELINE, jadi ia **menggelembungkan** Δ.
Subset bersih karenanya memberi estimasi yang lebih konservatif — dan itulah angka yang
lebih layak dipakai di naskah.

Pemakaian:

    python analisis_truncation.py

Butuh `Data/processed_id/test.jsonl` dan `results/preds_p6/preds/*.preds.json`.
Menulis `results/eval_truncation.json`.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import sys

from eval import rougeL, token_f1  # definisi metrik KANONIK (\w+ + lower)

for _aliran in (sys.stdout, sys.stderr):
    try:
        _aliran.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

TEST_FILE = os.path.join("Data", "processed_id", "test.jsonl")
PREDS_DIR = os.path.join("results", "preds_p6", "preds")
OUT = os.path.join("results", "eval_truncation.json")
UKURAN = ("0.8b", "2b", "4b")
SEED = 42

# "Selesai utuh" = diakhiri tanda baca penutup kalimat. Sengaja longgar (termasuk kutip
# dan kurung penutup) supaya yang dianggap TERPOTONG benar-benar terpotong.
_SELESAI = re.compile(r"[.!?:)\]\"']\s*$")


def utuh(teks: str) -> bool:
    return bool(_SELESAI.search((teks or "").strip()))


def muat_test() -> list[str]:
    ref = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if not baris:
                continue
            msg = json.loads(baris)["messages"]
            ref.append(next(m["content"] for m in msg if m["role"] == "assistant"))
    return ref


def muat_preds(label: str) -> list[str]:
    path = os.path.join(PREDS_DIR, f"{label}.preds.json")
    if not os.path.exists(path):
        raise SystemExit(f"Tidak ada {path}. Lihat results/preds_p6/README.md")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["preds"]


def skor(preds: list[str], refs: list[str], idxs: list[int]) -> dict:
    return {
        "f1": statistics.mean(token_f1(preds[i], refs[i]) for i in idxs),
        "rougeL": statistics.mean(rougeL(preds[i], refs[i]) for i in idxs),
    }


def bootstrap_delta(base: list[str], ft: list[str], refs: list[str], idxs: list[int],
                    metrik, n_iter: int = 2000) -> list[float]:
    """CI 95% untuk delta BERPASANGAN (ft - base) via bootstrap item."""
    per_item = [(metrik(base[i], refs[i]), metrik(ft[i], refs[i])) for i in idxs]
    rng = random.Random(SEED)
    m = len(per_item)
    deltas = []
    for _ in range(n_iter):
        s = 0.0
        for _ in range(m):
            b, f_ = per_item[rng.randrange(m)]
            s += f_ - b
        deltas.append(s / m)
    deltas.sort()
    return [deltas[int(0.025 * n_iter)], deltas[int(0.975 * n_iter) - 1]]


def main() -> None:
    refs = muat_test()
    n = len(refs)
    hasil = {}

    print(f"Test set: {n} item\n")
    print("TINGKAT TRUNCATION (jawaban tak diakhiri tanda baca penutup):")
    preds = {}
    for u in UKURAN:
        for k in ("baseline", "finetuned"):
            lbl = f"qwen{u}_{k}"
            preds[lbl] = muat_preds(lbl)
            if len(preds[lbl]) != n:
                raise SystemExit(f"Panjang tak cocok: {lbl}")
            r = sum(not utuh(t) for t in preds[lbl]) / n
            print(f"  {lbl:22s} {r * 100:5.1f}%")

    print(f"\n{'':8s}{'subset':>10s}{'n':>7s}"
          f"{'Δ token-F1':>13s}{'CI95':>22s}{'Δ ROUGE-L':>13s}{'CI95':>22s}")

    for u in UKURAN:
        base, ft = preds[f"qwen{u}_baseline"], preds[f"qwen{u}_finetuned"]
        semua = list(range(n))
        # subset BERSIH: kedua kondisi selesai utuh -> tak ada yang dirugikan batas panjang
        bersih = [i for i in semua if utuh(base[i]) and utuh(ft[i])]

        baris = {}
        for nama, idxs in (("semua", semua), ("bersih", bersih)):
            sb, sf = skor(base, refs, idxs), skor(ft, refs, idxs)
            ci_f1 = bootstrap_delta(base, ft, refs, idxs, token_f1)
            ci_rl = bootstrap_delta(base, ft, refs, idxs, rougeL)
            d_f1, d_rl = sf["f1"] - sb["f1"], sf["rougeL"] - sb["rougeL"]
            baris[nama] = {
                "n": len(idxs),
                "baseline": sb, "finetuned": sf,
                "delta_f1": d_f1, "ci_f1": ci_f1,
                "delta_rougeL": d_rl, "ci_rougeL": ci_rl,
            }
            tanda = "signifikan" if (ci_f1[0] > 0) == (ci_f1[1] > 0) else "TIDAK signifikan"
            print(f"  {u:6s}{nama:>10s}{len(idxs):7d}"
                  f"{d_f1:+13.4f}  [{ci_f1[0]:+.4f},{ci_f1[1]:+.4f}]"
                  f"{d_rl:+13.4f}  [{ci_rl[0]:+.4f},{ci_rl[1]:+.4f}]  {tanda}")
        hasil[u] = baris

    print("\nPERBANDINGAN (bersih - semua), yaitu berapa besar Δ berubah setelah")
    print("item terpotong dikeluarkan:")
    for u in UKURAN:
        a, b = hasil[u]["semua"], hasil[u]["bersih"]
        print(f"  {u:6s} token-F1 {b['delta_f1'] - a['delta_f1']:+.4f}   "
              f"ROUGE-L {b['delta_rougeL'] - a['delta_rougeL']:+.4f}   "
              f"(n {a['n']} -> {b['n']}, tersisa {b['n'] / a['n'] * 100:.0f}%)")

    print("\nCARA MEMBACA: truncation merugikan BASELINE (terpotong 90% vs 39-63%), jadi ia")
    print("MENGGELEMBUNGKAN Δ. Bila Δ pada subset bersih tetap positif & signifikan, keberatan")
    print("'Δ Anda cuma artefak batas panjang' GUGUR — dan angka subset bersih itulah estimasi")
    print("konservatif yang lebih layak dilaporkan.")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n_test": n,
            "max_new_tokens_saat_generate": 256,
            "truncation_rate": {lbl: sum(not utuh(t) for t in p) / n for lbl, p in preds.items()},
            "per_ukuran": hasil,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDisimpan -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
