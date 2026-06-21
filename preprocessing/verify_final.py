"""
verify_final.py  —  VERIFIKASI MUTU dataset final (Data/processed_final/*).

Mengecek 3 klaim "emas" langsung pada FILE FINAL (bukan angka tengah pipeline):
  1. Dedup & anti-leakage antar split (exact, by signature user+assistant).
  2. Proporsi bahasa Indonesia final tiap split (is_id heuristik yang SAMA).
  3. Over-stripping: clean_greeting tidak lagi membuang konten + distribusi panjang
     jawaban (tidak ada yang ter-gutting jadi sangat pendek).

Pakai:  python preprocessing/verify_final.py
"""

import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocessing"))

from preprocess_dataset import load_jsonl, sig, is_id, msgs  # noqa: E402
from chat_format import clean_greeting  # noqa: E402

FINAL = ROOT / "Data" / "processed_final"
SPLITS = {"train": "train_final.jsonl", "val": "val_final.jsonl", "test": "test_final.jsonl"}


def banner(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def assistant_answers(samples):
    out = []
    for s in samples:
        for m in msgs(s):
            if m.get("role") == "assistant":
                out.append(m.get("content", ""))
    return out


def main():
    data = {k: load_jsonl(FINAL / v) for k, v in SPLITS.items()}
    sigs = {k: [sig(s) for s in v] for k, v in data.items()}
    sets = {k: set(v) for k, v in sigs.items()}

    # ---- 1. DEDUP & ANTI-LEAKAGE ----
    banner("1. DEDUP & ANTI-LEAKAGE (exact, file final)")
    for k in SPLITS:
        dup = len(sigs[k]) - len(sets[k])
        print(f"  [{k:5s}] n={len(data[k]):6d} | duplikat internal exact: {dup}")
    pairs = [("train", "test"), ("train", "val"), ("val", "test")]
    for a, b in pairs:
        overlap = len(sets[a] & sets[b])
        print(f"  leakage {a} & {b}: {overlap}")

    # ---- 2. PROPORSI BAHASA INDONESIA ----
    banner("2. PROPORSI BAHASA INDONESIA FINAL (is_id)")
    for k in SPLITS:
        idn = sum(is_id(s) for s in data[k])
        n = len(data[k])
        print(f"  [{k:5s}] ID {idn:6d} / {n:6d}  ({idn/n*100:5.2f}%)  | EN {n-idn} ({(n-idn)/n*100:5.2f}%)")

    # ---- 3. OVER-STRIPPING ----
    banner("3. OVER-STRIPPING (clean_greeting idempoten + panjang jawaban)")
    for k in SPLITS:
        ans = assistant_answers(data[k])
        lens = [len(a) for a in ans]
        over = sum(1 for a in ans if len(a) - len(clean_greeting(a)) > 50)
        short = sum(1 for L in lens if L < 20)
        print(f"  [{k:5s}] jawaban={len(ans):6d} | clean_greeting masih buang >50c: {over} "
              f"| <20 char: {short} | panjang min/median/max: "
              f"{min(lens)}/{int(statistics.median(lens))}/{max(lens)}")

    banner("VERDICT")
    ok_leak = all(len(sets[a] & sets[b]) == 0 for a, b in pairs)
    ok_dup = all(len(sigs[k]) == len(sets[k]) for k in SPLITS)
    ok_over = all(
        sum(1 for a in assistant_answers(data[k]) if len(a) - len(clean_greeting(a)) > 50) == 0
        for k in SPLITS
    )
    print(f"  Tidak ada leakage antar split : {'YA' if ok_leak else 'TIDAK'}")
    print(f"  Tidak ada duplikat internal   : {'YA' if ok_dup else 'TIDAK'}")
    print(f"  Tidak ada over-strip tersisa  : {'YA' if ok_over else 'TIDAK'}")


if __name__ == "__main__":
    main()
