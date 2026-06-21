"""
analyze_language_distribution.py — menghitung proporsi bahasa (Indonesia vs Inggris)
dan distribusi sumber pada dataset hasil build_dataset.py.

Jalankan setelah `python preprocessing/build_dataset.py`:
  python preprocessing/analyze_language_distribution.py
"""

import json
from collections import Counter
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "Data" / "processed"

# Pemetaan source -> bahasa, berdasarkan asal masing-masing dataset
# (lihat docstring di tiap preprocessing/process_*.py)
LANG_MAP = {
    "indonesia_qna": "Indonesia",   # Alodokter QnA
    "bioasq":        "Inggris",
    "chatdoctor":    "Inggris",
    "hidr":          "Inggris",     # WHO HIDR indicators
    "icd11":         "Inggris",
    "medfit":        "Inggris",
    "medmcqa":       "Inggris",
    "medquad":       "Inggris",
    "pubmedqa":      "Inggris",
    "wikidoc":       "Inggris",
}


def count_sources(path: Path) -> Counter:
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["source"]] += 1
    return counts


def main():
    splits = ["train", "val", "test"]
    per_split = {}
    total_source = Counter()

    for split in splits:
        path = PROCESSED_DIR / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} tidak ditemukan — jalankan build_dataset.py dulu")
        counts = count_sources(path)
        per_split[split] = counts
        total_source.update(counts)

    grand_total = sum(total_source.values())

    print("=" * 60)
    print(f"TOTAL keseluruhan (train+val+test): {grand_total:,}")
    print("=" * 60)

    print("\nDistribusi per SOURCE (gabungan train+val+test):")
    for src, cnt in total_source.most_common():
        lang = LANG_MAP.get(src, "?")
        print(f"  {src:<15} {lang:<10} {cnt:>7,}  ({cnt/grand_total*100:5.1f}%)")

    total_lang = Counter()
    for src, cnt in total_source.items():
        total_lang[LANG_MAP.get(src, "?")] += cnt

    print("\nDistribusi BAHASA (gabungan):")
    for lang, cnt in total_lang.most_common():
        print(f"  {lang:<10} {cnt:>7,}  ({cnt/grand_total*100:5.1f}%)")

    print("\nPer split:")
    for split, counts in per_split.items():
        n     = sum(counts.values())
        id_n  = sum(cnt for s, cnt in counts.items() if LANG_MAP.get(s) == "Indonesia")
        en_n  = n - id_n
        print(f"  {split:<6} total={n:>7,}   "
              f"Indonesia={id_n:>6,} ({id_n/n*100:4.1f}%)   "
              f"Inggris={en_n:>6,} ({en_n/n*100:4.1f}%)")


if __name__ == "__main__":
    main()
