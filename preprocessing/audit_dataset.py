"""
audit_dataset.py — AUDIT-ONLY (read-only) untuk brief Indonesia-only (Pivot 4, catatan bagian 3).

Tidak mengubah / menulis ulang dataset apa pun. Hanya MEMBACA file yang ada,
mendeteksi bahasa per-sampel (langdetect), menghitung distribusi sumber/domain,
dan mendeteksi artefak yang diketahui (ICD-11 lookup, sapaan, <think></think> kosong,
campuran bahasa dalam satu sampel).

Output:
  - results/audit_dataset.json   (semua angka, machine-readable)
  - ringkasan teks dicetak ke layar + spot-check sampel acak

Jalankan:
  python preprocessing/audit_dataset.py --debug          # subset 200/file (cepat)
  python preprocessing/audit_dataset.py --sample 5000    # batasi deteksi bahasa per-file
  python preprocessing/audit_dataset.py                  # full (deteksi bahasa semua sampel)

CPU-safe, seed=42.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 42
random.seed(42)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# File yang diaudit (yang ada di repo). Label deskriptif -> path.
FILES = {
    "processed/train": ROOT / "Data/processed/train.jsonl",
    "processed/val": ROOT / "Data/processed/val.jsonl",
    "processed/test": ROOT / "Data/processed/test.jsonl",
    "processed_final/train_final": ROOT / "Data/processed_final/train_final.jsonl",
    "processed_final/val_final": ROOT / "Data/processed_final/val_final.jsonl",
    "processed_final/test_final": ROOT / "Data/processed_final/test_final.jsonl",
    "processed_shared/train_reduced": ROOT / "Data/processed_shared/train_reduced.jsonl",
    "processed_shared/val_reduced": ROOT / "Data/processed_shared/val_reduced.jsonl",
}

# --- Artefak (bagian 4 brief) -------------------------------------------------
ICD_PATTERNS = re.compile(
    r"(ICD[- ]?11 code|ICD[- ]?10 code|kode ICD|ICD[- ]?11 classification|"
    r"what (condition|disease) does .* represent|ICD[- ]?11 code for)",
    re.IGNORECASE,
)
# Sapaan + nama placeholder / nama kapital di awal jawaban
GREETING_NAME = re.compile(
    r"^\s*(hi|hai|halo|hello|dear|hallo)\b[\s,]+[A-Z][a-z]+", re.IGNORECASE
)
GREETING_LEAD = re.compile(r"^\s*(hi|hai|halo|hello|dear|hallo)\b", re.IGNORECASE)
EMPTY_THINK = re.compile(r"<think>\s*</think>|<thinking>\s*</thinking>", re.IGNORECASE)


def get_text(msgs, role):
    for m in msgs:
        if m.get("role") == role:
            return m.get("content", "") or ""
    return ""


def safe_detect(text):
    text = text.strip()
    if len(text) < 12:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def audit_file(label, path, sample):
    if not path.exists():
        return {"label": label, "exists": False, "path": str(path)}

    n = 0
    sources = Counter()
    user_lang = Counter()
    asst_lang = Counter()
    mixed = 0  # user lang != asst lang (di antara id/en)
    icd_hits = 0
    greet_name = 0
    greet_lead = 0
    empty_think = 0
    asst_lens = []
    user_lens = []
    empty_answer = 0
    detect_count = 0
    src_lang_id = Counter()  # per-source: jumlah jawaban terdeteksi id
    src_lang_total = Counter()
    reservoir = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n += 1
            msgs = o.get("messages", [])
            src = o.get("source", "<none>")
            sources[src] += 1

            u = get_text(msgs, "user")
            a = get_text(msgs, "assistant")
            user_lens.append(len(u))
            asst_lens.append(len(a))
            if not a.strip():
                empty_answer += 1

            # artefak
            if ICD_PATTERNS.search(u) or ICD_PATTERNS.search(a) or src in ("icd11", "hidr"):
                icd_hits += 1
            if GREETING_NAME.search(a):
                greet_name += 1
            if GREETING_LEAD.search(a):
                greet_lead += 1
            if EMPTY_THINK.search(a) or EMPTY_THINK.search(u):
                empty_think += 1

            # deteksi bahasa pada subset (sampel) agar cepat di file besar
            if sample is None or detect_count < sample:
                lu = safe_detect(u)
                la = safe_detect(a)
                user_lang[lu] += 1
                asst_lang[la] += 1
                if lu in ("id", "en") and la in ("id", "en") and lu != la:
                    mixed += 1
                src_lang_total[src] += 1
                if la == "id":
                    src_lang_id[src] += 1
                detect_count += 1

            # reservoir sampling utk spot-check
            if len(reservoir) < 25:
                reservoir.append((src, u, a))
            else:
                j = random.randint(0, n - 1)
                if j < 25:
                    reservoir[j] = (src, u, a)

    def pct(c, tot):
        return round(100 * c / tot, 2) if tot else 0.0

    def stat(xs):
        xs = sorted(xs)
        if not xs:
            return {}
        return {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]}

    src_id_pct = {s: pct(src_lang_id[s], src_lang_total[s]) for s in src_lang_total}

    return {
        "label": label,
        "exists": True,
        "path": str(path),
        "n": n,
        "sources": dict(sources.most_common()),
        "lang_detect_n": detect_count,
        "user_lang": dict(user_lang.most_common()),
        "assistant_lang": dict(asst_lang.most_common()),
        "user_lang_pct": {k: pct(v, detect_count) for k, v in user_lang.most_common()},
        "assistant_lang_pct": {k: pct(v, detect_count) for k, v in asst_lang.most_common()},
        "assistant_id_pct_by_source": src_id_pct,
        "mixed_lang_user_vs_asst": mixed,
        "mixed_lang_pct": pct(mixed, detect_count),
        "artifacts": {
            "icd_or_hidr_lookup": icd_hits,
            "icd_or_hidr_lookup_pct": pct(icd_hits, n),
            "greeting_with_name": greet_name,
            "greeting_leading": greet_lead,
            "greeting_leading_pct": pct(greet_lead, n),
            "empty_think_tags": empty_think,
            "empty_answer": empty_answer,
        },
        "user_len_chars": stat(user_lens),
        "assistant_len_chars": stat(asst_lens),
        "_reservoir": reservoir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="batasi jumlah deteksi bahasa per file (default: semua)")
    ap.add_argument("--debug", action="store_true", help="alias --sample 200")
    args = ap.parse_args()
    sample = 200 if args.debug else args.sample

    report = {}
    for label, path in FILES.items():
        print(f"[audit] {label} ...", flush=True)
        report[label] = audit_file(label, path, sample)

    # tulis JSON (tanpa reservoir besar)
    out = {"_mode": "debug(200/file)" if args.debug else (f"sample={sample}" if sample else "full")}
    for label, r in report.items():
        rr = dict(r)
        rr.pop("_reservoir", None)
        out[label] = rr
    (RESULTS / "audit_dataset.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- ringkasan layar ----
    print("\n" + "=" * 78)
    print(f"RINGKASAN AUDIT DATASET (read-only)  mode={out['_mode']}")
    print("=" * 78)
    for label, r in report.items():
        if not r.get("exists"):
            print(f"\n## {label}: TIDAK ADA ({r['path']})")
            continue
        print(f"\n## {label}  n={r['n']:,}  (deteksi bahasa pada {r['lang_detect_n']:,} sampel)")
        ul = r["assistant_lang_pct"]
        print(f"   bahasa JAWABAN  : id={ul.get('id',0)}%  en={ul.get('en',0)}%  "
              f"lain={round(100-ul.get('id',0)-ul.get('en',0),2)}%")
        ulu = r["user_lang_pct"]
        print(f"   bahasa PERTANYAAN: id={ulu.get('id',0)}%  en={ulu.get('en',0)}%")
        print(f"   campur bahasa (user!=asst): {r['mixed_lang_user_vs_asst']:,} ({r['mixed_lang_pct']}%)")
        print(f"   id% per-source  : {r['assistant_id_pct_by_source']}")
        a = r["artifacts"]
        print(f"   ICD/HIDR lookup : {a['icd_or_hidr_lookup']:,} ({a['icd_or_hidr_lookup_pct']}%)")
        print(f"   sapaan di awal  : {a['greeting_leading']:,} ({a['greeting_leading_pct']}%) "
              f"[dgn nama: {a['greeting_with_name']:,}]")
        print(f"   <think></think> kosong: {a['empty_think_tags']:,}   jawaban kosong: {a['empty_answer']:,}")
        print(f"   sumber: {r['sources']}")

    # ---- spot-check ----
    print("\n" + "=" * 78)
    print("SPOT-CHECK (sampel acak dari processed_final/train_final)")
    print("=" * 78)
    pf = report.get("processed_final/train_final")
    if pf and pf.get("_reservoir"):
        for i, (src, u, a) in enumerate(pf["_reservoir"][:25], 1):
            print(f"\n[{i}] source={src}")
            print(f"  Q: {u[:200]}")
            print(f"  A: {a[:200]}")

    print(f"\n[audit] JSON ditulis ke {RESULTS / 'audit_dataset.json'}")


if __name__ == "__main__":
    main()
