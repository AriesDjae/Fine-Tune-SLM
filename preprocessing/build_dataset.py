"""
build_dataset.py — merges all processed data sources into a single unified dataset.

Local sources:
  1. BioASQ training13b + 13B[1-4] golden
  2. Indonesia Medical QnA (Alodokter)
  3. ICD-11 MMS Excel
  4. HIDR indicators Excel

HuggingFace sources (run `python Data/data.py` first):
  5. PubMedQA, 6. MedQuAD, 7. WikiDoc, 8. MedMCQA, 9. MedFit, 10. ChatDoctor

Quality pipeline (applied globally before split):
  1. Clean HTML tags + URLs from all text
  2. Deduplicate by question text (global, cross-source)
  3. Cap repeated identical answers (max 5 occurrences per answer prefix)
  4. Remove very short answers (<80 chars)
  5. Remove ChatDoctor generic-greeting-only responses

Output: Data/processed/{train,val,test}.jsonl  — split 80/10/10

Run:
  python preprocessing/build_dataset.py
"""

import json
import re
import random
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from process_bioasq        import load_samples as load_bioasq
from process_indonesia_qna import load_samples as load_id_qna
from process_icd11         import load_samples as load_icd11
from process_hidr          import load_samples as load_hidr
from process_pubmedqa      import load_samples as load_pubmedqa
from process_medquad       import load_samples as load_medquad
from process_wikidoc       import load_samples as load_wikidoc
from process_medmcqa       import load_samples as load_medmcqa
from process_medfit        import load_samples as load_medfit
from process_chatdoctor    import load_samples as load_chatdoctor

SEED            = 42
TRAIN_FRAC      = 0.80
VAL_FRAC        = 0.10
MAX_ANS_REPEAT  = 5    # max occurrences of the same answer prefix (first 120 chars)
MIN_ANS_LEN     = 80   # drop answers shorter than this

OUT_DIR = Path(__file__).parent.parent / "Data" / "processed"

# ── Cleaning helpers ──────────────────────────────────────────────────────────
_HTML_TAG    = re.compile(r"<[^>]{1,80}>")
_URL         = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL    = re.compile(r"\n{3,}")

# ChatDoctor generic greetings with no real medical content
_GENERIC_GREETINGS = re.compile(
    r"^(Hello[,.]?\s*Thank you for (asking|consulting)|"
    r"Hi+[.,]?\s*(Thank you|Hope this message|I have gone through)|"
    r"Hi+[.,]?\s*Thank you for consulting in Chat Doctor\.\s*$)",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _clean_sample(s: dict) -> dict | None:
    msgs = []
    for m in s["messages"]:
        cleaned = dict(m)
        cleaned["content"] = _clean_text(m["content"])
        msgs.append(cleaned)
    return {**s, "messages": msgs}


# ── Quality filters ───────────────────────────────────────────────────────────

def deduplicate_by_question(samples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in samples:
        q = next((m["content"] for m in s["messages"] if m["role"] == "user"), "").strip().lower()
        if q and q not in seen:
            seen.add(q)
            out.append(s)
    removed = len(samples) - len(out)
    print(f"  dedup by question   : removed {removed:,} ({removed/len(samples)*100:.1f}%)")
    return out


def cap_repeated_answers(samples: list[dict], max_repeat: int = MAX_ANS_REPEAT) -> list[dict]:
    counts, out = Counter(), []
    for s in samples:
        a   = next((m["content"] for m in s["messages"] if m["role"] == "assistant"), "")
        key = a.strip()[:120]
        if counts[key] < max_repeat:
            counts[key] += 1
            out.append(s)
    removed = len(samples) - len(out)
    print(f"  cap repeated answers: removed {removed:,} ({removed/len(samples)*100:.1f}%)")
    return out


def remove_short_answers(samples: list[dict], min_len: int = MIN_ANS_LEN) -> list[dict]:
    out = [
        s for s in samples
        if len(next((m["content"] for m in s["messages"] if m["role"] == "assistant"), "")) >= min_len
    ]
    removed = len(samples) - len(out)
    print(f"  short answer filter : removed {removed:,} ({removed/len(samples)*100:.1f}%)")
    return out


def remove_generic_greetings(samples: list[dict]) -> list[dict]:
    out = []
    for s in samples:
        a = next((m["content"] for m in s["messages"] if m["role"] == "assistant"), "")
        # Keep if answer has real medical content beyond the greeting (>150 chars after greeting)
        if _GENERIC_GREETINGS.match(a) and len(a) < 150:
            continue
        out.append(s)
    removed = len(samples) - len(out)
    print(f"  generic greetings   : removed {removed:,} ({removed/len(samples)*100:.1f}%)")
    return out


# ── Split / Write ─────────────────────────────────────────────────────────────

def split_data(samples: list, train_f: float, val_f: float):
    n       = len(samples)
    n_train = int(n * train_f)
    n_val   = int(n * val_f)
    return samples[:n_train], samples[n_train:n_train + n_val], samples[n_train + n_val:]


def write_jsonl(path: Path, samples: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
    print(f"  wrote {len(samples):,} -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Loading data sources...")
    print("=" * 60)

    all_samples = []
    all_samples.extend(load_bioasq())
    all_samples.extend(load_id_qna())
    all_samples.extend(load_icd11())
    all_samples.extend(load_hidr())
    all_samples.extend(load_pubmedqa())
    all_samples.extend(load_medquad())
    all_samples.extend(load_wikidoc())
    all_samples.extend(load_medmcqa())
    all_samples.extend(load_medfit())
    all_samples.extend(load_chatdoctor())

    print()
    print(f"Raw total: {len(all_samples):,}")

    dist = Counter(s["source"] for s in all_samples)
    print("Source distribution (raw):")
    for src, cnt in dist.most_common():
        print(f"  {src:<25} {cnt:>7,}  ({cnt / len(all_samples) * 100:.1f}%)")

    # ── Global quality pipeline ───────────────────────────────────────────────
    print()
    print("Applying quality pipeline...")

    # 1. Clean HTML + URLs
    all_samples = [_clean_sample(s) for s in all_samples]

    # 2. Remove very short answers first (before dedup, cheaper)
    all_samples = remove_short_answers(all_samples)

    # 3. Remove generic-only greetings
    all_samples = remove_generic_greetings(all_samples)

    # 4. Global deduplication by question
    random.seed(SEED)
    random.shuffle(all_samples)   # shuffle before dedup so no source is systematically favoured
    all_samples = deduplicate_by_question(all_samples)

    # 5. Cap repeated identical answers
    all_samples = cap_repeated_answers(all_samples)

    print()
    print(f"After quality pipeline: {len(all_samples):,}")
    print()

    dist_clean = Counter(s["source"] for s in all_samples)
    print("Source distribution (clean):")
    for src, cnt in dist_clean.most_common():
        print(f"  {src:<25} {cnt:>7,}  ({cnt / len(all_samples) * 100:.1f}%)")

    # ── Split ─────────────────────────────────────────────────────────────────
    random.shuffle(all_samples)
    train, val, test = split_data(all_samples, TRAIN_FRAC, VAL_FRAC)

    print()
    print(f"Split 80/10/10: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print()

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl",   val)
    write_jsonl(OUT_DIR / "test.jsonl",  test)

    print()
    print("Done. Dataset saved to Data/processed/")


if __name__ == "__main__":
    main()
