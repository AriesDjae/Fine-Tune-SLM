"""
BioASQ processor — converts training13b.json and 13B[1-4]_golden.json
into ChatML samples for fine-tuning a medical chatbot.

Each question type is handled differently to produce the most informative answer:
  - summary  : use ideal_answer directly
  - yesno    : prefix Yes/No, then ideal_answer
  - factoid  : list exact entities, then ideal_answer
  - list     : enumerate exact items, then ideal_answer
"""

import json
import os
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a medical AI assistant. When asked about medical conditions, symptoms, "
    "treatments, or research findings, provide a detailed and accurate explanation. "
    "Cover the condition's definition, underlying mechanisms, key findings, and clinical "
    "relevance where applicable. Always advise users to consult a qualified healthcare "
    "professional for personal medical decisions."
)

DATA_DIR = Path(__file__).parent.parent / "Data"
BIOASQ_FILES = [
    "training13b.json",
    "13B1_golden.json",
    "13B2_golden.json",
    "13B3_golden.json",
    "13B4_golden.json",
]


def _format_answer(q: dict) -> str:
    ideal = q.get("ideal_answer", [])
    ideal_text = (ideal[0] if isinstance(ideal, list) else ideal) if ideal else ""
    if not ideal_text:
        return ""

    qtype = q.get("type", "summary")
    exact = q.get("exact_answer")

    if qtype == "yesno" and exact:
        verdict = "Yes" if str(exact).lower() == "yes" else "No"
        return f"{verdict}. {ideal_text}"

    if qtype == "factoid" and exact:
        try:
            # exact_answer is a list of lists or list of strings
            flat = [x[0] if isinstance(x, list) else x for x in exact]
            entities = ", ".join(str(e) for e in flat[:5])  # cap at 5 entities
            return f"{entities}. {ideal_text}"
        except Exception:
            pass

    if qtype == "list" and exact:
        try:
            flat = [x[0] if isinstance(x, list) else x for x in exact]
            items = "; ".join(str(e) for e in flat[:10])
            return f"These include: {items}. {ideal_text}"
        except Exception:
            pass

    return ideal_text


def load_samples() -> list[dict]:
    samples = []
    for fname in BIOASQ_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found")
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", data if isinstance(data, list) else [])
        for q in questions:
            body = q.get("body", "").strip()
            answer = _format_answer(q)
            if not body or not answer:
                continue
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": body},
                    {"role": "assistant", "content": answer},
                ],
                "source": "bioasq",
            })
    print(f"  [bioasq] loaded {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    print(f"Total: {len(samples)}")
    import json as _j
    print(_j.dumps(samples[0], indent=2, ensure_ascii=False))
