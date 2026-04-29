"""
MedQuAD processor — converts lavita/MedQuAD into ChatML samples.
~47,000 medical QA pairs. Filters rows with empty answers.
Download first: python Data/data.py
"""

from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. Answer questions accurately based on clinical "
    "guidelines and established medical knowledge. Always recommend consulting a "
    "healthcare professional for personal medical decisions."
)

HF_CACHE = str(Path(__file__).parent.parent / "Data" / "hf_cache")


def load_samples() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [medquad] skip — datasets not installed")
        return []

    try:
        ds = load_dataset("lavita/MedQuAD", cache_dir=HF_CACHE)
    except Exception as e:
        print(f"  [medquad] skip — {e}")
        return []

    split = ds.get("train", list(ds.values())[0])

    samples = []
    for item in split:
        question = str(item.get("question", "")).strip()
        answer   = str(item.get("answer", "")).strip()

        if not question or not answer or answer.lower() in ("none", "nan", ""):
            continue
        if len(answer) < 50:
            continue
        if len(answer) > 2000:
            answer = answer[:2000].rsplit(" ", 1)[0] + "..."

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "medquad",
        })

    print(f"  [medquad] loaded {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
