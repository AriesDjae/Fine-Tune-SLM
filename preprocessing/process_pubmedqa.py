"""
PubMedQA processor — converts qiaojin/PubMedQA (pqa_labeled) into ChatML samples.
Uses question, long_answer, and final_decision columns.
Download first: python Data/data.py
"""

from pathlib import Path

SYSTEM_PROMPT = (
    "You are a medical AI assistant specialized in biomedical research. "
    "Answer questions based on published medical literature accurately. "
    "Always recommend consulting healthcare professionals for personal medical decisions."
)

HF_CACHE = str(Path(__file__).parent.parent / "Data" / "hf_cache")


def load_samples() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [pubmedqa] skip — datasets not installed")
        return []

    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir=HF_CACHE)
    except Exception as e:
        print(f"  [pubmedqa] skip — {e}")
        return []

    split = ds.get("train", ds.get("test", list(ds.values())[0]))

    samples = []
    for item in split:
        question    = str(item.get("question", "")).strip()
        long_answer = str(item.get("long_answer", "")).strip()
        decision    = str(item.get("final_decision", "")).strip()

        if not question or not long_answer or long_answer == "nan":
            continue

        answer = f"{decision.capitalize()}. {long_answer}" if decision and decision != "nan" else long_answer

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "pubmedqa",
        })

    print(f"  [pubmedqa] loaded {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
