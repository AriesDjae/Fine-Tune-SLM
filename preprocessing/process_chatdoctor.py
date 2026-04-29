"""
ChatDoctor processor — converts lavita/medical-qa-datasets (chatdoctor_healthcaremagic)
into ChatML samples.
Dataset has ~226k rows; capped at MAX_SAMPLES to prevent imbalance.
Download first: python Data/data.py
"""

from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. Answer patient questions with accurate, "
    "empathetic responses based on established clinical knowledge. Always recommend "
    "consulting a healthcare professional for proper diagnosis and treatment."
)

HF_CACHE    = str(Path(__file__).parent.parent / "Data" / "hf_cache")
MAX_SAMPLES = 40_000
MIN_Q_LEN   = 20
MIN_A_LEN   = 100
MAX_A_LEN   = 2000
_EMPTY      = {"", "none", "nan"}


def load_samples() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [chatdoctor] skip — datasets not installed")
        return []

    try:
        ds = load_dataset(
            "lavita/medical-qa-datasets",
            name="chatdoctor_healthcaremagic",
            cache_dir=HF_CACHE,
        )
    except Exception as e:
        print(f"  [chatdoctor] skip — {e}")
        return []

    split = ds.get("train", list(ds.values())[0])
    print(f"  [chatdoctor] features: {list(split.features.keys())}")

    samples = []
    for item in split:
        if len(samples) >= MAX_SAMPLES:
            break

        # Column names vary across lavita dataset configs
        question = str(
            item.get("input", item.get("question", item.get("instruction", "")))
        ).strip()
        answer = str(
            item.get("output", item.get("answer", item.get("response", "")))
        ).strip()

        if not question or not answer or answer.lower() in _EMPTY:
            continue
        if len(question) < MIN_Q_LEN or len(answer) < MIN_A_LEN:
            continue
        if len(answer) > MAX_A_LEN:
            answer = answer[:MAX_A_LEN].rsplit(" ", 1)[0] + "..."

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "chatdoctor",
        })

    print(f"  [chatdoctor] loaded {len(samples):,} samples (cap={MAX_SAMPLES:,})")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
