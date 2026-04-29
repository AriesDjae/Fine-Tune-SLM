"""
WikiDoc Medical processor — converts medalpaca/medical_meadow_wikidoc into ChatML samples.
~10,000 medical knowledge QA in Alpaca instruction format.
Download first: python Data/data.py
"""

from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. Provide accurate, detailed answers to medical "
    "questions based on established clinical knowledge. Always recommend consulting a "
    "healthcare professional for personal medical decisions."
)

HF_CACHE = str(Path(__file__).parent.parent / "Data" / "hf_cache")
_EMPTY = {"", "none", "nan"}


def load_samples() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [wikidoc] skip — datasets not installed")
        return []

    try:
        ds = load_dataset("medalpaca/medical_meadow_wikidoc", cache_dir=HF_CACHE)
    except Exception as e:
        print(f"  [wikidoc] skip — {e}")
        return []

    split = ds.get("train", list(ds.values())[0])

    samples = []
    for item in split:
        instruction = str(item.get("instruction", "")).strip()
        input_text  = str(item.get("input", "")).strip()
        output      = str(item.get("output", "")).strip()

        if not output or output.lower() in _EMPTY:
            continue
        if len(output) < 50:
            continue
        if len(output) > 2000:
            output = output[:2000].rsplit(" ", 1)[0] + "..."

        question = (
            f"{instruction}\n{input_text}".strip()
            if input_text and input_text.lower() not in _EMPTY
            else instruction
        )
        if not question:
            continue

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": output},
            ],
            "source": "wikidoc",
        })

    print(f"  [wikidoc] loaded {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
