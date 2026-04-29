"""
MedFit processor — converts mlx-community/medfit-dataset into ChatML samples.
Schema is detected at runtime (supports instruction/input/output, question/answer,
prompt/completion, and messages formats).
Download first: python Data/data.py
"""

from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. Provide accurate, detailed answers to medical "
    "questions based on established clinical knowledge. Always recommend consulting a "
    "healthcare professional for personal medical decisions."
)

HF_CACHE = str(Path(__file__).parent.parent / "Data" / "hf_cache")
_EMPTY   = {"", "none", "nan"}


def _extract_qa(item: dict) -> tuple[str, str]:
    # Alpaca-style: instruction / input / output
    if "instruction" in item and "output" in item:
        instr  = str(item.get("instruction", "")).strip()
        inp    = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()
        q = f"{instr}\n{inp}".strip() if inp and inp.lower() not in _EMPTY else instr
        return q, output

    # Simple QA
    if "question" in item and "answer" in item:
        return str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()

    # Prompt / completion
    if "prompt" in item and "completion" in item:
        return str(item.get("prompt", "")).strip(), str(item.get("completion", "")).strip()

    # OpenAI messages format
    if "messages" in item:
        msgs      = item["messages"]
        user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
        asst_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if user_msgs and asst_msgs:
            return str(user_msgs[0]).strip(), str(asst_msgs[0]).strip()

    return "", ""


def load_samples() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [medfit] skip — datasets not installed")
        return []

    try:
        ds = load_dataset("mlx-community/medfit-dataset", cache_dir=HF_CACHE)
    except Exception as e:
        print(f"  [medfit] skip — {e}")
        return []

    split = ds.get("train", list(ds.values())[0])
    print(f"  [medfit] features: {list(split.features.keys())}")

    features = list(split.features.keys())

    # Dataset has a single pre-formatted 'text' column — skip (can't split into Q/A)
    if features == ["text"] or (len(features) == 1 and "text" in features):
        print("  [medfit] skip — single 'text' column, cannot extract Q/A pairs")
        return []

    samples = []
    for item in split:
        question, answer = _extract_qa(item)
        if not question or not answer or answer.lower() in _EMPTY:
            continue

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": "medfit",
        })

    print(f"  [medfit] loaded {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
