"""
MedMCQA processor — converts openlifescienceai/medmcqa into ChatML samples.
Takes first MAX_SAMPLES from training split. Formats as MCQ with explanation.
Download first: python Data/data.py
"""

from pathlib import Path

SYSTEM_PROMPT = (
    "You are a medical AI assistant helping with clinical knowledge questions. "
    "Answer multiple choice medical questions accurately and explain the reasoning. "
    "Always recommend consulting healthcare professionals for personal medical decisions."
)

HF_CACHE    = str(Path(__file__).parent.parent / "Data" / "hf_cache")
MAX_SAMPLES = 5_000
_LABELS     = ["A", "B", "C", "D"]
_EMPTY      = {"", "none", "nan"}


def load_samples() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [medmcqa] skip — datasets not installed")
        return []

    try:
        ds = load_dataset(
            "openlifescienceai/medmcqa",
            split=f"train[:{MAX_SAMPLES}]",
            cache_dir=HF_CACHE,
        )
    except Exception as e:
        print(f"  [medmcqa] skip — {e}")
        return []

    samples = []
    for item in ds:
        question = str(item.get("question", "")).strip()
        opa = str(item.get("opa", "")).strip()
        opb = str(item.get("opb", "")).strip()
        opc = str(item.get("opc", "")).strip()
        opd = str(item.get("opd", "")).strip()
        cop = item.get("cop")
        exp = str(item.get("exp", "")).strip()

        if not question or cop is None:
            continue

        q_text = (
            f"{question}\n"
            f"A) {opa}\n"
            f"B) {opb}\n"
            f"C) {opc}\n"
            f"D) {opd}"
        )

        options       = [opa, opb, opc, opd]
        correct_label = _LABELS[cop] if 0 <= cop < 4 else "?"
        correct_text  = options[cop] if 0 <= cop < 4 else ""
        answer        = f"The correct answer is **{correct_label}) {correct_text}**."
        if exp and exp.lower() not in _EMPTY:
            answer += f" {exp}"

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": q_text},
                {"role": "assistant", "content": answer},
            ],
            "source": "medmcqa",
        })

    print(f"  [medmcqa] loaded {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
