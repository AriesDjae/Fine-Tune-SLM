"""
ICD-11 processor — converts SimpleTabulation-ICD-11-MMS-en.xlsx into ChatML samples.

Quality filters:
  - Only ClassKind == "category" (no blocks/chapters)
  - Exclude Extension codes (ChapterNo == "X") — modifiers, not diagnoses
  - Skip very short titles (len < 4)

QA types per entry:
  1. Code lookup  : "What is the ICD-11 code for [disease]?"
  2. Reverse      : "A patient has code [code]. What condition?"
  3. CodingNote   : "Describe [disease] per ICD-11." (only if CodingNote exists)

Capped at MAX_SAMPLES, prioritising entries with CodingNote.
"""

import random
import pandas as pd
from pathlib import Path

SYSTEM_EN = (
    "You are a medical AI assistant with expertise in ICD-11 disease classification. "
    "Provide accurate information about disease codes, categories, and clinical context "
    "when asked about specific conditions. Always recommend consulting a healthcare "
    "professional for personal medical advice."
)

DATA_DIR    = Path(__file__).parent.parent / "Data"
XLSX_PATH   = DATA_DIR / "SimpleTabulation-ICD-11-MMS-en.xlsx"
MAX_SAMPLES = 20_000
SEED        = 42

ICD11_CHAPTERS = {
    "01": "Certain infectious or parasitic diseases",
    "02": "Neoplasms",
    "03": "Diseases of the blood or blood-forming organs",
    "04": "Diseases of the immune system",
    "05": "Endocrine, nutritional or metabolic diseases",
    "06": "Mental, behavioural or neurodevelopmental disorders",
    "07": "Sleep-wake disorders",
    "08": "Diseases of the nervous system",
    "09": "Diseases of the visual system",
    "10": "Diseases of the ear or mastoid process",
    "11": "Diseases of the circulatory system",
    "12": "Diseases of the respiratory system",
    "13": "Diseases of the digestive system",
    "14": "Diseases of the skin",
    "15": "Diseases of the musculoskeletal system or connective tissue",
    "16": "Diseases of the genitourinary system",
    "17": "Conditions related to sexual health",
    "18": "Pregnancy, childbirth or the perinatal period",
    "19": "Certain conditions originating in the perinatal period",
    "20": "Developmental anomalies",
    "21": "Symptoms, signs or clinical findings, not elsewhere classified",
    "22": "Injury, poisoning or certain other consequences of external causes",
    "23": "External causes of morbidity or mortality",
    "24": "Factors influencing health status or contact with health services",
    "25": "Codes for special purposes",
    "26": "Supplementary chapter Traditional Medicine conditions",
    "V":  "Supplementary section for functioning assessment",
}


def _make_pairs(title, code, ch_key, ch_name, coding_note="") -> list[dict]:
    pairs = []

    pairs.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_EN},
            {"role": "user",      "content": f"What is the ICD-11 code for {title}?"},
            {"role": "assistant", "content": (
                f"The ICD-11 code for **{title}** is **{code}**. "
                f"It is classified under Chapter {ch_key}: {ch_name}."
            )},
        ],
        "source": "icd11",
    })

    pairs.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_EN},
            {"role": "user",      "content": f"A patient has been diagnosed with ICD-11 code {code}. What condition does this represent?"},
            {"role": "assistant", "content": (
                f"ICD-11 code **{code}** represents **{title}**. "
                f"This condition is classified under Chapter {ch_key}: {ch_name} "
                f"in the ICD-11 international classification system. "
                f"For detailed clinical information and patient-specific advice, "
                f"please consult a qualified healthcare professional."
            )},
        ],
        "source": "icd11",
    })

    if coding_note and len(coding_note) > 20:
        pairs.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_EN},
                {"role": "user",      "content": f"Describe {title} according to ICD-11 classification guidelines."},
                {"role": "assistant", "content": (
                    f"**{title}** (ICD-11: {code}) — {coding_note} "
                    f"This condition falls under Chapter {ch_key}: {ch_name}."
                )},
            ],
            "source": "icd11",
        })

    return pairs


def load_samples() -> list[dict]:
    df = pd.read_excel(XLSX_PATH)

    df = df[
        (df["Code"].notna()) &
        (df["Code"].astype(str).str.strip() != "") &
        (df["ClassKind"] == "category")
    ].copy()

    # Exclude Extension codes (Chapter X)
    df = df[df["ChapterNo"].astype(str).str.strip() != "X"]

    df["Title"] = df["Title"].astype(str).str.replace(r"^[\-\s]+", "", regex=True).str.strip()
    df["Code"]  = df["Code"].astype(str).str.strip()
    df = df[df["Title"].str.len() > 3].reset_index(drop=True)

    if "CodingNote" in df.columns:
        df["CodingNote"] = df["CodingNote"].fillna("").astype(str).str.strip()
    else:
        df["CodingNote"] = ""

    # Split: with note (richer) vs without
    mask_note  = df["CodingNote"].str.len() > 20
    with_note  = df[mask_note]
    no_note    = df[~mask_note]

    rng = random.Random(SEED)
    samples = []

    # All entries with CodingNote first (richer QA)
    for _, row in with_note.iterrows():
        ch_key  = str(row["ChapterNo"]).strip()
        ch_name = ICD11_CHAPTERS.get(ch_key, "Unknown chapter")
        samples.extend(_make_pairs(row["Title"], row["Code"], ch_key, ch_name, row["CodingNote"]))

    # Fill quota from remaining entries (shuffled)
    no_note_rows = no_note.to_dict("records")
    rng.shuffle(no_note_rows)
    for row in no_note_rows:
        if len(samples) >= MAX_SAMPLES:
            break
        ch_key  = str(row["ChapterNo"]).strip()
        ch_name = ICD11_CHAPTERS.get(ch_key, "Unknown chapter")
        samples.extend(_make_pairs(row["Title"], row["Code"], ch_key, ch_name))

    # Final cap + shuffle
    if len(samples) > MAX_SAMPLES:
        rng.shuffle(samples)
        samples = samples[:MAX_SAMPLES]

    print(f"  [icd11] loaded {len(samples):,} samples "
          f"(X-codes excluded, cap {MAX_SAMPLES:,})")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
