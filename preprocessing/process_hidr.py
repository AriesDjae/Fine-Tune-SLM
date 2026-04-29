"""
HIDR indicators processor — converts hidr_indicators.xlsx into ChatML samples.
Generates 3 QA types per indicator:
  1. What does indicator X measure?
  2. Which dataset covers indicator X?
  3. What topic area does X fall under in WHO HIDR?
"""

import pandas as pd
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a medical AI assistant with expertise in global health indicators and WHO data. "
    "Provide accurate information about health indicators, their measurements, and data sources. "
    "Always recommend consulting healthcare professionals for personal medical decisions."
)

DATA_DIR = Path(__file__).parent.parent / "Data"
XLSX_PATH = DATA_DIR / "hidr_indicators.xlsx"


def load_samples() -> list[dict]:
    df = pd.read_excel(XLSX_PATH)
    print(f"  [hidr] columns: {list(df.columns)}")

    samples = []
    for _, row in df.iterrows():
        indicator_name = str(row.get("indicator_name", "")).strip()
        topic_area     = str(row.get("topic_area", "")).strip()
        dataset_name   = str(row.get("dataset_name", "")).strip()
        dimension      = str(row.get("dimension", "")).strip()
        indicator_abbr = str(row.get("indicator_abbr", row.get("indicator_id", ""))).strip()
        dataset_id     = str(row.get("dataset_id", "")).strip()

        if not indicator_name or indicator_name == "nan":
            continue

        # QA type 1: abbreviation → full name + dimension
        if indicator_abbr and indicator_abbr != "nan":
            dim_suffix = f", categorized under the dimension: {dimension}." if dimension and dimension != "nan" else "."
            samples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": f"What does the health indicator '{indicator_abbr}' measure?"},
                    {"role": "assistant", "content": f"The indicator '{indicator_abbr}' measures **{indicator_name}**{dim_suffix}"},
                ],
                "source": "hidr",
            })

        # QA type 2: indicator name → dataset
        if dataset_name and dataset_name != "nan":
            id_suffix = f" (ID: {dataset_id})" if dataset_id and dataset_id != "nan" else ""
            samples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": f"Which dataset covers the health indicator '{indicator_name}'?"},
                    {"role": "assistant", "content": f"The indicator **{indicator_name}** is covered by the **{dataset_name}** dataset{id_suffix}."},
                ],
                "source": "hidr",
            })

        # QA type 3: topic area
        if topic_area and topic_area != "nan":
            samples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": f"What topic area does '{indicator_name}' fall under in WHO HIDR?"},
                    {"role": "assistant", "content": f"**{indicator_name}** falls under the **'{topic_area}'** topic area in the WHO Health Indicator and Data Repository (HIDR)."},
                ],
                "source": "hidr",
            })

    print(f"  [hidr] loaded {len(samples):,} samples from {len(df):,} indicators")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
