"""
Indonesia Medical QnA processor — converts indonesia-medical-qna/qna.csv
(Alodokter patient-doctor conversations) into ChatML samples.

Filters applied (based on data quality audit):
  - Drop null question/answer
  - Drop HTML tags, decode HTML entities, strip URLs
  - Drop question < 20 char or answer < 50 char
  - Drop rows where question == answer (70 corrupt rows)
  - Drop answers that are user complaints, not doctor responses
    (e.g. "Tolong dibalas dok", "Gmna dok?" — 6,582 rows)
  - Deduplicate by question text (269,854 duplicates = 39.6%)
  - Cap at MAX_SAMPLES after dedup
"""

import html
import re
import pandas as pd
from pathlib import Path

MAX_SAMPLES = 30_000  # quality cap; dataset has 681k total

SYSTEM_PROMPT = (
    "Anda adalah asisten dokter AI yang membantu menjelaskan keluhan medis pasien "
    "secara detail dan akurat. Untuk setiap keluhan, berikan penjelasan yang mencakup: "
    "kemungkinan kondisi yang dialami, gejala yang relevan, penyebab umum, kapan harus "
    "segera ke dokter, serta saran penanganan awal yang aman. Selalu ingatkan pengguna "
    "untuk berkonsultasi langsung dengan dokter untuk diagnosis dan penanganan yang tepat."
)

DATA_DIR = Path(__file__).parent.parent / "Data"
CSV_PATH = DATA_DIR / "indonesia-medical-qna" / "qna.csv"

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"\s{2,}")
_HREF_REFS = re.compile(r"https?://\S+")

# Answers matching these patterns are user follow-ups, not doctor responses
_NOT_ANSWER = re.compile(
    r"tolong\s*(?:d\s*)?balas|tolong\s*di\s*jawab|gmna\s*dok|belum\s*di\s*balas|"
    r"kenapa\s*belum|mohon\s*d\s*balas|dong\s*dok|ma['\s]?af.*belum|"
    r"dok\s*tolong|harap\s*di\s*balas",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)        # decode &amp; &lt; etc.
    text = _HREF_REFS.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def load_samples() -> list[dict]:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # Prefer clean columns; fall back to raw if needed
    q_col = "question_clean" if "question_clean" in df.columns else "question"
    a_col = "answer_clean"   if "answer_clean"   in df.columns else "answer"

    df = df[[q_col, a_col]].dropna()
    df[q_col] = df[q_col].astype(str).apply(_clean)
    df[a_col] = df[a_col].astype(str).apply(_clean)

    # Drop rows that are too short or too long to be meaningful
    df = df[
        (df[q_col].str.len() >= 20) &
        (df[a_col].str.len() >= 100) &
        (df[a_col].str.len() <= 2000)
    ]

    # Drop corrupt rows where question text == answer text
    df = df[df[q_col] != df[a_col]]

    # Drop answers that are user complaints/follow-ups, not doctor responses
    df = df[~df[a_col].str.contains(_NOT_ANSWER, na=False)]

    # Deduplicate by question text, then cap
    df = df.drop_duplicates(subset=[q_col])
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": row[q_col]},
                {"role": "assistant", "content": row[a_col]},
            ],
            "source": "indonesia_qna",
        })

    print(f"  [indonesia_qna] loaded {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    samples = load_samples()
    print(f"Total: {len(samples)}")
    import json
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
