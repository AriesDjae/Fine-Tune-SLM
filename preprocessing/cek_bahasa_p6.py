"""
cek_bahasa_p6.py — persentase pertanyaan & jawaban berbahasa Indonesia pada data P6.

Kenapa ada: angka bahasa di naskah (dulu 99,97–99,99%) berasal dari `audit_dataset.py`,
yang mengaudit korpus LAMA (Data/processed*, gabungan 10 sumber), bukan data P6.
Skrip ini memakai detektor yang sama (`safe_detect`, langdetect seed 42) pada data yang
benar-benar dipakai: latih/validasi `processed_id_final`, uji `processed_id/test.jsonl`.

    python preprocessing/cek_bahasa_p6.py   ->  results/bahasa_p6.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocessing"))
from audit_dataset import safe_detect  # noqa: E402  (langdetect, DetectorFactory.seed = 42)

SPLITS = {
    "train": ROOT / "Data/processed_id_final/train.jsonl",
    "val": ROOT / "Data/processed_id_final/val.jsonl",
    "test": ROOT / "Data/processed_id/test.jsonl",
}


def main():
    out = {}
    for name, path in SPLITS.items():
        cq, ca, n = Counter(), Counter(), 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
                cq[safe_detect(m["user"])] += 1
                ca[safe_detect(m["assistant"])] += 1
                n += 1
        out[name] = {
            "n": n,
            "question_id_pct": round(100 * cq["id"] / n, 2),
            "answer_id_pct": round(100 * ca["id"] / n, 2),
            "question_lang": dict(cq.most_common()),
            "answer_lang": dict(ca.most_common()),
        }
        print(name, n, "Q id", out[name]["question_id_pct"], "A id", out[name]["answer_id_pct"])
    dst = ROOT / "results/bahasa_p6.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", dst)


if __name__ == "__main__":
    main()
