"""
cek_prompt_sama.py — bukti bahwa prompt pelatihan, evaluasi, dan pengukuran di HP identik.

Tiga jalur kode membangun prompt secara terpisah:
  - pelatihan  : serverless/train_p6.py -> tokenizer.apply_chat_template(enable_thinking=False)
  - evaluasi   : chat_format.split_prompt_reference (dipakai eval.py)
  - HP / demo  : benchmark_ondevice.bangun_prompt (string ChatML ditulis manual)
Skrip ini merender N pertanyaan uji lewat ketiganya dan membandingkan string-nya.

    python cek_prompt_sama.py            ->  results/prompt_sama.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

import chat_format as cf  # noqa: E402
from benchmark_ondevice import SYSTEM_PROMPT, bangun_prompt  # noqa: E402

N = 50


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    rows = [json.loads(l) for l in open(ROOT / "Data/processed_id/test.jsonl", encoding="utf-8")][:N]
    mode = cf.MODEL_MODES.get("qwen")
    sama_eval = sama_hp = sama_system = 0
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        latih = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        evalp, _ = cf.split_prompt_reference(r, mode, tok)
        hp = bangun_prompt(next(m["content"] for m in msgs if m["role"] == "user"))
        sama_eval += latih == evalp
        sama_hp += latih == hp
        sama_system += next(m["content"] for m in msgs if m["role"] == "system") == SYSTEM_PROMPT
    out = {"n": len(rows), "latih_sama_eval": sama_eval, "latih_sama_hp": sama_hp,
           "system_prompt_data_sama_hp": sama_system,
           "ekor_prompt": latih[-40:]}
    (ROOT / "results/prompt_sama.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
