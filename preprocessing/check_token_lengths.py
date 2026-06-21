"""
check_token_lengths.py — Pengecekan panjang TOKEN (acuan MAX_SEQ_LENGTH) per model <=1B.

char != token. Tokenisasi seluruh sampel final (system+user+assistant via chat_template)
dengan tokenizer MASING-MASING model, lapor median/p90/p95/p99/max, tetapkan MAX_SEQ_LENGTH
SAMA utk ketiganya, dan hitung berapa sampel ter-truncate.

Model (coba mirror unsloth ungated dulu, lalu resmi/gated):
  Gemma 3 1B IT · Qwen3.5 0.8B · Llama 3.2 1B Instruct

Jalankan:  python preprocessing/check_token_lengths.py
Output  :  results/token_length_<tanggal>.json + ringkasan layar.
Catatan  :  model gated (Gemma/Llama) butuh `huggingface-cli login` / HF_TOKEN bila mirror gagal.
"""
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "Data" / "processed_id"
RESULTS = ROOT / "results"

MODELS = {
    "gemma-3-1b-it":      ["unsloth/gemma-3-1b-it", "google/gemma-3-1b-it"],
    "qwen3.5-0.8b":       ["unsloth/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", "unsloth/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
    "llama-3.2-1b-instr": ["unsloth/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"],
}
CANDIDATE_MAXLEN = [512, 768, 1024, 1536, 2048]


def load_tok(ids):
    for mid in ids:
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            if tok.chat_template is None:
                continue
            return mid, tok
        except Exception as e:
            print(f"   (gagal {mid}: {type(e).__name__})")
    return None, None


def main():
    rows = []
    for sp in ("train", "val", "test"):
        rows += [json.loads(l) for l in open(OUT_DIR / f"{sp}.jsonl", encoding="utf-8")]
    msgs = [r["messages"] for r in rows]
    print(f"sampel total: {len(msgs):,}")

    report = {"n": len(msgs), "models": {}}
    for name, ids in MODELS.items():
        print(f"\n[{name}] memuat tokenizer ...")
        mid, tok = load_tok(ids)
        if tok is None:
            print(f"   SKIP — tidak ada tokenizer yang bisa dimuat (gated? perlu HF_TOKEN).")
            report["models"][name] = {"status": "skipped", "tried": ids}
            continue
        lens = []
        for m in msgs:
            try:
                # transformers 5.x: tokenize=True default mengembalikan BatchEncoding (len()=2 keys,
                # BUKAN panjang token). return_dict=False -> list[int] token ids yang benar.
                ids_ = tok.apply_chat_template(
                    m, tokenize=True, add_generation_prompt=False, return_dict=False)
                if ids_ and isinstance(ids_[0], list):   # jaga-jaga bila ter-batch
                    ids_ = ids_[0]
            except Exception:
                # fallback: render string lalu encode (hindari dobel special token)
                s = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                ids_ = tok(s, add_special_tokens=False).input_ids
            lens.append(len(ids_))
        a = np.array(lens)
        pct = {f"p{p}": int(np.percentile(a, p)) for p in (50, 90, 95, 99)}
        trunc = {str(L): int((a > L).sum()) for L in CANDIDATE_MAXLEN}
        report["models"][name] = {
            "status": "ok", "model_id": mid,
            "median": int(np.median(a)), **pct, "max": int(a.max()),
            "truncated_at": trunc,
        }
        print(f"   model={mid}")
        print(f"   median={int(np.median(a))}  p90={pct['p90']}  p95={pct['p95']}  p99={pct['p99']}  max={int(a.max())}")
        print(f"   ter-truncate @: " + "  ".join(f"{L}:{trunc[str(L)]}" for L in CANDIDATE_MAXLEN))

    # rekomendasi MAX_SEQ_LENGTH (sama utk semua): terkecil dari kandidat yg menampung p99 semua model
    oks = [m for m in report["models"].values() if m.get("status") == "ok"]
    if oks:
        p99max = max(m["p99"] for m in oks)
        rec = next((L for L in CANDIDATE_MAXLEN if L >= p99max), CANDIDATE_MAXLEN[-1])
        total_trunc = {str(L): sum(m["truncated_at"][str(L)] for m in oks) for L in CANDIDATE_MAXLEN}
        report["recommendation"] = {"max_p99_across_models": p99max,
                                    "MAX_SEQ_LENGTH": rec,
                                    "total_truncated_at_rec": total_trunc[str(rec)]}
        print("\n" + "=" * 60)
        print(f"REKOMENDASI MAX_SEQ_LENGTH (sama utk 3 model): {rec}")
        print(f"  (p99 tertinggi antar-model = {p99max}; ter-truncate @ {rec} = {total_trunc[str(rec)]} sampel/model-total)")

    (RESULTS / f"token_length_{__import__('datetime').date.today():%Y%m%d}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetail: results/token_length_*.json")


if __name__ == "__main__":
    main()
