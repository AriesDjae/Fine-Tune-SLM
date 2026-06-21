"""
eval.py  —  BAGIAN 3 (MASTER NOTE FINAL): evaluasi fair, multi-metrik, per-bahasa.

Mengevaluasi SATU model pada test_final.jsonl dan menyimpan hasil JSON. Untuk
membuktikan "peningkatan" (RQ1), jalankan untuk KEEMPAT konfigurasi dgn protokol
identik, lalu ringkas deltanya:

    # baseline (sebelum fine-tune) + finetuned, dua-duanya:
    python eval.py --model unsloth/Qwen3.5-2B               --model_type qwen  --label qwen_baseline
    python eval.py --model outputs/merged/qwen35-2b-medical --model_type qwen  --label qwen_finetuned
    python eval.py --model unsloth/gemma-4-E2B-it           --model_type gemma --label gemma_baseline
    python eval.py --model outputs/merged/gemma4-e2b-medical --model_type gemma --label gemma_finetuned
    python eval.py --summarize results                      # tabel perbandingan + delta

Protokol FAIR (3.4): test_final SAMA, n_eval SAMA, greedy (do_sample=False),
max_new_tokens=256, no_repeat_ngram_size=3, format prompt SAMA (chat_format.py).
Optimasi kecepatan (3.5): padding kiri + batching, TANPA empty_cache di loop.

Metrik (3.2):
  - MCQA berhuruf (medmcqa)      -> Exact Match huruf opsi (accuracy)
  - Yes/No/Maybe (pubmedqa,...)  -> Exact Match ternormalisasi (accuracy)
  - Open-ended (sisanya)         -> ROUGE-L, ROUGE-1, token-F1
Dipecah per BAHASA (ID vs EN) (3.3) dan per bucket.
"""
import argparse
import glob
import json
import os
import re
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from chat_format import MODEL_MODES, split_prompt_reference, clean_greeting  # noqa: E402

# --------------------------------------------------------------------------- #
# Deteksi bahasa (ringkas; selaras dgn preprocessing)
# --------------------------------------------------------------------------- #
_ID_SOURCES = ("indonesia_qna", "alodokter", "ppk", "kemenkes", "indonesia", "id_med")
_ID_KW = ["puskesmas", "dokter", "pasien", "obat", "demam", "rumah sakit", "kesehatan",
          "penyakit", "gejala", "keluhan", "saya", "yang", "dan", "tidak", "dengan"]


def is_id(sample):
    src = sample.get("source", "").lower()
    if any(x in src for x in _ID_SOURCES):
        return True
    c = " ".join(m.get("content", "") for m in sample.get("messages", [])).lower()
    return sum(w in c for w in _ID_KW) >= 3


# --------------------------------------------------------------------------- #
# Metrik (self-contained, tanpa dependensi eksternal)
# --------------------------------------------------------------------------- #
_TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return _TOK.findall(s.lower())


def _lcs(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def _f1(common, plen, rlen):
    if common == 0 or plen == 0 or rlen == 0:
        return 0.0
    p, r = common / plen, common / rlen
    return 2 * p * r / (p + r)


def rouge1(pred, ref):
    pt, rt = toks(pred), toks(ref)
    from collections import Counter
    overlap = sum((Counter(pt) & Counter(rt)).values())
    return _f1(overlap, len(pt), len(rt))


def rougeL(pred, ref):
    pt, rt = toks(pred), toks(ref)
    return _f1(_lcs(pt, rt), len(pt), len(rt))


def token_f1(pred, ref):
    from collections import Counter
    pt, rt = toks(pred), toks(ref)
    common = sum((Counter(pt) & Counter(rt)).values())
    return _f1(common, len(pt), len(rt))


_LETTER = [
    re.compile(r"correct answer is\s*\**\s*\(?\s*([A-E])\b", re.I),
    re.compile(r"answer\s*[:=]?\s*\**\s*\(?\s*([A-E])\b", re.I),
    re.compile(r"^\s*\(?\s*([A-E])[).:]", re.I),
    re.compile(r"\b([A-E])\)", re.I),
]


def extract_letter(text):
    for pat in _LETTER:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def extract_yesno(text):
    m = re.match(r"\s*\**\s*(yes|no|maybe|ya|tidak|mungkin)\b", text.strip(), re.I)
    if not m:
        return None
    w = m.group(1).lower()
    return {"ya": "yes", "tidak": "no", "mungkin": "maybe"}.get(w, w)


def bucket_of(sample, ref):
    src = sample.get("source", "")
    user = " ".join(m["content"] for m in sample["messages"] if m["role"] == "user")
    if src == "medmcqa" or re.search(r"\n\s*[A-E]\)", user):
        return "mcqa"
    if src == "pubmedqa" or extract_yesno(ref) in ("yes", "no", "maybe"):
        return "yesno"
    return "open"


# --------------------------------------------------------------------------- #
# Loading model
# --------------------------------------------------------------------------- #
def load_model(args):
    """Kembalikan (kind, handle, tokenizer). kind in {'hf','gguf'}.
    GGUF (Bagian 4): handle=llama_cpp.Llama; tokenizer HF dipakai HANYA utk format prompt
    (template chat identik dgn eval 16-bit -> verifikasi konsistensi template 4.4)."""
    from transformers import AutoTokenizer
    if args.gguf:
        from llama_cpp import Llama
        llm = Llama(model_path=args.gguf, n_ctx=4096, n_gpu_layers=-1,
                    n_threads=os.cpu_count(), verbose=False, logits_all=False)
        tok = AutoTokenizer.from_pretrained(args.model)  # merged dir -> hanya utk template
        return "gguf", llm, tok

    if args.loader == "unsloth":
        import unsloth
        # KEDUA model (Qwen3.5-2B & Gemma 4 E2B) multimodal -> FastVisionModel (sama spt training).
        Loader = unsloth.FastVisionModel
        try:
            model, tok = Loader.from_pretrained(
                model_name=args.model, max_seq_length=args.max_seq_length,
                load_in_4bit=args.load_in_4bit, dtype=None)
        except TypeError:   # versi Unsloth yg tak terima max_seq_length pada loader vision
            model, tok = Loader.from_pretrained(
                model_name=args.model, load_in_4bit=args.load_in_4bit, dtype=None)
        Loader.for_inference(model)
        # FastVisionModel bisa mengembalikan processor (mis. Gemma 4) -> ambil tokenizer teks
        # di dalamnya (eval ini text-only; chat template ikut di tokenizer).
        tok = getattr(tok, "tokenizer", tok)
    else:
        import torch
        from transformers import AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return "hf", model, tok


# --------------------------------------------------------------------------- #
# Generate (greedy, identik antar model utk fair comparison)
# --------------------------------------------------------------------------- #
def generate_all(kind, handle, tok, prompts, batch_size, max_new_tokens):
    preds = []
    if kind == "gguf":
        for i, p in enumerate(prompts):
            o = handle.create_completion(
                p, max_tokens=max_new_tokens, temperature=0.0,  # greedy
                repeat_penalty=1.0, top_k=1)
            preds.append(o["choices"][0]["text"])
            print(f"  generated {i + 1}/{len(prompts)}", end="\r")
        print()
        return preds

    import torch
    dev = next(handle.parameters()).device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(dev)
        with torch.no_grad():
            out = handle.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                no_repeat_ngram_size=3, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return preds


# --------------------------------------------------------------------------- #
# Evaluasi satu model
# --------------------------------------------------------------------------- #
def evaluate(args):
    import random
    mode = MODEL_MODES[args.model_type]

    samples = [json.loads(l) for l in open(args.test_file, encoding="utf-8")]
    random.seed(args.seed)
    random.shuffle(samples)
    samples = samples[:args.n_eval]
    print(f"[{args.label}] model={args.model} n={len(samples)}")

    kind, handle, tok = load_model(args)

    prompts, refs, buckets, langs = [], [], [], []
    for s in samples:
        p, r = split_prompt_reference(s, mode, tok)
        prompts.append(p)
        refs.append(r)
        buckets.append(bucket_of(s, r))
        langs.append("id" if is_id(s) else "en")

    preds = generate_all(kind, handle, tok, prompts, args.batch_size, args.max_new_tokens)

    # Task 4 — JARING PENGAMAN: post-processing sapaan pd OUTPUT model (cleaner
    # BERSAMA chat_format, sama spt training & deployment). Transparan & konsisten;
    # bisa dimatikan dgn --no_postclean utk ablasi (lapor di metodologi).
    if not args.no_postclean:
        preds = [clean_greeting(p) for p in preds]

    # akumulasi metrik per (bucket) dan per (bucket, lang) dan overall lang
    agg = defaultdict(lambda: defaultdict(list))   # key -> metric -> [values]
    examples = []
    for pred, ref, b, lg in zip(preds, refs, buckets, langs):
        keys = [f"bucket:{b}", f"bucket:{b}|lang:{lg}", f"lang:{lg}", "overall"]
        if b == "mcqa":
            hit = float(extract_letter(pred) is not None and
                        extract_letter(pred) == extract_letter(ref))
            for k in keys:
                agg[k]["accuracy"].append(hit)
        elif b == "yesno":
            gp, gr = extract_yesno(pred), extract_yesno(ref)
            hit = float(gp is not None and gp == gr)
            for k in keys:
                agg[k]["accuracy"].append(hit)
        else:
            rl, r1, f1 = rougeL(pred, ref), rouge1(pred, ref), token_f1(pred, ref)
            for k in keys:
                agg[k]["rougeL"].append(rl)
                agg[k]["rouge1"].append(r1)
                agg[k]["token_f1"].append(f1)
        if len(examples) < 12:
            examples.append({"bucket": b, "lang": lg, "pred": pred[:300], "ref": ref[:300]})

    def summarize(d):
        return {m: round(sum(v) / len(v), 4) for m, v in d.items() if v}, \
               {m + "_n": len(v) for m, v in d.items() if v}

    results = {}
    for k, d in sorted(agg.items()):
        means, counts = summarize(d)
        results[k] = {**means, **counts}

    out = {
        "label": args.label,
        "model": args.model,
        "model_type": args.model_type,
        "config": {
            "n_eval": len(samples), "max_new_tokens": args.max_new_tokens,
            "do_sample": False, "no_repeat_ngram_size": 3, "batch_size": args.batch_size,
            "quant": "Q4_K_M(gguf)" if args.gguf else ("4bit" if args.load_in_4bit else "16bit"),
            "loader": "gguf" if args.gguf else args.loader, "gguf": args.gguf,
            "test_file": args.test_file, "seed": args.seed,
        },
        "metrics": results,
        "examples": examples,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")
    _print_one(out)


def _print_one(out):
    print(f"\n=== {out['label']} ===")
    for k in ("overall", "bucket:mcqa", "bucket:yesno", "bucket:open",
              "lang:id", "lang:en"):
        if k in out["metrics"]:
            print(f"  {k:16s} {out['metrics'][k]}")


# --------------------------------------------------------------------------- #
# Ringkasan banyak hasil -> tabel + delta (finetuned - baseline)
# --------------------------------------------------------------------------- #
def summarize_results(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    runs = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        runs[d["label"]] = d["metrics"]
    if not runs:
        print("Tak ada hasil JSON di", folder)
        return

    cols = ["overall", "bucket:mcqa", "bucket:yesno", "bucket:open", "lang:id", "lang:en"]

    def metric_for(m):
        return "accuracy" if m in ("bucket:mcqa", "bucket:yesno") else "rougeL"

    print(f"\n{'label':22s} | " + " | ".join(f"{c.split(':')[-1]:>8s}" for c in cols))
    print("-" * 100)
    for label, met in runs.items():
        cells = []
        for c in cols:
            mk = metric_for(c)
            v = met.get(c, {}).get(mk)
            cells.append(f"{v:8.4f}" if isinstance(v, (int, float)) else f"{'-':>8s}")
        print(f"{label:22s} | " + " | ".join(cells))

    # delta untuk pasangan *_finetuned vs *_baseline
    print("\n--- PENINGKATAN (finetuned - baseline) ---")
    for ft in [l for l in runs if l.endswith("_finetuned")]:
        base = ft.replace("_finetuned", "_baseline")
        if base not in runs:
            continue
        cells = []
        for c in cols:
            mk = metric_for(c)
            a = runs[ft].get(c, {}).get(mk)
            b = runs[base].get(c, {}).get(mk)
            cells.append(f"{a - b:+8.4f}" if isinstance(a, (int, float))
                         and isinstance(b, (int, float)) else f"{'-':>8s}")
        print(f"{ft.replace('_finetuned',''):22s} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", metavar="FOLDER", help="ringkas semua *.json di folder")
    ap.add_argument("--model", help="path model merged ATAU HF id (baseline)")
    ap.add_argument("--model_type", choices=["qwen", "gemma"])
    ap.add_argument("--label", default="run")
    ap.add_argument("--test_file", default="Data/processed_final/test_final.jsonl")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--loader", choices=["unsloth", "hf"], default="unsloth")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="eval pada bobot 4-bit (default 16-bit utk Bagian 3)")
    ap.add_argument("--gguf", default=None,
                    help="path file .gguf (Q4_K_M) -> eval terkuantisasi (Bagian 4); "
                         "--model dipakai sbg direktori tokenizer utk format prompt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_postclean", action="store_true",
                    help="matikan post-processing sapaan pd output (Task 4) -> ablasi")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.summarize:
        summarize_results(args.summarize)
        return
    assert args.model and args.model_type, "--model dan --model_type wajib"
    if args.out is None:
        args.out = f"results/eval_{args.label}.json"
    evaluate(args)


if __name__ == "__main__":
    main()
