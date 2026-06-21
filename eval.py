"""
eval.py  —  BAGIAN 3: evaluasi fair, multi-metrik (PIVOT 5: single-model baseline vs fine-tuned).

OBJEKTIF (revisi dosen): SATU model `Qwen3.5-0.8B` dibandingkan dgn DIRINYA SENDIRI —
baseline (pre-trained) vs setelah fine-tuning (QLoRA). Jalankan dua konfigurasi dgn
protokol identik lalu ringkas deltanya:

    # FULL test set Indonesia (processed_id, 2998 sampel) -> --n_eval 3000.
    # 1) baseline (pre-trained)       -> buat delta:
    python eval.py --model unsloth/Qwen3.5-0.8B --label qwen08_baseline --n_eval 3000
    # 2) fine-tuned 16-bit (referensi quantization gap) -- adapter QLoRA langsung:
    python eval.py --model outputs/checkpoints/qwen35-0.8b-train --label qwen08_finetuned --n_eval 3000
    # 3) fine-tuned Q4_K_M (angka deployment jujur) -- butuh merge+GGUF dulu:
    python eval.py --gguf outputs/gguf/qwen35-0.8b-medical-Q4_K_M.gguf \
                   --model outputs/merged/qwen35-0.8b-medical \
                   --label qwen08_finetuned_q4 --loader gguf --n_eval 3000
    # 4) ringkas -> tabel + delta + cek ROUGE-L>=0.30:
    python eval.py --summarize results

  Catatan: model fine-tuned bisa berupa direktori adapter QLoRA (Unsloth FastLanguageModel
  auto-load base + adapter) ATAU direktori merged 16-bit. `--test_file` default sudah
  `Data/processed_id/test.jsonl`. Label HARUS berpola `*_baseline` / `*_finetuned`
  (prefix sama) agar delta otomatis terhitung. Tambah `_q4` utk eval terkuantisasi.

Protokol FAIR (3.4): test set SAMA, n_eval SAMA, greedy (do_sample=False),
max_new_tokens=256, no_repeat_ngram_size=3, format prompt SAMA (chat_format.py — ChatML
Qwen + scaffold <think></think> via template, identik train=eval=deploy).

Metrik (3.2) — EM/MCQA DI-DROP utk Pivot 4/5 (data native open-ended saja):
  - Open-ended (semua sampel processed_id) -> token-F1, ROUGE-L, ROUGE-1
  (bucket mcqa/yesno tetap didukung utk dataset lama, tapi processed_id 100% "open").
Dipecah per BAHASA (ID vs EN; processed_id mayoritas ID) dan per bucket.
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
    # processed_id (Pivot 4/5) menyertakan field eksplisit `source_lang` -> pakai itu dulu.
    sl = str(sample.get("source_lang", "")).lower()
    if sl in ("id", "ms"):
        return True
    if sl in ("en",):
        return False
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


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def add_bertscore(agg, preds, refs, buckets, langs, args):
    """BERTScore F1 (semantik) pd sampel OPEN-ENDED, dihitung BATCH lalu di-agregasi
    ke key yg sama spt metrik lain. Multilingual (default bert-base-multilingual-cased)
    agar relevan utk teks Indonesia. Butuh paket `bert-score`."""
    pairs = [(p, r, [f"bucket:{b}", f"bucket:{b}|lang:{lg}", f"lang:{lg}", "overall"])
             for p, r, b, lg in zip(preds, refs, buckets, langs) if b == "open"]
    if not pairs:
        return
    from bert_score import score as _bs
    print(f"[bertscore] model={args.bertscore_model} n={len(pairs)} ...")
    _, _, F = _bs([x[0] for x in pairs], [x[1] for x in pairs],
                  model_type=args.bertscore_model, num_layers=args.bertscore_layers,
                  batch_size=args.batch_size, verbose=False,
                  device=("cuda" if _has_cuda() else "cpu"))
    for (_, _, keys), f in zip(pairs, F.tolist()):
        for k in keys:
            agg[k]["bertscore"].append(f)


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
    # processed_id (Pivot 4/5) menandai tipe eksplisit -> native QA SELALU "open"
    # (cegah salah-bucket jawaban yg kebetulan diawali "Ya,"/"Tidak," jadi yes/no).
    if str(sample.get("type", "")).lower() == "open" or \
            sample.get("source", "") == "indonesia_qna":
        return "open"
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
        # Qwen3.5-0.8B = model TEKS -> FastLanguageModel (SAMA spt training Pivot 4/5),
        # BUKAN FastVisionModel (itu utk varian 2B multimodal Pivot 3). FastLanguageModel
        # meng-auto-load base + adapter bila --model menunjuk direktori adapter QLoRA.
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit, dtype=None)
        FastLanguageModel.for_inference(model)   # inferensi 2x lebih cepat
    else:
        import torch
        from transformers import AutoModelForCausalLM
        # --adapter: base(--model) + LoRA via PEFT (tanpa unsloth). Tokenizer dari folder
        # adapter (memuat chat_template hasil training) bila ada, agar template == training.
        tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
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
        # text=batch (BUKAN posisi): bila tok adalah Processor multimodal, argumen
        # posisi pertama = `images` -> teks akan salah-diproses sbg gambar (load_image
        # gagal "Incorrect image source"). Keyword `text` aman utk tokenizer & processor.
        enc = tok(text=batch, return_tensors="pt", padding=True, truncation=True,
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

    if args.bertscore:
        add_bertscore(agg, preds, refs, buckets, langs, args)

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
            "bertscore": args.bertscore,
            "bertscore_model": args.bertscore_model if args.bertscore else None,
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

    # Pivot 5: data processed_id 100% OPEN-ENDED -> metrik = token-F1 & ROUGE-L.
    # (EM/MCQA/yes-no di-drop; bucket mcqa/yesno tidak relevan utk objektif ini.)
    cols = ["overall", "lang:id", "lang:en", "bucket:open"]

    has_bs = any("bertscore" in met.get(c, {}) for met in runs.values() for c in cols)

    def cell(met, c):
        d = met.get(c, {})
        f1, rl, bs = d.get("token_f1"), d.get("rougeL"), d.get("bertscore")
        if isinstance(f1, (int, float)) and isinstance(rl, (int, float)):
            s = f"{f1:.3f}/{rl:.3f}"
            if has_bs:
                s += f"/{bs:.3f}" if isinstance(bs, (int, float)) else "/  -  "
            return s
        return f"{'-':>15s}"

    w = 19 if has_bs else 13
    note = "tokenF1 / rougeL" + (" / BERTScore" if has_bs else "")
    print(f"\n(sel open-ended = {note})")
    print(f"\n{'label':22s} | " + " | ".join(f"{c.split(':')[-1]:>{w}s}" for c in cols))
    print("-" * 120)
    for label, met in runs.items():
        print(f"{label:22s} | " + " | ".join(cell(met, c) for c in cols))

    # delta untuk pasangan *_finetuned vs *_baseline (token-F1 / ROUGE-L [/ BERTScore])
    def val(met, c, mk):
        return met.get(c, {}).get(mk)

    def delta_cell(amet, bmet, c):
        af, bf = val(amet, c, "token_f1"), val(bmet, c, "token_f1")
        ar, br = val(amet, c, "rougeL"), val(bmet, c, "rougeL")
        if not all(isinstance(x, (int, float)) for x in (af, bf, ar, br)):
            return f"{'-':>15s}"
        s = f"{af - bf:+.3f}/{ar - br:+.3f}"
        if has_bs:
            asc, bsc = val(amet, c, "bertscore"), val(bmet, c, "bertscore")
            s += (f"/{asc - bsc:+.3f}" if isinstance(asc, (int, float))
                  and isinstance(bsc, (int, float)) else "/  -  ")
        return s

    print("\n--- PENINGKATAN (finetuned - baseline) ---")
    for ft in [l for l in runs if l.endswith("_finetuned")]:
        base = ft.replace("_finetuned", "_baseline")
        if base not in runs:
            continue
        cells = [delta_cell(runs[ft], runs[base], c) for c in cols]
        print(f"{ft.replace('_finetuned',''):22s} | " + " | ".join(cells))

    # Quantization gap: fine-tuned Q4_K_M (label *_q4) vs fine-tuned 16-bit.
    print("\n--- QUANTIZATION GAP (Q4_K_M - 16bit, fine-tuned) ---")
    any_q4 = False
    for q4 in [l for l in runs if l.endswith("_q4")]:
        ref16 = q4[:-3]  # buang "_q4" -> label 16-bit
        if ref16 not in runs:
            continue
        any_q4 = True
        cells = [delta_cell(runs[q4], runs[ref16], c) for c in cols]
        print(f"{ref16:22s} | " + " | ".join(cells))
    if not any_q4:
        print("  (belum ada hasil *_q4 -> jalankan eval GGUF dulu)")

    # Cek target ROUGE-L >= 0.30 (acuan open-ended; subset ID paling relevan Puskesmas).
    TARGET = 0.30
    print(f"\n--- CEK TARGET ROUGE-L >= {TARGET:.2f} ---")
    for label, met in runs.items():
        for c in ("overall", "lang:id"):
            rl = val(met, c, "rougeL")
            if isinstance(rl, (int, float)):
                flag = "OK " if rl >= TARGET else "BELUM"
                print(f"  [{flag}] {label:22s} {c:10s} ROUGE-L = {rl:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", metavar="FOLDER", help="ringkas semua *.json di folder")
    ap.add_argument("--model", help="path model merged ATAU HF id (baseline)")
    # PIVOT 5 single-model: hanya Qwen3.5-0.8B (gemma/llama di-drop).
    ap.add_argument("--model_type", choices=["qwen"], default="qwen")
    ap.add_argument("--label", default="run")
    ap.add_argument("--test_file", default="Data/processed_id/test.jsonl")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--loader", choices=["unsloth", "hf"], default="unsloth")
    ap.add_argument("--adapter", default=None,
                    help="path adapter LoRA (QLoRA): base(--model)+adapter via PEFT (loader hf)")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="eval pada bobot 4-bit (default 16-bit utk Bagian 3)")
    ap.add_argument("--gguf", default=None,
                    help="path file .gguf (Q4_K_M) -> eval terkuantisasi (Bagian 4); "
                         "--model dipakai sbg direktori tokenizer utk format prompt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_postclean", action="store_true",
                    help="matikan post-processing sapaan pd output (Task 4) -> ablasi")
    ap.add_argument("--bertscore", action="store_true",
                    help="hitung BERTScore F1 (semantik) pd open-ended; butuh paket bert-score")
    ap.add_argument("--bertscore_model", default="bert-base-multilingual-cased",
                    help="model BERTScore (multilingual utk teks Indonesia)")
    ap.add_argument("--bertscore_layers", type=int, default=None,
                    help="num_layers BERTScore (default lookup otomatis dari model)")
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
