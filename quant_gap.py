"""
quant_gap.py — quantization gap RQ3: model fine-tuned Q4_K_M (GGUF) vs 16-bit.

Protokol = Tingkat 1 (eval.py): greedy, max_new_tokens=256, pemblokir 3-gram,
clean_greeting pada keluaran, token-F1/ROUGE-L dari eval.py. Jawaban 16-bit TIDAK
dibangkitkan ulang: diambil dari results/preds_p6/preds/<label>.preds.json, yang
sejajar dengan urutan baris Data/processed_id/test.jsonl. Keduanya dibandingkan
pada subset yang sama (seed 42), selisihnya diuji dengan paired bootstrap.

    # 1) bangkitkan Q4_K_M (resumable; python yang punya llama_cpp)
    python quant_gap.py gen --ukuran 0.8b 2b 4b --n 500
    # 2) ringkas (+ BERTScore bila paket bert-score ada, mis. .venv-gpu)
    .venv-gpu/Scripts/python.exe quant_gap.py ringkas --bertscore

Keluaran: results/quant_gap/q4_<ukuran>.jsonl (teks pasien → gitignored) dan
results/quant_gap/ringkasan.json (ter-track).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

from chat_format import clean_greeting
from eval import NO_REPEAT_NGRAM, _ngram_blocker, rougeL, token_f1

TEST = "Data/processed_id/test.jsonl"
PREDS = "results/preds_p6/preds/qwen{u}_finetuned.preds.json"
GGUF = "outputs/gguf/Qwen3.5-{U}.Q4_K_M.gguf"
OUT = "results/quant_gap"
MAX_NEW = 256
SEED = 42


def ukuran_ke_berkas(u):
    return GGUF.format(U=u.upper())


def subset(n, total):
    return sorted(random.Random(SEED).sample(range(total), n))


def prompt_dari(sample):
    # identik dengan template Qwen3.5 yang dipakai 16-bit (chat_format.RESPONSE_PART)
    sys_ = next(m["content"] for m in sample["messages"] if m["role"] == "system")
    usr = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    return (f"<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{usr}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def referensi(sample):
    return next(m["content"] for m in sample["messages"] if m["role"] == "assistant")


def gen(args):
    from llama_cpp import Llama, LogitsProcessorList
    baris = [json.loads(x) for x in open(TEST, encoding="utf-8")]
    idx = subset(args.n, len(baris))
    os.makedirs(OUT, exist_ok=True)
    for u in args.ukuran:
        path = os.path.join(OUT, f"q4_{u}.jsonl")
        sudah = set()
        if os.path.exists(path):
            sudah = {json.loads(x)["idx"] for x in open(path, encoding="utf-8")}
        sisa = [i for i in idx if i not in sudah]
        print(f"[{u}] {len(sudah)} sudah, {len(sisa)} sisa", flush=True)
        if not sisa:
            continue
        llm = Llama(model_path=ukuran_ke_berkas(u), n_ctx=2048, n_gpu_layers=0,
                    n_threads=args.threads, verbose=False)
        proc = LogitsProcessorList([_ngram_blocker(NO_REPEAT_NGRAM)])
        t_mulai = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for k, i in enumerate(sisa, 1):
                o = llm.create_completion(prompt_dari(baris[i]), max_tokens=MAX_NEW,
                                          temperature=0.0, top_k=1, repeat_penalty=1.0,
                                          logits_processor=proc)
                f.write(json.dumps({"idx": i, "pred": o["choices"][0]["text"],
                                    "selesai": o["choices"][0].get("finish_reason")},
                                   ensure_ascii=False) + "\n")
                f.flush()
                laju = (time.time() - t_mulai) / k
                print(f"[{u}] {k}/{len(sisa)}  ~{laju * (len(sisa) - k) / 60:.0f} menit lagi",
                      flush=True)
        del llm


def bootstrap(selisih, b=5000):
    rng = random.Random(SEED)
    n = len(selisih)
    rata = sorted(sum(selisih[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [rata[int(0.025 * b)], rata[int(0.975 * b) - 1]]


def ringkas(args):
    baris = [json.loads(x) for x in open(TEST, encoding="utf-8")]
    hasil = {"protokol": {"test_file": TEST, "seed_subset": SEED, "max_new_tokens": MAX_NEW,
                          "no_repeat_ngram_size": NO_REPEAT_NGRAM, "greedy": True,
                          "postclean": "clean_greeting", "bootstrap": 5000,
                          "q4": "GGUF Q4_K_M, llama.cpp CPU",
                          "sixteen_bit": "results/preds_p6/preds (Tingkat 1)"},
             "per_ukuran": {}}
    for u in ["0.8b", "2b", "4b"]:
        path = os.path.join(OUT, f"q4_{u}.jsonl")
        if not os.path.exists(path):
            continue
        q4 = {r["idx"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
        p16 = json.load(open(PREDS.format(u=u), encoding="utf-8"))["preds"]
        assert len(p16) == len(baris), "preds 16-bit tidak sejajar test.jsonl"
        idx = sorted(q4)
        refs = [referensi(baris[i]) for i in idx]
        a = [clean_greeting(q4[i]["pred"]) for i in idx]
        b = [clean_greeting(p16[i]) for i in idx]
        blok = {"n": len(idx), "q4_terpotong_256": sum(q4[i]["selesai"] == "length" for i in idx),
                "identik_persis": sum(x == y for x, y in zip(a, b))}
        metrik = {"token_f1": token_f1, "rougeL": rougeL}
        skor = {m: ([f(x, r) for x, r in zip(a, refs)], [f(y, r) for y, r in zip(b, refs)])
                for m, f in metrik.items()}
        if args.bertscore:
            from bert_score import score as _bs
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _, _, fa = _bs(a, refs, model_type="bert-base-multilingual-cased", device=dev,
                           batch_size=16, verbose=False)
            _, _, fb = _bs(b, refs, model_type="bert-base-multilingual-cased", device=dev,
                           batch_size=16, verbose=False)
            skor["bertscore"] = (fa.tolist(), fb.tolist())
        for m, (sa, sb) in skor.items():
            d = [x - y for x, y in zip(sa, sb)]
            blok[m] = {"q4": round(sum(sa) / len(sa), 4), "16bit": round(sum(sb) / len(sb), 4),
                       "delta": round(sum(d) / len(d), 4),
                       "sk95": [round(v, 4) for v in bootstrap(d)]}
        hasil["per_ukuran"][u] = blok
        print(u, json.dumps(blok, ensure_ascii=False))
    with open(os.path.join(OUT, "ringkasan.json"), "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=1, ensure_ascii=False)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--ukuran", nargs="+", default=["0.8b", "2b", "4b"])
    g.add_argument("--n", type=int, default=500)
    g.add_argument("--threads", type=int, default=12)
    r = sub.add_parser("ringkas")
    r.add_argument("--bertscore", action="store_true")
    args = ap.parse_args()
    gen(args) if args.cmd == "gen" else ringkas(args)


if __name__ == "__main__":
    main()
