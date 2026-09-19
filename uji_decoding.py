# -*- coding: utf-8 -*-
"""
uji_decoding.py — adu cara mencegah pengulangan pada jalur deploy (llama.cpp, GGUF).

Latar: pemblokir 3-gram yang dipakai eval.py bekerja pada TOKEN, bukan kata. Tokenizer
Qwen memecah kata jadi potongan, sehingga istilah yang wajar diulang ("puting payudara")
ikut terlarang dan model terpaksa memilih potongan lain -> kata rusak ("benjola",
"putING"). Skrip ini mengukur apakah cara lain memutus loop tanpa merusak kata.

Varian (semua greedy, tanpa batas token selain n_ctx):
  A token3   : no_repeat_ngram_size=3 berbasis token — protokol yang menghasilkan angka naskah
  B polos    : tanpa pengaman apa pun — untuk melihat seberapa parah loop-nya
  C kalimat  : generasi bebas; begitu kalimat utuh terulang atau rangkaian 6 KATA muncul
               ketiga kalinya, generasi dihentikan dan teks dipotong di awal pengulangan
  D dry      : hukuman DRY (mult 0.8, base 1.75, allowed_length 2) — pengulangan pendek
               dibiarkan, perpanjangan urutan berulang dihukum eksponensial. Pendekatan:
               tanpa "sequence breakers" llama.cpp.
  E dry_plus : D + larangan mengulang KATA yang baru saja ditulis ("sering sering") +
               pendeteksi C sebagai pengaman terakhir (hanya memotong bila loop tetap lolos)

Ukuran per varian: token-F1 & ROUGE-L (definisi eval.py), "kata asing" per 100 kata
(kata yang tak pernah muncul di data latih — proksi kata rusak), sisa loop (rangkaian 6 kata
berulang di teks akhir), jumlah terpotong n_ctx, latensi.

    python uji_decoding.py --gguf outputs/gguf/Qwen3.5-0.8B.Q4_K_M.gguf --n 100
    python uji_decoding.py --ringkas results/uji_decoding
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time

import numpy as np

from eval import rougeL, token_f1, _ngram_blocker, NO_REPEAT_NGRAM

TEST = "Data/processed_id/test.jsonl"
LATIH = "Data/processed_id_final/train.jsonl"
KATA = re.compile(r"\w+")
N_KALIMAT = 6          # panjang rangkaian kata yang dianggap loop
VARIAN = ["token3", "polos", "kalimat", "dry", "dry_plus"]


def prompt_dari(sample):
    sys_ = next(m["content"] for m in sample["messages"] if m["role"] == "system")
    usr = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    ref = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
    p = (f"<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{usr}<|im_end|>\n"
         f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return p, ref


KALIMAT = re.compile(r"[^.!?\n]+[.!?\n]")


def ada_loop(teks, n=N_KALIMAT):
    """Posisi karakter tempat teks harus dipotong karena loop, atau None.

    Frasa 6 kata yang muncul DUA kali masih wajar ("Demam pada anak dapat disebabkan
    oleh ..." sebagai pembuka dua kalimat berbeda). Yang dianggap loop:
      (1) kalimat utuh (>= 4 kata) yang persis sama muncul lagi -> potong di awal ulangannya;
      (2) rangkaian n kata muncul untuk KETIGA kalinya -> potong di awal kemunculan kedua
          (menangkap loop tanpa tanda baca: "bisul ini juga dapat terjadi ... bisul ini ...").
    """
    lihat = set()
    for k in KALIMAT.finditer(teks):
        kunci = " ".join(KATA.findall(k.group().lower()))
        if kunci.count(" ") >= 3:
            if kunci in lihat:
                return k.start()
            lihat.add(kunci)
    m = list(KATA.finditer(teks.lower()))
    w = [x.group() for x in m]
    muncul = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        muncul.setdefault(g, []).append(i)
        if len(muncul[g]) == 3:
            return m[muncul[g][1]].start()
    return None


def potong_kalimat(teks, pos):
    """Buang semuanya mulai titik pengulangan."""
    return teks[:pos].rstrip(" ,;:")


def dry_processor(mult=0.8, base=1.75, allowed=2, cap=50):
    def proses(input_ids, scores):
        ids = np.asarray(input_ids)
        n = ids.size
        if n < 3:
            return scores
        last = ids[-1]
        hukum = {}
        for i in np.flatnonzero(ids[:-1] == last):
            L = 1
            while L < cap and i - L >= 0 and ids[i - L] == ids[n - 1 - L]:
                L += 1
            if L >= allowed:
                t = int(ids[i + 1])
                hukum[t] = max(hukum.get(t, 0.0), mult * base ** (L - allowed))
        for t, h in hukum.items():
            scores[t] -= h
        return scores
    return proses


def gagap_processor(llm):
    """Larang token awal-kata yang sama persis dengan token sebelumnya ("memang memang").

    Hanya token yang diawali spasi dan berisi huruf; angka dan potongan kata tak disentuh,
    jadi "100" (token '1','0','0') dan reduplikasi berhubung ("anak-anak") tetap bisa.
    """
    cache = {}

    def kata_utuh(t):
        if t not in cache:
            s = llm.detokenize([t]).decode("utf-8", "ignore")
            cache[t] = s.startswith(" ") and s.strip().isalpha()
        return cache[t]

    def proses(input_ids, scores):
        t = int(input_ids[-1])
        if kata_utuh(t):
            scores[t] = -np.inf
        return scores
    return proses


def generate(llm, prompt, varian, n_ctx):
    from llama_cpp import LogitsProcessorList, StoppingCriteriaList
    kw = dict(max_tokens=None, temperature=0.0, top_k=1, repeat_penalty=1.0)
    if varian == "token3":
        kw["logits_processor"] = LogitsProcessorList([_ngram_blocker(NO_REPEAT_NGRAM)])
    elif varian == "dry":
        kw["logits_processor"] = LogitsProcessorList([dry_processor()])
    elif varian == "dry_plus":
        kw["logits_processor"] = LogitsProcessorList([dry_processor(), gagap_processor(llm)])
    if varian in ("kalimat", "dry_plus"):
        plen = len(llm.tokenize(prompt.encode("utf-8"), special=True))

        def berhenti(input_ids, logits):
            if (len(input_ids) - plen) % 4:          # cukup periksa tiap 4 token
                return False
            t = llm.detokenize(list(input_ids[plen:])).decode("utf-8", "ignore")
            return ada_loop(t) is not None
        kw["stopping_criteria"] = StoppingCriteriaList([berhenti])
    t0 = time.time()
    o = llm.create_completion(prompt, **kw)
    dt = time.time() - t0
    teks = o["choices"][0]["text"]
    if varian in ("kalimat", "dry_plus"):
        pos = ada_loop(teks)
        if pos is not None:
            teks = potong_kalimat(teks, pos)
    return teks, o["choices"][0].get("finish_reason"), o["usage"]["completion_tokens"], dt


def jalankan(args):
    from llama_cpp import Llama
    baris = [json.loads(x) for x in open(TEST, encoding="utf-8")]
    idx = sorted(random.Random(42).sample(range(len(baris)), args.n))
    llm = Llama(model_path=args.gguf, n_ctx=args.n_ctx, n_gpu_layers=0,
                n_threads=args.threads, verbose=False)
    os.makedirs(args.out_dir, exist_ok=True)
    for v in args.varian:
        path = os.path.join(args.out_dir, f"{v}.jsonl")
        sudah = {json.loads(x)["idx"] for x in open(path, encoding="utf-8")} if os.path.exists(path) else set()
        with open(path, "a", encoding="utf-8") as f:
            for k, i in enumerate(idx, 1):
                if i in sudah:
                    continue
                p, ref = prompt_dari(baris[i])
                teks, fin, ntok, dt = generate(llm, p, v, args.n_ctx)
                f.write(json.dumps({"idx": i, "pred": teks, "ref": ref, "selesai": fin,
                                    "token": ntok, "latensi_s": round(dt, 2)},
                                   ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {v} {k}/{len(idx)}  {ntok} tok  {dt:5.1f}s  {fin}   ", end="\r")
        print()


def kosakata_latih():
    kos = set()
    for x in open(LATIH, encoding="utf-8"):
        for m in json.loads(x)["messages"]:
            kos.update(KATA.findall(m["content"].lower()))
    return kos


def ringkas(folder):
    kos = kosakata_latih()
    hasil = {}
    for v in VARIAN:
        path = os.path.join(folder, f"{v}.jsonl")
        if not os.path.exists(path):
            continue
        r = [json.loads(x) for x in open(path, encoding="utf-8")]
        asing, total, contoh = 0, 0, []
        for x in r:
            w = [k for k in KATA.findall(x["pred"].lower()) if k.isalpha()]
            a = [k for k in w if k not in kos]
            asing += len(a)
            total += len(w)
            contoh += a
        hasil[v] = {
            "n": len(r),
            "token_f1": round(statistics.mean(token_f1(x["pred"], x["ref"]) for x in r), 4),
            "rougeL": round(statistics.mean(rougeL(x["pred"], x["ref"]) for x in r), 4),
            "kata_asing_per100": round(100 * asing / max(total, 1), 2),
            "sisa_loop": sum(ada_loop(x["pred"]) is not None for x in r),
            "terpotong_nctx": sum(x["selesai"] == "length" for x in r),
            "panjang_kata_mean": round(total / len(r), 1),
            "latensi_s_mean": round(statistics.mean(x["latensi_s"] for x in r), 2),
            "contoh_kata_asing": sorted(set(contoh))[:25],
        }
    json.dump(hasil, open(os.path.join(folder, "ringkasan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"{'varian':8s} {'n':>4s} {'F1':>7s} {'RL':>7s} {'asing%':>7s} {'loop':>5s} "
          f"{'potong':>6s} {'kata':>6s} {'lat_s':>6s}")
    for v, h in hasil.items():
        print(f"{v:8s} {h['n']:4d} {h['token_f1']:7.4f} {h['rougeL']:7.4f} "
              f"{h['kata_asing_per100']:7.2f} {h['sisa_loop']:5d} {h['terpotong_nctx']:6d} "
              f"{h['panjang_kata_mean']:6.1f} {h['latensi_s_mean']:6.2f}")
    for v, h in hasil.items():
        print(f"  {v}: {', '.join(h['contoh_kata_asing'])}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--varian", nargs="+", default=VARIAN, choices=VARIAN)
    ap.add_argument("--out_dir", default="results/uji_decoding")
    ap.add_argument("--ringkas", metavar="FOLDER")
    a = ap.parse_args()
    if a.ringkas:
        ringkas(a.ringkas)
    else:
        jalankan(a)
        ringkas(a.out_dir)


if __name__ == "__main__":
    main()
