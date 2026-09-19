# -*- coding: utf-8 -*-
"""
pengaman_decoding.py — pengaman pengulangan untuk jalur DEPLOY (llama.cpp / GGUF).

Tanpa pengaman, 0.8B fine-tuned greedy terkunci mengulang kalimat pada 73 dari 100 soal.
Pemblokir 3-gram eval.py memutus loop tapi bekerja pada TOKEN, sehingga istilah yang wajar
diulang ikut terlarang dan kata jadi rusak ("benjola", "putING"). DRY+ adalah pemenang
`uji_decoding.py` (0.8B, n=100): 0 loop, kata asing 0,85/100 kata (token-3: 3,58),
token-F1 setara (−0,004, CI melewati nol).

DRY+ = tiga lapis:
  1. hukuman DRY (mult 0.8, base 1.75, allowed_length 2): pengulangan pendek dibiarkan,
     perpanjangan urutan yang sudah pernah muncul dihukum eksponensial. Tanpa "sequence
     breakers" llama.cpp.
  2. larangan token awal-kata yang sama dengan token sebelumnya ("sering sering").
  3. jaring terakhir: bila kalimat utuh terulang atau 6 kata muncul ketiga kalinya,
     generasi dihentikan dan teks dipotong di awal pengulangan.

Hanya butuh numpy + llama_cpp, jadi jalan di Termux.

    kw, rapikan = dry_plus(llm, prompt)
    o = llm.create_completion(prompt, max_tokens=None, temperature=0.0, top_k=1, **kw)
    teks = rapikan(o["choices"][0]["text"])
"""
from __future__ import annotations

import re

import numpy as np

KATA = re.compile(r"\w+")
KALIMAT = re.compile(r"[^.!?\n]+[.!?\n]")
N_KALIMAT = 6          # panjang rangkaian kata yang dianggap loop


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


def potong_loop(teks):
    """Buang semuanya mulai titik pengulangan (bila ada)."""
    pos = ada_loop(teks)
    return teks if pos is None else teks[:pos].rstrip(" ,;:")


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


def berhenti_saat_loop(llm, prompt):
    """StoppingCriteria: hentikan begitu teks jawaban memuat loop (diperiksa tiap 4 token)."""
    plen = len(llm.tokenize(prompt.encode("utf-8"), special=True))

    def berhenti(input_ids, logits):
        if (len(input_ids) - plen) % 4:
            return False
        t = llm.detokenize(list(input_ids[plen:])).decode("utf-8", "ignore")
        return ada_loop(t) is not None
    return berhenti


def dry_plus(llm, prompt):
    """(kwargs untuk create_completion, fungsi perapi teks akhir)."""
    from llama_cpp import LogitsProcessorList, StoppingCriteriaList
    kw = {
        "repeat_penalty": 1.0,
        "logits_processor": LogitsProcessorList([dry_processor(), gagap_processor(llm)]),
        "stopping_criteria": StoppingCriteriaList([berhenti_saat_loop(llm, prompt)]),
    }
    return kw, potong_loop
