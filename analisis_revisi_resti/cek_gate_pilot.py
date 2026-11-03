"""Hitung ulang gerbang B3a/B3b pilot dari pilot_generations.txt (logika = serverless/train_p6.py:461-503)."""
import sys
from langdetect import DetectorFactory, detect
DetectorFactory.seed = 42
for s in ['0.8b', '2b']:
    t = open(f'analisis_revisi_resti/hf_p6hasil/qwen35_{s}_p6_pilot/results/pilot_generations.txt', encoding='utf-8').read()
    blocks = [b for b in t.split('-' * 60) if 'A: ' in b]
    idok = deg = 0
    for b in blocks:
        a = b.split('A: ', 1)[1].strip()
        try:
            idok += detect(a[:500]) == 'id'
        except Exception:
            pass
        toks = a.split(); g = [tuple(toks[j:j + 4]) for j in range(len(toks) - 3)]
        deg += ((1 - len(set(g)) / len(g)) if g else 0) > 0.30
    print(s, len(blocks), 'id', idok, 'degenerate', deg)
