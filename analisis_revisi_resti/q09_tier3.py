"""Q09: rincian 25 penilaian dokter (ukuran, dokter, ganda) + Safe rate dokter vs GPT-4o pada pasangan yang sama.
Hanya membaca (dataset HF privat + results/llm_judge); tidak menulis ke results/."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
import analisis_tier3 as t3
sys.stdout.reconfigure(encoding='utf-8')
idx_file = json.load(open(t3.INDEKS, encoding='utf-8'))
indeks = {p['id']: p for p in idx_file['pasangan']}
print('kunci indeks:', [k for k in idx_file if k != 'pasangan'], {k: idx_file[k] for k in idx_file if k != 'pasangan'})
dok = [r for r in t3.muat_dokter(None) if r['id'] in indeks]
print('n penilaian', len(dok), '| dokter', len({r['email'] for r in dok}), '| pasangan unik', len({r['id'] for r in dok}))
print('per dokter', sorted(Counter(r['email'] for r in dok).values(), reverse=True))
print('per ukuran', Counter(indeks[r['id']]['ukuran'] for r in dok))
print('target pasangan yg dinilai', Counter(indeks[r['id']].get('target') for r in dok))
tier2 = {u: t3.muat_tier2(u) for u in t3.UKURAN}
sd = Counter(); sg = Counter()
for r in dok:
    p = indeks[r['id']]
    g = tier2[p['ukuran']][p['idx_test']]['nilai']
    kunci = p['kunci']
    d_ = t3.per_kondisi({'A': r['A'], 'B': r['B']}, kunci)
    g_ = t3.per_kondisi({'A': g['A'], 'B': g['B']}, kunci)
    for k in ('baseline', 'finetuned'):
        sd[k] += d_[k]['aman'] == 'ya'; sg[k] += g_[k]['aman'] == 'ya'
n = len(dok)
print('Safe dokter      ', {k: f'{sd[k]}/{n}={sd[k]/n:.0%}' for k in ('baseline', 'finetuned')})
print('Safe GPT-4o (sama)', {k: f'{sg[k]}/{n}={sg[k]/n:.0%}' for k in ('baseline', 'finetuned')})
print('contoh entri indeks:', {k: v for k, v in list(indeks.values())[0].items()})
