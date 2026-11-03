"""Table 2 token-F1/ROUGE-L dengan vs tanpa clean_greeting (5.000 iterasi, seed 42), seluruh test set."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analisis_revisi import muat_test, muat_preds, bootstrap, LABEL, UKURAN, clean_greeting, token_f1, rougeL
sys.stdout.reconfigure(encoding='utf-8')
refs = [t['assistant'] for t in muat_test()]
out = {}
for mode in ('tanpa_clean', 'dengan_clean'):
    P = {l: [clean_greeting(p) if mode == 'dengan_clean' else p for p in muat_preds(l)] for l in LABEL}
    for u in UKURAN:
        b, f = P[f'qwen{u}_baseline'], P[f'qwen{u}_finetuned']
        for m, fn in (('f1', token_f1), ('rougeL', rougeL)):
            sb = [fn(x, r) for x, r in zip(b, refs)]; sf = [fn(x, r) for x, r in zip(f, refs)]
            out[f'{mode}|{u}|{m}'] = {'baseline': round(sum(sb) / len(sb), 4), 'finetuned': round(sum(sf) / len(sf), 4),
                                     **bootstrap([y - x for x, y in zip(sb, sf)])}
            print(mode, u, m, out[f'{mode}|{u}|{m}'])
json.dump(out, open('analisis_revisi_resti/table2_clean_greeting.json', 'w', encoding='utf-8'), indent=1)
