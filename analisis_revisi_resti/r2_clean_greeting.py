"""
r2_clean_greeting.py — putaran 2 (R2-06, R2-02): periksa clean_greeting di semua analisis.

Hanya membaca keluaran yang sudah ada; tidak membangkitkan jawaban dan tidak menulis ke results/.
Keluaran: analisis_revisi_resti/r2_clean_greeting.json

    python analisis_revisi_resti/r2_clean_greeting.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analisis_revisi import LABEL, UKURAN, bootstrap, clean_greeting, muat_preds, muat_test, rougeL, token_f1  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
METRIK = (("f1", token_f1), ("rougeL", rougeL))
rerata = lambda v: sum(v) / len(v)

refs = [t["assistant"] for t in muat_test()]
mentah = {l: muat_preds(l) for l in LABEL}
bersih = {l: [clean_greeting(p) for p in v] for l, v in mentah.items()}
out = {}

# ---- R2-06: rerata 6 desimal + CI, dan berapa jawaban yang disentuh clean_greeting ----
tabel = {}
for u in UKURAN:
    b, f = f"qwen{u}_baseline", f"qwen{u}_finetuned"
    for m, fn in METRIK:
        sb = [fn(x, r) for x, r in zip(bersih[b], refs)]
        sf = [fn(x, r) for x, r in zip(bersih[f], refs)]
        tabel[f"{u}|{m}"] = {"baseline": round(rerata(sb), 6), "finetuned": round(rerata(sf), 6),
                             "delta_6d": round(rerata(sf) - rerata(sb), 6),
                             **bootstrap([y - x for x, y in zip(sb, sf)])}
        print(u, m, tabel[f"{u}|{m}"])
out["R2_06_table2"] = tabel

out["jawaban_diubah_clean_greeting"] = {
    l: sum((a or "").strip() != b for a, b in zip(mentah[l], bersih[l])) for l in LABEL}
print(out["jawaban_diubah_clean_greeting"])

# ---- R2-02e: referensi. Apakah test.jsonl masih bereaksi terhadap clean_greeting? ----
refs_bersih = [clean_greeting(r) for r in refs]
berubah = [i for i, (a, b) in enumerate(zip(refs, refs_bersih)) if a.strip() != b]
out["referensi"] = {"n": len(refs), "berubah_bila_diclean_lagi": len(berubah),
                    "contoh": [refs[i][:70] for i in berubah[:5]]}
sens = {}
for l in LABEL:
    sens[l] = {m: [round(rerata([fn(p, r) for p, r in zip(bersih[l], refs)]), 4),
                   round(rerata([fn(p, r) for p, r in zip(bersih[l], refs_bersih)]), 4)]
               for m, fn in METRIK}
out["sensitivitas_refs_diclean_lagi"] = sens
print(out["referensi"], sens)

# ---- R2-02d: uji decoding 100 soal, tanpa vs dengan clean_greeting ----
dec = {}
for v in ("polos", "token3", "kalimat", "dry", "dry_plus"):
    r = [json.loads(x) for x in open(f"results/uji_decoding/{v}.jsonl", encoding="utf-8")]
    baris = {"n": len(r), "diubah": sum((x["pred"] or "").strip() != clean_greeting(x["pred"]) for x in r),
             "ref_sama_test": all(x["ref"] == refs[x["idx"]] for x in r)}
    for m, fn in METRIK:
        baris[m] = [round(rerata([fn(x["pred"], x["ref"]) for x in r]), 4),
                    round(rerata([fn(clean_greeting(x["pred"]), x["ref"]) for x in r]), 4)]
    dec[v] = baris
    print(v, baris)
out["R2_02d_uji_decoding"] = dec

# ---- Uji-silang quant gap: angka tersimpan terulang dari keluaran GGUF? ----
qg = {}
for u in UKURAN:
    q = {x["idx"]: x for x in map(json.loads, open(f"results/quant_gap/q4_{u}.jsonl", encoding="utf-8"))}
    idx = sorted(q)
    p16 = mentah[f"qwen{u}_finetuned"]
    qg[u] = {"n": len(idx)}
    for m, fn in METRIK:
        a = [fn(clean_greeting(q[i]["pred"]), refs[i]) for i in idx]
        b = [fn(clean_greeting(p16[i]), refs[i]) for i in idx]
        qg[u][m] = {"q4": round(rerata(a), 4), "16bit": round(rerata(b), 4), "delta": round(rerata(a) - rerata(b), 4)}
out["R2_02c_quant_gap_hitung_ulang"] = qg
print(qg)

json.dump(out, open("analisis_revisi_resti/r2_clean_greeting.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
