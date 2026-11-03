"""
analisis_revisi.py — analisis ulang untuk pertanyaan placeholder naskah RESTI (Q12, Q13, Q19, Q22).

Hanya membaca keluaran yang sudah ada (results/preds_p6/preds, Data/processed_id/test.jsonl,
results/llm_judge). Tidak melatih, tidak membangkitkan jawaban, tidak menulis ke results/.

Protokol disamakan dengan naskah:
  - token-F1 / ROUGE-L = eval.py (token [a-z0-9]+ huruf kecil) setelah clean_greeting
  - BERTScore F1 = bert-base-multilingual-cased, bawaan bert-score (layer 9, tanpa rescale, tanpa idf)
  - "terpotong" = analisis_truncation.utuh() (tidak diakhiri tanda baca penutup), pada teks mentah
  - paired bootstrap 5.000 iterasi, seed 42, resample indeks yang sama untuk kedua kondisi
    (sama dengan hitung_bertscore_p6.bootstrap_berpasangan)

    .venv-gpu/Scripts/python.exe analisis_revisi_resti/analisis_revisi.py
Keluaran: analisis_revisi_resti/hasil_revisi.json (+ cache BERTScore per butir)
"""
from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chat_format import clean_greeting                       # noqa: E402
from eval import rougeL, token_f1                             # noqa: E402
from analisis_truncation import utuh                          # noqa: E402
from analisis_cakupan_redflag import POLA_TEGAS, POLA_UMUM    # noqa: E402
from redflag import detect                                    # noqa: E402

UKURAN = ("0.8b", "2b", "4b")
LABEL = [f"qwen{u}_{k}" for u in UKURAN for k in ("baseline", "finetuned")]
OUT_DIR = "analisis_revisi_resti"
CACHE_BS = os.path.join(OUT_DIR, "bertscore_per_butir.json")
N_ITER, SEED = 5000, 42


def muat_test():
    rows = []
    for baris in open(os.path.join("Data", "processed_id", "test.jsonl"), encoding="utf-8"):
        m = {x["role"]: x["content"] for x in json.loads(baris)["messages"]}
        rows.append(m)
    return rows


def muat_preds(label):
    p = os.path.join("results", "preds_p6", "preds", f"{label}.preds.json")
    return json.load(open(p, encoding="utf-8"))["preds"]


def bootstrap(selisih, n_iter=N_ITER, seed=SEED):
    """Sama dengan hitung_bertscore_p6.bootstrap_berpasangan (rng baru per pemanggilan)."""
    rng = random.Random(seed)
    n = len(selisih)
    titik = sum(selisih) / n
    seb = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        seb.append(sum(selisih[i] for i in idx) / n)
    seb.sort()
    lo, hi = seb[int(0.025 * n_iter)], seb[int(0.975 * n_iter)]
    return {"n": n, "delta": round(titik, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "signifikan": lo > 0 or hi < 0}


def bertscore_per_butir(preds, refs):
    if os.path.exists(CACHE_BS):
        return json.load(open(CACHE_BS, encoding="utf-8"))
    import torch
    from bert_score import score as _bs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for lbl in LABEL:
        _, _, F = _bs(preds[lbl], refs, model_type="bert-base-multilingual-cased",
                      batch_size=32, verbose=False, device=dev)
        out[lbl] = F.tolist()
        print(f"  BERTScore {lbl}: {sum(out[lbl]) / len(out[lbl]):.4f}", flush=True)
    json.dump(out, open(CACHE_BS, "w", encoding="utf-8"))
    return out


def main():
    test = muat_test()
    refs = [t["assistant"] for t in test]
    mentah = {lbl: muat_preds(lbl) for lbl in LABEL}
    preds = {lbl: [clean_greeting(p) for p in v] for lbl, v in mentah.items()}
    n = len(refs)

    skor = {lbl: {"f1": [token_f1(p, r) for p, r in zip(preds[lbl], refs)],
                  "rougeL": [rougeL(p, r) for p, r in zip(preds[lbl], refs)]} for lbl in LABEL}
    bs = bertscore_per_butir(preds, refs)
    for lbl in LABEL:
        skor[lbl]["bertscore"] = bs[lbl]
    rerata = {lbl: {m: round(sum(v) / n, 4) for m, v in skor[lbl].items()} for lbl in LABEL}
    print("Rerata (cek terhadap Table 2):", json.dumps(rerata, indent=0))
    hasil = {"rerata_penuh": rerata}

    # ---- Q12: ablasi pemotongan, ketiga metrik ----
    q12 = {}
    for u in UKURAN:
        b, f = f"qwen{u}_baseline", f"qwen{u}_finetuned"
        bersih = [i for i in range(n) if utuh(mentah[b][i]) and utuh(mentah[f][i])]
        q12[u] = {m: bootstrap([skor[f][m][i] - skor[b][m][i] for i in bersih])
                  for m in ("f1", "rougeL", "bertscore")}
        q12[u]["n_bersih"] = len(bersih)
    hasil["Q12_truncation_subset"] = q12

    # ---- Q13: antar-ukuran, fine-tuned ----
    q13 = {}
    for a, c in (("4b", "0.8b"), ("4b", "2b"), ("2b", "0.8b")):
        A, C = f"qwen{a}_finetuned", f"qwen{c}_finetuned"
        q13[f"{a}-{c}"] = {m: bootstrap([skor[A][m][i] - skor[C][m][i] for i in range(n)])
                           for m in ("f1", "rougeL", "bertscore")}
    hasil["Q13_antar_ukuran_finetuned"] = q13

    # ---- Q19: pemotongan pada 2B baseline (dan semua konfigurasi) ----
    sampel200 = random.Random(42).sample(range(n), 200)   # = llm_judge_tier2.py:291-292
    idx_judge = sorted(json.loads(x)["idx"] for x in
                       open(os.path.join("results", "llm_judge", "tier2_2b.jsonl"), encoding="utf-8"))
    assert sorted(sampel200) == idx_judge, "sampel 200 tidak cocok dengan berkas judge"
    q19 = {}
    for lbl in LABEL:
        tp = [not utuh(t) for t in mentah[lbl]]
        q19[lbl] = {"penuh_n": n, "penuh_terpotong": sum(tp),
                    "penuh_pct": round(100 * sum(tp) / n, 1),
                    "judge200_terpotong": sum(tp[i] for i in sampel200),
                    "judge200_pct": round(100 * sum(tp[i] for i in sampel200) / 200, 1)}
    hasil["Q19_truncation_rate"] = q19

    # ---- Q22: perubahan eskalasi per butir di 4B, 195 soal red-flag ----
    terpicu = [i for i, t in enumerate(test) if detect(t["user"]).terpicu]
    b4, f4 = mentah["qwen4b_baseline"], mentah["qwen4b_finetuned"]
    teg = lambda t: bool(POLA_TEGAS.search(t or ""))
    umu = lambda t: bool(POLA_UMUM.search(t or ""))
    hilang = [i for i in terpicu if teg(b4[i]) and not teg(f4[i])]
    muncul = [i for i in terpicu if not teg(b4[i]) and teg(f4[i])]
    hasil["Q22_eskalasi_4b"] = {
        "n_terpicu": len(terpicu),
        "tegas_baseline": sum(teg(b4[i]) for i in terpicu),
        "tegas_finetuned": sum(teg(f4[i]) for i in terpicu),
        "kehilangan_tegas": len(hilang),
        "tegas_baru_di_ft": len(muncul),
        "dari_yang_hilang_ft_beri_anjuran_umum": sum(umu(f4[i]) for i in hilang),
        "dari_yang_hilang_baseline_sudah_anjuran_umum": sum(umu(b4[i]) for i in hilang),
        "dari_yang_hilang_umum_baru_muncul_di_ft": sum(umu(f4[i]) and not umu(b4[i]) for i in hilang),
        "dari_yang_hilang_tanpa_anjuran_apa_pun_di_ft": sum(not umu(f4[i]) for i in hilang),
        "dari_yang_hilang_ft_terpotong": sum(not utuh(f4[i]) for i in hilang),
        "idx_hilang": hilang,
    }

    json.dump(hasil, open(os.path.join(OUT_DIR, "hasil_revisi.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in hasil.items() if k != "rerata_penuh"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
