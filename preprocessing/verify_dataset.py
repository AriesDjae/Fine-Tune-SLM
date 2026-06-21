"""
verify_dataset.py — VERIFIER INDEPENDEN (dibangun ulang pasca audit manual).

Memeriksa SELURUH train/val/test hasil strict_clean_v2 untuk SISA artefak per kategori
B1-B6. Khusus NAMA: pakai NER (cahya IndoBERT), BUKAN cuma regex greeting (penyebab
klaim "0 leaks" palsu di v1). Lapor jumlah & PERSEN sisa per kategori; target 0% (B1-B5),
B6 mendekati 0.

Jalankan:  python preprocessing/verify_dataset.py
Output  :  results/verify_v2_<tanggal>.json + ringkasan layar.
"""
from __future__ import annotations
import argparse, datetime, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strict_clean_v2 as S   # pakai POLA B1-B6 yang SAMA + NER

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "Data" / "processed_id"
RESULTS = ROOT / "results"

# pola deteksi = pola pembersih (B1-B6) + greeting
import re
_GREET_LEAD = re.compile(r"(?i)^\s*(hai|halo|alo|hallo|hello|hi|hey|selamat\s+(pagi|siang|sore|malam))\b")
CHECKS = {
    "B1_drName": S._DR_MID,                 # "dr. Nama" tersisa
    "B1_drName_trail": S._DR_TRAIL,
    "B1_titled_name": S._TITLED,
    "B1_lead_bang": S._LEAD_BANG,
    "B1_trail_name": S._TRAIL_NAME,
    "B1_greeting_lead": _GREET_LEAD,
    "B1_leaked_name": S._LEAKED_RE,         # nama bocor terkonfirmasi (blocklist) — wajib 0
    "B2_placeholder": S._PLACEHOLDER,
    "B3_article_tail": S._ARTICLE_TAIL,
    "B4_citation": S._CITATION,
    "B5_emoji": S._EMOJI,
    "B5_emoticon": S._EMOTICON,
}


def glued_count(t):
    return len(re.findall(r"[a-z][A-Z]", t))


def main():
    splits = {}
    for sp in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(OUT_DIR / f"{sp}.jsonl", encoding="utf-8")]
        splits[sp] = rows

    report = {"date": S._dt.datetime.now().isoformat(timespec="seconds"), "splits": {}}
    print("=" * 78)
    print("VERIFIER INDEPENDEN v2 (NER + pola B1-B6) — sisa artefak per kategori")
    print("=" * 78)
    for sp, rows in splits.items():
        n = len(rows)
        cnt = Counter()
        glued = 0
        qs = [r["messages"][1]["content"] for r in rows]
        as_ = [r["messages"][2]["content"] for r in rows]
        for q, a in zip(qs, as_):
            for name, pat in CHECKS.items():
                # dr/greeting/trail/article/emoji dicek di jawaban; lainnya di q & a TERPISAH
                # (cek q,a terpisah — BUKAN q+"\n"+a — agar tak ada false-match lintas-batas,
                #  mis. q diakhiri "dr" + a diawali Kapital terbaca "dr Kapital" palsu).
                if name in ("B1_drName_trail", "B1_trail_name", "B3_article_tail", "B1_greeting_lead"):
                    m = pat.search(a)
                else:
                    m = pat.search(q) or pat.search(a)
                # B1_trail_name: JANGAN hitung sign-off ("Sekian"/"Semoga Bermanfaat"/...) sbg nama
                if name == "B1_trail_name" and m and m.group(1).lower() in S._TRAIL_NAME_STOP:
                    m = None
                if m:
                    cnt[name] += 1
            if glued_count(q) + glued_count(a) > 0:
                glued += 1
        # NER PER (seluruh split)
        print(f"[{sp}] NER {n:,}x2 field (cahya) ...", flush=True)
        q_per = S.ner_has_person(qs)
        a_per = S.ner_has_person(as_)
        name_ner = sum(1 for i in range(n) if q_per[i] or a_per[i])

        d = {"n": n, "glued_words": glued, "glued_pct": round(100 * glued / n, 2),
             "NER_person_residual": name_ner, "NER_person_pct": round(100 * name_ner / n, 3)}
        for name in CHECKS:
            d[name] = cnt[name]
            d[name + "_pct"] = round(100 * cnt[name] / n, 3)
        report["splits"][sp] = d

        print(f"\n## {sp}  n={n:,}")
        print(f"   NER PERSON residual : {name_ner}  ({d['NER_person_pct']}%)  <- target 0%")
        for name in CHECKS:
            flag = "" if cnt[name] == 0 else "  <-- SISA!"
            print(f"   {name:20} {cnt[name]:5}  ({d[name+'_pct']}%){flag}")
        print(f"   B6_glued_words      {glued:5}  ({d['glued_pct']}%)")

    (RESULTS / f"verify_v2_{S._dt.datetime.now():%Y%m%d}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # verdict
    allcats = []
    for sp, d in report["splits"].items():
        for k, v in d.items():
            if k.endswith("_pct") and not k.startswith(("glued", "B6")):
                allcats.append((sp, k, v))
    bad = [(s, k, v) for s, k, v in allcats if v > 0]
    print("\n" + "=" * 78)
    if not bad:
        print("VERDICT: BERSIH — 0% sisa di semua kategori B1-B5 + NER. ")
    else:
        print(f"VERDICT: MASIH ADA SISA di {len(bad)} kategori:")
        for s, k, v in bad:
            print(f"   {s}/{k} = {v}%")
    print(f"\nDetail: {RESULTS / f'verify_v2_{S._dt.datetime.now():%Y%m%d}.json'}")


# =====================================================================================
# VERIFIER v2 FIXED (TUGAS 4) — tambah RECALL + coverage-gap, POLA INDEPENDEN dari stripper.
# =====================================================================================
# Pelajaran metodologis v1->v2: verifier lama (main() di atas) MEMAKAI POLA YANG SAMA dgn
# cleaner (import strict_clean_v2). Ia hanya mengukur PRECISION ("tak ada token jelek yg
# pola-X tinggalkan") -> sirkular, melaporkan 0% palsu. Verifier FIXED ini:
#   (a) memakai DETEKTOR INDEPENDEN dari audit_recall_v2 (regex sendiri, bukan B1-B6), dan
#   (b) menambah cek RECALL (OVER_STRIP) yang sebelumnya HILANG sehingga verdict "0%"
#       menyesatkan. Plus coverage-gap: DR_NAME_LEAK case-insensitive & BARE_CITATION_TAIL.
# Text-only & cepat (tanpa NER/align); angka align-confirmed ada di audit_recall_v2_*.json.

import audit_recall_v2 as AR  # detektor INDEPENDEN (regex sendiri)


def _qa(rec):
    q = a = ""
    for m in rec.get("messages", []):
        if m["role"] == "user":
            q = m["content"]
        elif m["role"] == "assistant":
            a = m["content"]
    return q, a


def main_fixed():
    date = f"{datetime.datetime.now():%Y%m%d}"
    cats = ["OVER_STRIP_HIGH", "OVER_STRIP_LOW", "BARE_CITATION_TAIL", "DR_NAME_LEAK", "TRUNCATION"]
    report = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
              "note": "independent recall+coverage verifier; precision-only B1-B6 ada di verify_v2_<date>.json",
              "splits": {}}
    print("=" * 80)
    print("VERIFIER v2 FIXED — RECALL + coverage-gap (pola INDEPENDEN dari cleaner)")
    print("=" * 80)
    for sp in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(OUT_DIR / f"{sp}.jsonl", encoding="utf-8")]
        n = len(rows)
        cnt = Counter()
        for r in rows:
            q, a = _qa(r)
            # OVER_STRIP (RECALL) — hanya pada jawaban (head-strip jawaban)
            conf, _ = AR.detect_over_strip_textonly(a)
            if conf == "high":
                cnt["OVER_STRIP_HIGH"] += 1
            elif conf == "low":
                cnt["OVER_STRIP_LOW"] += 1
            # coverage-gap leak + truncation — q & a
            for text in (q, a):
                if AR.detect_dr_name_leak(text)[0]:
                    cnt["DR_NAME_LEAK"] += 1
                    break
            for text in (q, a):
                if AR.detect_bare_citation_tail(text)[0]:
                    cnt["BARE_CITATION_TAIL"] += 1
                    break
            for text in (q, a):
                if AR.detect_truncation(text)[0]:
                    cnt["TRUNCATION"] += 1
                    break
        d = {"n": n}
        for c in cats:
            d[c] = cnt[c]
            d[c + "_pct"] = round(100 * cnt[c] / n, 3) if n else 0
        report["splits"][sp] = d
        print(f"\n## {sp}  n={n:,}")
        for c in cats:
            print(f"   {c:20} {cnt[c]:6}  ({d[c+'_pct']}%)")

    out = RESULTS / f"verify_v2_FIXED_{date}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- bandingkan dgn verifier LAMA (precision-only) utk tunjukkan recall/coverage gap ---
    old_path = RESULTS / f"verify_v2_{date}.json"
    print("\n" + "=" * 80)
    print("PERBANDINGAN vs verifier LAMA (precision-only, pola = cleaner):")
    if old_path.exists():
        old = json.load(open(old_path, encoding="utf-8"))
        otr = old["splits"].get("train", {})
        ntr = report["splits"]["train"]
        print(f"   LAMA  train B1_drName     = {otr.get('B1_drName_pct', 'n/a')}%   "
              f"(pola = stripper -> sirkular)")
        print(f"   FIXED train DR_NAME_LEAK  = {ntr['DR_NAME_LEAK_pct']}%   (regex independen, case-insensitive)")
        print(f"   LAMA  train B4_citation   = {otr.get('B4_citation_pct', 'n/a')}%")
        print(f"   FIXED train BARE_CITATION = {ntr['BARE_CITATION_TAIL_pct']}%   (ekor telanjang B3/B4)")
        print(f"   LAMA  (tidak ada cek RECALL over-strip sama sekali)")
        print(f"   FIXED train OVER_STRIP_HIGH = {ntr['OVER_STRIP_HIGH_pct']}% | LOW = {ntr['OVER_STRIP_LOW_pct']}%")
    else:
        print(f"   (verify_v2_{date}.json tidak ditemukan — jalankan `python preprocessing/verify_dataset.py` dulu)")
    print(f"\nDetail FIXED: {out}")
    print("Catatan: verifier deteksi-leak hanya ukur PRECISION; RECALL (over-strip) butuh cek")
    print("terpisah + align ke sumber asli (lihat results/audit_recall_v2_<date>.json).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", action="store_true",
                    help="jalankan verifier RECALL+coverage independen (tanpa NER) -> verify_v2_FIXED")
    args = ap.parse_args()
    if args.fixed:
        main_fixed()
    else:
        main()
