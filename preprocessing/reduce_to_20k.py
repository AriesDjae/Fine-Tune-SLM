# -*- coding: utf-8 -*-
"""reduce_to_20k.py — Reduksi train ke 20.000 record TERBERSIH + amankan ke folder final.

Sumber : Data/processed_id_clean/  (sudah clean + diaudit + remediasi)
Output : Data/processed_id_final/  (siap-train)
  - train : buang SEMUA record ber-noise, lalu pilih 20.000 terbaik (stratified per-domain,
            ranking quality_score + bonus keterbacaan). Distribusi domain dipertahankan.
  - val/test : hanya buang record ber-noise (ukuran tidak diciutkan; eval stabil).

Asli Data/processed_id/ dan Data/processed_id_clean/ TIDAK diubah.
"""
import json, re, os, sys, io, collections, math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "Data", "processed_id_clean")
DST  = os.path.join(ROOT, "Data", "processed_id_final")
os.makedirs(DST, exist_ok=True)
TARGET_TRAIN = 20000

NOISE = re.compile(
    r"(alodokter|halodoc|halaman (?:berikut|lain)|diskusi (?:terkait|untuk dibaca|lebih lanjut)|"
    r"artikel (?:ini|berikut|lanjut)|baca juga|dapat (?:anda|Anda) baca|silakan baca|"
    r"klik|aplikasi|unduh|download|live\s*chat|fitur\s*chat)", re.I)
GREET = re.compile(r"\b(halo|hallo|hai|selamat (pagi|siang|sore|malam)|"
                   r"terima kasih (atas|telah)|terimakasih (atas|telah))\b", re.I)
URL = re.compile(r"(https?://|www\.|\.com|\.id\b|@\w+\.)", re.I)

def asst(r): return " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
def usr(r):  return " ".join(m["content"] for m in r["messages"] if m["role"] == "user")
def nf(s):   return re.sub(r"[^a-z0-9]", "", s.lower())   # normalisasi exact-match
def toks(s): return set(re.findall(r"[a-z]+", s.lower()))
def jac(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0
NEARDUP = 0.8          # ambang Jaccard token = near-duplicate
def pkey(r): return nf(asst(r))[:200]   # bucket prefix (kandidat near-dup, murah)

def noise_reason(r):
    a = asst(r)
    if NOISE.search(a):  return "platform_ref"
    if GREET.search(a):  return "greeting"
    if URL.search(a):    return "url"
    if len(a) < 120:     return "too_short"
    return None

def quality(r):
    """Skor seleksi: quality_score + bonus keterbacaan (panjang wajar, diakhiri tanda baca)."""
    a = asst(r); q = float(r.get("quality_score", 0.0))
    len_ok = 1.0 if 200 <= len(a) <= 2000 else 0.0
    end_ok = 1.0 if a.rstrip()[-1:] in ".!?" else 0.0
    return q + 0.02 * len_ok + 0.01 * end_ok

def load(sp):
    return [json.loads(l) for l in open(os.path.join(SRC, f"{sp}.jsonl"), encoding="utf-8") if l.strip()]

def write(sp, rows):
    with open(os.path.join(DST, f"{sp}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def drop_noise(rows):
    dropped = collections.Counter(); clean = []
    for r in rows:
        why = noise_reason(r)
        if why: dropped[why] += 1
        else:   clean.append(r)
    return clean, dropped

def dedup_answer(rows):
    """Buang jawaban exact-dup lalu near-dup (Jaccard>=0.8); sisakan quality tertinggi."""
    best = {}
    for r in rows:
        k = nf(asst(r))
        if k not in best or quality(r) > quality(best[k]): best[k] = r
    buckets = collections.defaultdict(list)
    for r in best.values(): buckets[pkey(r)].append(r)
    keep = []
    for lst in buckets.values():
        lst.sort(key=quality, reverse=True); sel = []
        for r in lst:
            if all(jac(asst(r), asst(s)) < NEARDUP for s in sel): sel.append(r)
        keep.extend(sel)
    return keep

def clean_eval(rows, ban_ans, ban_q, ban_bucket):
    """Buang noise + dedup/near-dup internal + kebocoran (exact & near-dup) vs ban set."""
    rows, dn = drop_noise(rows)
    seen_a = set(); seen_bucket = collections.defaultdict(list)
    out = collections.Counter(); kept = []
    for r in rows:
        a, q, A = nf(asst(r)), nf(usr(r)), asst(r)
        if a in ban_ans:  out["leak_ans"] += 1; continue
        if q in ban_q:    out["leak_q"]   += 1; continue
        if any(jac(A, c) >= NEARDUP for c in ban_bucket.get(pkey(r), [])):
            out["leak_near"] += 1; continue
        if a in seen_a:   out["dup_ans"]  += 1; continue
        if any(jac(A, asst(s)) >= NEARDUP for s in seen_bucket.get(pkey(r), [])):
            out["dup_near"] += 1; continue
        seen_a.add(a); seen_bucket[pkey(r)].append(r); kept.append(r)
    out["noise"] = sum(dn.values())
    return kept, out

def bucketize(rows):
    b = collections.defaultdict(list)
    for r in rows: b[pkey(r)].append(asst(r))
    return b

def main():
    rep = []
    P = lambda *a: (print(*a), rep.append(" ".join(str(x) for x in a)))

    # ---- train: buang noise -> dedup jawaban -> stratified top-20000 per-domain by quality
    rows = load("train")
    clean, dropped = drop_noise(rows)
    P(f"train: {len(rows)} -> buang noise {sum(dropped.values())} ({dict(dropped)}) -> {len(clean)}")
    before = len(clean); clean = dedup_answer(clean)
    P(f"  dedup jawaban (exact+near-dup>={NEARDUP}): {before} -> {len(clean)} (buang {before-len(clean)})")
    if len(clean) < TARGET_TRAIN:
        P(f"  PERINGATAN: sisa bersih < {TARGET_TRAIN}; pakai semua.")
        chosen = clean
    else:
        by_dom = collections.defaultdict(list)
        for r in clean: by_dom[r.get("domain", "?")].append(r)
        ratio = TARGET_TRAIN / len(clean)
        # alokasi awal (floor) + bagikan sisa ke domain dgn pecahan terbesar
        alloc, frac = {}, {}
        for d, lst in by_dom.items():
            exact = len(lst) * ratio
            alloc[d] = min(len(lst), int(math.floor(exact))); frac[d] = exact - math.floor(exact)
        remain = TARGET_TRAIN - sum(alloc.values())
        for d in sorted(frac, key=lambda x: frac[x], reverse=True):
            if remain <= 0: break
            if alloc[d] < len(by_dom[d]): alloc[d] += 1; remain -= 1
        chosen = []
        for d, lst in by_dom.items():
            lst.sort(key=quality, reverse=True)
            chosen.extend(lst[:alloc[d]])
        # koreksi presisi bila masih meleset (akibat clamp)
        if len(chosen) != TARGET_TRAIN:
            pool = [r for r in clean if r not in chosen]
            pool.sort(key=quality, reverse=True)
            if len(chosen) < TARGET_TRAIN:
                chosen.extend(pool[:TARGET_TRAIN - len(chosen)])
            else:
                chosen.sort(key=quality, reverse=True); chosen = chosen[:TARGET_TRAIN]
    write("train", chosen)
    P(f"train final: {len(chosen)} record")

    # ---- val/test: buang noise + dedup/near-dup internal + kebocoran (exact+near-dup) vs train (& val utk test)
    tr_a = {nf(asst(r)) for r in chosen}; tr_q = {nf(usr(r)) for r in chosen}
    tr_b = bucketize(chosen)
    val, vo = clean_eval(load("val"), tr_a, tr_q, tr_b)
    write("val", val)
    P(f"\nval: {len(val)} record ({dict(vo)})")
    va_a = tr_a | {nf(asst(r)) for r in val}; va_q = tr_q | {nf(usr(r)) for r in val}
    vt_b = collections.defaultdict(list, {k: list(v) for k, v in tr_b.items()})
    for r in val: vt_b[pkey(r)].append(asst(r))
    test, to = clean_eval(load("test"), va_a, va_q, vt_b)
    write("test", test)
    P(f"test: {len(test)} record ({dict(to)})")

    # distribusi domain before/after
    dom_before = collections.Counter(r.get("domain", "?") for r in clean)
    dom_after  = collections.Counter(r.get("domain", "?") for r in chosen)
    qs = [float(r.get("quality_score", 0)) for r in chosen]
    P("\nDistribusi domain (bersih -> final 20k):")
    for d, c in dom_before.most_common():
        P(f"  {d:14} {c:6} -> {dom_after[d]:6}  ({dom_after[d]/c*100:.0f}%)")
    P(f"\nquality_score final: min={min(qs):.3f} mean={sum(qs)/len(qs):.3f} max={max(qs):.3f}")

    open(os.path.join(DST, "REDUCTION_REPORT.txt"), "w", encoding="utf-8").write("\n".join(rep))
    print(f"\n[OK] -> {DST}/  (train/val/test + REDUCTION_REPORT.txt)")

if __name__ == "__main__":
    main()
