# -*- coding: utf-8 -*-
"""fix_typos_kbbi.py (v3) — KBBI (HF Lyon28) + frekuensi korpus + stemmer afiks + proteksi nama.
Koreksi typo jawaban assistant. GUARD: target di KBBI, len>=4, |Δlen|<=2, freq target>=30,
bukan kata-berinfleksi-sah (akar di KBBI), bukan nama (pernah kapital di tengah kalimat).
IN/OUT: Data/processed_id_clean/*.jsonl (asli Data/processed_id/ aman)."""
import json, re, csv, os, collections, sys, io
from huggingface_hub import hf_hub_download
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"); csv.field_size_limit(10**7)
DST="Data/processed_id_clean"; WORD=re.compile(r"[A-Za-z]+")
def toks(t): return WORD.findall(t or "")

path=hf_hub_download('Lyon28/kamus-besar-bahasa-indonesia','data.csv',repo_type='dataset')
VALID=set(); OFFICIAL={}
with open(path,encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f):
        for col in ("nama","kata_dasar","varian","kata_turunan","gabungan_"):
            for w in toks((row.get(col) or "").lower()):
                if len(w)>=2: VALID.add(w)
        nb=toks((row.get("bentuk_tidak_baku") or "").lower()); nm=toks((row.get("nama") or "").lower())
        if len(nb)==1 and len(nm)==1:
            a,b=nb[0],nm[0]
            if len(b)>=4 and a[0]==b[0] and abs(len(a)-len(b))<=2 and a!=b: OFFICIAL.setdefault(a,b)

PRE=["meng","meny","mem","men","peng","peny","pem","pen","ber","ter","per","di","ke","se","me","be","pe"]
SUF=["kannya","annya","nya","kan","lah","kah","an","i"]
def morph_valid(w):
    if w in VALID: return True
    for p in PRE:
        if w.startswith(p) and len(w)-len(p)>=3:
            r=w[len(p):]
            if r in VALID: return True
            for s in SUF:
                if r.endswith(s) and len(r)-len(s)>=3 and r[:-len(s)] in VALID: return True
    for s in SUF:
        if w.endswith(s) and len(w)-len(s)>=3 and w[:-len(s)] in VALID: return True
    return False

freq=collections.Counter(); midcap=collections.Counter(); rows_by={}
MIDCAP=re.compile(r"(?<=[a-z,])\s([A-Z][a-z]+)")
for sp in ["train","val","test"]:
    rows=[json.loads(l) for l in open(os.path.join(DST,f"{sp}.jsonl"),encoding="utf-8") if l.strip()]
    rows_by[sp]=rows
    for r in rows:
        for m in r["messages"]:
            if m["role"]=="assistant":
                c=m["content"]; freq.update(w.lower() for w in toks(c))
                for mm in MIDCAP.finditer(c): midcap[mm.group(1).lower()]+=1
CORPUS_OK=5; RARE=2
def is_valid(w): return w in VALID or freq[w]>=CORPUS_OK or morph_valid(w)
AB="abcdefghijklmnopqrstuvwxyz"
def edits1(w):
    s=[(w[:i],w[i:]) for i in range(len(w)+1)]; o=set()
    o.update(L+R[1:] for L,R in s if R); o.update(L+R[1]+R[0]+R[2:] for L,R in s if len(R)>1)
    o.update(L+c+R[1:] for L,R in s if R for c in AB); o.update(L+c+R for L,R in s for c in AB)
    return o
def ok(t,f): return f in VALID and len(f)>=4 and abs(len(t)-len(f))<=2 and t!=f
NEVER={"salma","nova","tante","kakak","adik","bapak","budi","intan","ratna","sari","dewi","putri","rama"}
MANUAL={"telfon":"telepon"}
corr={}
for w,c in freq.items():
    if c>RARE or len(w)<4 or len(w)>18 or is_valid(w): continue
    if midcap[w]>=1 or w in NEVER: continue           # proteksi nama
    if w in MANUAL: corr[w]=MANUAL[w]; continue
    if w in OFFICIAL and ok(w,OFFICIAL[w]): corr[w]=OFFICIAL[w]; continue
    best=None; bf=-1
    for cand in edits1(w):
        if cand in VALID and freq[cand]>=30 and ok(w,cand) and freq[cand]>bf: best=cand; bf=freq[cand]
    if best: corr[w]=best
for a,b in OFFICIAL.items():
    if a in freq and a not in corr and not is_valid(a) and midcap[a]==0 and ok(a,b): corr[a]=b
corr={k:v for k,v in corr.items() if ok(k,v)}
print(f"KBBI {len(VALID):,} kata | OFFICIAL {len(OFFICIAL)} | koreksi final {len(corr):,} tipe")

applied=collections.Counter()
def repl(m):
    w=m.group(0); low=w.lower()
    if low in corr: fx=corr[low]; applied[(low,fx)]+=1; return fx.capitalize() if w[0].isupper() else fx
    return w
for sp,rows in rows_by.items():
    with open(os.path.join(DST,f"{sp}.jsonl"),"w",encoding="utf-8") as fo:
        for r in rows:
            for m in r["messages"]:
                if m["role"]=="assistant": m["content"]=WORD.sub(repl,m["content"])
            fo.write(json.dumps(r,ensure_ascii=False)+"\n")
total=sum(applied.values())
open(os.path.join(DST,"TYPO_CORRECTIONS.txt"),"w",encoding="utf-8").write(
    f"KBBI v3 — {len(applied):,} tipe, {total:,} penggantian\n"+"="*50+"\n"+
    "\n".join(f"{c:>4}x  {a} -> {b}" for (a,b),c in applied.most_common()))
print(f"Diterapkan {total:,} penggantian; cek FP yg tadi:")
for w in ["salma","nova","kadek","ditegaskan","telfon"]:
    print(f"   {w}: ", {b for (a,b) in applied if a==w} or "TIDAK dikoreksi (aman)")
print("\nContoh 30:")
for (a,b),c in applied.most_common(30): print(f"  {c:>3}x  {a} -> {b}")
