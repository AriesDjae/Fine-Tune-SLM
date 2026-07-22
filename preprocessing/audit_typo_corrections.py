# -*- coding: utf-8 -*-
"""audit_typo_corrections.py — AUDIT STRICT LOKAL (gratis, offline) dataset hasil cleaning.

Tujuan: sebelum keluar uang untuk training, pastikan koreksi typo (assistant-only) di
Data/processed_id_clean/ TIDAK mengubah makna medis. Membandingkan SETIAP baris
Data/processed_id/ (asli) vs Data/processed_id_clean/ (hasil cleaning), per-kata.

Tidak mengubah data apa pun. Hanya menulis laporan:
  Data/processed_id_clean/AUDIT_STRICT.txt   — ringkasan + daftar flag berkategori
  Data/processed_id_clean/audit_flagged.tsv  — machine-readable (before, after, kategori, n, contoh)

KBBI dibaca dari cache HF lokal (offline). Tidak perlu jaringan / huggingface_hub.
"""
import json, re, csv, os, sys, io, collections, difflib, glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
csv.field_size_limit(10**7)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "Data", "processed_id")
DST  = os.path.join(ROOT, "Data", "processed_id_clean")
WORD = re.compile(r"[A-Za-z]+")
SPLITS = ["train", "val", "test"]

# ---------------------------------------------------------------- KBBI (offline)
def load_kbbi():
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Lyon28--kamus-besar-bahasa-indonesia/"
        "snapshots/*/data.csv")
    hits = glob.glob(pat)
    if not hits:
        sys.exit("KBBI cache tidak ditemukan. Jalankan fix_typos_kbbi.py dulu (sekali, untuk cache).")
    valid, baku = set(), {}
    with open(hits[0], encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            for col in ("nama", "kata_dasar", "varian", "kata_turunan", "gabungan_"):
                for w in WORD.findall((row.get(col) or "").lower()):
                    if len(w) >= 2:
                        valid.add(w)
            nb = WORD.findall((row.get("bentuk_tidak_baku") or "").lower())
            nm = WORD.findall((row.get("nama") or "").lower())
            if len(nb) == 1 and len(nm) == 1:
                baku.setdefault(nb[0], nm[0])
    return valid, baku

# ---------------------------------------------------- morfologi (akar berimbuhan)
PRE = ["meng","meny","mem","men","peng","peny","pem","pen","ber","ter","per",
       "di","ke","se","me","be","pe"]
SUF = ["kannya","annya","nya","kan","lah","kah","an","i"]
def morph_valid(w, VALID):
    if w in VALID: return True
    for p in PRE:
        if w.startswith(p) and len(w)-len(p) >= 3:
            r = w[len(p):]
            if r in VALID: return True
            for s in SUF:
                if r.endswith(s) and len(r)-len(s) >= 3 and r[:-len(s)] in VALID: return True
    for s in SUF:
        if w.endswith(s) and len(w)-len(s) >= 3 and w[:-len(s)] in VALID: return True
    return False

# ---------------------------------------------------------- risiko istilah medis
# Sufiks/pola morfologi klinis (Indonesia + serapan Latin/Yunani). Kalau before-word
# cocok pola ini DAN bukan kata KBBI biasa -> kemungkinan istilah medis yg salah dikoreksi.
MED_SUFFIX = ("itis","osis","oma","emia","aemia","uria","algia","pati","patik",
              "ektomi","tomi","ostomi","plasti","gram","grafi","skopi","skopik",
              "sentesis","penia","sitosis","trofi","plasia","megali","ritma",
              "dema","edema","fagia","plegia","paresis","sklerosis","tropik")
MED_HINT = ("hordeolum","kalazion","blefaritis","nefrotik","sindrom","hipertensi",
            "diabetes","melitus","anemia","vertigo","migrain","gastritis","dispepsia",
            "tukak","ulkus","abses","sianosis","ikterus","edema","trombosis","emboli",
            "iskemia","infark","sepsis","meningitis","ensefalitis","epilepsi","stroke",
            "tumor","karsinoma","sarkoma","limfoma","leukemia","metastasis","biopsi",
            "endoskopi","kolonoskopi","rontgen","ultrasonografi","elektrokardiogram")
def looks_medical(w):
    lw = w.lower()
    if lw in MED_HINT: return True
    return any(lw.endswith(s) and len(lw) - len(s) >= 2 for s in MED_SUFFIX)

# akronim/singkatan klinis yg tak boleh disentuh (huruf, biasa ALL-CAPS di teks asli)
ACRONYM = {"dbd","usg","bab","bak","mri","ct","hb","tbc","ispa","hiv","aids","ekg",
           "igd","ugd","icu","nicu","ppok","gerd","ckd","dm","ht","asi","mpasi",
           "kb","iud","pcr","rt","pcr","spo","wbc","rbc","ph"}

# kata kritis yg perubahannya berbahaya secara klinis
CRITICAL = {"tidak","tanpa","bukan","jangan","kecuali","hindari","dilarang",
            "boleh","harus","wajib","segera","darurat"}

def lev(a, b):
    if a == b: return 0
    la, lb = len(a), len(b)
    if not la: return lb
    if not lb: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1,
                         prev[j-1] + (a[i-1] != b[j-1]))
        prev = cur
    return prev[lb]

# ------------------------------------------------------------------------- main
def sentence_of(text, word):
    """Potong kalimat asli yang memuat `word` (untuk konteks audit)."""
    for part in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\b" + re.escape(word) + r"\b", part, re.I):
            return part.strip()[:240]
    idx = text.lower().find(word.lower())
    return text[max(0, idx-80):idx+80].strip()

def assistant_msgs(rec):
    return [m["content"] for m in rec.get("messages", []) if m.get("role") == "assistant"]

def user_caps_set(records):
    """Kata yg pernah muncul Kapital di tengah kalimat user -> kandidat nama."""
    midcap = re.compile(r"(?<=[a-z,])\s([A-Z][a-z]+)")
    s = set()
    for rec in records:
        for m in rec.get("messages", []):
            if m.get("role") == "user":
                for mm in midcap.finditer(m.get("content", "")):
                    s.add(mm.group(1).lower())
    return s

def main():
    VALID, BAKU = load_kbbi()
    print(f"KBBI dimuat: {len(VALID):,} kata | baku-map {len(BAKU):,}")

    subs    = collections.Counter()          # (before,after) -> n
    sub_ctx = {}                             # (before,after) -> contoh kalimat
    deletions = []                           # (split, idx, teks dihapus)
    shrunk    = []                           # record yg assistant-nya menyusut drastis
    corpus_freq = collections.Counter()      # frekuensi token di assistant CLEAN
    n_rec = n_asst = 0
    all_user_records = []

    data = {}
    for sp in SPLITS:
        orig = [json.loads(l) for l in open(os.path.join(SRC, f"{sp}.jsonl"), encoding="utf-8") if l.strip()]
        clean = [json.loads(l) for l in open(os.path.join(DST, f"{sp}.jsonl"), encoding="utf-8") if l.strip()]
        assert len(orig) == len(clean), f"{sp}: jumlah record beda!"
        data[sp] = (orig, clean)
        all_user_records.extend(clean)
        for r in clean:
            for c in assistant_msgs(r):
                corpus_freq.update(w.lower() for w in WORD.findall(c))

    NAME_CAND = user_caps_set(all_user_records)

    for sp in SPLITS:
        orig, clean = data[sp]
        for i, (ro, rc) in enumerate(zip(orig, clean)):
            n_rec += 1
            ao, ac = assistant_msgs(ro), assistant_msgs(rc)
            for co, cc in zip(ao, ac):
                n_asst += 1
                # token (lowercase) untuk alignment; simpan original utk konteks/kapital
                to = WORD.findall(co); tc = WORD.findall(cc)
                lo = [w.lower() for w in to]; lc = [w.lower() for w in tc]
                sm = difflib.SequenceMatcher(a=lo, b=lc, autojunk=False)
                for tag, a0, a1, b0, b1 in sm.get_opcodes():
                    if tag == "equal":
                        continue
                    if tag == "replace" and (a1-a0) == (b1-b0):
                        # alignment 1:1 -> koreksi kata
                        for k in range(a1-a0):
                            wa, wb = lo[a0+k], lc[b0+k]
                            if wa == wb or not wa.isalpha() or not wb.isalpha():
                                continue
                            subs[(wa, wb)] += 1
                            if (wa, wb) not in sub_ctx:
                                sub_ctx[(wa, wb)] = sentence_of(co, wa)
                    elif tag == "delete":
                        seg = " ".join(to[a0:a1])
                        if len(seg) >= 12:   # abaikan hapusan sepele
                            deletions.append((sp, i, seg[:200]))
                # deteksi penyusutan konten drastis
                if len(cc) < 0.6 * len(co) and len(co) > 120:
                    shrunk.append((sp, i, len(co), len(cc), co[:120]))

    # --------------------------------------------------- klasifikasi tiap koreksi
    def classify(wa, wb, n):
        flags = []
        if looks_medical(wa):
            flags.append("MEDIS")                       # istilah klinis dikoreksi -> bahaya
        if wa in VALID or morph_valid(wa, VALID):
            flags.append("BUKAN_TYPO")                  # before kata valid -> salah koreksi
        if wa in ACRONYM or (len(wa) <= 4 and wa not in VALID):
            if wa in ACRONYM:
                flags.append("AKRONIM")
        if wa in CRITICAL or wb in CRITICAL:
            flags.append("KRITIS")                      # negasi/kuantitas klinis
        if wa in NAME_CAND:
            flags.append("NAMA?")
        d = lev(wa, wb)
        if d >= 3 or wa[0] != wb[0] or abs(len(wa)-len(wb)) >= 2:
            flags.append(f"JAUH(d={d})")
        if wb not in VALID:
            flags.append("TARGET_TDK_BAKU")             # target koreksi bukan kata KBBI
        if corpus_freq[wa] >= 8:
            flags.append(f"SERING({corpus_freq[wa]})")  # before sering muncul -> mungkin valid
        return flags

    rows = []
    for (wa, wb), n in subs.items():
        fl = classify(wa, wb, n)
        if fl:
            rows.append((wa, wb, n, fl, sub_ctx[(wa, wb)]))

    # prioritas: MEDIS > BUKAN_TYPO > KRITIS > NAMA? > TARGET_TDK_BAKU > sisanya
    PRIO = {"MEDIS":0,"BUKAN_TYPO":1,"KRITIS":2,"AKRONIM":2,"NAMA?":3,"TARGET_TDK_BAKU":4}
    def sortkey(r):
        base = min((PRIO.get(f.split("(")[0], 9) for f in r[3]), default=9)
        return (base, -r[2])
    rows.sort(key=sortkey)

    total_sub_types = len(subs)
    total_sub_apply = sum(subs.values())
    flagged_types   = len(rows)
    flagged_apply   = sum(r[2] for r in rows)

    cat = collections.Counter()
    for r in rows:
        for f in r[3]:
            cat[f.split("(")[0]] += 1

    # --------------------------------------------------------------- tulis report
    out = io.StringIO()
    P = lambda *a: print(*a, file=out)
    P("AUDIT STRICT — Data/processed_id  vs  Data/processed_id_clean")
    P("=" * 64)
    P(f"records diaudit         : {n_rec:,}  (assistant msgs: {n_asst:,})")
    P(f"tipe koreksi kata       : {total_sub_types:,}  ({total_sub_apply:,} penerapan)")
    P(f"tipe DIFLAG (risiko)    : {flagged_types:,}  ({flagged_apply:,} penerapan)")
    P(f"  -> aman (tak diflag)  : {total_sub_types - flagged_types:,} tipe")
    P(f"konten dihapus (stage1) : {len(deletions):,} segmen >=12 char")
    P(f"assistant menyusut >40% : {len(shrunk):,} record")
    P("")
    P("Distribusi kategori flag (per tipe koreksi):")
    for k, v in cat.most_common():
        P(f"  {v:>5}  {k}")
    P("")
    P("LEGEND: MEDIS=istilah klinis dikoreksi | BUKAN_TYPO=before kata valid KBBI |")
    P("        KRITIS=negasi/kuantitas | AKRONIM=singkatan | NAMA?=kandidat nama |")
    P("        TARGET_TDK_BAKU=hasil koreksi bukan kata KBBI | JAUH=edit-distance besar |")
    P("        SERING=before sering muncul di korpus (mungkin sebenarnya valid)")
    P("=" * 64)

    def dump(title, predicate):
        sel = [r for r in rows if predicate(r)]
        P(f"\n### {title}  ({len(sel)} tipe)")
        for wa, wb, n, fl, ctx in sel[:400]:
            P(f"  [{n:>3}x] {wa} -> {wb}   |{' '.join(fl)}|")
            P(f"         konteks: …{ctx}…")

    dump("PRIORITAS 1 — koreksi istilah MEDIS (cek manual!)", lambda r: "MEDIS" in r[3])
    dump("PRIORITAS 2 — before BUKAN typo (kata valid KBBI)", lambda r: "BUKAN_TYPO" in r[3] and "MEDIS" not in r[3])
    dump("PRIORITAS 3 — kata KRITIS / AKRONIM / NAMA?",
         lambda r: any(f in r[3] for f in ("KRITIS","AKRONIM","NAMA?")) and not ({"MEDIS","BUKAN_TYPO"} & set(r[3])))
    dump("PRIORITAS 4 — target hasil koreksi bukan kata baku KBBI",
         lambda r: "TARGET_TDK_BAKU" in r[3] and not any(f in r[3] for f in ("MEDIS","BUKAN_TYPO","KRITIS","AKRONIM","NAMA?")))

    P(f"\n### KONTEN DIHAPUS (sampel 60 dari {len(deletions)})")
    for sp, i, seg in deletions[:60]:
        P(f"  [{sp}#{i}] …{seg}…")

    P(f"\n### ASSISTANT MENYUSUT >40% (sampel 40 dari {len(shrunk)})")
    for sp, i, lo_, lc_, head in shrunk[:40]:
        P(f"  [{sp}#{i}] {lo_}->{lc_} char | {head}…")

    report = out.getvalue()
    with open(os.path.join(DST, "AUDIT_STRICT.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    with open(os.path.join(DST, "audit_flagged.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["before", "after", "n", "flags", "context"])
        for wa, wb, n, fl, ctx in rows:
            w.writerow([wa, wb, n, ",".join(fl), ctx])

    # ringkas ke stdout
    print(report[:4000])
    print(f"\n[OK] Report -> {os.path.join(DST,'AUDIT_STRICT.txt')}")
    print(f"[OK] TSV    -> {os.path.join(DST,'audit_flagged.tsv')}")

if __name__ == "__main__":
    main()
