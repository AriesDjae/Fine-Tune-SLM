# -*- coding: utf-8 -*-
"""remediate_typos.py — Batalkan HANYA false-positive terkonfirmasi dari audit strict.

Sumber kebenaran FP: hasil audit_typo_corrections.py (cek manual per-konteks).
Koreksi benar lainnya (sinusitis, konjungtivitis, tidak/harus/tanpa, dll) DIPERTAHANKAN.

Bekerja posisional: untuk tiap pesan assistant, diff token asli (processed_id) vs hasil
(processed_id_clean); kalau sebuah koreksi 1:1 cocok daftar FP, token itu (dan hanya token
di posisi itu) dikembalikan ke kata yang benar. Menulis ulang processed_id_clean (asli
processed_id/ tetap utuh sebagai backup) + REMEDIATION_REPORT.txt.
"""
import json, re, os, sys, io, collections, difflib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "Data", "processed_id")        # asli (backup)
DST  = os.path.join(ROOT, "Data", "processed_id_clean")  # hasil cleaning (akan di-remediasi)
WORD = re.compile(r"[A-Za-z]+")
SPLITS = ["train", "val", "test"]

# (before, after_yang_salah) -> kata yang BENAR (perbaikan makna)
MANUAL_FIX = {
    ("garus",  "harus"): "garis",   # "manipulasi berlebihan garis yang muncul"
    ("tanpak", "tanpa"): "tampak",  # "penglihatan tampak buram"
    ("dann",   "daun"):  "dan",     # "leher, dan pangkal paha"
    ("haris",  "hari"):  "harus",   # "harus dicocokkan dengan kondisi"
    ("sempet", "sempit"):"sempat",  # "darah keluar sempat melalui pencernaan"
}
# (before, after_yang_salah) -> kembalikan ke token ASLI (before): istilah medis/Latin/
# English/merek obat/nama/kata-valid yang salah dikoreksi.
REVERT = {
    ("urogram", "program"),   # CT urogram (pencitraan ginjal)
    ("mamma",   "mama"),      # "Mamma kanan" (USG payudara, Latin)
    ("tine",    "tipe"),      # "tinea versicolor"
    ("optical", "optimal"),   # "optical coherence tomography (OCT)"
    ("acids",   "aids"),      # "alpha hydroxy acids (AHA)"
    ("rbbb",    "rbb"),       # RBBB = Right Bundle Branch Block
    ("lasal",   "asal"),      # Lasal = merek obat batuk
    ("marus",   "harus"),     # marus = makanan (dideh)
    ("ligma",   "lima"),      # "penyakit bernama ligma" (kata yg dibahas)
    ("selasa",  "selama"),    # selasa = hari (kata valid)
    ("herma",   "herba"),     # nama (sapaan)
    ("karin",   "kain"),      # nama
    ("anna",    "anda"),      # nama/username "Anna01"
    ("rika",    "jika"),      # nama/username "Rika_"
    ("yanti",   "anti"),      # bagian nama "Rahma_yanti20"
    ("ghilang", "hilang"),    # nama (sapaan)
}

def cap_like(src, target):
    return target.capitalize() if src[:1].isupper() else target

def assistant_idx(rec):
    return [i for i, m in enumerate(rec.get("messages", [])) if m.get("role") == "assistant"]

def remediate_text(co, cc):
    """co=teks asli, cc=teks clean. Kembalikan (cc_baru, list[(salah->benar)])."""
    to = WORD.findall(co); lo = [w.lower() for w in to]
    matches = list(WORD.finditer(cc)); lc = [m.group(0).lower() for m in matches]
    sm = difflib.SequenceMatcher(a=lo, b=lc, autojunk=False)
    fixes = {}   # clean_token_index -> kata pengganti (sudah cased)
    log = []
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "replace" and (a1 - a0) == (b1 - b0):
            for k in range(a1 - a0):
                wa, wb = lo[a0 + k], lc[b0 + k]
                key = (wa, wb)
                if key in MANUAL_FIX:
                    tgt = MANUAL_FIX[key]
                elif key in REVERT:
                    tgt = wa
                else:
                    continue
                orig_tok = matches[b0 + k].group(0)
                fixes[b0 + k] = cap_like(orig_tok, tgt)
                log.append((orig_tok, fixes[b0 + k]))
    if not fixes:
        return cc, log
    out, prev = [], 0
    for j, m in enumerate(matches):
        if j in fixes:
            out.append(cc[prev:m.start()]); out.append(fixes[j]); prev = m.end()
    out.append(cc[prev:])
    return "".join(out), log

def main():
    applied = collections.Counter()
    n_rec_changed = 0
    data = {}
    for sp in SPLITS:
        orig = [json.loads(l) for l in open(os.path.join(SRC, f"{sp}.jsonl"), encoding="utf-8") if l.strip()]
        clean = [json.loads(l) for l in open(os.path.join(DST, f"{sp}.jsonl"), encoding="utf-8") if l.strip()]
        assert len(orig) == len(clean)
        data[sp] = (orig, clean)

    for sp in SPLITS:
        orig, clean = data[sp]
        changed = 0
        for ro, rc in zip(orig, clean):
            oi, ci = assistant_idx(ro), assistant_idx(rc)
            rec_touched = False
            for io_, ic_ in zip(oi, ci):
                co = ro["messages"][io_]["content"]
                cc = rc["messages"][ic_]["content"]
                new, log = remediate_text(co, cc)
                if log:
                    rc["messages"][ic_]["content"] = new
                    for a, b in log:
                        applied[(a.lower(), b.lower())] += 1
                    rec_touched = True
            if rec_touched:
                changed += 1
        n_rec_changed += changed
        with open(os.path.join(DST, f"{sp}.jsonl"), "w", encoding="utf-8") as f:
            for rc in clean:
                f.write(json.dumps(rc, ensure_ascii=False) + "\n")
        print(f"{sp}: {changed} record diremediasi")

    total = sum(applied.values())
    lines = [f"REMEDIASI FALSE-POSITIVE — {len(applied)} tipe, {total} penggantian, "
             f"{n_rec_changed} record", "=" * 60]
    for (a, b), c in applied.most_common():
        kind = "FIX " if any(a == k[0] and k in MANUAL_FIX for k in MANUAL_FIX) else "RVRT"
        lines.append(f"  {c:>3}x  [{kind}] {a} -> {b}")
    open(os.path.join(DST, "REMEDIATION_REPORT.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[OK] {total} penggantian -> processed_id_clean (asli processed_id/ tetap utuh)")

if __name__ == "__main__":
    main()
