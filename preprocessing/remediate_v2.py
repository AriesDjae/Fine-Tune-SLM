"""
remediate_v2.py — REMEDIASI TERTARGET (TUGAS 3), TANPA rebuild pipeline.

Hanya dijalankan SETELAH peneliti ACC angka audit_recall_v2 (2026-06-17: ACC = remediasi).
Memperbaiki HANYA cacat ter-flag (tidak menyentuh record bersih), in-place per split
(record TIDAK pindah split -> cross-split dedup tetap 0/0 by construction). Backup dulu.

Aksi (jawaban = field assistant; pertanyaan hanya scrub nama demi privasi):
  1. OVER_STRIP konten   -> kembalikan kepala istilah medis dari 06_scored (mis. "Kista
                            Bartholin"), buang token nama/sapaan di depannya.
  2. DR_NAME_LEAK        -> normalkan "dokter/dr <Nama>" -> "dokter" (q & a).
  3. BARE_CITATION_TAIL  -> potong ekor ajakan-baca telanjang di ujung jawaban.
  4. TRUNCATION (a)      -> potong klausa menggantung ke batas kalimat terakhir;
                            kalau tak ada batas / jadi terlalu pendek -> DROP record.
Catatan: truncation/citation di PERTANYAAN TIDAK disentuh (gaya native Alodokter
informal — keputusan native-data). Typo sumber tetap dipertahankan.

Jalankan:  python preprocessing/remediate_v2.py            # tulis hasil + backup
           python preprocessing/remediate_v2.py --dry-run  # hitung saja, tak menulis

Output: Data/processed_id/{train,val,test}.jsonl (di-update; backup di
_intermediate/pre_remediation_<tanggal>/) + results/remediation_v2_<tanggal>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "Data" / "processed_id"
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).parent))
import audit_recall_v2 as AR  # detektor + alignment INDEPENDEN

MIN_ANSWER_CHARS = 25  # di bawah ini jawaban dianggap rusak -> drop
_SENT_END_RE = re.compile(r"[.!?](?=\s|$)")
_TERMINAL_RE = AR._TERMINAL_RE


def _qa_indices(rec):
    qi = ai = None
    for i, m in enumerate(rec.get("messages", [])):
        if m["role"] == "user":
            qi = i
        elif m["role"] == "assistant":
            ai = i
    return qi, ai


# ---------------------------------------------------------------------------
# Aksi perbaikan
# ---------------------------------------------------------------------------
def scrub_dr_name(text: str):
    """Normalkan nama dokter bocor -> 'dokter'. Iterasi sampai stabil agar nama
    beruntun ('dr. irna cecilia') ter-scrub semua. Return (text, n_fix)."""
    n = 0

    def _known(m):
        nonlocal n
        n += 1
        return "dokter"

    def _abbr(m):
        nonlocal n
        if m.group(2).lower() in AR._DR_ABBR_STOP:
            return m.group(0)
        n += 1
        return "dokter"

    t = text
    for _ in range(4):
        before = t
        t = AR._DR_KNOWN_RE.sub(_known, t)
        t = AR._DR_ABBR_RE.sub(_abbr, t)
        # "dokter dokter" hasil scrub beruntun -> satu
        t = re.sub(r"\bdokter\s+dokter\b", "dokter", t, flags=re.IGNORECASE)
        if t == before:
            break
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t, n


def restore_med_head(orig_a: str, final_a: str):
    """Untuk OVER_STRIP konten: ambil kepala dari original, buang token nama/sapaan
    di depan, sisakan dari istilah medis pertama. Return text baru atau None."""
    head, _ = AR.find_head_removed(orig_a, final_a)
    if head is None:
        return None
    toks = head.split()
    # cari token medis pertama; buang nama/sapaan sebelumnya
    keep_from = None
    for i, tk in enumerate(toks):
        base = re.sub(r"[^A-Za-z]", "", tk).lower()
        if base in AR._MED_HEAD_LEX:
            keep_from = i
            break
    if keep_from is None:
        return None  # tak ada istilah medis -> murni nama, biarkan (sudah benar di-strip)
    med_head = " ".join(toks[keep_from:]).strip(" .,:;-")
    if not med_head:
        return None
    return f"{med_head} {final_a}".strip()


def trim_citation_tail(text: str):
    """Potong ekor ajakan-baca telanjang di ujung. Return (text, trimmed_bool)."""
    changed = False
    for _ in range(3):  # bisa bertumpuk
        t = text.rstrip()
        cut = None
        for rx in (AR._BARE_TAIL_RE, AR._DANGLING_TAIL_RE, AR._INFO_LEAD_RE):
            m = rx.search(t)
            if m and m.start() > 0:
                cut = m.start() if cut is None else min(cut, m.start())
        if cut is None:
            break
        text = t[:cut].rstrip(" ,:;-")
        changed = True
    if changed and text and not _TERMINAL_RE.search(text):
        text = text + "."  # tutup kalimat yg terpotong dari ekor sitasi
    return text, changed


def trim_truncation(text: str):
    """Potong klausa menggantung ke batas kalimat terakhir. Return (text, status)
    status: 'kept' (terpotong rapi) | 'drop' (tak bisa diselamatkan) | 'noop'."""
    t = text.rstrip()
    if _TERMINAL_RE.search(t):
        return text, "noop"
    m = AR._LAST_WORD_RE.search(t)
    last = m.group(1).lower() if m else ""
    if last in AR._SIGNOFF:
        return text, "noop"  # sign-off tanpa titik = normal
    if last not in AR._DANGLING_END and last:
        return text, "noop"  # bukan menggantung di kata fungsi
    # cari batas kalimat terakhir
    ends = list(_SENT_END_RE.finditer(t))
    if ends:
        cut = ends[-1].end()
        trimmed = t[:cut].strip()
        if len(trimmed) >= MIN_ANSWER_CHARS:
            return trimmed, "kept"
    return text, "drop"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="hitung saja, tidak menulis")
    args = ap.parse_args()
    date = dt.datetime.now().strftime("%Y%m%d")

    # index 06_scored utk restore (hanya signature yg dibutuhkan)
    splits = {sp: [json.loads(l) for l in open(DATA_DIR / f"{sp}.jsonl", encoding="utf-8")]
              for sp in ("train", "val", "test")}
    needed = set()
    for rows in splits.values():
        for r in rows:
            qi, _ = _qa_indices(r)
            if qi is not None:
                needed.add(AR.qsig(r["messages"][qi]["content"]))
    print(f"[align] index pool utk {len(needed):,} signature ...", flush=True)
    idx, _, _ = AR.build_pool_index(needed)

    stats = Counter()
    out_splits = {}
    for sp, rows in splits.items():
        kept = []
        for r in rows:
            qi, ai = _qa_indices(r)
            if ai is None:
                kept.append(r)
                continue
            q = r["messages"][qi]["content"] if qi is not None else ""
            a = r["messages"][ai]["content"]
            drop = False

            # 1. OVER_STRIP konten -> restore kepala medis (align-confirmed, tak bergantung
            #    pada kata pembuka: head bisa hilang tanpa menyisakan konjungsi di depan)
            cands = idx.get(AR.qsig(q)) if q else None
            if cands:
                for orig in cands:
                    new_a = restore_med_head(orig, a)
                    if new_a:
                        a = new_a
                        stats["over_strip_restored"] += 1
                        break

            # 2. DR_NAME_LEAK scrub (q & a)
            if qi is not None:
                q2, nq = scrub_dr_name(q)
                if nq:
                    q = q2
                    stats["dr_name_scrub_q"] += 1
            a2, na = scrub_dr_name(a)
            if na:
                a = a2
                stats["dr_name_scrub_a"] += 1

            # 3. BARE_CITATION_TAIL trim (a)
            a3, cut = trim_citation_tail(a)
            if cut:
                a = a3
                stats["citation_tail_trimmed"] += 1

            # 4. TRUNCATION trim (a)
            a4, status = trim_truncation(a)
            if status == "kept":
                a = a4
                stats["truncation_trimmed"] += 1
            elif status == "drop":
                drop = True
                stats["dropped_truncation"] += 1

            # final sanity
            if not drop and len(a.strip()) < MIN_ANSWER_CHARS:
                drop = True
                stats["dropped_too_short"] += 1

            if drop:
                continue
            if qi is not None:
                r["messages"][qi]["content"] = q
            r["messages"][ai]["content"] = a
            kept.append(r)
        out_splits[sp] = kept
        stats[f"{sp}_in"] = len(rows)
        stats[f"{sp}_out"] = len(kept)

    # --- tulis ---
    report = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "actions": {k: v for k, v in stats.items() if not k.endswith(("_in", "_out"))},
        "counts": {sp: {"in": stats[f"{sp}_in"], "out": stats[f"{sp}_out"],
                        "dropped": stats[f"{sp}_in"] - stats[f"{sp}_out"]}
                   for sp in ("train", "val", "test")},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"remediation_v2_{date}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"REMEDIASI v2 {'(DRY-RUN)' if args.dry_run else ''}")
    print("=" * 70)
    for k, v in report["actions"].items():
        print(f"   {k:26} {v:6}")
    print("-" * 70)
    for sp, c in report["counts"].items():
        print(f"   {sp:6} {c['in']:6} -> {c['out']:6}  (drop {c['dropped']})")
    print("-" * 70)

    if not args.dry_run:
        backup = DATA_DIR / "_intermediate" / f"pre_remediation_{date}"
        backup.mkdir(parents=True, exist_ok=True)
        for sp in ("train", "val", "test"):
            shutil.copy2(DATA_DIR / f"{sp}.jsonl", backup / f"{sp}.jsonl")
            with open(DATA_DIR / f"{sp}.jsonl", "w", encoding="utf-8") as f:
                for r in out_splits[sp]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   Backup : {backup}")
        print(f"   Updated: Data/processed_id/{{train,val,test}}.jsonl")
    else:
        print("   (dry-run: tidak ada file ditulis)")
    print(f"   Report : {RESULTS / f'remediation_v2_{date}.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
