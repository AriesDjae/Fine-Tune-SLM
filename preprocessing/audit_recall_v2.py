"""
audit_recall_v2.py — AUDIT RECALL (read-only) untuk dataset ID v2 (STRICT CLEAN v2).

KONTEKS (brief 2026-06-17): verifier lama (verify_dataset.py) hanya mengukur PRECISION
("tidak ada token jelek tersisa") dan, lebih buruk, MEMAKAI POLA YANG SAMA dengan cleaner
(import strict_clean_v2) sehingga verdict "0% leak" menyesatkan. Skrip ini mengukur RECALL:
apakah ada token BAGUS yang ikut terbuang (over-strip), plus beberapa coverage-gap leak yang
verifier lama lewatkan. SEMUA POLA DI SINI INDEPENDEN dari strict_clean_v2 (sengaja).

JANGAN re-run pipeline 12-gerbang. Skrip ini HANYA membaca file final + pool orisinal,
men-tag tiap record, dan melaporkan ANGKA POPULASI per kategori. TIDAK ada auto-drop/auto-fix.

Empat detektor (tiap record dicek q DAN a TERPISAH, satu record bisa kena >1 kategori):
  1. OVER_STRIP          — head jawaban/ pertanyaan kemungkinan ke-strip (recall; baru)
  2. BARE_CITATION_TAIL  — ekor ajakan baca telanjang tanpa konten (coverage gap B3/B4)
  3. DR_NAME_LEAK        — nama dokter tersisa, case-insensitive (coverage gap)
  4. TRUNCATION          — teks terpotong di ujung (truncate@1024)

Alignment ke pool orisinal (06_scored.jsonl: field q/a/title, TANPA id) memakai
SIGNATURE pertanyaan (ekor 80 char alnum-lower — bagian yang tidak disentuh leading-strip).

Output:
  - results/audit_recall_v2_<tanggal>.json     (ringkasan angka, machine-readable)
  - results/audit_recall_v2_flagged.jsonl      (semua record ter-flag + span/alasan)
  - tabel ringkasan ke stdout

Jalankan:
  python preprocessing/audit_recall_v2.py                 # full
  python preprocessing/audit_recall_v2.py --no-align      # skip alignment (cepat, tanpa DETECTOR1-align)
  python preprocessing/audit_recall_v2.py --debug 500     # subset N record/split
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "Data" / "processed_id"
POOL_FILE = DATA_DIR / "_intermediate" / "06_scored.jsonl"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MAX_SEQ_LENGTH = 1024  # selaras strict_clean v2 — untuk cross-check truncation

# Nama bocor terkonfirmasi (kurasi). Sengaja didefinisikan ULANG di sini (independen);
# dipakai HANYA untuk MENGECUALIKAN head_removed yang memang nama dari OVER_STRIP.
NAMESET = {"ainul"}

# ---------------------------------------------------------------------------
# DETECTOR 1 — OVER_STRIP (recall)
# ---------------------------------------------------------------------------
# Penghubung pembuka. DIBAGI per presisi (audit jujur):
#  HIGH = pola apositif/definisi yang HAMPIR TAK PERNAH mengawali kalimat utuh
#         ("X adalah/ialah/merupakan Y", "X atau Y") -> subjek/kepala ke-strip.
#  LOW  = penghubung yang LAZIM mengawali kalimat ID (untuk/pada/dari/dengan/...)
#         -> banyak false-positive, dilaporkan TERPISAH sbg kandidat lemah.
_CONNECTORS_HIGH = {"adalah", "merupakan", "ialah", "atau", "sehingga"}
_CONNECTORS_LOW = {
    "dan", "yang", "dengan", "untuk", "pada", "dari", "namun", "tetapi",
    "serta", "maka", "karena",
}
_LEAD_PUNCT_RE = re.compile(r'^[\s"\'`.,;:!?·•\-–—()\[\]]+')
_FIRST_WORD_RE = re.compile(r"^([A-Za-z]+)")
# kalimat pertama: <preposisi> ... <kopula> di awal -> subjek nominal kemungkinan hilang
_PREP_COPULA_RE = re.compile(
    r"^(pada|di|dari|untuk|dengan)\b[^.!?]{1,80}?\b(merupakan|adalah|ialah)\b",
    re.IGNORECASE,
)


def _strip_lead_punct(t: str) -> str:
    return _LEAD_PUNCT_RE.sub("", t or "")


def detect_over_strip_textonly(text: str):
    """Sinyal OVER_STRIP tanpa alignment. Return (conf, reasons).
    conf in {None, 'low', 'high'} -- 'high' presisi tinggi, 'low' kandidat lemah."""
    reasons = []
    t = _strip_lead_punct(text)
    if not t:
        return None, reasons
    m = _FIRST_WORD_RE.match(t)
    first = m.group(1).lower() if m else ""
    conf = None
    if first in _CONNECTORS_HIGH:
        reasons.append(f"lead_connector_high:{first}")
        conf = "high"
    elif first in _CONNECTORS_LOW:
        reasons.append(f"lead_connector_low:{first}")
        conf = "low"
    # preposisi + kopula di awal -> subjek nominal kemungkinan hilang (naikkan ke high)
    if _PREP_COPULA_RE.match(t):
        reasons.append("prep_copula_lead")
        conf = "high"
    # kalimat diawali huruf kecil (jawaban normal diawali kapital) -> minimal low
    if t[0].islower():
        reasons.append("lowercase_start")
        conf = conf or "low"
    return conf, reasons


# ---------------------------------------------------------------------------
# DETECTOR 2 — BARE_CITATION_TAIL (coverage gap B3/B4)
# ---------------------------------------------------------------------------
_BARE_TAIL_RE = re.compile(
    r"\b(silakan|silahkan|baca|lihat|klik|kunjungi|simak|ikuti)\b"
    r"[^.!?]{0,40}?"
    r"(di\s?sini|disini|berikut|tautan|link|diskusi|selengkapnya)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
# kata kerja ajakan menggantung tepat di ujung ("...baca", "...berikut", "...silakan Anda")
_DANGLING_TAIL_RE = re.compile(
    r"\b(silakan|silahkan|baca|lihat|klik|kunjungi|simak|ikuti|berikut|tautan|link)"
    r"(\s+(anda|kamu|di\s?sini|disini))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
# pengantar sitasi menggantung tanpa konten: "...Informasi lebih lanjut/terkait ... pada/berikut$"
_INFO_LEAD_RE = re.compile(
    r"\b(informasi\s+(lebih\s+lanjut|terkait)|lebih\s+rincinya|info\s+(lebih\s+lanjut|terkait))\b"
    r"[^.!?]{0,40}?(pada|berikut|di\s?sini|disini)?\s*[.!:]?\s*$",
    re.IGNORECASE,
)


def detect_bare_citation_tail(text: str):
    t = (text or "").rstrip()
    if not t:
        return False, None
    # ambil ~80 char terakhir sebagai konteks ekor
    tail = t[-90:]
    m = _BARE_TAIL_RE.search(tail) or _DANGLING_TAIL_RE.search(tail) or _INFO_LEAD_RE.search(tail)
    if m:
        return True, m.group(0).strip()
    return False, None


# ---------------------------------------------------------------------------
# DETECTOR 3 — DR_NAME_LEAK (coverage gap, case-insensitive)
# ---------------------------------------------------------------------------
# Nama dokter bocor yang TERKONFIRMASI dari korpus (Alodokter sign-off berulang).
# "dokter <kata>" sangat ambigu (kata = istilah umum), maka kita pakai 2 jalur PRESISI:
#   (1) abbreviation "dr."/"drg." + kata  -> hampir selalu nama (kecuali stoplist spesialisasi)
#   (2) "dokter"/"dr"/"drg" + nama dalam KNOWN_DR_NAMES (kurasi, case-insensitive)
KNOWN_DR_NAMES = {
    "william", "irna", "cecilia", "caecilia", "danny", "ulfi", "yosephine",
    "nofrina", "jati", "kresnawati",
}
_DR_ABBR_RE = re.compile(r"\b(dr|drg)\.\s*([A-Za-z][a-z]+)\b", re.IGNORECASE)
_DR_KNOWN_RE = re.compile(
    r"\b(dr\.?|drg\.?|dokter)\s+(" + "|".join(sorted(KNOWN_DR_NAMES)) + r")\b",
    re.IGNORECASE,
)
# kata setelah "dr."/"drg." yang BUKAN nama (spesialisasi / fungsi) -> jangan flag
_DR_ABBR_STOP = {
    "spesialis", "bedah", "dok", "paru", "anak", "spog", "saya", "umum", "gigi",
    "nya", "spkk", "sp", "obgyn", "internis", "kandungan", "kulit", "mata", "jiwa",
    "saraf", "jantung", "gizi", "kelamin", "dalam", "tht", "ahli", "muda", "hewan",
}


def detect_dr_name_leak(text: str):
    t = text or ""
    m = _DR_KNOWN_RE.search(t)
    if m:
        return True, m.group(0).strip()
    for m in _DR_ABBR_RE.finditer(t):
        name = m.group(2).lower()
        if name in _DR_ABBR_STOP:
            continue
        return True, m.group(0).strip()
    return False, None


# ---------------------------------------------------------------------------
# DETECTOR 4 — TRUNCATION
# ---------------------------------------------------------------------------
_TERMINAL_RE = re.compile(r'[.!?…]["\'\)\]]?\s*$')
_LAST_WORD_RE = re.compile(r"([A-Za-z]+)[\"'\)\]]?\s*$")
# Sign-off khas Alodokter — berakhir tanpa titik itu NORMAL, BUKAN truncation.
_SIGNOFF = {
    "bermanfaat", "membantu", "terimakasih", "terima", "kasih", "sehat", "sembuh",
    "semoga", "salam", "wassalam", "demikian", "sekian", "ya", "yaa", "banyak",
    "selalu", "menjawab", "jawaban", "berguna", "jelas", "berkenan", "membantumu",
}
# Kata fungsi/penghubung di ujung -> kalimat MENGGANTUNG (sinyal kuat truncation/cut).
_DANGLING_END = {
    "pada", "dan", "atau", "untuk", "dengan", "yang", "ke", "di", "dari", "dalam",
    "serta", "namun", "maka", "secara", "agar", "supaya", "bila", "jika", "akan",
    "dapat", "adalah", "merupakan", "juga", "lebih", "karena", "sehingga", "tetapi",
    "seperti", "yaitu", "antara", "bahwa", "tentang", "hingga", "sambil", "saat",
}


def approx_token_len(text: str) -> int:
    # heuristik ringan (tanpa load tokenizer): ~1.3 token / kata utk teks ID
    return int(len(text.split()) * 1.3)


def detect_truncation(text: str):
    """Truncation/cut presisi: TIDAK diakhiri tanda terminal, BUKAN sign-off,
    dan berakhir dgn kata fungsi menggantung (kalimat terpotong). Sign-off tanpa
    titik (khas Alodokter) sengaja TIDAK di-flag (itu normal, bukan truncation)."""
    t = (text or "").rstrip()
    if not t:
        return False, None
    ends_with = t[-30:]
    if _TERMINAL_RE.search(t):
        return False, ends_with
    m = _LAST_WORD_RE.search(t)
    last = m.group(1).lower() if m else ""
    if last in _SIGNOFF:
        return False, ends_with  # sign-off tanpa titik -> normal
    if last in _DANGLING_END:
        return True, ends_with    # menggantung di kata fungsi -> terpotong
    # ujung tanpa kata huruf sama sekali (mis. koma/kutip) juga mencurigakan
    if not last:
        return True, ends_with
    return False, ends_with


# ---------------------------------------------------------------------------
# Alignment ke pool orisinal (untuk OVER_STRIP strong-align)
# ---------------------------------------------------------------------------
_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_ALNUM_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyz")


def norm_alnum(s: str) -> str:
    return _ALNUM_RE.sub("", (s or "").lower())


def qsig(q: str) -> str:
    """Signature pertanyaan = ekor 80 char alnum-lower (bagian stabil thd leading-strip)."""
    n = norm_alnum(q)
    if len(n) < 40:
        return n  # terlalu pendek -> pakai utuh (mungkin tak unik, ditandai)
    return n[-80:]


def build_pool_index(needed_keys: set):
    """Pass tunggal streaming pool; simpan original answer utk key yang dibutuhkan saja."""
    idx = defaultdict(list)  # key -> list of original answers
    matched = 0
    total = 0
    with open(POOL_FILE, encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = r.get("q") or r.get("title") or ""
            k = qsig(q)
            if k in needed_keys:
                idx[k].append(r.get("a") or "")
                matched += 1
    return idx, total, matched


# Klasifikasi head yang terbuang (align) -> bedakan strip SAH (nama/sapaan) vs over-strip KONTEN.
_GREET_HEAD = {
    "halo", "hai", "alo", "hallo", "hello", "hi", "hey", "pagi", "siang", "sore",
    "malam", "selamat", "salam", "assalamualaikum", "assalamualaykum", "assalam",
    "waalaikumsalam", "wassalam", "waalaikumsalamwarahmatullah", "terimakasih",
    "terima", "kasih", "permisi", "maaf", "dok", "dokter", "yth",
}


_FORUM_HDR_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\b"
    r"|\b\d{1,2}:\d{2}\b|\b\d{4}\b|\b(am|pm)\b", re.IGNORECASE)


def classify_head(head: str) -> str:
    """Kembalikan 'forum'|'greeting'|'name'|'content'. Hanya 'content' = over-strip KONTEN nyata."""
    toks = re.findall(r"[A-Za-z]+", head)
    if not toks:
        return "forum"
    low = [t.lower() for t in toks]
    # header forum (username + tanggal/jam) -> pembersihan SAH
    if _FORUM_HDR_RE.search(head):
        return "forum"
    if any(t in _GREET_HEAD for t in low):
        return "greeting"
    # mayoritas token Kapital (proper noun) & tak ada istilah medis -> nama (strip B1 disengaja)
    cap = sum(1 for t in toks if t[0].isupper())
    has_med = any(t in _MED_HEAD_LEX for t in low)
    if cap >= max(1, len(toks) - 1) and not has_med:
        return "name"
    return "content"


# leksikon kepala medis/istilah (dipakai utk TIDAK menandai nama bila sebetulnya istilah medis)
_MED_HEAD_LEX = {
    "kista", "hernia", "lupus", "cacar", "bisul", "tumor", "kanker", "polip", "miom",
    "vertigo", "asma", "tbc", "hiv", "tipes", "demam", "diabetes", "hipertensi",
    "anemia", "migrain", "sinusitis", "gastritis", "vitiligo", "psoriasis", "eksim",
}


def find_head_removed(orig_a: str, final_a: str):
    """Jika norm(final_a) adalah SUFFIX dari norm(orig_a) & != -> head terpotong.
    Return (head_removed_text, delta_char) atau (None, 0)."""
    fo, ff = norm_alnum(orig_a), norm_alnum(final_a)
    if not ff or not fo or ff == fo:
        return None, 0
    if not fo.endswith(ff):
        return None, 0
    cut_norm = len(fo) - len(ff)  # banyak char alnum yg dibuang dari head
    if cut_norm <= 0:
        return None, 0
    # map cut_norm (index di ruang alnum) -> index char di orig_a
    seen = 0
    cut_char = len(orig_a)
    for i, ch in enumerate(orig_a):
        if ch.lower() in _ALNUM_CHARS:  # ch adalah alnum (case-insensitive)
            seen += 1
            if seen >= cut_norm:
                cut_char = i + 1
                break
    head = orig_a[:cut_char].strip()
    return head, len(orig_a) - len(final_a)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def load_split(name: str, limit=None):
    rows = []
    with open(DATA_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def get_qa(rec):
    msgs = rec.get("messages", [])
    q = a = ""
    for m in msgs:
        if m["role"] == "user":
            q = m["content"]
        elif m["role"] == "assistant":
            a = m["content"]
    return q, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-align", action="store_true", help="skip alignment ke pool")
    ap.add_argument("--debug", type=int, default=0, help="batasi N record/split")
    args = ap.parse_args()
    limit = args.debug or None

    date = dt.datetime.now().strftime("%Y%m%d")
    splits = {sp: load_split(sp, limit) for sp in ("train", "val", "test")}

    # --- kumpulkan key yg dibutuhkan utk alignment ---
    align_idx = {}
    pool_total = pool_matched = 0
    if not args.no_align:
        needed = set()
        for rows in splits.values():
            for r in rows:
                q, _ = get_qa(r)
                needed.add(qsig(q))
        print(f"[align] membangun index pool utk {len(needed):,} signature unik ...", flush=True)
        align_idx, pool_total, pool_matched = build_pool_index(needed)
        print(f"[align] pool dibaca {pool_total:,} baris; {pool_matched:,} cocok ke signature.", flush=True)

    flagged = []  # records ter-flag (utk jsonl)
    summary = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "max_seq_length": MAX_SEQ_LENGTH,
        "align": (not args.no_align),
        "pool_lines": pool_total,
        "pool_signature_matched": pool_matched,
        "splits": {},
        "delta_char": {},
        "top_head_removed_content": {},
    }

    CATS = ["OVER_STRIP_HIGH", "OVER_STRIP_LOW", "BARE_CITATION_TAIL", "DR_NAME_LEAK", "TRUNCATION"]
    deltas = []
    head_counter = Counter()
    align_head_types = Counter()
    align_hit = align_attempt = align_confirmed = src_identical_high = 0

    for sp, rows in splits.items():
        n = len(rows)
        # count[cat][field] -> int ; field in {q,a,any}
        cnt = {c: Counter() for c in CATS}
        for r in rows:
            q, a = get_qa(r)
            rec_cats = {}

            # DETECTOR 1: OVER_STRIP pada q & a (text-only tiered + align-confirmed)
            for field, text in (("q", q), ("a", a)):
                conf, reasons = detect_over_strip_textonly(text)
                head_removed = None
                delta_char = 0
                head_type = None
                src_identical = False
                # align (hanya untuk jawaban) -> bedakan: sumber identik (native),
                # head konten terbuang (over-strip NYATA), atau strip nama/sapaan (SAH)
                if field == "a" and not args.no_align:
                    cands = align_idx.get(qsig(q))
                    if cands:
                        align_attempt += 1
                        na = norm_alnum(a)
                        for orig_a in cands:
                            if norm_alnum(orig_a) == na:
                                src_identical = True
                            if head_removed is None:
                                hr, dc = find_head_removed(orig_a, a)
                                if hr is not None and norm_alnum(hr):
                                    head_removed, delta_char = hr, dc
                                    head_type = classify_head(hr)
                        if head_removed is not None:
                            align_hit += 1
                            align_head_types[head_type] += 1
                            if head_type == "content":
                                reasons = reasons + ["align_suffix_content"]
                                conf = "high"          # over-strip konten terkonfirmasi
                                align_confirmed += 1
                            else:
                                # strip nama/sapaan = pembersihan SAH, jangan di-flag over-strip
                                reasons = reasons + [f"align_{head_type}_legit"]
                        if src_identical and conf == "high":
                            reasons = reasons + ["source_identical"]
                            src_identical_high += 1
                if conf == "high":
                    cnt["OVER_STRIP_HIGH"][field] += 1
                elif conf == "low":
                    cnt["OVER_STRIP_LOW"][field] += 1
                if conf:
                    rec_cats.setdefault("OVER_STRIP", []).append({
                        "field": field, "confidence": conf, "reasons": reasons,
                        "head_removed": head_removed, "head_type": head_type,
                        "delta_char": delta_char, "source_identical": src_identical,
                    })
                    if head_removed and head_type == "content":
                        deltas.append(delta_char)
                        first_tok = head_removed.split()[0] if head_removed.split() else head_removed
                        head_counter[first_tok.strip(".,;:!?-")[:30]] += 1

            # DETECTOR 2-4 pada q & a
            for field, text in (("q", q), ("a", a)):
                bt, span = detect_bare_citation_tail(text)
                if bt:
                    cnt["BARE_CITATION_TAIL"][field] += 1
                    rec_cats.setdefault("BARE_CITATION_TAIL", []).append({"field": field, "span": span})
                dr, span = detect_dr_name_leak(text)
                if dr:
                    cnt["DR_NAME_LEAK"][field] += 1
                    rec_cats.setdefault("DR_NAME_LEAK", []).append({"field": field, "span": span})
                tr, ew = detect_truncation(text)
                if tr:
                    cnt["TRUNCATION"][field] += 1
                    rec_cats.setdefault("TRUNCATION", []).append({"field": field, "ends_with": ew})

            if rec_cats:
                flagged.append({
                    "split": sp,
                    "source": r.get("source"),
                    "domain": r.get("domain"),
                    "categories": rec_cats,
                    "q": q[:300],
                    "a_head": a[:200],
                    "a_tail": a[-120:],
                })

        # ringkas per split
        sp_sum = {"n": n}
        for c in CATS:
            q_n, a_n = cnt[c]["q"], cnt[c]["a"]
            any_n = q_n + a_n  # catatan: q & a dihitung terpisah (bisa double utk 1 record)
            sp_sum[c] = {
                "q": q_n, "a": a_n,
                "q_rate_pct": round(100 * q_n / n, 3) if n else 0,
                "a_rate_pct": round(100 * a_n / n, 3) if n else 0,
                "total_fields": any_n,
            }
        summary["splits"][sp] = sp_sum

    # distribusi delta_char OVER_STRIP (align)
    if deltas:
        ds = sorted(deltas)
        summary["delta_char"] = {
            "n": len(ds), "min": ds[0], "median": int(statistics.median(ds)),
            "p95": ds[int(0.95 * (len(ds) - 1))], "max": ds[-1],
            "mean": round(statistics.mean(ds), 1),
        }
    summary["align_attempt"] = align_attempt
    summary["align_hit_total"] = align_hit
    summary["align_head_types"] = dict(align_head_types)  # greeting/name/content
    summary["over_strip_content_confirmed"] = align_confirmed  # over-strip KONTEN nyata (align)
    summary["over_strip_high_source_identical"] = src_identical_high  # HIGH yg ternyata native
    summary["top_head_removed_content"] = dict(head_counter.most_common(20))
    # total record union per kategori (lintas split)
    tot = {c: {"q": 0, "a": 0} for c in CATS}
    for s in summary["splits"].values():
        for c in CATS:
            tot[c]["q"] += s[c]["q"]
            tot[c]["a"] += s[c]["a"]
    N = sum(s["n"] for s in summary["splits"].values())
    summary["total"] = {"n": N, **{c: {**tot[c],
        "a_rate_pct": round(100 * tot[c]["a"] / N, 3) if N else 0,
        "q_rate_pct": round(100 * tot[c]["q"] / N, 3) if N else 0} for c in CATS}}

    # --- tulis output ---
    out_json = RESULTS / f"audit_recall_v2_{date}.json"
    out_jsonl = RESULTS / "audit_recall_v2_flagged.jsonl"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in flagged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- stdout tabel (baris=kategori, kolom=split) ---
    print("\n" + "=" * 88)
    print("AUDIT RECALL v2 - angka populasi per kategori (read-only, tidak ada auto-fix)")
    print("=" * 88)
    print(f"Pool 06_scored : {pool_total:,} baris | qsig match: {pool_matched:,}")
    print(f"Align (answer) : attempt={align_attempt:,} | suffix-hit={align_hit:,} "
          f"{dict(align_head_types)}")
    print(f"  -> over-strip KONTEN nyata (head=istilah, bukan nama/sapaan): {align_confirmed:,}")
    print(f"  -> OVER_STRIP_HIGH yg ternyata IDENTIK sumber asli (native, bukan pipeline): "
          f"{src_identical_high:,}")
    cols = ["train", "val", "test", "TOTAL"]
    print("-" * 88)
    print(f"{'kategori':<22}{'fld':<4}" + "".join(f"{c:>15}" for c in cols))
    print(f"{'':22}{'':4}" + "".join(f"{'n (rate%)':>15}" for _ in cols))
    print("-" * 88)
    for c in CATS:
        for fld in ("a", "q"):
            cells = []
            for sp in ("train", "val", "test"):
                s = summary["splits"][sp][c]
                rate = s[f"{fld}_rate_pct"]
                cells.append(f"{s[fld]} ({rate}%)")
            t = summary["total"][c]
            cells.append(f"{t[fld]} ({t[f'{fld}_rate_pct']}%)")
            label = c if fld == "a" else ""
            print(f"{label:<22}{fld:<4}" + "".join(f"{x:>15}" for x in cells))
    print("-" * 88)
    if summary["delta_char"]:
        d = summary["delta_char"]
        print(f"OVER_STRIP delta_char (char head terbuang, align): n={d['n']} "
              f"min={d['min']} median={d['median']} p95={d['p95']} max={d['max']} mean={d['mean']}")
    if summary["top_head_removed_content"]:
        print("Top head-removed KONTEN (istilah medis ke-strip, dari align):")
        line = "   " + " | ".join(f"{k!r}:{v}" for k, v in list(summary["top_head_removed_content"].items())[:20])
        print(line)
    print("-" * 88)
    print(f"Total record ter-flag (>=1 kategori): {len(flagged):,} / {N:,}")
    print("Catatan: OVER_STRIP_HIGH = presisi tinggi (definisi/apositif/align-confirmed);")
    print("         OVER_STRIP_LOW = kandidat lemah (penghubung yg lazim mengawali kalimat ID).")
    print(f"JSON    : {out_json}")
    print(f"FLAGGED : {out_jsonl}")
    print("=" * 88)


if __name__ == "__main__":
    main()
