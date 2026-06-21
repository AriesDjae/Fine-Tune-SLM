"""
strict_clean_v2.py — Pembersihan STRICT v2 (dasar: review manual 30 sampel).

HEMAT KUOTA: TIDAK embedding ulang. Bahan baku = pool yang SUDAH ber-skor (final 36K +
borderline 20K = 56,483 sampel, masing-masing punya komposit v1 pada skala sama). Cleaning
hanya operasi teks → relevansi Q-A tak berubah berarti → re-select pakai komposit v1.

Aturan (catat in/out/dropped + persen tiap aturan):
  B5 emoji/emotikon · B4 sitasi · B6 kata-nyambung (split lower->Upper; buang bila terlalu rusak)
  B3 ekor artikel/promo · B1 nama: regex sign-off "dr. Nama" (segala bentuk, trailing+tengah),
     titled name, leading greeting+nama, trailing bare name + NER (cahya IndoBERT, drop PER residual)
  B2 placeholder "[NAMA]" dibersihkan; PII = HAPUS nama (bukan placeholder)
  B7 pasca-strip: re-cek panjang (G4) + anti-deflection (G9), buang yang jadi pendek
Lalu re-select top 36K by komposit v1 -> re-split 30002/2999/2999 -> cross-split dedup.

Jalankan:
  python preprocessing/strict_clean_v2.py --debug
  python preprocessing/strict_clean_v2.py
CPU-safe, seed=42.
"""
from __future__ import annotations
import argparse, datetime as _dt, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
_np_array_orig = np.array
np.array = lambda *a, **k: _np_array_orig(*a, **{**k, "copy": None}) if k.get("copy") is False else _np_array_orig(*a, **k)

sys.path.insert(0, str(Path(__file__).parent.parent))   # ROOT (utk chat_format)
sys.path.insert(0, str(Path(__file__).parent))           # preprocessing (utk build_id_dataset)
from build_id_dataset import (  # noqa: E402  reuse helper v1
    normalize_text, strip_lead_address, strip_signoff, clean_greeting,
    medical_terms_count, word_count, tag_domain, _minhash, _MULTISP,
    SYSTEM_MSG, NEAR_DUP_TH, MINHASH_PERM, MIN_ANSWER_WORDS, MAX_ANSWER_WORDS,
    MIN_QUESTION_CHARS, _DEFLECTION, _NAMED_GREETING,
)

SEED = 42
random.seed(SEED)
ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "Data" / "processed_id"
INT_DIR = OUT_DIR / "_intermediate"
RESULTS = ROOT / "results"
NER_MODEL = "cahya/bert-base-indonesian-NER"

# ============================================================ B-rules
# B5 emoji + emotikon
_EMOJI = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F0FF"
    "\U0000FE00-\U0000FE0F" "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "\U0000FE0F" "]+")
_EMOTICON = re.compile(r"(?<!\w)(:-?\)|:-?\(|:-?D|;-?\)|=\)|:'\(|:\||:[pP]|<3|:3|\^\^)(?!\w)")
# B4 sitasi sumber lepas
_CITATION = re.compile(r"\s*\((NCBI|WHO|CDC|NIH|PubMed|Kemenkes|WebMD|Mayo\s*Clinic|Medscape|Healthline|IDAI|POGI)\)", re.I)
# B3 ekor artikel / link promosi -> potong sampai akhir
_ARTICLE_TAIL = re.compile(
    r"(?i)\s*("
    r"baca\s+juga|baca\s+selengkapnya|berikut(\s+ini)?\s+(artikel|beberapa\s+artikel|informasi\s+terkait)|"
    r"klik\s+(artikel|link|tautan|di\s*sini)|artikel\s+(terkait|lainnya|berikut)|"
    r"(berikut\s+)?artikel\s+yang\s+dapat\s+(anda|kamu)\s+baca|silakan\s+baca\s+artikel|"
    r"untuk\s+informasi\s+(lebih\s+)?lanjut[^.!?]{0,25}(artikel|baca|link)"
    r").*$")
# B1 dr-name sign-off (TRAILING) — segala bentuk, termasuk tanpa spasi "dr.Theresia".
# Keyword dua-kasus; bagian NAMA wajib KAPITAL sungguhan (tanpa (?i)) agar "dr. spesialis" (kecil) TIDAK ikut.
_DR_TRAIL = re.compile(
    r"[\s.,!?·•\-–—]*\b(?:[Dd][Rr]|[Dd]rg|[Pp]rof|[Ss][Pp])\.?\s*[A-Z][a-zA-Z]*"
    r"(\.?\s*[A-Z][a-zA-Z]*){0,4}(\s*,?\s*[Ss][Pp]\.?[A-Za-z.()\-]*)?\s*$")
# B1 dr-name di TENGAH -> ganti "dokter" (nama wajib kapital)
_DR_MID = re.compile(r"\b(?:[Dd][Rr]|[Dd]rg|[Pp]rof)\.?\s*[A-Z][a-zA-Z]+(\s+[A-Z][a-zA-Z.]+){0,3}")
# B1 titled name -> buang gelar+nama (nama wajib kapital; "ibu hamil"/"mas kawin" TIDAK kena)
_TITLED = re.compile(
    r"\b(?:[Bb]apak|[Ii]bu|[Ss]dr|[Ss]audara|[Ss]audari|[Mm]bak|[Mm]as|[Tt]uan|[Nn]yonya|[Nn]ona)\s+"
    r"[A-Z][a-z]+(\s+[A-Z][a-z]+)?")
# B1 "Dani!" pembuka
_LEAD_BANG = re.compile(r"^\s*([A-Z][a-z]+)\s*!\s*")
# B1 trailing bare name (1-2 kapital setelah akhir kalimat) — true-capital
_TRAIL_NAME = re.compile(r"(?<=[.!?])\s+([A-Z][a-z]+)(\s+[A-Z][a-z]+)?\s*$")
# B-extra: forum header / timestamp bocor (mis. "Irfan Rasta January 17, 2016 at 10:52 am")
_TIMESTAMP = re.compile(
    r"(?i)\b(jan(uari)?|feb(ruari)?|mar(et|ch)?|apr(il)?|may|mei|jun[ei]?|jul[iy]?|"
    r"aug(ust)?|agu(stus)?|sep(tember)?|o[kc]t(ober)?|nov(ember)?|des(ember)?|dec(ember)?)\s+"
    r"\d{1,2},?\s+\d{4}(\s+at\s+\d{1,2}[:.]\d{2}\s*([ap]m)?)?")
_CLOCK = re.compile(r"\b\d{1,2}[:.]\d{2}\s*([ap]m)\b", re.I)
_AT_PREFIX = re.compile(r"^\s*at\s+", re.I)

# B2 placeholder — termasuk intro "Saya/nama saya [NAMA]." dibuang utuh, bukan sisa "Saya ."
_PLACEHOLDER_INTRO = re.compile(r"(?i)\b(nama\s+saya|perkenalkan,?\s*saya|saya)\s*\[\s*nama\s*\]\s*[.,;!]?\s*")
_PLACEHOLDER = re.compile(r"\s*\[\s*nama\s*\]|\s*\{\s*nama\s*\}|\s*<\s*nama\s*>|\s*\[\s*name\s*\]", re.I)
_ORPHAN_PUNCT = re.compile(r"\s+([.,;:!?])")

# B2/B7 PII intro nama di pertanyaan (v1 lewatkan nama huruf-kecil "nama saya widya")
_PII_INTRO_V2 = re.compile(r"(?i)\b(nama\s+saya|perkenalkan,?\s*saya)\s+[A-Za-z][a-zA-Z'’.]+(\s+[A-Za-z][a-z]+){0,2}\s*[.,;]?\s*")
_PII_SAYA_AGE = re.compile(r"(?i)\bsaya\s+([A-Za-z][a-z]{2,})(?=\s+(?:umur|usia|berumur|thn|tahun|th\b))")

_TRAIL_NAME_STOP = {
    "terima", "kasih", "salam", "hormat", "demikian", "semoga", "ya", "amin", "aamiin",
    "sekian", "regards", "thanks", "ok", "oke", "konsultasi", "dokter", "umum",
}

# Nama pasien BOCOR frekuensi-tinggi yang DIKONFIRMASI via inspeksi manual: dipakai dokter
# sbg sapaan ("Ainul telinga berdenging...", "Sebaiknya Ainul segera ke..."), 235 mention,
# 0 di question. TIDAK terdeteksi NER (konteks medis menekan skor) & TIDAK terpisah dari
# istilah medis (Kista/Tahi/Flek) oleh statistik korpus → blocklist kurasi (high-precision).
# Tambah nama lain HANYA setelah konfirmasi manual (jangan tebak — risiko hapus kata medis).
_LEAKED_NAMES = {"ainul"}
_LEAKED_RE = re.compile(r"\b(?:" + "|".join(sorted(_LEAKED_NAMES)) + r")\b", re.I)
_LEAD_PUNCT = re.compile(r"^[\s.,;:!?·•\-–—]+")   # rapikan tanda baca di awal pasca-strip
_REPEAT_PUNCT = re.compile(r"([.,;:!?])(?:\s*[.,;:!?])+")   # ".. " / ". ." -> satu tanda


def fix_glued(t):
    n = len(re.findall(r"[a-z][A-Z]", t))
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    return t, n


def strict_clean_field(text, is_answer):
    """Terapkan B2-B6 + B1(regex) ke satu field. Return (text_bersih, n_glued_boundaries)."""
    t = normalize_text(text)
    t = _LEAKED_RE.sub(" ", t)                            # nama bocor terkonfirmasi (kurasi)
    t = _ANS_GREET.sub("", t)                             # sapaan pembuka "Halo dok," dst
    t = _TIMESTAMP.sub(" ", t); t = _CLOCK.sub(" ", t)    # B-extra: timestamp/forum header
    t = _PLACEHOLDER_INTRO.sub(" ", t)                    # B2 "Saya [NAMA]." -> buang utuh
    t = _PLACEHOLDER.sub(" ", t)                          # B2 sisa "[NAMA]"
    t = _EMOJI.sub("", t); t = _EMOTICON.sub("", t)       # B5
    t = _CITATION.sub("", t)                              # B4
    t, nglue = fix_glued(t)                               # B6
    if is_answer:
        t = _ARTICLE_TAIL.sub("", t)                      # B3 (potong ekor)
    else:
        t = _PII_INTRO_V2.sub(" ", t)                     # B2/B7 "nama saya X" (q, termasuk huruf kecil)
        t = _PII_SAYA_AGE.sub("saya", t)                  # "saya <nama> umur.." -> "saya umur.."
    # B1 (regex)
    t = strip_lead_address(t)                             # sapaan+nama / yang baik / elipsis
    t = _LEAD_BANG.sub("", t)                             # "Dani! "
    t = _TITLED.sub(" ", t)                               # gelar+nama -> hapus
    for _ in range(3):                                    # trailing dr-name (multi)
        t2 = _DR_TRAIL.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    t = _DR_MID.sub("dokter", t)                          # dr-name tengah -> "dokter"
    if is_answer:
        m = _TRAIL_NAME.search(t)                         # trailing bare name
        if m and m.group(1).lower() not in _TRAIL_NAME_STOP:
            t = t[:m.start()].strip()
    t = clean_greeting(t, min_keep=15)
    t = strip_lead_address(t)
    t = _ORPHAN_PUNCT.sub(r"\1", t)                       # rapikan " ." / " ," sisa pembuangan
    t = _REPEAT_PUNCT.sub(r"\1", t)                       # ".. " -> "." (pasca-strip nama)
    t = _LEAD_PUNCT.sub("", t)                            # buang tanda baca di awal (pasca-strip nama)
    return _MULTISP.sub(" ", t).strip(), nglue


# ============================================================ NER (cahya IndoBERT)
_NER = None
_NER_DEVICE = None  # diisi saat pipeline dibuat: 0=GPU, -1=CPU
def ner_device():
    """0 (GPU) bila CUDA tersedia, else -1 (CPU). Aman bila torch CPU-only."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1
def ner_pipe():
    global _NER, _NER_DEVICE
    if _NER is None:
        from transformers import pipeline
        dev = ner_device()
        _NER = pipeline("token-classification", model=NER_MODEL,
                        aggregation_strategy="simple", device=dev)
        _NER_DEVICE = dev
        print(f"[NER] device={'GPU(cuda:0)' if dev == 0 else 'CPU'} model={NER_MODEL}", flush=True)
    return _NER


# cahya NER salah-tag token gaul/sapaan/singkatan ID sbg PER -> stoplist + ambang ketat.
_NER_STOP = {
    "dr", "drg", "dokter", "dok", "bapak", "ibu", "tuhan", "allah", "anda", "saya", "sya",
    "kami", "kita", "ass", "assalamualaikum", "assalamu", "alaikum", "warahmatullahi",
    "wabarakatuh", "wassalam", "wass", "salam", "trims", "trimakasih", "trimakas", "makasih",
    "mksh", "thx", "kmarin", "kemarin", "pagi", "siang", "sore", "malam", "mlm", "bing",
    "gan", "sis", "bro", "min", "kak", "mbak", "mas", "sob", "yth", "mohon", "tolong",
    "maaf", "permisi", "halo", "hai", "oke", "sip", "mas", "bu", "pak", "om", "tante",
    "prim", "wid", "fyi", "btw", "dll", "dsb", "amin", "aamiin", "wr", "wb",
    "maka", "trim", "trimk", "namun", "demikian", "begitu", "soalnya", "intinya",
}
NER_SCORE_TH = 0.90
NER_MIN_LEN = 4


def _ner_persons(ents):
    return [e["word"].strip() for e in ents
            if e["entity_group"] == "PER" and e["score"] >= NER_SCORE_TH
            and e["word"].strip().lower() not in _NER_STOP and len(e["word"].strip()) >= NER_MIN_LEN]


def ner_raw_batch(texts, batch=None):
    """Return list[list[dict]] — entitas MENTAH (dgn start/end/score) per teks."""
    pipe = ner_pipe()
    if batch is None:
        batch = 128 if _NER_DEVICE == 0 else 32   # GPU lebih besar; CPU kecil
    return list(pipe([t[:1000] for t in texts], batch_size=batch))


def ner_persons_batch(texts, batch=None):
    """Return list[list[str]] — entitas PER (terfilter) per teks."""
    return [_ner_persons(ents) for ents in ner_raw_batch(texts, batch)]


def ner_has_person(texts, batch=None):
    return [len(p) > 0 for p in ner_persons_batch(texts, batch)]


# --- leading-name strip (mis. "Ainul" ×205): NER pada teks penuh GAGAL menandai nama di
# awal kalimat (konteks medis menekan skor → kosong). Solusi teruji: NER kata-pertama
# TERISOLASI memisahkan nama (PER>=0.5) dari kata medis (Perlu/Vaksin/Anemia/... = None,
# 0 false-positive pada 24 kata uji). Kandidat = answer diawali "<Nama> <Kapital>".
LEAD_NAME_TH = 0.5
_LEAD_NAME_CAND = re.compile(r"^([A-Z][a-z]{2,11})\s+[A-Z]")   # gate: <Nama> <Kapital...>
# fragmen nama tersisa pasca-strip: "Art, Obat" / "Ctx. Terima" (Kata,/. lalu Kapital).
# Sempit (wajib koma/titik) → tak menyentuh kalimat medis sah "Pemberian vaksin ...".
_RESID_LEAD = re.compile(r"^[A-Z][a-z]{1,11}[,.]\s+[A-Z]")
# kandidat token-1 (gate) + token-2 (utk nama 2-kata "Zie Boenda")
_LEAD_TWO = re.compile(r"^([A-Z][a-z]{2,11})\s+([A-Z][a-zA-Z.]{1,11})")


def build_leadname_set(texts):
    """Kumpulkan 1-2 token kapital terdepan, NER terisolasi, kembalikan set kata = NAMA."""
    cands = set()
    for t in texts:
        m1 = _LEAD_NAME_CAND.match(t)
        if not m1:
            continue
        cands.add(m1.group(1))                       # token-1 (gate)
        m2 = _LEAD_TWO.match(t)
        if m2:                                        # token-2 (nama 2-kata), bila >=2 huruf
            cands.add(m2.group(2).rstrip("."))
    cands = sorted(c for c in cands if c)
    if not cands:
        return set()
    names = set()
    for w, ents in zip(cands, ner_raw_batch(cands)):
        if any(e.get("entity_group") == "PER" and float(e.get("score", 0)) >= LEAD_NAME_TH
               for e in ents):
            names.add(w)
    return names


def strip_leading_name(text, nameset):
    """Buang nama berurutan di awal teks. Gate: token-1 harus nama. Return teks (mungkin sama)."""
    if not _LEAD_NAME_CAND.match(text):
        return text
    toks = text.split()
    n = 0
    while n < 3 and n < len(toks):
        w = toks[n].strip(",.!?:;-")
        if w in nameset:
            n += 1
        else:
            break
    if n == 0:
        return text
    rest = text.split(None, n)[n] if len(text.split(None, n)) > n else ""
    rest = rest.lstrip(" ,.!?:;-–—\t")
    return rest if len(rest) >= 15 else text


def high_person(ents):
    """True bila ada PER kuat tersisa (threshold drop = NER_SCORE_TH, + stoplist/min-len)."""
    return any(e.get("entity_group") == "PER" and float(e.get("score", 0)) >= NER_SCORE_TH
               and e["word"].strip().lower() not in _NER_STOP
               and len(e["word"].strip()) >= NER_MIN_LEN for e in ents)


# ============================================================ echo q≈a (answer = ulangan question)
# Skoring relevansi q-vs-a v1 menilai echo (a≈q) sbg MAKSIMAL relevan → echo ber-skor tinggi
# & lolos ke final. Ini SAMPAH (bukan jawaban dokter) + sering bawa nama pasien. Drop.
_ECHO_GREET = re.compile(r"(?i)^\s*(hai|halo|alo|hallo|hello|hi|hey|pagi|malam|malem|siang|sore|"
                         r"selamat\s+(pagi|siang|sore|malam)|permisi|assalamualaikum|waalaikumsalam)[\s,.!]*")


def _norm_echo(s):
    s = _ECHO_GREET.sub("", s)
    s = re.sub(r"(?i)\bdok(ter)?\b", "", s)
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def is_echo(q, a):
    """True bila answer ~ ulangan question (substring near-prefix setelah normalisasi)."""
    nq = re.sub(r"\s+", " ", _norm_echo(q)).strip()
    na = re.sub(r"\s+", " ", _norm_echo(a)).strip()
    return len(nq) >= 40 and (nq[:80] in na or na[:80] in nq)


# Sapaan pembuka (answer/question): "Halo dok," / "Selamat pagi dokter," / "Assalamualaikum".
# Di-strip (bukan drop) — sapaan dokter sah jadi bersih; utterance pasien lanjut ke filter.
_ANS_GREET = re.compile(
    r"(?i)^\s*(?:(?:hai|halo+|alo|hallo|hello|hi|hey|selamat\s+(?:pagi|siang|sore|malam)|"
    r"assalamu'?alaikum(?:\s+wr\.?\s*wb\.?)?|wa'?alaikum\s*salam|wassalam|permisi|"
    r"pagi|siang|sore|malam|malem)\b[\s,.!]*)+(?:dok(?:ter)?\b[\s,.!]*)*")

# Answer yang sebenarnya UTTERANCE PASIEN (bukan jawaban dokter): "saya" + deskripsi diri
# (mau tanya / umur / usia / wanita / <nama> umur). Sapaan OPSIONAL (krn bisa sudah di-strip).
# Dokter tak memperkenalkan diri dgn umur/gender → FP rendah. Sering bawa nama/umur pasien.
_PATIENT_UTTER = re.compile(
    r"(?i)^\s*(?:(?:hai|halo+|alo|hallo|hello|hi|hey|selamat\s+\w+|assalamu'?alaikum|"
    r"permisi|pagi|siang|sore|malam|malem)\b[\s,.!]*)*(?:dok(?:ter)?\b[\s,.!]*)*"
    r"(?:saya|sya|aku)\s+(?:mau\s+(?:tanya|bertanya|konsul\w*|nanya)|ingin\s+(?:tanya|bertanya|konsul\w*)|"
    r"(?:seorang\s+)?(?:wanita|perempuan|laki|pria|cowok|cewek|ibu|mahasiswa|remaja)|"
    r"(?:ber)?umur\b|(?:ber)?usia\b|[a-z]+\s+(?:ber)?(?:umur|usia)\b)")


def is_patient_utterance(a):
    return bool(_PATIENT_UTTER.match(a))


# ============================================================ load pool
SCORED_POOL = INT_DIR / "06_scored.jsonl"   # full deduped+scored pool (393K, TANPA embedding)


def load_pool():
    """Sumber utama = _intermediate/06_scored.jsonl (393K, sudah ber-komposit, NO re-embed).
    Fallback = final + borderline bila pool penuh tak ada."""
    if SCORED_POOL.exists():
        pool = []
        for o in (json.loads(l) for l in open(SCORED_POOL, encoding="utf-8")):
            pool.append({"q": o["q"], "a": o["a"], "title": o.get("title", ""),
                         "composite": o.get("scores", {}).get("composite", 0.0)})
        return pool, "06_scored(393K full pool, no re-embed)"
    pool = []
    for sp in ("train", "val", "test"):
        for o in (json.loads(l) for l in open(OUT_DIR / f"{sp}.jsonl", encoding="utf-8")):
            pool.append({"q": o["messages"][1]["content"], "a": o["messages"][2]["content"],
                         "title": "", "domain": o.get("domain", "umum"), "composite": o.get("quality_score", 0.0)})
    bl = INT_DIR / "review_borderline.jsonl"
    if bl.exists():
        for o in (json.loads(l) for l in open(bl, encoding="utf-8")):
            pool.append({"q": o["q"], "a": o["a"], "title": "", "domain": o.get("domain", "umum"),
                         "composite": o.get("scores", {}).get("composite", 0.0)})
    return pool, "final+borderline(fallback)"


LOG = {"pipeline": "strict_clean_v2", "date": _dt.datetime.now().isoformat(timespec="seconds"),
       "seed": SEED, "basis": "review manual 30 sampel; reuse pool ber-skor (no re-embed)",
       "rules": []}


def rule(name, before, after, **extra):
    pct = round(100 * (before - after) / before, 2) if before else 0
    rec = {"rule": name, "in": before, "out": after, "dropped": before - after, "dropped_pct": pct}
    rec.update(extra); LOG["rules"].append(rec)
    print(f"[{name}] {before:,} -> {after:,}  (buang {before-after:,} = {pct}%)" + (f"  {extra}" if extra else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--train", type=int, default=30002)
    ap.add_argument("--val", type=int, default=2999)
    ap.add_argument("--test", type=int, default=2999)
    args = ap.parse_args()
    n_tr, n_va, n_te = args.train, args.val, args.test
    if args.debug:
        n_tr, n_va, n_te = 400, 60, 60
    need = n_tr + n_va + n_te

    pool, src = load_pool()
    LOG["pool_source"] = src
    if args.debug:
        pool = sorted(pool, key=lambda r: r["composite"], reverse=True)[:6000]
    rule("load_pool", len(pool), len(pool), source=src)

    # --- B2-B6 + B1(regex) pada SELURUH pool (cheap) ---
    before = len(pool); kept = []; affected = 0; glued_drop = 0
    for r in pool:
        q0, a0 = r["q"], r["a"]
        q, nq = strict_clean_field(q0, is_answer=False)
        a, na = strict_clean_field(a0, is_answer=True)
        if max(nq, na) > 4:                 # B6: terlalu banyak kata-nyambung -> teks rusak
            glued_drop += 1; continue
        if q != q0 or a != a0:
            affected += 1
        r["q"], r["a"] = q, a
        r["domain"] = tag_domain(r.get("title", ""), q)
        kept.append(r)
    rule("B1_B6_clean", before, len(kept), text_modified=affected, glued_too_broken=glued_drop)
    pool = kept

    # --- B7: re-cek panjang + deflection + named-greeting residu ---
    before = len(pool); kept = []; reasons = Counter()
    for r in pool:
        if len(r["q"]) < MIN_QUESTION_CHARS:
            reasons["question_short"] += 1; continue
        wc = word_count(r["a"])
        if wc < MIN_ANSWER_WORDS:
            reasons["answer_short"] += 1; continue
        if wc > MAX_ANSWER_WORDS:
            reasons["answer_long"] += 1; continue
        if wc < 45 and _DEFLECTION.search(r["a"]) and medical_terms_count(r["a"]) < 3:
            reasons["deflection_only"] += 1; continue
        if _NAMED_GREETING.search(r["a"]):
            reasons["named_greeting"] += 1; continue
        kept.append(r)
    rule("B7_length_deflection", before, len(kept), reasons=dict(reasons))
    pool = kept

    # --- echo q≈a + patient-utterance: answer bukan jawaban dokter -> drop ---
    before = len(pool); echo_n = utter_n = 0; kept = []
    for r in pool:
        if is_echo(r["q"], r["a"]):
            echo_n += 1; continue
        if is_patient_utterance(r["a"]):
            utter_n += 1; continue
        kept.append(r)
    rule("echo_qa_dup", before, len(kept), echo=echo_n, patient_utterance=utter_n,
         note="answer = ulangan/utterance pasien (bukan jawaban dokter)")
    pool = kept

    # --- PRE-CAP by komposit sebelum NER (NER ~31 doc/dtk di CPU; batasi set) ---
    precap = need * 2 if args.debug else 44000   # 44K -> NER drop ~11% -> ~39K -> pilih 36K
    if len(pool) > precap:
        pool.sort(key=lambda r: r["composite"], reverse=True)
        rule("precap_for_NER", len(pool), precap, note=f"top-{precap} komposit sebelum NER (efisiensi CPU)")
        pool = pool[:precap]

    # --- B1 leading-name strip ('Ainul'-type): NER first-word terisolasi (Q & A) ---
    before = len(pool)
    print(f"[B1_NER] bangun nameset (NER first-word terisolasi) ...", flush=True)
    nameset = build_leadname_set([r["a"] for r in pool] + [r["q"] for r in pool])
    strip_a = strip_q = 0
    strippeda = [False] * len(pool)
    for idx, r in enumerate(pool):
        a2 = strip_leading_name(r["a"], nameset)
        q2 = strip_leading_name(r["q"], nameset)
        if a2 != r["a"]:
            r["a"] = a2; strip_a += 1; strippeda[idx] = True
        if q2 != r["q"]:
            r["q"] = q2; strip_q += 1
    # re-strip sapaan yg TER-EKSPOS pasca leading-name strip ("Budi Halo dok," -> "Halo dok,")
    for r in pool:
        r["a"] = _LEAD_PUNCT.sub("", _ANS_GREET.sub("", r["a"])).strip()
        r["q"] = _LEAD_PUNCT.sub("", _ANS_GREET.sub("", r["q"])).strip()
    print(f"[B1_NER] leading-name: nameset={len(nameset)} strip a={strip_a:,} q={strip_q:,}", flush=True)

    # --- GERBANG AKHIR (post-semua-modifikasi teks): jamin lolos verifier apa pun quirk hulu.
    #     drop sisa nama kuat NER, dr-name, fragmen, nama bocor, echo, utterance pasien. ---
    print(f"[B1_NER] NER {before:,}x2 field — gerbang akhir ...", flush=True)
    a_ents = ner_raw_batch([r["a"] for r in pool])
    q_ents = ner_raw_batch([r["q"] for r in pool])
    kept = []; drop_a = drop_q = drop_drmid = drop_resid = drop_leak = drop_utter = 0
    for i, r in enumerate(pool):
        if high_person(a_ents[i]):
            drop_a += 1; continue
        if high_person(q_ents[i]):
            drop_q += 1; continue
        if _DR_MID.search(r["a"]) or _DR_MID.search(r["q"]):   # residual "dr. Nama"
            drop_drmid += 1; continue
        if _LEAKED_RE.search(r["a"]) or _LEAKED_RE.search(r["q"]):   # nama bocor tersisa (safety)
            drop_leak += 1; continue
        if is_echo(r["q"], r["a"]) or is_patient_utterance(r["a"]):  # echo/utterance pasca-strip
            drop_utter += 1; continue
        if strippeda[i] and _RESID_LEAD.match(r["a"]):          # fragmen nama tersisa
            drop_resid += 1; continue
        kept.append(r)
    rule("B1_NER_person", before, len(kept), nameset=len(nameset),
         strip_lead_a=strip_a, strip_lead_q=strip_q, dropped_a=drop_a, dropped_q=drop_q,
         dropped_drmid=drop_drmid, dropped_leak=drop_leak, dropped_utter=drop_utter,
         dropped_resid=drop_resid,
         note="leading-name strip + gerbang akhir (PER>=0.90 + drName + leaked + echo/utterance + fragmen)")
    pool = kept

    # --- re-select top-need by komposit v1, stratified split ---
    pool.sort(key=lambda r: r["composite"], reverse=True)
    if len(pool) < need:
        print(f"WARNING: pool {len(pool):,} < target {need:,} — semua dipakai.")
    sel = pool[:need]
    LOG["selection"] = {"clean_pool": len(pool), "selected": len(sel),
                        "composite_cut": sel[-1]["composite"] if sel else None}
    by = defaultdict(list)
    for r in sel:
        by[r["domain"]].append(r)
    rng = random.Random(SEED)
    train, val, test = [], [], []
    for dom, items in by.items():
        rng.shuffle(items); n = len(items)
        nv = round(n * n_va / need); nte = round(n * n_te / need)
        val += items[:nv]; test += items[nv:nv + nte]; train += items[nv + nte:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    # cross-split dedup
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=NEAR_DUP_TH, num_perm=MINHASH_PERM)
    for i, r in enumerate(train):
        lsh.insert(f"tr{i}", _minhash(r["q"]))
    def filt(split, nm):
        out, d = [], 0
        for r in split:
            mh = _minhash(r["q"])
            if lsh.query(mh): d += 1
            else: lsh.insert(f"{nm}{len(out)}", mh); out.append(r)
        return out, d
    val, dv = filt(val, "v"); test, dt = filt(test, "t")
    LOG["cross_split_dedup"] = {"val_removed": dv, "test_removed": dt}
    print(f"[split] train={len(train):,} val={len(val):,} test={len(test):,}  (cross-dedup val-{dv} test-{dt})")

    write_outputs(train, val, test, suffix="_debug" if args.debug else "")
    stamp = _dt.datetime.now().strftime("%Y%m%d")
    suffix = "_debug" if args.debug else ""
    (RESULTS / f"strict_clean_v2_log_{stamp}{suffix}.json").write_text(
        json.dumps(LOG, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLog: {RESULTS/f'strict_clean_v2_log_{stamp}{suffix}.json'}")


def to_final(r):
    return {"messages": [{"role": "system", "content": SYSTEM_MSG},
                         {"role": "user", "content": r["q"]},
                         {"role": "assistant", "content": r["a"]}],
            "domain": r["domain"], "type": "open", "source": "indonesia_qna",
            "source_lang": "id", "translated": False, "quality_score": r["composite"]}


def write_outputs(train, val, test, suffix=""):
    # debug TIDAK menimpa produksi -> tulis ke _intermediate/*_debug.jsonl
    base = INT_DIR if suffix else OUT_DIR
    stats = {"date": LOG["date"], "seed": SEED, "version": "strict_clean_v2",
             "pool_source": LOG.get("pool_source"), "system_message": SYSTEM_MSG, "splits": {}}
    for nm, rows in {"train": train, "val": val, "test": test}.items():
        with open(base / f"{nm}{suffix}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_final(r), ensure_ascii=False) + "\n")
        ac = sorted(len(r["a"]) for r in rows)
        stats["splits"][nm] = {"n": len(rows),
            "domain_distribution": dict(Counter(r["domain"] for r in rows).most_common()),
            "answer_chars": {"min": ac[0], "median": ac[len(ac)//2], "max": ac[-1]} if ac else {}}
    if not suffix:
        (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
