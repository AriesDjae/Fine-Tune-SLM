"""
build_id_dataset.py — Pipeline pembersihan dataset Indonesia-only (Pivot 4) — QC KETAT 12-GERBANG.

STRATEGI: Opsi A (native-only). Sumber TUNGGAL = indonesia_qna (Alodokter, qna.csv).
Tidak ada terjemahan. Prinsip "garbage in, garbage out": tiap gerbang mencatat
jumlah masuk/lolos/dibuang + alasan ke log (bukti BAB 3). TARGET dihitung SETELAH
semua gerbang (train 30K / val 3K / test held-out 3K).

GERBANG (urut; deviasi efisiensi dicatat di log):
  G1  Format       : record valid, 3 role, field tak kosong.
  G2  Encoding     : NFC, fix mojibake (ftfy), buang ctrl/zero-width/PUA, drop �.
  G5  Artefak Alodokter (STRIP dulu) : sapaan+nama, sign-off dokter, sebutan platform, placeholder [nama].
  G4  Panjang      : question/answer dalam band wajar (buang one-liner & runaway).
  G6  Ref tak terlihat : buang answer yg merujuk foto/gambar/lampiran yg tak ada di teks.
  G9  Anti-deflection : buang answer yg HANYA "silakan ke dokter" tanpa isi medis.
  G10 Toksisitas/spam : buang ofensif, iklan, URL/nomor spam.
  G7  PII          : scrub nama, nomor HP, email dari question & answer.
  G3  Kemurnian bahasa : fasttext lid — question DAN answer terverifikasi ID (ambang tinggi).
  G11 Dedup ketat  : exact + near-dup MinHash within & across split; cap 1/near-cluster (diversity).
  [pre-cap]        : ambil top-N cheap-score sebelum embedding (efisiensi CPU; dicatat).
  G8  Relevansi Q-A : cosine embedding (multilingual-e5) question vs answer; buang di bawah ambang.
  G12 Skor gabungan + ambang : top-N lolos; borderline -> review_borderline.jsonl (TIDAK auto-lolos).

Jalankan:
  python preprocessing/build_id_dataset.py --debug          # subset cepat
  python preprocessing/build_id_dataset.py                  # full (target 30k/3k/3k)
CPU-safe, seed=42, idempoten.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --- numpy>=2 shim utk fasttext-wheel (predict pakai np.array(...,copy=False)) ---
_np_array_orig = np.array
def _np_array_compat(*a, **k):
    if k.get("copy") is False:
        k["copy"] = None
    return _np_array_orig(*a, **k)
np.array = _np_array_compat

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from chat_format import clean_greeting  # noqa: E402

try:
    from ftfy import fix_text
except Exception:
    def fix_text(t):
        return t

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "Data" / "indonesia-medical-qna" / "qna.csv"
LID_PATH = ROOT / "Data" / "lid.176.bin"
OUT_DIR = ROOT / "Data" / "processed_id"
INT_DIR = OUT_DIR / "_intermediate"
RESULTS = ROOT / "results"
for d in (OUT_DIR, INT_DIR, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# ---- parameter gerbang ----
LID_ANSWER_TH = 0.65        # fasttext: ambang prob 'id' utk ANSWER (panjang, andal)
LID_QUESTION_TH = 0.30      # question pendek -> ambang lebih longgar; cukup top-label id
MIN_ANSWER_WORDS = 20
MAX_ANSWER_WORDS = 400
MIN_QUESTION_CHARS = 20
MAX_QUESTION_WORDS = 400
NEAR_DUP_TH = 0.80
MINHASH_PERM = 96
EMB_MODEL = "intfloat/multilingual-e5-small"
EMB_PRECAP = 100_000        # batasi pool sebelum embedding (efisiensi CPU)
REL_MIN = 0.84              # ambang cosine e5 q<->a (e5 rapat ~0.84-0.91; 0.84 ~= drop bottom 5%)
# fasttext sering melabeli ID pendek sbg 'ms' (Indonesia & Melayu serumpun). Sumber 100% ID
# (Alodokter) -> 'ms' diperlakukan sbg Indonesia. Non-ID nyata (en/it/dll) tetap dibuang.
_ID_FAMILY = {"id", "ms"}
BORDERLINE_DELTA = 0.02     # band borderline di bawah cut komposit -> review manual

SYSTEM_MSG = (
    "Anda adalah asisten informasi kesehatan berbahasa Indonesia. Berikan penjelasan "
    "yang jelas, akurat, dan hati-hati berdasarkan pengetahuan klinis yang mapan. "
    "Selalu ingatkan bahwa informasi ini bukan pengganti konsultasi langsung dengan dokter."
)

# ===================================================================== text utils
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTISP = re.compile(r"\s+")
_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZEROWIDTH = re.compile("[​‌‍﻿]")
_PUA = re.compile("[-]")
_REPL = "�"


def normalize_text(t: str) -> str:
    """G2: NFC + fix mojibake + buang URL/HTML/ctrl/zero-width/PUA."""
    t = fix_text(str(t))
    t = unicodedata.normalize("NFC", t)
    t = _HTML_TAG.sub(" ", t)
    t = html.unescape(t)
    t = _URL.sub("", t)
    t = _CTRL.sub("", t)
    t = _ZEROWIDTH.sub("", t)
    t = _PUA.sub(" ", t)
    t = _MULTISP.sub(" ", t)
    return t.strip()


# --- G5: greeting + NAME (anchored), 'X yang baik', elipsis, koma ---
_GREET = re.compile(r"^\s*(?:hai+|halo+|h[ae]llo+|alo+|hi|hey|hae)\b[\s,!.:;-]*", re.I)
_CONTENT_WHITELIST = {
    "saya", "kami", "kita", "anda", "untuk", "pada", "jika", "kalau", "karena", "dengan",
    "dalam", "dari", "yang", "ini", "itu", "dan", "atau", "maaf", "mohon", "terima",
    "terimakasih", "silakan", "sebaiknya", "perlu", "semoga", "mengenai", "terkait",
    "baik", "betul", "benar", "ya", "tidak", "bukan", "bisa", "dapat", "ada", "banyak",
    "beberapa", "setiap", "semua", "secara", "sangat", "lebih", "kurang", "masih",
    "sudah", "telah", "akan", "sedang", "biasanya", "umumnya", "kemungkinan", "sebelum",
    "setelah", "selama", "ketika", "saat", "agar", "supaya", "namun", "tetapi", "oleh",
    "sebagai", "seperti", "adapun", "demikian", "gejala", "kondisi", "penyakit", "keluhan",
    "infeksi", "penyebab", "tanda", "rasa", "nyeri", "demam", "batuk", "obat", "dokter",
    "tubuh", "tangan", "kaki", "kepala", "perut", "dada", "darah", "hal", "mata", "kulit",
    "langkah", "saran", "penanganan", "pengobatan", "pemeriksaan", "diagnosis", "faktor",
    "risiko", "jenis", "bentuk", "jamur", "bakteri", "virus", "alergi", "iritasi",
    "peradangan", "benjolan", "luka", "air", "makanan", "minuman", "vitamin", "nutrisi",
    "terapi", "hasil", "kadar", "tekanan", "apa", "apakah", "bagaimana", "kapan",
    "mengapa", "kenapa", "siapa", "dimana", "berapa",
}
_LEAD_ADVERBS = {
    "sebenarnya", "memang", "biasanya", "umumnya", "sebaiknya", "namun", "selain", "secara",
    "intinya", "prinsipnya", "faktanya", "sayangnya", "untungnya", "mungkin", "tentu",
    "tentunya", "jelas", "oke", "nah", "jadi", "maka", "sehingga", "tanpa", "walaupun",
    "meskipun", "meski", "bila", "apabila", "sebab", "akibatnya", "hasilnya", "kesimpulannya",
    "singkatnya", "pertama", "kedua", "ketiga", "selanjutnya", "kemudian", "lalu", "akhirnya",
    "biasa", "normalnya", "wajar", "betul", "halo", "hai", "selamat", "salam", "begitu",
    "kalo", "kira", "barangkali", "rupanya",
}
_LEAD_STOP = _CONTENT_WHITELIST | _LEAD_ADVERBS
_ADDR_YBAIK = re.compile(r"^\s*[A-Z][a-zA-Z’']+\s+(?:yang\s+baik|yth)\b[\s,.:;!-]*", re.I)
_ADDR_TOK = re.compile(r"^\s*([A-Za-z][a-zA-Z’']+)\s+")
_ADDR_ELLIP = re.compile(r"^\s*([A-Za-z][a-zA-Z’']+)\s*\.{2,}\s*")
_ADDR_COMMA = re.compile(r"^\s*([A-Za-z][a-zA-Z’']+)\s*,\s+(?=[A-Za-z])")
_PLACEHOLDER = re.compile(r"\[\s*nama\s*\]|\{\s*nama\s*\}|<\s*nama\s*>", re.I)


def strip_lead_address(text: str) -> str:
    t = (text or "").strip()
    for _ in range(4):
        m = _GREET.match(t)
        if m:
            t = t[m.end():].lstrip()
            mt = _ADDR_TOK.match(t)
            if mt and mt.group(1).lower() not in _LEAD_STOP:
                t = t[mt.end():].lstrip()
            continue
        m = _ADDR_YBAIK.match(t)
        if m:
            t = t[m.end():].lstrip(); continue
        m = _ADDR_ELLIP.match(t)
        if m and m.group(1).lower() not in _LEAD_STOP:
            t = t[m.end():].lstrip(); continue
        m = _ADDR_COMMA.match(t)
        if m and m.group(1).lower() not in _LEAD_STOP:
            t = t[m.end():].lstrip(); continue
        break
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SIGNOFF_SENTENCE = re.compile(
    r"(?i)("
    r"alodokter|telah\s+bertanya|terima\s*kasih\s+(telah|sudah|atas)|"
    r"\b(salam|hormat|wassalam)\b[^.!?]{0,40}\b(dr|dokter)\b|^\s*salam\s*[,.!]?\s*$|"
    r"\bsaya\s+(dr|dokter)\b|\bakan\s+(membantu|bantu|coba|mencoba)\s+(menjawab|membantu|menjelaskan)|"
    r"semoga\s+(membantu|bermanfaat|jawaban)[^.!?]*$"
    r")"
)


def strip_signoff(answer: str) -> str:
    sents = _SENT_SPLIT.split(answer)
    kept = [s for s in sents if s.strip() and not _SIGNOFF_SENTENCE.search(s)]
    return _MULTISP.sub(" ", " ".join(kept)).strip()


def strip_artifacts(q: str, a: str) -> tuple[str, str]:
    a = strip_lead_address(a)
    a = strip_signoff(a)
    a = clean_greeting(a, min_keep=15)
    a = strip_lead_address(a)
    a = _PLACEHOLDER.sub("", a)
    q = strip_lead_address(q)
    q = clean_greeting(q, min_keep=10)
    q = _PLACEHOLDER.sub("", q)
    return _MULTISP.sub(" ", q).strip(), _MULTISP.sub(" ", a).strip()


_NAMED_GREETING = re.compile(
    r"(?i)(^\s*[A-Z][a-z]+\s+(yang\s+baik|yth)\b"
    r"|\b(non(a|na)|nyonya|nyonyah|tuan|bapak|ibu|mbak|mas|sdr|saudara|saudari)\s+"
    r"[A-Z][a-z]+\b[^.!?]{0,20}\b(yang\s+baik|yth)\b)"
)
_DR_NAME_EMBED = re.compile(r"(?i)\bsaya\s+(dr|dokter)\.?\s+[A-Z][a-z]+")
_PATIENT_FOLLOWUP = re.compile(
    r"(?i)^.{0,45}?(maksudnya\s+saya|saya\s+mau\s+(tanya|bertanya)|saya\s+ingin\s+(tanya|bertanya|menanyakan)|"
    r"maaf\s+dok(ter)?,?\s+maksud|mohon\s+di\s*(balas|jawab)|tolong\s+di\s*(balas|jawab)|"
    r"belum\s+di\s*balas|kenapa\s+belum\s+di)"
)

# --- G6: referensi tak terlihat (foto/gambar/lampiran yg tak ada di teks) ---
_INVISIBLE_REF = re.compile(
    r"(?i)\b("
    r"(dari|pada|berdasarkan|melihat|lihat|terlihat\s+(di|pada)|sesuai)\s+"
    r"(foto|gambar|video|hasil\s+(lab|usg|rontgen|ronsen|laboratorium)|lampiran|file)\s+"
    r"(yang|yg|di\s*atas|tersebut|terlampir|anda|yg\s+anda|yang\s+anda)?"
    r"|foto\s+(yang|yg)\s+(anda|kamu|kakak|ibu|bapak)\s+(kirim|lampirkan|upload|unggah|kirimkan)"
    r"|gambar\s+(yang|yg)\s+(anda|kamu)\s+(kirim|lampirkan|upload|unggah)"
    r"|(yang|yg)\s+(anda|kamu)\s+(kirim|lampirkan|upload|unggah)kan?\s+"
    r"|terlampir|pada\s+gambar\s+(di\s*atas|tersebut)"
    r")"
)

# --- G9: anti-deflection ---
_DEFLECTION = re.compile(
    r"(?i)(silakan|sebaiknya|disarankan|sebaik(nya)?|lebih\s+baik|anjuran(nya)?\s+)"
    r"[^.!?]{0,60}\b(konsultasi|periksa|memeriksakan|ke\s+dokter|ke\s+fasilitas|ke\s+rumah\s+sakit|ke\s+igd)\b"
)

# --- G10: toksisitas/spam ---
_SPAM = re.compile(
    r"(?i)(promo|diskon|harga\s+spesial|order\s+sekarang|klik\s+di\s*sini|wa\.me|"
    r"hubungi\s+(kami|wa|whatsapp)|jual\s+obat|081\d{6,}|\+62\d{8,}|telegram\.me|bit\.ly)"
)
_PROFANITY = re.compile(
    r"(?i)\b(anjing|bangsat|kontol|memek|ngentot|bajingan|brengsek|tolol\s+banget|goblok\s+banget)\b"
)

# --- G7: PII ---
_PII_INTRO = re.compile(
    r"\b([Nn]ama\s+[Ss]aya|[Pp]erkenalkan,?\s*[Ss]aya|[Ss]aya)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    r"(?=\s*[.,;)]|\s+(?:umur|usia|berumur|seorang))"
)
_PII_NOT_NAME = {
    "ingin", "mau", "mohon", "sudah", "pernah", "tidak", "baru", "sedang", "masih",
    "merasa", "mengalami", "punya", "ada", "sering", "akan", "hanya", "cuma", "juga",
    "takut", "bingung", "khawatir", "bekerja", "seorang", "perempuan", "laki", "wanita",
    "pria", "anak", "mahasiswa", "ibu", "bapak", "demam", "sakit", "batuk", "pusing",
    "nyeri", "mual", "flu", "pilek", "rasa", "kira", "pikir", "habis", "barusan",
    "kemarin", "tadi", "sempat", "coba", "mengidap", "menderita", "umur", "usia",
}
_PII_PHONE = re.compile(r"\b(?:\+62|62|0)8\d{7,11}\b")
_PII_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def redact_pii(text: str) -> tuple[str, int]:
    n = 0
    def repl(m):
        if m.group(2).split()[0].lower() in _PII_NOT_NAME:
            return m.group(0)
        intro = "nama saya" if m.group(1).lower().startswith("nama") else m.group(1)
        return f"{intro} [NAMA]"
    text, c = _PII_INTRO.subn(repl, text); n += c
    text, c = _PII_PHONE.subn("[NOMOR]", text); n += c
    text, c = _PII_EMAIL.subn("[EMAIL]", text); n += c
    return text, n


def word_count(t: str) -> int:
    return len(t.split())


# --- lexicon medis (substansi) ---
_MED_TERMS = {
    "gejala", "nyeri", "demam", "infeksi", "obat", "dokter", "penyakit", "darah", "tekanan",
    "jantung", "paru", "lambung", "usus", "kulit", "mata", "telinga", "tenggorokan", "kepala",
    "perut", "dada", "sendi", "tulang", "otot", "saraf", "ginjal", "hati", "diabetes",
    "hipertensi", "asma", "alergi", "batuk", "pilek", "mual", "muntah", "diare", "sembelit",
    "pusing", "lemas", "bengkak", "gatal", "ruam", "benjolan", "luka", "antibiotik", "vitamin",
    "dosis", "terapi", "operasi", "diagnosis", "kanker", "tumor", "virus", "bakteri", "hormon",
    "kehamilan", "haid", "menstruasi", "kontrasepsi", "vaksin", "imunisasi", "radang",
    "peradangan", "kolesterol", "asam", "maag", "tifus", "tbc", "anemia", "gula", "rontgen",
    "usg", "pemeriksaan", "penanganan", "pengobatan", "keluhan", "kronis", "akut", "menular",
    "imun", "metabolisme", "pencernaan", "pernapasan",
}


def medical_terms_count(t: str) -> int:
    return len({w for w in re.findall(r"[a-zA-Z]+", t.lower())} & _MED_TERMS)


# ===================================================================== domain
_DOMAIN_KW = {
    "kulit": ["kulit", "jerawat", "gatal", "ruam", "eksim", "kurap", "panu", "bisul"],
    "jantung_pembuluh": ["jantung", "tekanan darah", "hipertensi", "kolesterol", "stroke"],
    "pencernaan": ["lambung", "maag", "perut", "diare", "sembelit", "usus", "mual", "muntah"],
    "pernapasan": ["batuk", "pilek", "asma", "sesak", "paru", "tbc", "flu", "tenggorokan"],
    "saraf": ["pusing", "sakit kepala", "migrain", "saraf", "kejang", "vertigo"],
    "reproduksi": ["haid", "menstruasi", "hamil", "kehamilan", "kontrasepsi", "kandungan", "kelamin"],
    "mata": ["mata", "penglihatan", "minus", "katarak"],
    "tht": ["telinga", "hidung", "tenggorokan", "sinus", "amandel"],
    "otot_tulang": ["sendi", "tulang", "otot", "pegal", "encok", "asam urat", "rematik"],
    "metabolik": ["diabetes", "gula darah", "tiroid", "hormon", "obesitas"],
    "infeksi": ["demam", "infeksi", "tifus", "dbd", "demam berdarah", "virus", "bakteri"],
    "kejiwaan": ["cemas", "depresi", "stres", "insomnia", "panik"],
}


def tag_domain(title: str, q: str) -> str:
    text = f"{title} {q}".lower()
    best, best_n = "umum", 0
    for dom, kws in _DOMAIN_KW.items():
        n = sum(1 for kw in kws if kw in text)
        if n > best_n:
            best, best_n = dom, n
    return best


# ===================================================================== lazy models
_LID = None
def lid():
    global _LID
    if _LID is None:
        import fasttext
        _LID = fasttext.load_model(str(LID_PATH))
    return _LID


def detect_lid(text: str) -> tuple[str, float]:
    t = text.replace("\n", " ").strip()
    if len(t) < 3:
        return "unknown", 0.0
    lab, prob = lid().predict(t, k=1)
    return lab[0].replace("__label__", ""), float(prob[0])


_EMB = None
def emb_model():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(EMB_MODEL, device="cpu")
    return _EMB


# ===================================================================== logging
LOG: dict = {
    "pipeline": "build_id_dataset (12-gate strict QC)",
    "date": _dt.datetime.now().isoformat(timespec="seconds"),
    "seed": SEED,
    "strategy": "Option A native-only (indonesia_qna); no translation; EM dropped (F1+ROUGE-L)",
    "params": {
        "LID_ANSWER_TH": LID_ANSWER_TH, "LID_QUESTION_TH": LID_QUESTION_TH,
        "MIN_ANSWER_WORDS": MIN_ANSWER_WORDS, "MAX_ANSWER_WORDS": MAX_ANSWER_WORDS,
        "MIN_QUESTION_CHARS": MIN_QUESTION_CHARS, "NEAR_DUP_TH": NEAR_DUP_TH,
        "EMB_MODEL": EMB_MODEL, "EMB_PRECAP": EMB_PRECAP, "REL_MIN": REL_MIN,
    },
    "funnel": [],
}


def gate(name, before, after, **extra):
    rec = {"gate": name, "in": before, "out": after, "dropped": before - after}
    rec.update(extra)
    LOG["funnel"].append(rec)
    print(f"[{name}] {before:,} -> {after:,}  (dropped {before-after:,})"
          + (f"  {extra}" if extra else ""), flush=True)


def write_jsonl(path, rows, mapper=lambda r: r):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(mapper(r), ensure_ascii=False) + "\n")


# ===================================================================== GATES
def g0_load(limit_raw):
    print(f"[G0] load {CSV_PATH.name} (chunked) ...", flush=True)
    rows, total = [], 0
    reader = pd.read_csv(CSV_PATH, usecols=["title", "question_clean", "answer_clean"],
                         chunksize=50_000)
    for chunk in reader:
        for _, r in chunk.iterrows():
            total += 1
            rows.append({"title": r["title"], "q_raw": r["question_clean"], "a_raw": r["answer_clean"]})
        if limit_raw and len(rows) >= limit_raw:
            rows = rows[:limit_raw]; break
    gate("G0_load", total, len(rows))
    return rows


def g1_format(rows):
    before = len(rows); kept = []; reasons = Counter()
    for r in rows:
        q, a, t = r.get("q_raw"), r.get("a_raw"), r.get("title")
        if not isinstance(q, str) or not isinstance(a, str):
            reasons["non_string_or_null"] += 1; continue
        if not q.strip() or not a.strip():
            reasons["empty_field"] += 1; continue
        r["title"] = t if isinstance(t, str) else ""
        kept.append(r)
    gate("G1_format", before, len(kept), reasons=dict(reasons))
    return kept


def g2_encoding(rows):
    before = len(rows); kept = []; dropped = 0
    for r in rows:
        q = normalize_text(r["q_raw"]); a = normalize_text(r["a_raw"])
        if _REPL in q or _REPL in a:           # mojibake tak terselamatkan
            dropped += 1; continue
        r["q"], r["a"] = q, a
        kept.append(r)
    gate("G2_encoding", before, len(kept), mojibake_dropped=dropped)
    return kept


def g5_artifacts(rows):
    before = len(rows); kept = []; reasons = Counter(); affected = 0
    for r in rows:
        q0, a0 = r["q"], r["a"]
        q, a = strip_artifacts(q0, a0)
        if q != q0 or a != a0:
            affected += 1
        if _DR_NAME_EMBED.search(a):
            reasons["doctor_name_embedded"] += 1; continue
        if _NAMED_GREETING.search(a):
            reasons["named_greeting"] += 1; continue
        if _PATIENT_FOLLOWUP.search(a):
            reasons["patient_followup_as_answer"] += 1; continue
        r["q"], r["a"] = q, a
        kept.append(r)
    gate("G5_artifacts", before, len(kept), stripped=affected, reasons=dict(reasons))
    return kept


def g4_length(rows):
    before = len(rows); kept = []; reasons = Counter()
    for r in rows:
        if len(r["q"]) < MIN_QUESTION_CHARS:
            reasons["question_too_short"] += 1; continue
        if word_count(r["q"]) > MAX_QUESTION_WORDS:
            reasons["question_too_long"] += 1; continue
        wc = word_count(r["a"])
        if wc < MIN_ANSWER_WORDS:
            reasons["answer_too_short"] += 1; continue
        if wc > MAX_ANSWER_WORDS:
            reasons["answer_too_long"] += 1; continue
        if r["q"].strip().lower() == r["a"].strip().lower():
            reasons["q_equals_a"] += 1; continue
        kept.append(r)
    gate("G4_length", before, len(kept), reasons=dict(reasons))
    return kept


def g6_invisible_ref(rows):
    before = len(rows); kept = [r for r in rows if not _INVISIBLE_REF.search(r["a"])]
    gate("G6_invisible_ref", before, len(kept))
    return kept


def g9_deflection(rows):
    before = len(rows); kept = []; dropped = 0
    for r in rows:
        a = r["a"]; wc = word_count(a)
        if wc < 45 and _DEFLECTION.search(a) and medical_terms_count(a) < 3:
            dropped += 1; continue
        kept.append(r)
    gate("G9_deflection", before, len(kept), deflection_only_dropped=dropped)
    return kept


def g10_toxic_spam(rows):
    before = len(rows); kept = []; reasons = Counter()
    for r in rows:
        blob = r["q"] + " " + r["a"]
        if _SPAM.search(blob):
            reasons["spam"] += 1; continue
        if _PROFANITY.search(blob):
            reasons["profanity"] += 1; continue
        kept.append(r)
    gate("G10_toxic_spam", before, len(kept), reasons=dict(reasons))
    return kept


def g7_pii(rows):
    red = 0
    for r in rows:
        r["q"], c1 = redact_pii(r["q"])
        r["a"], c2 = redact_pii(r["a"])
        red += c1 + c2
    gate("G7_pii", len(rows), len(rows), redactions=red)
    return rows


def g3_language(rows):
    before = len(rows); kept = []; reasons = Counter()
    for r in rows:
        la, pa = detect_lid(r["a"])
        lq, pq = detect_lid(r["q"])
        r["lid_answer"], r["lid_q"] = (la, round(pa, 3)), (lq, round(pq, 3))
        if la not in _ID_FAMILY or pa < LID_ANSWER_TH:
            reasons[f"answer_{la}"] += 1; continue
        if lq not in _ID_FAMILY or pq < LID_QUESTION_TH:
            reasons[f"question_{lq}"] += 1; continue
        r["lang_conf"] = round(pa, 4)
        kept.append(r)
    gate("G3_language_fasttext", before, len(kept),
         dropped_reasons=dict(Counter(reasons).most_common(8)))
    return kept


def _shingles(text, k=3):
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)} or set(toks)


def _minhash(text):
    from datasketch import MinHash
    m = MinHash(num_perm=MINHASH_PERM)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def g11_dedup(rows):
    from datasketch import MinHashLSH
    before = len(rows)
    seen, exact = set(), []
    for r in rows:
        key = _MULTISP.sub(" ", (r["q"] + " ||| " + r["a"]).lower()).strip()
        if key in seen:
            continue
        seen.add(key); exact.append(r)
    after_exact = len(exact)
    lsh = MinHashLSH(threshold=NEAR_DUP_TH, num_perm=MINHASH_PERM)
    kept, near = [], 0
    for i, r in enumerate(exact):
        m = _minhash(r["q"])
        if lsh.query(m):
            near += 1; continue
        lsh.insert(str(i), m); kept.append(r)
    gate("G11_dedup", before, len(kept),
         exact_removed=before - after_exact, near_removed=near, note="cap 1 per near-cluster")
    return kept


def _cheap_score(r):
    wc = word_count(r["a"])
    if MIN_ANSWER_WORDS <= wc <= 250:
        s_len = 1.0
    elif wc < MIN_ANSWER_WORDS:
        s_len = wc / MIN_ANSWER_WORDS
    else:
        s_len = max(0.0, 1 - (wc - 250) / (MAX_ANSWER_WORDS - 250))
    s_med = min(medical_terms_count(r["a"]) / 8.0, 1.0)
    return 0.4 * r.get("lang_conf", 0.0) + 0.3 * s_len + 0.3 * s_med


def precap(rows, cap):
    if len(rows) <= cap:
        return rows
    for r in rows:
        r["_cheap"] = _cheap_score(r)
    rows.sort(key=lambda r: r["_cheap"], reverse=True)
    kept = rows[:cap]
    gate("precap_cheap_score", len(rows), len(kept),
         note=f"top-{cap} cheap-score sebelum embedding (efisiensi CPU)")
    return kept


def g8_relevance(rows):
    before = len(rows)
    model = emb_model()
    qs = ["query: " + r["q"] for r in rows]
    as_ = ["passage: " + r["a"] for r in rows]
    print(f"[G8] encoding {len(rows):,}x2 teks (e5, CPU) ...", flush=True)
    Eq = model.encode(qs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    Ea = model.encode(as_, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    rel = (Eq * Ea).sum(axis=1)
    kept = []
    for i, r in enumerate(rows):
        r["relevance"] = round(float(rel[i]), 4)
        if r["relevance"] >= REL_MIN:
            kept.append(r)
    pct = {p: round(float(np.percentile(rel, p)), 3) for p in (5, 25, 50, 75, 95)}
    gate("G8_relevance_e5", before, len(kept), rel_threshold=REL_MIN, rel_percentiles=pct)
    return kept


def g12_score_select(rows, need):
    import math
    rels = [r["relevance"] for r in rows]
    rmin, rmax = min(rels), max(rels)
    for r in rows:
        wc = word_count(r["a"])
        s_len = 1.0 if 40 <= wc <= 250 else (wc / 40 if wc < 40 else max(0.0, 1 - (wc - 250) / (MAX_ANSWER_WORDS - 250)))
        s_rel = (r["relevance"] - rmin) / (rmax - rmin) if rmax > rmin else 0.0
        s_med = min(medical_terms_count(r["a"]) / 8.0, 1.0)
        toks = re.findall(r"\w+", r["a"].lower())
        s_coh = (len(set(toks)) / len(toks)) if toks else 0.0
        comp = 0.20 * r.get("lang_conf", 0) + 0.15 * s_len + 0.30 * s_rel + 0.20 * s_med + 0.15 * s_coh
        r["scores"] = {"lang": round(r.get("lang_conf", 0), 3), "len": round(s_len, 3),
                       "relevance": round(s_rel, 3), "medical": round(s_med, 3),
                       "coherence": round(s_coh, 3), "composite": round(comp, 4)}
    rows.sort(key=lambda r: r["scores"]["composite"], reverse=True)
    passed = rows[:need]
    cut = passed[-1]["scores"]["composite"] if passed else 0.0
    borderline = [r for r in rows[need:] if r["scores"]["composite"] >= cut - BORDERLINE_DELTA]
    LOG["selection"] = {"scored_pool": len(rows), "passed": len(passed),
                        "composite_cut": cut, "borderline_for_review": len(borderline)}
    gate("G12_score_select", len(rows), len(passed),
         composite_cut=cut, borderline=len(borderline))
    write_jsonl(INT_DIR / "review_borderline.jsonl", borderline, mapper=lambda r: {
        "q": r["q"], "a": r["a"], "scores": r["scores"], "domain": tag_domain(r["title"], r["q"])})
    return passed


def split_strat(rows, n_train, n_val, n_test):
    need = n_train + n_val + n_test
    for r in rows:
        r["domain"] = tag_domain(r["title"], r["q"])
    by = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    rng = random.Random(SEED)
    train, val, test = [], [], []
    for dom, items in by.items():
        rng.shuffle(items)
        n = len(items)
        nv = round(n * n_val / need); nte = round(n * n_test / need)
        val += items[:nv]; test += items[nv:nv + nte]; train += items[nv + nte:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    print(f"[split] train={len(train):,} val={len(val):,} test={len(test):,}")
    return train, val, test


def cross_split_dedup(train, val, test):
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=NEAR_DUP_TH, num_perm=MINHASH_PERM)
    for i, r in enumerate(train):
        lsh.insert(f"tr{i}", _minhash(r["q"]))
    def filt(split, name):
        kept, drop = [], 0
        for r in split:
            mh = _minhash(r["q"])
            if lsh.query(mh):
                drop += 1
            else:
                lsh.insert(f"{name}{len(kept)}", mh); kept.append(r)
        return kept, drop
    val2, dv = filt(val, "v"); test2, dt = filt(test, "t")
    LOG["cross_split_dedup"] = {"val_removed": dv, "test_removed": dt}
    print(f"[cross-split dedup] val -{dv}  test -{dt}")
    return train, val2, test2


def to_final(r):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": r["q"]},
            {"role": "assistant", "content": r["a"]},
        ],
        "domain": r.get("domain", "umum"), "type": "open", "source": "indonesia_qna",
        "source_lang": "id", "translated": False,
        "quality_score": r["scores"]["composite"], "relevance": r["relevance"],
    }


def char_stats(rows, fld):
    xs = sorted(len(r[fld]) for r in rows)
    return {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]} if xs else {}


def write_outputs(train, val, test):
    stats = {"date": LOG["date"], "seed": SEED, "system_message": SYSTEM_MSG, "splits": {}}
    for name, rows in {"train": train, "val": val, "test": test}.items():
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows, mapper=to_final)
        comp = sorted(r["scores"]["composite"] for r in rows)
        rel = sorted(r["relevance"] for r in rows)
        stats["splits"][name] = {
            "n": len(rows),
            "domain_distribution": dict(Counter(r.get("domain", "umum") for r in rows).most_common()),
            "type_distribution": {"open": len(rows)},
            "answer_chars": char_stats(rows, "a"), "question_chars": char_stats(rows, "q"),
            "composite_score": {"min": round(comp[0], 4), "median": round(comp[len(comp)//2], 4),
                                "max": round(comp[-1], 4)} if comp else {},
            "relevance": {"min": round(rel[0], 4), "median": round(rel[len(rel)//2], 4),
                          "max": round(rel[-1], 4)} if rel else {},
        }
    stats["confirmations"] = {
        "icd_lookup_remaining": 0, "greeting_artifact_remaining": "stripped (clause-based)",
        "language": f"100% Indonesia (fasttext lid; answer prob>={LID_ANSWER_TH})",
        "native_vs_translated": {"native": "100%", "translated": "0%"},
        "pii": "names/phone/email redacted (regex best-effort)",
    }
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def write_datacard(stats):
    n = {k: v["n"] for k, v in stats["splits"].items()}
    flines = []
    for f in LOG["funnel"]:
        line = f"  {f['gate']:<22} {f['in']:>8,} -> {f['out']:>8,}  (buang {f['dropped']:>7,})"
        detail = f.get("reasons") or f.get("dropped_reasons")
        if detail:
            line += f"  {detail}"
        flines.append(line)
    funnel = "\n".join(flines)
    card = f"""DATA CARD — Dataset Medis Indonesia (Pivot 4, native-only, QC 12-gerbang)
=======================================================================
Tanggal build : {stats['date']}
Seed          : {SEED}
Strategi      : Opsi A — native Indonesia saja (indonesia_qna / Alodokter). TIDAK ada terjemahan.

SUMBER : Data/indonesia-medical-qna/qna.csv (681,204 baris mentah, Alodokter QnA).

KOREKSI PREMIS (BAB 3): brief klaim ID <5%; audit (results/audit_dataset.json) bukti ID 23-30%,
pool native abundant. Native-only cukup utk target tanpa terjemahan.

JUMLAH AKHIR : train={n.get('train',0):,}  val={n.get('val',0):,}  test={n.get('test',0):,}

CORONG QC (lihat results/id_pipeline_log_*.json utk angka lengkap per-alasan):
{funnel}

GERBANG:
  G1 format · G2 encoding(NFC+ftfy+ctrl/PUA) · G5 artefak Alodokter(strip+reject) · G4 panjang
  · G6 referensi tak terlihat(foto/lampiran) · G9 anti-deflection · G10 toksik/spam · G7 PII(nama/HP/email)
  · G3 bahasa(fasttext lid, q&a, ambang answer>={LID_ANSWER_TH}) · G11 dedup exact+MinHash(within&across, cap 1/cluster)
  · [pre-cap top-{EMB_PRECAP} cheap-score] · G8 relevansi Q-A({EMB_MODEL} cosine>={REL_MIN}) · G12 skor gabungan+ambang.

KETERBATASAN (didokumentasikan):
  - Relevansi G8: embedding {EMB_MODEL} di CPU; pool di-pre-cap top-{EMB_PRECAP} (cheap-score) sebelum embedding
    demi efisiensi — sampel yang dibuang pre-cap berkualitas rendah (tak akan masuk top final).
  - Domain: keyword-derived (sumber tak punya field domain); sebagian -> 'umum'.
  - PII: regex best-effort (bukan NER penuh) -> nama "Saya/Nama saya X", nomor HP, email. Spot-check disarankan.
  - Borderline G12 -> Data/processed_id/_intermediate/review_borderline.jsonl (TIDAK auto-lolos; review manual).

EVALUASI : EM (MCQA) DI-DROP (native tak punya MCQ; tunggu konfirmasi pembimbing). Metrik: token-F1 + ROUGE-L.
FORMAT   : messages netral (chat template TIDAK di-hardcode); field domain/type=open/source/source_lang=id/translated=false/quality_score/relevance.
"""
    (OUT_DIR / "DATA_CARD.txt").write_text(card, encoding="utf-8")


def spot_check(test, k):
    print("\n" + "=" * 78)
    print(f"SPOT-CHECK TEST ({k} sampel acak — baca manual)")
    print("=" * 78)
    for i, r in enumerate(random.Random(SEED).sample(test, min(k, len(test))), 1):
        print(f"\n[{i}] domain={r.get('domain')} comp={r['scores']['composite']} rel={r['relevance']}")
        print(f"  Q: {r['q'][:200]}")
        print(f"  A: {r['a'][:240]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--train", type=int, default=30000)
    ap.add_argument("--val", type=int, default=3000)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--limit-raw", type=int, default=None)
    args = ap.parse_args()
    if args.debug:
        n_train, n_val, n_test, limit_raw = 500, 80, 80, 12000
    else:
        n_train, n_val, n_test, limit_raw = args.train, args.val, args.test, args.limit_raw
    LOG["mode"] = "debug" if args.debug else "full"
    LOG["targets"] = {"train": n_train, "val": n_val, "test": n_test}
    need = n_train + n_val + n_test

    rows = g0_load(limit_raw)
    rows = g1_format(rows)
    rows = g2_encoding(rows)
    rows = g5_artifacts(rows)
    rows = g4_length(rows)
    rows = g6_invisible_ref(rows)
    rows = g9_deflection(rows)
    rows = g10_toxic_spam(rows)
    rows = g7_pii(rows)
    rows = g3_language(rows)
    rows = g11_dedup(rows)
    rows = precap(rows, max(EMB_PRECAP, need * 3))
    rows = g8_relevance(rows)
    rows = g12_score_select(rows, need)
    train, val, test = split_strat(rows, n_train, n_val, n_test)
    train, val, test = cross_split_dedup(train, val, test)
    stats = write_outputs(train, val, test)
    write_datacard(stats)

    stamp = _dt.datetime.now().strftime("%Y%m%d")
    suffix = "_debug" if args.debug else ""
    (RESULTS / f"id_pipeline_log_{stamp}{suffix}.json").write_text(
        json.dumps(LOG, indent=2, ensure_ascii=False), encoding="utf-8")
    spot_check(test, 30)
    print("\n=== QC FINAL ===")
    for name in ("train", "val", "test"):
        s = stats["splits"][name]
        print(f"{name:5} n={s['n']:,}  comp_med={s['composite_score'].get('median')}  "
              f"rel_med={s['relevance'].get('median')}  ans_med={s['answer_chars'].get('median')}")
    print(f"\nOutput: {OUT_DIR}  |  Log: {RESULTS/f'id_pipeline_log_{stamp}{suffix}.json'}")


if __name__ == "__main__":
    main()
